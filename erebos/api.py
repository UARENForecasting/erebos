import asyncio
import datetime as dt
from decimal import Decimal
import logging
import os
from typing import Optional, Union, Dict


from fastapi import FastAPI, HTTPException, BackgroundTasks
import pandas as pd
import psutil
from pvlib import clearsky, solarposition
from pydantic import BaseModel, FilePath, Json
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from starlette.requests import Request
from starlette_prometheus import PrometheusMiddleware, metrics
import xarray as xr
from uvicorn.main import Server as UvicornServer
from uvicorn.workers import UvicornWorker


from erebos import __version__, tasks, config
import erebos.adapters  # NOQA


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


app = FastAPI()
SentryAsgiMiddleware(app)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics/", metrics)
subapi = FastAPI(openapi_prefix=config.PROXY)


@app.get("/ping")
async def ping(request: Request):
    return "pong"


class SeriesResponse(BaseModel):
    version: str = __version__
    lon: float
    lat: float
    run_date: dt.date
    variable: str
    nan_value: int = -999
    results: Dict[dt.datetime, Decimal]


def _open_zds(run_date: dt.date):
    path = config.ZARR_DIR / run_date.strftime("%Y/%m/%d")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for {run_date}")
    zds = xr.open_zarr(str(path), consolidated=True)
    return zds


def _process_ser(ser, variable, run_date, lat, lon):
    ser = ser.tz_localize("UTC").sort_index().round(decimals=1).fillna(-999)
    ser = ser[~ser.index.duplicated()]
    out = {
        "variable": variable,
        "run_date": run_date,
        "lat": lat,
        "lon": lon,
        "results": ser,
    }
    return out


@subapi.get("/series/adjghi", response_model=SeriesResponse)
def get_series(
    run_date: dt.date,
    lon: float,
    lat: float,
    precipitable_water: float = 1.0,
    aod700: float = 0.05,
):
    # aod, pw based on 2019 at Tucson aeronet (not monsoon)
    zds = _open_zds(run_date)
    variables = ["ghi", "cloud_mask"]
    for variable in variables:
        if variable not in zds:
            raise HTTPException(
                status_code=404, detail=f"No variable {variable} in file"
            )
    data = zds.erebos.select_nearest(lon, lat)[variables].isel(z=0).load()
    df = data.to_dataframe()[variables]
    solpos = solarposition.get_solarposition(df.index, lat, lon)
    clr = clearsky.simplified_solis(
        solpos["elevation"], aod700=aod700, precipitable_water=precipitable_water
    )
    ser = df["ghi"] * df["cloud_mask"] + clr["ghi"] * (1 - df["cloud_mask"])
    # clip to up to 15% over solis clearsky
    ser = ser.clip(upper=clr["ghi"] * 1.15)
    return _process_ser(ser, "clearsky_adjusted_ghi", run_date, lat, lon)


@subapi.get("/series/{variable}", response_model=SeriesResponse)
def get_series(variable: str, run_date: dt.date, lon: float, lat: float):
    zds = _open_zds(run_date)
    if variable not in zds:
        raise HTTPException(status_code=404, detail=f"No variable {variable} in file")
    data = zds.erebos.select_nearest(lon, lat)[variable].isel(z=0).load()
    ser = data.to_dataframe()[variable]
    return _process_ser(ser, variable, run_date, lat, lon)


@subapi.get("/lastupdate")
def get_last_update(run_date: dt.date):
    zds = _open_zds(run_date)
    return pd.Timestamp(zds.t.max().values, tz=zds.timezone)


class NewFile(BaseModel):
    path: FilePath


@subapi.post("/process/ghiprediction")
def process_combined_file(
    newfile: NewFile, request: Request, background_tasks: BackgroundTasks
):
    tasks.process_combined_file.send(str(newfile.path))


class SNSMessage(BaseModel):
    Message: Union[Json, str]
    MessageId: str
    Signature: Optional[str]
    SignatureVersion: Optional[str]
    SigningCertURL: Optional[str]
    Subject: Optional[str]
    SubscribeURL: Optional[str]
    Timestamp: dt.datetime
    Token: Optional[str]
    TopicArn: str
    Type: str
    UnsubscribeURL: Optional[str]


@subapi.post("/process/s3file")
def process_s3_file(
    sns_message: SNSMessage, request: Request, background_tasks: BackgroundTasks
):
    rec = sns_message.Message
    logger.debug("SNS Message is: %s", rec)
    for record in rec["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if key.startswith(config.S3_PREFIX):
            tasks.generate_combined_file.send(key, bucket)


app.mount(config.PROXY, subapi)


class Server(UvicornServer):
    async def on_tick(self, counter):
        if counter % 100 == 0 and counter > 0:  # once every 10 seconds
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            for child in proc.children():
                rss += child.memory_info().rss
            rss_mb = rss / 1024 ** 2
            logger.debug("Process group currently using %0.2f MiB RSS", rss_mb)
            if rss_mb > config.MEM_LIMIT:
                logger.info("Restarting workers with RSS of %0.2f MiB RSS...", rss_mb)
                self.should_exit = True
        return await super().on_tick(counter)


class Worker(UvicornWorker):
    def run(self):
        self.config.app = self.wsgi
        server = Server(config=self.config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server.serve(sockets=self.sockets))
