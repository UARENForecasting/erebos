import datetime as dt
from decimal import Decimal
import logging
from typing import Optional, Union, Dict


from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, FilePath, Json
from starlette.requests import Request
import xarray as xr


from erebos import __version__, tasks, config
import erebos.adapters  # NOQA


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


app = FastAPI()
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


@subapi.get("/series/{variable}", response_model=SeriesResponse)
def get_series(variable: str, run_date: str, lon: float, lat: float):
    run_date = dt.date.fromisoformat(run_date)
    path = config.ZARR_DIR / run_date.strftime("%Y/%m/%d")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for {run_date}")
    zds = xr.open_zarr(str(path), consolidated=True)
    if variable not in zds:
        raise HTTPException(status_code=404, detail=f"No variable {variable} in file")
    data = zds.erebos.select_nearest(lon, lat)[variable].isel(z=0).load()
    ser = data.to_dataframe()[variable].tz_localize("UTC").sort_index()
    ser = ser.round(decimals=1).fillna(-999)
    out = {
        "variable": variable,
        "run_date": run_date,
        "lat": lat,
        "lon": lon,
        "results": ser,
    }
    return out


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
    Signature: str
    SignatureVersion: str
    SigningCertURL: str
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
