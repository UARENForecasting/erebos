import datetime as dt
from decimal import Decimal
import logging
from pathlib import Path
from typing import Optional, Union, Dict


from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, FilePath, Json
import requests
from starlette.config import Config
from starlette.requests import Request
import xarray as xr


from erebos import __version__
import erebos.adapters  # NOQA
from erebos.ml_models import predict
from erebos.custom_multichannel_generation import generate_combined_file

config = Config(".env")
sub = config("PROXY", default="/erebos")
logger = logging.getLogger(__name__)
ZARR_DIR = config("ZARR_DIR", cast=Path, default="/d4/uaren/goes/G16/erebos/zarr/")
S3_PREFIX = config("S3_PREFIX", default="ABI-L2-MCMIPC")
MULTI_DIR = config(
    "MULTI_DIR", cast=Path, default="/d4/uaren/goes/G16/multichannel/1km"
)
logger.setLevel(config("LOG_LEVEL", default="INFO"))


app = FastAPI()
subapi = FastAPI(openapi_prefix=sub)


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
    path = ZARR_DIR / run_date.strftime("%Y/%m/%d")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for {run_date}")
    zds = xr.open_zarr(str(path), consolidated=True)
    if variable not in zds:
        raise HTTPException(status_code=404, detail=f"No variable {variable} in file")
    data = zds.erebos.select_nearest(lon, lat)[variable].isel(z=0).load()
    ser = data.to_dataframe()[variable].tz_localize("UTC").sort_index()
    ser.iloc[-1] = None
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
    background_tasks.add_task(predict.full_prediction, newfile.path, zarr_dir=ZARR_DIR)


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


def _generate_combined(key, bucket, request):
    final_path = generate_combined_file(key, MULTI_DIR, bucket, overwrite=False)
    if final_path is None:
        return
    headers = {"content-type": "application/json"}
    if "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]
    url = request.url_for("process_combined_file")
    logger.info("Posting %s to %s", final_path, url)
    requests.post(url, json={"path": str(final_path)}, headers=headers)


@subapi.post("/process/s3file")
def process_s3_file(
    sns_message: SNSMessage, request: Request, background_tasks: BackgroundTasks
):

    if sns_message.Type == "SubscriptionConfirmation":
        requests.get(sns_message.SubscribeURL)
        return

    rec = sns_message.Message
    logger.debug("SNS Message is: %s", rec)
    for record in rec["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if key.startswith(S3_PREFIX):
            background_tasks.add_task(_generate_combined, key, bucket, request)


app.mount(sub, subapi)
