import datetime as dt
import logging
from pathlib import Path
from typing import Optional, Union


from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, FilePath, Json
import requests
from starlette.config import Config
from starlette.requests import Request
import xarray as xr


import erebos.adapters  # NOQA
from erebos.ml_models import predict
from erebos.custom_multichannel_generation import generate_combined_file


app = FastAPI()
logger = logging.getLogger(__name__)
config = Config(".env")
ZARR_DIR = config("ZARR_DIR", cast=Path, default="/d4/uaren/goes/G16/erebos/zarr/")
S3_PREFIX = config("S3_PREFIX", default="ABI-L2-MCMIPC")
MULTI_DIR = config(
    "MULTI_DIR", cast=Path, default="/d4/uaren/goes/G16/multichannel/1km"
)
logger.setLevel(config("LOG_LEVEL", default="INFO"))


@app.get("/series/{variable}")
def get_series(variable: str, run_date: str, lon: float, lat: float):
    run_date = dt.date.fromisoformat(run_date)
    path = ZARR_DIR / run_date.strftime("%Y/%m/%d")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for {run_date}")
    zds = xr.open_zarr(str(path), consolidated=True)
    if variable not in zds:
        raise HTTPException(status_code=404, detail=f"No variable {variable} in file")
    data = zds.erebos.select_nearest(lon, lat)[variable].isel(z=0).load()
    df = data.to_dataframe()[[variable]].tz_localize("UTC")
    df.index.name = "time"
    return df.to_dict()


class NewFile(BaseModel):
    path: FilePath


@app.post("/process/ghiprediction")
def process_combined_file(newfile: NewFile, request: Request):
    predict.full_prediction(newfile.path, zarr_dir=ZARR_DIR)


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
    headers = {"content-type": "application/json"}
    if "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]
    requests.post(
        request.url_for("process_combined_file"),
        json={"path": str(final_path)},
        headers=headers,
    )


@app.post("/process/s3file")
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
