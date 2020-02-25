from pathlib import Path


import sentry_dramatiq
import sentry_sdk
from starlette.config import Config


config = Config(".env")

SENTRY_DSN = config("SENTRY_DSN", default=None)
PROXY = config("PROXY", default="/erebos")
DASHBOARD_PATH = config("DASHBOARD_PATH", default="/erebos/drama")
LOG_LEVEL = config("LOG_LEVEL", default="INFO")
ZARR_DIR = config("ZARR_DIR", cast=Path, default="/d4/uaren/goes/G16/erebos/zarr/")
S3_PREFIX = config("S3_PREFIX", default="ABI-L2-MCMIPC")
MULTI_DIR = config(
    "MULTI_DIR", cast=Path, default="/d4/uaren/goes/G16/multichannel/1km"
)
REDIS_HOST = config("REDIS_HOST", default="localhost")
REDIS_PORT = config("REDIS_PORT", default=6379)
MEM_LIMIT = config("RSS_MEMORY_LIMIT", default=384.0, cast=float)


if SENTRY_DSN is not None:
    sentry_sdk.init(SENTRY_DSN, integrations=[sentry_dramatiq.DramatiqIntegration()])
