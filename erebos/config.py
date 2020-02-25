from pathlib import Path


from starlette.config import Config


config = Config(".env")

PROXY = config("PROXY", default="/erebos")
LOG_LEVEL = config("LOG_LEVEL", default="INFO")
ZARR_DIR = config("ZARR_DIR", cast=Path, default="/d4/uaren/goes/G16/erebos/zarr/")
S3_PREFIX = config("S3_PREFIX", default="ABI-L2-MCMIPC")
MULTI_DIR = config(
    "MULTI_DIR", cast=Path, default="/d4/uaren/goes/G16/multichannel/1km"
)
REDIS_HOST = config("REDIS_HOST", default="localhost")
REDIS_PORT = config("REDIS_PORT", default=6379)
MEM_LIMIT = config('RSS_MEMORY_LIMIT', default=384.0, cast=float)
