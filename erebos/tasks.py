import logging
import os
from pathlib import Path
import psutil
import signal


import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from starlette.config import Config


from erebos import custom_multichannel_generation
from erebos.ml_models import predict


config = Config(".env")
rabbitmq_broker = RabbitmqBroker(
    host=config("RABBITMQ_HOST", default="localhost"),
    port=config("RABBITMQ_PORT", default=5672),
)


class RestartMiddleware(dramatiq.Middleware):
    def __init__(self, memory_limit):
        self.logger = dramatiq.logging.get_logger(__name__, type(self))
        self.memory_limit = memory_limit

    def after_process_message(self, broker, message, *, result=None, exception=None):
        proc = psutil.Process(os.getppid())
        rss = proc.memory_info().rss
        for child in proc.children():
            rss += child.memory_info().rss
        rss_mb = rss / 1024**2
        self.logger.debug('Process group currently using %0.2f MiB RSS', rss_mb)
        if rss_mb > self.memory_limit:
            self.logger.info('Restarting workers with RSS of %0.2f MiB RSS...', rss_mb)
            os.kill(os.getppid(), signal.SIGHUP)


MEM_LIMIT = config('RSS_MEMORY_LIMIT', default=384.0, cast=float)
rabbitmq_broker.add_middleware(RestartMiddleware(MEM_LIMIT))
dramatiq.set_broker(rabbitmq_broker)


logger = logging.getLogger(__name__)
ZARR_DIR = config("ZARR_DIR", cast=Path, default="/d4/uaren/goes/G16/erebos/zarr/")
S3_PREFIX = config("S3_PREFIX", default="ABI-L2-MCMIPC")
MULTI_DIR = config(
    "MULTI_DIR", cast=Path, default="/d4/uaren/goes/G16/multichannel/1km"
)
logger.setLevel(config("LOG_LEVEL", default="INFO"))


@dramatiq.actor
def process_combined_file(combined_file_path):
    predict.full_prediction(Path(combined_file_path), zarr_dir=ZARR_DIR)


@dramatiq.actor
def generate_combined_file(key, bucket):
    final_path = custom_multichannel_generation.generate_combined_file(
        key, MULTI_DIR, bucket, overwrite=False
    )
    process_combined_file.send(str(final_path))
