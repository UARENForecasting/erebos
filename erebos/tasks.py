import logging
import os
from pathlib import Path
import psutil
import signal


import dramatiq
from dramatiq.brokers.redis import RedisBroker


from erebos import custom_multichannel_generation, config
from erebos.ml_models import predict


class RestartMiddleware(dramatiq.Middleware):
    def __init__(self, memory_limit):
        self.logger = dramatiq.logging.get_logger(__name__, type(self))
        self.memory_limit = memory_limit

    def after_process_message(self, broker, message, *, result=None, exception=None):
        proc = psutil.Process(os.getppid())
        rss = proc.memory_info().rss
        for child in proc.children():
            rss += child.memory_info().rss
        rss_mb = rss / 1024 ** 2
        self.logger.debug("Process group currently using %0.2f MiB RSS", rss_mb)
        if rss_mb > self.memory_limit:
            self.logger.info("Restarting workers with RSS of %0.2f MiB RSS...", rss_mb)
            os.kill(os.getppid(), signal.SIGHUP)


redis_broker = RedisBroker(host=config.REDIS_HOST, port=config.REDIS_PORT)
redis_broker.add_middleware(RestartMiddleware(config.MEM_LIMIT))
dramatiq.set_broker(redis_broker)
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


@dramatiq.actor
def process_combined_file(combined_file_path):
    predict.full_prediction(Path(combined_file_path), zarr_dir=config.ZARR_DIR)


@dramatiq.actor
def generate_combined_file(key, bucket):
    final_path = custom_multichannel_generation.generate_combined_file(
        key, config.MULTI_DIR, bucket, overwrite=False
    )
    process_combined_file.send(str(final_path))
