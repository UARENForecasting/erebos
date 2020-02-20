from concurrent.futures import ProcessPoolExecutor
import datetime as dt
from functools import partial, wraps
import logging
import os
from pathlib import Path
import sys
import time


import click
from croniter import croniter
import pytz


from erebos import __version__


def handle_exception(logger, exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def basic_logging_config():
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    sentry_dsn = os.getenv("SENTRY_DSN", None)
    if sentry_dsn is not None:
        try:
            import sentry_sdk
        except ImportError:
            logging.error("Cannot monitor with sentry")
        else:
            sentry_sdk.init(dsn=sentry_dsn, release=f"erebos@{__version__}")


sys.excepthook = partial(handle_exception, logging.getLogger())
basic_logging_config()
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def set_log_level(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        verbose = kwargs.pop("verbose", 0)
        if verbose == 1:
            loglevel = "INFO"
        elif verbose > 1:
            loglevel = "DEBUG"
        else:
            loglevel = "WARNING"
        logging.getLogger().setLevel(loglevel)
        return f(*args, **kwargs)

    return wrapper


verbose = click.option("-v", "--verbose", count=True, help="Increase logging verbosity")


def schedule_options(cmd):
    """Combine scheduling options into one decorator"""

    def wrapper(f):
        decs = [
            click.option("--cron", help="Run the script on a cron schedule"),
            click.option(
                "--cron-tz",
                help="Timezone to use for cron scheduling",
                show_default=True,
                default="UTC",
            ),
        ]
        for dec in reversed(decs):
            f = dec(f)
        return f

    return wrapper(cmd)


def _now(tz):
    return dt.datetime.now(tz=pytz.timezone(tz))


def run_times(cron, cron_tz):
    now = _now(cron_tz)
    iter = croniter(cron, now)
    while True:
        next_time = iter.get_next(dt.datetime)
        if next_time > _now(cron_tz):
            yield next_time


def silent_exit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            out = f(*args, **kwargs)
        except (SystemExit, KeyboardInterrupt):
            pass
        else:
            return out

    return wrapper


@silent_exit
def run_loop(fnc, *args, cron, cron_tz, **kwargs):
    if cron is None:
        return fnc(*args, **kwargs)
    for rt in run_times(cron, cron_tz):
        sleep_length = (rt - _now(cron_tz)).total_seconds()
        logging.info("Sleeping for %0.1f s to next run time at %s", sleep_length, rt)
        if sleep_length > 0:
            time.sleep(sleep_length)
        with ProcessPoolExecutor(1) as exc:
            fut = exc.submit(fnc, *args, **kwargs)
            fut.result()


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def cli():
    """
    The erebos command line tool.
    """
    pass  # pragma: no cover


class PathParamType(click.Path):
    def convert(self, value, param, ctx):
        p = super().convert(value, param, ctx)
        return Path(p)


@cli.command()
@verbose
@schedule_options
@set_log_level
@click.argument("sqs_url")
@click.argument(
    "save_directory",
    type=PathParamType(exists=True, writable=True, resolve_path=True, file_okay=False),
)
def process_files(sqs_url, save_directory, cron, cron_tz):
    """
    Process new files in SQS_URL and save the high-res combined NetCDF
    to SAVE_DIRECTORY
    """
    from erebos.input_file import get_process_and_save

    run_loop(get_process_and_save, sqs_url, save_directory, cron=cron, cron_tz=cron_tz)
