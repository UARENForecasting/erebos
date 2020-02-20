from concurrent.futures import ThreadPoolExecutor, wait
import json
import logging
from pathlib import Path
import warnings
import tempfile
import time
import threading


import boto3
from scipy.ndimage import zoom
import xarray as xr


from erebos import __version__
from erebos.adapters.goes import GOESFilename


logger = logging.getLogger(__name__)


def generate_single_chan_prefixes(mcmip_file, bucket):
    """
    From a CMIP or MCMIP filename, find the s3 keys for the
    16 indv. channels made at the same time in bucket
    """
    fn = GOESFilename.from_path(mcmip_file)
    s3 = boto3.client("s3")

    for chan in range(1, 17):
        prefix = fn.to_s3_prefix(channel=chan, product="CMIP")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if resp["KeyCount"] == 0:
            raise KeyError(f"No keys with prefix {prefix}")
        key = resp["Contents"][0]["Key"]
        yield chan, key


def _download(bucket, key, path):
    logger.debug("Downloading %s", key)
    s3 = boto3.resource("s3")
    s3.Object(bucket, key).download_file(str(path))
    logger.debug("Done with %s", key)


def download_files(mcmip_file, bucket, tmpdir):
    """
    Download all 16 CMIP channels (using mcmip_file as a template)
    from bucket into tmpdir
    """
    out = {}
    logger.info("Downloading files...")
    with ThreadPoolExecutor(max_workers=4) as exc:
        futs = []
        for chan, key in generate_single_chan_prefixes(mcmip_file, bucket):
            path = tmpdir / f"{chan}.nc"
            futs.append(exc.submit(_download, bucket, key, path))
            out[chan] = path
        wait(futs)
    logger.info("Done downloading")
    return out


def prep_first_file(ds, chan):
    """
    Take the dataset for chan and drop spurious variables and
    rename the main variables
    """
    drop_vars = (
        "nominal_satellite_subpoint_lat",
        "nominal_satellite_subpoint_lon",
        "nominal_satellite_height",
        "geospatial_lat_lon_extent",
        "algorithm_dynamic_input_data_container",
        "earth_sun_distance_anomaly_in_AU",
        "processing_parm_version_container",
        "algorithm_product_version_container",
        "band_id",
        "band_wavelength",
        "esun",
        "kappa0",
        "max_reflectance_factor",
        "mean_reflectance_factor",
        "min_reflectance_factor",
        "outlier_pixel_count",
        "percent_uncorrectable_GRB_errors",
        "percent_uncorrectable_L0_errors",
        "planck_bc1",
        "planck_bc2",
        "planck_fk1",
        "planck_fk2",
        "std_dev_reflectance_factor",
        "total_number_of_points",
        "valid_pixel_count",
        "focal_plane_temperature_threshold_exceeded_count",
        "maximum_focal_plane_temperature",
        "focal_plane_temperature_threshold_increasing",
        "focal_plane_temperature_threshold_decreasing",
    )
    drop_attr = (
        "id",
        "production_data_source",
        "dataset_name",
        "title",
        "summary",
        "processing_level",
        "date_created",
    )
    out = ds.rename({"CMI": f"CMI_C{chan:02d}", "DQF": f"DQF_CMI_C{chan:02d}"})
    out.attrs["timezone"] = "UTC"
    out.attrs["datasets"] = (ds.dataset_name,)
    for attr in drop_attr:
        del out.attrs[attr]
    out = out.drop(drop_vars, errors="ignore").set_coords("goes_imager_projection")
    return out.load()


def add_primary_variables(ds, other, base_chan):
    """
    For other channels, add the main data to ds at the same resolution.
    For channels that have a higher resolution, subsample the lower left point.
    For channels that have a lower resolution, copy the value to mulitple cells.
    """
    cband = f"C{other.band_id.item():02d}"
    nvar = other.CMI
    ndqf = other.DQF
    comp_params = {
        k: getattr(ds, f"CMI_C{base_chan:02d}").encoding[k]
        for k in ("chunksizes", "shuffle", "zlib", "complevel", "fletcher32")
    }
    ve = nvar.encoding
    ve.update(comp_params)
    de = ndqf.encoding
    de.update(comp_params)
    if other.dims["x"] > ds.dims["x"]:
        i = other.dims["x"] // ds.dims["x"]
        nvar = nvar[::i, ::i]
        nvar["x"] = ds.x
        nvar["y"] = ds.y
        ndqf = ndqf[::i, ::i]
        ndqf["x"] = ds.x
        ndqf["y"] = ds.y
    elif other.dims["x"] < ds.dims["x"]:
        i = ds.dims["x"] // other.dims["x"]
        nvard = zoom(nvar, i, order=0)
        ndqfd = zoom(ndqf, i, order=0)

        nvar = xr.DataArray(
            nvard, coords=[ds.y, ds.x], dims=["y", "x"], attrs=nvar.attrs
        )
        ndqf = xr.DataArray(
            ndqfd, coords=[ds.y, ds.x], dims=["y", "x"], attrs=ndqf.attrs
        )
    nvar.encoding = ve
    ndqf.encoding = de
    out = ds.assign({f"CMI_{cband}": nvar, f"DQF_CMI_{cband}": ndqf})
    out.attrs["datasets"] = (*ds.attrs["datasets"], other.attrs["dataset_name"])
    return out.load()


def make_out_path(mcmip_file, out_dir):
    fn = GOESFilename.from_path(mcmip_file)
    ftime = fn.start + (fn.end - fn.start) / 2
    dir_ = Path(out_dir) / ftime.strftime("%Y/%m/%d")
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_ / ftime.strftime(
        f"erebos_{fn.product}{fn.sector}_{fn.satellite}_%Y%m%dT%H%M%SZ.nc"
    )


def generate_combined_file(
    mcmip_file, out_dir, bucket="noaa-goes16", base_chan=1, overwrite=False
):
    """
    Make one netCDF file like MCMIP with the resolution of base_chan from the
    specified bucket and save to out_dir
    """
    logger.info("Generating combined file based on %s", mcmip_file)
    final_path = make_out_path(mcmip_file, out_dir)
    if not overwrite and final_path.exists():
        logger.info("File already exists at %s, skipping", final_path)
        return
    tmpdir = tempfile.TemporaryDirectory()
    paths = download_files(mcmip_file, bucket, Path(tmpdir.name))
    logger.debug("Prepping file based on channel %s", base_chan)
    with xr.open_dataset(paths.pop(base_chan), engine="h5netcdf") as ds:
        out = prep_first_file(ds, base_chan)

    for chan, path in paths.items():
        logger.debug("Adding data from channel %s", chan)
        with xr.open_dataset(path, engine="h5netcdf") as nextds:
            out = add_primary_variables(out, nextds, base_chan)
    out.attrs["erebos_version"] = __version__
    logger.info("Saving file to %s", final_path)
    tmppath = Path(tempfile.mkstemp(dir=final_path.parent, prefix=".", suffix=".nc")[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out.to_netcdf(tmppath, engine="h5netcdf")
        except Exception:
            tmppath.unlink()
            raise
        else:
            tmppath.rename(final_path)
    final_path.chmod(0o664)
    out.close()
    tmpdir.cleanup()
    logger.debug("Done saving file")


def _update_visibility(message, timeout, local):
    while not local.stop:
        message.change_visibility(VisibilityTimeout=timeout)
        time.sleep(timeout / 2)


def get_sqs_keys(sqs_url, s3_prefix):
    sqs = boto3.resource("sqs")
    q = sqs.Queue(sqs_url)
    messages = q.receive_messages(MaxNumberOfMessages=10)
    while len(messages) > 0:
        for message in messages:
            # continuously update message visibility until processing
            # is complete
            with ThreadPoolExecutor() as exc:
                data = threading.local()
                data.stop = False
                fut = exc.submit(_update_visibility, message, 30, data)
                sns_msg = json.loads(message.body)
                rec = json.loads(sns_msg["Message"])
                for record in rec["Records"]:
                    bucket = record["s3"]["bucket"]["name"]
                    key = record["s3"]["object"]["key"]
                    if key.startswith(s3_prefix):
                        yield (bucket, key)
                data.stop = True
                logger.debug("stopping message visibility update")
                fut.cancel()
            message.delete()
            logger.debug("message deleted")
        messages = q.receive_messages(MaxNumberOfMessages=10)


def get_process_and_save(sqs_url, out_dir, overwrite, s3_prefix="ABI-L2-MCMIPC"):
    for bucket, key in get_sqs_keys(sqs_url, s3_prefix):
        generate_combined_file(key, out_dir, bucket, overwrite=overwrite)
