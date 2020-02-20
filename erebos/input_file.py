from concurrent.futures import ThreadPoolExecutor, wait
import logging
from pathlib import Path
import warnings
import tempfile


import boto3
from scipy.ndimage import zoom
import xarray as xr


from erebos import __version__
from erebos.adapters.goes import GOESFilename


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
    logging.debug("Downloading %s", key)
    s3 = boto3.resource("s3")
    s3.Object(bucket, key).download_file(str(path))
    logging.debug("Done with %s", key)


def download_files(mcmip_file, bucket, tmpdir):
    """
    Download all 16 CMIP channels (using mcmip_file as a template)
    from bucket into tmpdir
    """
    out = {}
    logging.info("Downloading files...")
    with ThreadPoolExecutor(max_workers=4) as exc:
        futs = []
        for chan, key in generate_single_chan_prefixes(mcmip_file, bucket):
            path = tmpdir / f"{chan}.nc"
            futs.append(exc.submit(_download, bucket, key, path))
            out[chan] = path
        wait(futs)
    logging.info("Done downloading")
    return out


def prep_first_file(ds, chan):
    """
    Take the dataset for chan and drop spurious variables and
    rename the main variables
    """
    drop_vars = (
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
    logging.info("Prepping first file %s", ds.dataset_name)
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


def generate_combined_file(mcmip_file, final_path, bucket="noaa-goes16", base_chan=1):
    """
    Make one netCDF file like MCMIP with the resolution of base_chan from the
    specified bucket and save to final_path
    """
    logging.info("Generating combined file based on %s", mcmip_file)
    tmpdir = tempfile.TemporaryDirectory()
    paths = download_files(mcmip_file, bucket, Path(tmpdir.name))
    logging.info("Prepping file based on channel %s", base_chan)
    with xr.open_dataset(paths.pop(base_chan), engine="h5netcdf") as ds:
        out = prep_first_file(ds, base_chan)

    for chan, path in paths.items():
        logging.info("Adding data from channel %s", chan)
        with xr.open_dataset(path, engine="h5netcdf") as nextds:
            out = add_primary_variables(out, nextds, base_chan)
    out.attrs["erebos_version"] = __version__
    logging.info("Saving file to %s", final_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.to_netcdf(final_path, engine="h5netcdf")
    out.close()
    tmpdir.cleanup()
    logging.info("Done saving file")
