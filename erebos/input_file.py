import logging
import warnings


import boto3
import s3fs
from scipy.ndimage import zoom
import xarray as xr


from erebos import __version__
from erebos.adapters.goes import GOESFilename


def generate_single_chan_prefixes(mcmip_file, bucket):
    fn = GOESFilename.from_path(mcmip_file)
    s3 = boto3.client("s3")

    out = {}
    for chan in range(1, 17):
        prefix = fn.to_s3_prefix(channel=chan, product="CMIP")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if resp["KeyCount"] == 0:
            raise ValueError(f"No keys with prefix {prefix}")
        key = resp["Contents"][0]["Key"]
        out[chan] = key
    return out


def prep_first_file(ds, chan):
    drop_vars = (
        "algorithm_dynamic_input_data_container",
        "algorithm_product_version_container",
        "band_id",
        "band_wavelength",
        "earth_sun_distance_anomaly_in_AU",
        "esun",
        "kappa0",
        "max_reflectance_factor",
        "mean_reflectance_factor",
        "min_reflectance_factor",
        "nominal_satellite_subpoint_lat",
        "nominal_satellite_subpoint_lon",
        "nominal_satellite_height",
        "geospatial_lat_lon_extent",
        "outlier_pixel_count",
        "percent_uncorrectable_GRB_errors",
        "percent_uncorrectable_L0_errors",
        "planck_bc1",
        "planck_bc2",
        "planck_fk1",
        "planck_fk2",
        "processing_parm_version_container",
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
        "keywords",
        "keywords_vocabulary",
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
    logging.info("Generating combined file based on %s", mcmip_file)
    s3_keys = generate_single_chan_prefixes(mcmip_file, bucket)
    fs = s3fs.S3FileSystem(anon=True)
    logging.info("Prepping file based on channel %s", base_chan)
    with fs.open(bucket + "/" + s3_keys.pop(base_chan)) as f:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            out = prep_first_file(ds, base_chan)

    for chan, s3key in s3_keys.items():
        logging.info("Adding data from channel %s", chan)
        with fs.open(bucket + "/" + s3key) as f:
            with xr.open_dataset(f, engine="h5netcdf") as nextds:
                out = add_primary_variables(out, nextds, base_chan)
    out.attrs["erebos_version"] = __version__
    logging.info("Saving file to %s", final_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.to_netcdf(final_path, engine="h5netcdf")
    out.close()
    logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level="INFO")
    mf = "OR_ABI-L2-MCMIPC-M6_G16_s20200502101161_e20200502103541_c20200502104096.nc"
    generate_combined_file(mf, "combo.nc")
