import logging
from pathlib import Path


import boto3
import cartopy.crs as ccrs
from numba import types, generated_jit
import numpy as np
import pandas as pd
import xarray as xr


from erebos import utils, __version__
from erebos.adapters.goes import GOESFilename


def get_s3_keys(bucket, prefix=""):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """
    s3 = boto3.client("s3")
    kwargs = {"Bucket": bucket}

    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)
        if resp["KeyCount"] == 0:
            break
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.startswith(prefix):
                yield key

        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


def get_closest_s3_key(
    bucket, timestamp, product_name, scan_mode="3", band=None, window="160s"
):
    prefix = str(
        Path(product_name)
        / timestamp.strftime("%Y/%j/%H")
        / f"OR_{product_name}-M{scan_mode}"
    )
    if band is not None:
        prefix += "C{str(band).zfill(2)}"
    keys = get_s3_keys(bucket, prefix)
    closest_key = None
    for key in keys:
        key_time = pd.to_datetime(
            key.split("_")[-3].split(".")[0][1:-1], format="%Y%j%H%M%S"
        ).tz_localize("UTC")
        if pd.Timedelta(f"-{window}") < key_time - timestamp < pd.Timedelta(window):
            closest_key = key
            break
    if closest_key is None and scan_mode != "6":
        return get_closest_s3_key(bucket, timestamp, product_name, "6")
    return closest_key


def match_calipso_goes_times(calipso_dir, goes_dir, goes_glob):
    out = []
    goes_files = [GOESFilename(f) for f in goes_dir.glob(goes_glob)]
    goes_files = {f.start.round("5min"): f.filename for f in goes_files}
    for cf in calipso_dir.glob("*.hdf"):
        with xr.open_dataset(cf, engine="pynio") as cds:
            ctime = pd.Timestamp(cds.erebos.mean_time).round("5min")
        if ctime in goes_files:
            out.append((cf, goes_files[ctime]))
        else:
            out.append((cf, None))
    return out


def match_goes_file(calipso_file, goes_files, max_diff="6min"):
    gt = np.asarray([f.start for f in goes_files])
    with xr.open_dataset(calipso_file, engine="pynio") as cds:
        ctime = pd.Timestamp(cds.erebos.mean_time, tz="UTC").round("5min")
    diff = np.abs(gt - np.array(ctime))
    if diff.min() < pd.Timedelta(max_diff):
        return goes_files[np.argmin(diff)].filename
    else:
        return None


def translate_calipso_locations(calipso_ds, goes_ds, fill_na=True, level=0):
    # nan means no cloud
    cloud_heights = calipso_ds.erebos.cloud_top_altitude[:, level].values
    if fill_na:
        cloud_heights = np.ma.fix_invalid(cloud_heights).filled(0)
    terrain_height = calipso_ds.erebos.surface_elevation[:, 0].values
    cloud_locations = utils.RotatedECRPosition.from_geodetic(
        calipso_ds.erebos.Latitude[:, 0].values,
        calipso_ds.erebos.Longitude[:, 0].values,
        0.0,
    )
    actual_cloud_pos = utils.find_actual_cloud_position(
        calipso_ds.erebos.spacecraft_location, cloud_locations, cloud_heights
    )
    apparent_cloud_pos = utils.find_apparent_cloud_position(
        goes_ds.erebos.spacecraft_location, actual_cloud_pos, terrain_height
    )
    alat, alon = apparent_cloud_pos.to_geodetic()
    goes_cloud_pos = goes_ds.erebos.crs.transform_points(ccrs.Geodetic(), alon, alat)
    return goes_cloud_pos[:, :2]


def _mapit(vals, index, first):
    ovals = []
    inds = []
    cts = []
    for i, val in enumerate(vals):
        ind = index[i]
        if ind not in inds:
            inds.append(ind)
            ovals.append(val)
            cts.append(1.0)
        else:
            # basically do a nanmean
            idx = inds.index(ind)
            if np.isnan(val):
                continue
            elif np.isnan(ovals[idx]):
                ovals[idx] = val
                cts[idx] = 1.0
            elif not first:
                ovals[idx] += val
                cts[idx] += 1.0
    out = np.zeros(len(ovals), vals.dtype)
    for i in range(len(ovals)):
        out[i] = ovals[i] / cts[i]
    ind_out = np.array(inds, index.dtype)
    j = np.argsort(ind_out)
    return out[j], ind_out[j]


@generated_jit(nopython=True)
def map_values_to_index_num(vals, index, first):
    if isinstance(vals, types.Array):
        return _mapit
    else:
        raise TypeError("vals must be a numpy array")


def calipso_indices(calipso_ds, goes_ds, level=0):
    cloud_pts = translate_calipso_locations(calipso_ds, goes_ds, level=level)
    dist, inds = goes_ds.erebos.kdtree.query(
        cloud_pts.astype("float32"), k=1, distance_upper_bound=3e3
    )
    return inds


def calipso_var_in_goes_coords(var, calipso_ds, goes_ds, level=0):
    inds = calipso_indices(calipso_ds, goes_ds, level)
    avg, adj_inds = map_values_to_index_num(var, inds, False)
    iy, ix = np.unravel_index(adj_inds, (goes_ds.dims["y"], goes_ds.dims["x"]))
    out = xr.DataArray(
        avg,
        coords={"x": ("rec", goes_ds.erebos.x[ix]), "y": ("rec", goes_ds.erebos.y[iy])},
        dims=("rec"),
    )
    return out


def make_combined_dataset(
    calipso_file,
    goes_file,
    calipso_mean_vars,
    calipso_first_vars,
    goes_channels,
    level=0,
):
    calipso_ds = xr.open_dataset(calipso_file, engine="pynio")
    goes_ds = xr.open_dataset(goes_file, engine="netcdf4")
    inds = calipso_indices(calipso_ds, goes_ds)
    adj_inds = np.unique(inds)
    vars_ = {}
    for v in calipso_mean_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, level].values.astype("float32")
        avg, _ = map_values_to_index_num(vals, inds, False)
        vars_[v] = xr.DataArray(avg, dims=("rec"), attrs=var.attrs)

    for v in calipso_first_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, 0].values
        avg, _ = map_values_to_index_num(vals, inds, True)
        vars_[v] = xr.DataArray(avg, dims=("rec"), attrs=var.attrs)

    for chan in goes_channels:
        name = f"CMI_C{chan:02}"
        var = goes_ds.erebos.variables[name]
        cmi = var.values.reshape(-1)[adj_inds]
        # anything != 0 is considered questionable quality
        dqf = (
            goes_ds.erebos.variables[f"DQF_C{chan:02}"]
            .values.reshape(-1)[adj_inds]
            .astype(bool)
        )
        da = xr.DataArray(np.ma.array(cmi, mask=dqf), dims=("rec"), attrs=var.attrs)
        vars_[name] = da
    iy, ix = np.unravel_index(adj_inds, (goes_ds.dims["y"], goes_ds.dims["x"]))
    coords = {
        "x": ("rec", goes_ds.erebos.x[ix].values),
        "y": ("rec", goes_ds.erebos.y[iy].values),
    }
    return xr.Dataset(
        vars_,
        coords=coords,
        attrs={
            "goes_time": str(goes_ds.erebos.t.values),
            "goes_file": str(goes_file),
            "calipso_file": str(calipso_file),
            "erebose_version": __version__,
        },
    )


def runit(calipso_dir, goes_dir, save_dir, goes_glob):
    calipso_files = calipso_dir.glob("*.hdf")
    goes_files = [GOESFilename(f) for f in goes_dir.glob(goes_glob)]
    for cfile in calipso_files:
        logging.info("Processing %s", cfile)
        gfile = match_goes_file(cfile, goes_files)
        if gfile is None:
            logging.warning("No matching GOES file for %s", cfile)
            continue
        ds = make_combined_dataset(
            cfile,
            gfile,
            [
                "cloud_top_altitude",
                "cloud_thickness",
                "cloud_base_altitude",
                "cloud_layers",
                "solar_azimuth",
                "solar_zenith",
            ],
            ["cloud_type", "day_night_flag", "surface_elevation"],
            range(1, 17),
        )
        filename = pd.Timestamp(ds.goes_time).strftime("%Y%m%dT%H%M%S_combined.nc")
        ds.to_netcdf(save_dir / filename)
        ds.close()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelno)s %(message)s", level="INFO")
    goes_glob = "*L2-MC*.nc"
    calipso_dir = Path("/storage/projects/goes_alg/calipso/southwest/1km_cloud")
    goes_dir = Path("/storage/projects/goes_alg/goes_data/southwest_adj/")
    save_dir = Path("/storage/projects/goes_alg/combined/southwest/")
    save_dir.mkdir(parents=True, exist_ok=True)

    runit(calipso_dir, goes_dir, save_dir, goes_glob)
