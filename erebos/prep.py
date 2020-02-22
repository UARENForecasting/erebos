from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import logging
from pathlib import Path
import warnings


import boto3
import cartopy.crs as ccrs
from numba import types, generated_jit
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
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
        prefix += f"C{str(band).zfill(2)}"
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
        return get_closest_s3_key(
            bucket, timestamp, product_name, "6", band=band, window=window
        )
    return closest_key


def match_calipso_goes_times(calipso_dir, goes_dir, goes_glob):
    out = []
    goes_files = [GOESFilename.from_path(f) for f in goes_dir.glob(goes_glob)]
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
        ctime = pd.Timestamp(cds.erebos.time.data.min(), tz="UTC")
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
        calipso_ds.erebos.spacecraft_ecr_position, cloud_locations, cloud_heights
    )
    apparent_cloud_pos = utils.find_apparent_cloud_position(
        goes_ds.erebos.spacecraft_ecr_position, actual_cloud_pos, terrain_height
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


def calipso_indices(calipso_ds, goes_ds, level=0, k=1, dub=1.3e3):
    cloud_pts = translate_calipso_locations(calipso_ds, goes_ds, level=level)
    dist, inds = goes_ds.erebos.kdtree.query(
        cloud_pts.astype("float32"), k=k, distance_upper_bound=dub
    )
    return inds


def calipso_var_in_goes_coords(var, calipso_ds, goes_ds, level=0):
    inds = calipso_indices(calipso_ds, goes_ds, level)
    avg, adj_inds = map_values_to_index_num(var, inds, False)
    invalid = adj_inds == goes_ds.dims["y"] * goes_ds.dims["x"]
    iy, ix = np.unravel_index(
        adj_inds[~invalid], (goes_ds.dims["y"], goes_ds.dims["x"])
    )
    out = xr.DataArray(
        avg[~invalid],
        coords={"x": ("rec", goes_ds.erebos.x[ix]), "y": ("rec", goes_ds.erebos.y[iy])},
        dims=("rec"),
    )
    return out


def make_combined_dataset(
    calipso_file, goes_file, calipso_mean_vars, calipso_first_vars, level=0
):
    calipso_ds = xr.open_dataset(calipso_file, engine="pynio")
    goes_ds = xr.open_dataset(goes_file, engine="netcdf4")
    # include the nearest 4 points in the goes file
    inds = calipso_indices(calipso_ds, goes_ds, k=4, dub=2e3)
    over_ind = goes_ds.dims["x"] * goes_ds.dims["y"]
    _, uniq = np.unique(inds[:, 0], return_index=True)
    do_not_include = np.logical_or.reduce(inds[uniq] == over_ind, axis=1)
    # basically, do not include points within 3 km of border
    adj_inds = inds[uniq][~do_not_include]
    vars_ = {}
    for v in calipso_mean_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, level].values.astype("float32")
        avg, ninds = map_values_to_index_num(vals, inds[:, 0], False)
        assert (ninds == inds[uniq, 0]).all()
        da = xr.DataArray(avg[~do_not_include], dims=("rec"), attrs=var.attrs)
        da.encoding = {"zlib": True, "complevel": 1, "shuffle": True}
        vars_[v] = da

    for v in calipso_first_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, 0].values
        avg, ninds = map_values_to_index_num(vals, inds[:, 0], True)
        assert (ninds == inds[uniq, 0]).all()
        da = xr.DataArray(avg[~do_not_include], dims=("rec"), attrs=var.attrs)
        da.encoding = {"zlib": True, "complevel": 1, "shuffle": True}
        vars_[v] = da

    for name, var in goes_ds.erebos.variables.items():
        if name.startswith("DQF") or "x" not in var.dims or "y" not in var.dims:
            continue
        cmi = var.values.reshape(-1)[adj_inds]
        # anything != 0 is considered questionable quality
        # not quite true for COD, but ok for now
        dqf = (
            goes_ds.erebos.variables[f"DQF_{name}"]
            .values.reshape(-1)[adj_inds]
            .astype(bool)
        )
        da = xr.DataArray(
            np.ma.array(cmi, mask=dqf), dims=("rec", "near"), attrs=var.attrs
        )
        da.encoding = var.encoding
        vars_[name] = da
    vars_["goes_imager_projection"] = goes_ds.goes_imager_projection
    iy, ix = np.unravel_index(adj_inds, (goes_ds.dims["y"], goes_ds.dims["x"]))
    coords = {
        "x": (("rec", "near"), goes_ds.erebos.x[ix.ravel()].values.reshape(ix.shape)),
        "y": (("rec", "near"), goes_ds.erebos.y[iy.ravel()].values.reshape(iy.shape)),
    }
    out = xr.Dataset(
        vars_,
        coords=coords,
        attrs={
            "goes_time": str(goes_ds.erebos.t.values),
            "goes_file": str(goes_file),
            "calipso_file": str(calipso_file),
            "erebose_version": __version__,
        },
    )
    calipso_ds.close()
    goes_ds.close()
    return out


def create_class_search_xml(
    goes_cmip_file, xml_dir, additional_l2_products=("ACHA", "ACM", "ACTP", "COD")
):
    import xml.etree.ElementTree as ET

    base_xml = b"""
<sc datatype_family="GRABIPRD">

  <item group="search_opt">SC</item><item group="nlat">90</item><item group="wlon">-180</item><item group="elon">180</item><item group="slat">-90</item><item group="minDiff">0.0</item><item group="data_start">2014-01-01</item><item group="data_end">2019-09-16</item><item group="max_days_val">366</item><item group="between_through">T</item><item group="max_sum_hits">4000</item><item group="lrg_max_sum_hits">10000</item><item group="brk_srch_hrs_qs">6</item><item group="bulk_order">N</item><item group="limit_search">Y</item><item group="max_lat_range">180</item><item group="max_lon_range">360</item><item group="datatype_family">GRABIPRD</item><item group="Datatype">ABIL2PROD</item></sc>"""
    base = io.BytesIO(base_xml)
    tree = ET.parse(base)
    root = tree.getroot()
    sat = ET.SubElement(root, "item", group="Satellite")
    sat.text = goes_cmip_file.satellite
    sec = ET.SubElement(root, "item", group="ABI Scan Sector")
    sec.text = goes_cmip_file.sector
    for dt in additional_l2_products:
        pt = ET.SubElement(root, "item", group="Product Type")
        pt.text = dt
    start = goes_cmip_file.start - pd.Timedelta("30s")
    end = goes_cmip_file.start + pd.Timedelta("30s")
    sd = ET.SubElement(root, "item", group="start_date")
    sd.text = start.strftime("%Y-%m-%d")
    st = ET.SubElement(root, "item", group="start_time")
    st.text = start.strftime("%H:%M:%S")
    ed = ET.SubElement(root, "item", group="end_date")
    ed.text = end.strftime("%Y-%m-%d")
    et = ET.SubElement(root, "item", group="end_time")
    et.text = end.strftime("%H:%M:%S")
    path = Path(xml_dir) / goes_cmip_file.start.strftime("%Y%m%d_%H%M%S.xml")
    tree.write(
        path, encoding="ISO-8859-1", xml_declaration=False, short_empty_elements=False
    )


def download_goes_files_at_time(bucket_name, time_, product_name, band, goes_dir):
    boto3.client("s3")
    s3 = boto3.resource("s3")
    gfile = get_closest_s3_key(bucket_name, time_, product_name, band=band)
    if gfile is None:
        logging.warning("No GOES file at time %s", time_)
        return
    gpath = goes_dir / Path(gfile).name
    if not gpath.exists():
        logging.info("Downloading GOES file %s to %s", gfile, gpath)
        s3.Object(bucket_name, gfile).download_file(str(gpath))
    else:
        logging.info("Skipping %s", gfile)
    return gpath


def download_corresponding_goes_files(
    calipso_dir,
    goes_dir,
    bucket_name="noaa-goes16",
    product_names_bands=("ABI-L2-MCMIPC", None),
    checkpoint=False,
    cglob="*.hdf",
):
    if checkpoint:
        chkpoints = open("/tmp/.checkpoint", "r").read().split("\n")
        chkfile = open(Path("/tmp/.checkpoint"), "a")
    goes_dir.mkdir(parents=True, exist_ok=True)
    for calipso_file in calipso_dir.glob(cglob):
        if checkpoint and str(calipso_file) in chkpoints:
            continue
        with xr.open_dataset(calipso_file, engine="pynio") as cds:
            ct = pd.Timestamp(cds.erebos.time.data.min(), tz="UTC")

        def f(pnb):
            return download_goes_files_at_time(
                bucket_name, ct, pnb[0], pnb[1], goes_dir
            )

        with ThreadPoolExecutor(max_workers=8) as exc:
            got = exc.map(f, product_names_bands)
        tuple(got)
        if checkpoint:
            chkfile.write(f"{str(calipso_file)}\n")
            chkfile.flush()
    chkfile.close()


def combine_calipso_goes_files(
    calipso_dir, goes_dir, save_dir, goes_glob, calipso_glob, limits=(0, None)
):
    calipso_files = list(calipso_dir.glob(calipso_glob))[slice(*limits)]
    goes_files = [
        GOESFilename(f, start=pd.Timestamp(f.name.split("_")[0], tz="UTC"))
        for f in goes_dir.glob(goes_glob)
    ]
    for cfile in calipso_files:
        logging.info("Processing %s", cfile)
        gfile = match_goes_file(cfile, goes_files)
        if gfile is None:
            logging.warning("No matching GOES file for %s", cfile)
            continue

        filename = save_dir / gfile.name

        if filename.exists():
            logging.info("File already exists at %s", filename)
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
        )
        logging.info("Saving file to %s", filename)
        ds.to_netcdf(filename, engine="netcdf4")
        ds.close()


def add_primary_variables(ds, other, var):
    try:
        cband = f"C{other.band_id.item():02d}"
    except AttributeError:
        cband = None
    nvar = getattr(other, var)
    ndqf = other.DQF
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
        ne = nvar.encoding
        nvar = xr.DataArray(
            nvard, coords=[ds.y, ds.x], dims=["y", "x"], attrs=nvar.attrs
        )
        nvar.encoding = ne
        de = ndqf.encoding
        ndqf = xr.DataArray(
            ndqfd, coords=[ds.y, ds.x], dims=["y", "x"], attrs=ndqf.attrs
        )
        ndqf.encoding = de
    if cband is not None:
        out = ds.assign({f"{var}_{cband}": nvar, f"DQF_{var}_{cband}": ndqf})
    else:
        out = ds.assign({var: nvar, f"DQF_{var}": ndqf})
    out.attrs["datasets"] = (*ds.attrs["datasets"], other.attrs["dataset_name"])
    return out


def add_empty_var(ds, var):
    encoding = {
        "chunksizes": (250, 250),
        "fletcher32": False,
        "shuffle": False,
        "zlib": True,
        "complevel": 1,
        "dtype": np.dtype("int16"),
        "_Unsigned": "true",
        "_FillValue": np.array([1023], dtype=np.uint16),
        "scale_factor": np.array([1], dtype=np.float32),
        "add_offset": np.array([0], dtype=np.float32),
    }
    nvar = xr.DataArray(
        np.ma.masked_equal(np.zeros((ds.dims["y"], ds.dims["x"]), dtype="float32"), 0),
        coords=[ds.y, ds.x],
        dims=["y", "x"],
    )
    nvar.encoding = encoding
    ndqf = xr.DataArray(
        np.ones((ds.dims["y"], ds.dims["x"]), dtype="float32"),
        coords=[ds.y, ds.x],
        dims=["y", "x"],
    )
    ndqf.encoding = encoding
    out = ds.assign({var: nvar, f"DQF_{var}": ndqf})
    return out


def prep_first_file(ds):
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
    out = ds.rename({"CMI": "CMI_C01", "DQF": "DQF_CMI_C01"})
    out.attrs["datasets"] = (ds.dataset_name,)
    for attr in drop_attr:
        del out.attrs[attr]
    out = out.drop(drop_vars, errors="ignore").set_coords("goes_imager_projection")
    return out


def add_variables_to_out(
    out, dir_, gfile, var, chan, empty_file="/tmp/no_goes_file", **kwargs
):
    globber = gfile.to_path(channel=chan, **kwargs, glob_ready=True).name
    plevel = kwargs.get("processing_level", gfile.processing_level)
    product = kwargs.get("product", gfile.product)
    sector = kwargs.get("sector", gfile.sector)
    logging.debug("Finding file like %s", globber)
    try:
        fname = list(dir_.glob(globber))[0]
    except IndexError:
        logging.warning("No file found like %s, trying to download", globber)
        dfile = download_goes_files_at_time(
            "noaa-goes16",
            gfile.start,
            f"ABI-{plevel}-{product}{sector}",
            chan or None,
            dir_,
        )
        if dfile is None:
            logging.error("No file found like %s", globber)
            out = add_empty_var(out, var)
            out.attrs["is_missing_data"] = 1
            if empty_file is not None:
                with open(empty_file, "a") as f:
                    f.write(f"{gfile.filename} {var}\n")
        else:
            other = xr.open_dataset(dfile, engine="netcdf4").load()
            out = add_primary_variables(out, other, var)
            other.close()
    else:
        other = xr.open_dataset(fname, engine="netcdf4").load()
        out = add_primary_variables(out, other, var)
        other.close()
    return out


def combine_goes_files_at_time(base_path, first_ds):
    var_map = {"ACHA": "HT", "ACM": "BCM", "ACTP": "Phase", "COD": "COD"}
    gfile = GOESFilename.from_path(base_path)
    out = first_ds.copy()
    dir_ = base_path.parent / "../CMIP"
    for chan in range(2, 17):
        out = add_variables_to_out(out, dir_, gfile, "CMI", chan)
    dir_ = base_path.parent / "../Rad"
    for chan in range(1, 17):
        out = add_variables_to_out(
            out, dir_, gfile, "Rad", chan, processing_level="L1b", product="Rad"
        )
    for prod, var in var_map.items():
        dir_ = base_path.parent / ".." / prod
        out = add_variables_to_out(out, dir_, gfile, var, chan=0, product=prod)
    return out


def combine_goes_files(base_dir):
    cmip_c01_files = (base_dir / "CMIP").glob("*C01*")
    outdir = base_dir / "combined"
    outdir.mkdir(parents=True, exist_ok=True)
    for afile in cmip_c01_files:
        final_path = outdir / GOESFilename.from_path(afile).start.strftime(
            "%Y%m%dT%H%M%S_combined.nc"
        )
        if final_path.is_file():
            logging.info("Path exists at %s", final_path)
            continue
        orig = xr.open_dataset(afile, engine="netcdf4").load()
        ds = prep_first_file(orig).load()
        orig.close()
        out = combine_goes_files_at_time(afile, ds)
        ds.close()
        out.attrs["erebos_version"] = __version__
        logging.info("Saving file to %s", final_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.to_netcdf(final_path, engine="netcdf4")
        out.close()
        logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelno)s %(message)s", level="INFO")
    base_dir = Path("/storage/projects/goes_alg/goes_data/west")
    combine_goes_files(base_dir)
