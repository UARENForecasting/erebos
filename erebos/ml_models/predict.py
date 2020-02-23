import logging
from pathlib import Path


import lightgbm
import numpy as np
import onnxruntime as rt
import pandas as pd
import xarray as xr


import erebos.adapters  # NOQA
from erebos.adapters import goes


logging = logging.getLogger(__name__)


def prepare_dataset(ds):
    return (
        ds.pipe(goes.assign_latlon)
        .pipe(goes.assign_surface_elevation)
        .pipe(goes.assign_solarposition_variables)
    )


def _predict(ds, vars_, pred_func):
    sel = {}
    for dim in ("t", "y", "x"):
        if dim not in ds.dims:
            ds = ds.expand_dims(dim)
            sel[dim] = 0

    arr = (
        ds[vars_]
        .to_array("var")
        .stack(z=("t", "y", "x"))
        .transpose("z", "var")
        .sel(var=vars_)
        .values
    )
    nans = np.isnan(arr).any(axis=1)
    X = arr[~nans]

    pred = pred_func(X)

    out = np.ma.ones(nans.shape, dtype="float32")
    out.mask = nans
    out[~nans] = pred.reshape(-1)
    out = xr.DataArray(
        out.reshape(ds.dims["t"], ds.dims["y"], ds.dims["x"]), dims=("t", "y", "x")
    )
    if sel:
        out = out.isel(sel)
    return out


def _predict_onnx(ds, onnx_model, vars_):
    sess = rt.InferenceSession(str(Path(__file__).parent.absolute() / onnx_model))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return _predict(ds, vars_, lambda x: sess.run([output_name], {input_name: x})[0])


def predict_cloud_mask(ds):
    logging.info("Predicting cloud mask...")
    vars_ = [f"CMI_C{c:02d}" for c in range(1, 17)]
    da = _predict_onnx(ds, "cloud_mask.onnx", vars_)
    da.name = "cloud_mask"
    da = da.assign_attrs(
        long_name="Erebos predicted binary Clear Sky Mask",
        standard_name="cloud_binary_mask",
        valid_range=[0, 1],
        flag_values=[0, 1],
        flag_meanings=["clear_or_probably_clear", "cloudy_or_probably_cloudy"],
    )
    da.encoding = {"dtype": "uint8", "_FillValue": 255, "zlib": True}
    return da


def predict_cloud_type(ds):
    logging.info("Predicting cloud type...")
    vars_ = [f"CMI_C{c:02d}" for c in range(1, 17)]
    da = _predict_onnx(ds, "cloud_type.onnx", vars_)
    da.name = "cloud_type"
    da = da.assign_attrs(
        long_name="Erebos Cloud Type Flag",
        valid_range=[0, 4],
        flag_values=[0, 1, 2, 3, 4],
        flag_meanings=[
            "no cloud",
            "low cloud",
            "mid-level cloud",
            "high cloud",
            "deep convective cloud",
        ],
    )
    da.encoding = {"dtype": "uint8", "_FillValue": 255, "zlib": True}
    return da


def predict_cloud_height(ds, cloud_mask=None, cloud_type=None):
    logging.info("Predicting cloud height...")
    vars_ = [f"CMI_C{c:02d}" for c in range(1, 17)]
    vars_ += ["cloud_type"]
    if cloud_mask is None:
        cloud_mask = predict_cloud_mask(ds)
    if cloud_type is None:
        cloud_type = predict_cloud_type(ds)
    nds = ds.assign(cloud_type=cloud_type * cloud_mask)
    da = _predict_onnx(nds, "cloud_height.onnx", vars_)
    da.name = "cloud_height"
    da = da.assign_attrs(
        long_name="Erebos predicted Cloud Top Height",
        standard_name="geopotential_cloud_top_height",
        units="km",
        valid_range=[0, 16],
    )
    da.encoding = {
        "dtype": "uint16",
        "scale_factor": 0.01,
        "_FillValue": 65535,
        "zlib": True,
    }
    return da


def predict_ghi(ds, cloud_mask=None, cloud_type=None, cloud_height=None):
    logging.info("Predicting GHI...")
    vars_ = [f"CMI_C{c:02d}" for c in range(1, 17)]
    vars_ += [
        "solar_zenith",
        "solar_azimuth",
        "solar_extra_radiation",
        "surface_elevation",
        "latitude",
        "longitude",
        "cloud_type",
        "cloud_height",
        "cloud_mask",
    ]
    if "longitude" not in ds:
        ds = prepare_dataset(ds)
    if cloud_mask is None:
        cloud_mask = predict_cloud_mask(ds)
    if cloud_type is None:
        cloud_type = predict_cloud_type(ds)
    if cloud_height is None:
        cloud_height = predict_cloud_height(ds, cloud_mask, cloud_type)
    nds = ds.assign(
        cloud_mask=cloud_mask,
        cloud_type=cloud_type * cloud_mask,
        cloud_height=cloud_height,
    ).reset_coords(("latitude", "longitude"))
    booster = lightgbm.Booster(
        model_file=str(Path(__file__).parent.absolute() / "ghi.lgbm")
    )
    da = _predict(nds, vars_, booster.predict)
    da = da.where(ds.solar_zenith < 90).fillna(0)
    da.name = "ghi"
    da = da.assign_attrs(
        units="W m-2",
        valid_range=[0, 1400],
        standard_name="surface_downwelling_shortwave_flux_in_air",
        long_name="Erebos predicted Global Horizontal Irradiance",
    )
    da.encoding = {
        "dtype": "uint16",
        "scale_factor": 0.1,
        "_FillValue": 65535,
        "zlib": True,
    }
    return da


def full_prediction(
    combined_path, nc_dir=None, zarr_dir=None, domain=(-115, -103, 30, 38)
):
    logging.info("Predicting quantities based on %s", combined_path)
    inp = xr.open_dataset(combined_path)
    restricted = goes.restrict_domain(inp, domain[:2], domain[2:])
    prepped = prepare_dataset(restricted)
    cmask = predict_cloud_mask(prepped)
    ctype = predict_cloud_type(prepped)
    cheight = predict_cloud_height(prepped, cmask, ctype)
    cghi = predict_ghi(prepped, cmask, ctype, cheight)
    out = restricted.assign(
        {cmask.name: cmask, ctype.name: ctype, cheight.name: cheight, cghi.name: cghi}
    )
    drop_vars = [f"CMI_C{c:02d}" for c in range(1, 17)] + [
        f"DQF_CMI_C{c:02d}" for c in range(1, 17)
    ]
    out = out.drop(drop_vars)
    nvars = {var: prepped[var] for var in prepped.data_vars if var not in drop_vars}
    out = out.assign(nvars)
    out.attrs["datasets"] += [str(combined_path.absolute())]
    mean_time = pd.Timestamp(prepped.erebos.mean_time)
    if nc_dir is not None:
        ncpath = nc_dir / mean_time.strftime(
            "%Y/%m/%d/erebos_prediction_%Y%m%dT%H%M%SZ.nc"
        )
        ncpath.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Saving NetCDF to %s", ncpath)
        out.erebos.to_netcdf(ncpath)
    if zarr_dir is not None:
        zarrpath = zarr_dir / mean_time.strftime("%Y/%m/%d")
        zarrpath.mkdir(parents=True, exist_ok=True)
        logging.info("Saving zarr to %s", zarrpath)
        if (zarrpath / ".zmetadata").exists():
            root = xr.open_zarr(str(zarrpath))
            if out.erebos.t.data in root.t.data:
                logging.warning("Time already present in zarr group, not saving")
            else:
                out.erebos.to_zarr(zarrpath, append_dim="t")
        else:
            out.erebos.to_zarr(zarrpath)
    inp.close()
    return out
