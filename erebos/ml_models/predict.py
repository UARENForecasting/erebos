from pathlib import Path


import lightgbm
import numpy as np
import onnxruntime as rt
import xarray as xr


import erebos.adapters  # NOQA
from erebos.adapters import goes


def prepare_dataset(ds):
    return (
        ds.pipe(goes.assign_latlon)
        .pipe(goes.assign_surface_elevation)
        .pipe(goes.assign_solarposition_variables)
    )


def _predict(ds, vars_, pred_func):
    arr = (
        ds[vars_]
        .to_array("var")
        .stack(z=("y", "x"))
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
    return xr.DataArray(out.reshape(ds.dims["y"], ds.dims["x"]), dims=("y", "x"))


def _predict_onnx(ds, onnx_model, vars_):
    sess = rt.InferenceSession(str(Path(__file__).parent.absolute() / onnx_model))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return _predict(ds, vars_, lambda x: sess.run([output_name], {input_name: x})[0])


def cloud_mask(ds):
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


def cloud_type(ds, cloud_mask):
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


def cloud_height(ds, cloud_mask, cloud_type):
    vars_ = [f"CMI_C{c:02d}" for c in range(1, 17)]
    vars_ += ["cloud_type"]
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


def ghi(ds, cloud_mask, cloud_type, cloud_height):
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
    nds = ds.assign(
        cloud_mask=cloud_mask,
        cloud_type=cloud_type * cloud_mask,
        cloud_height=cloud_height,
    ).reset_coords(("latitude", "longitude"))
    booster = lightgbm.Booster(
        model_file=str(Path(__file__).parent.absolute() / "ghi.lgbm")
    )
    da = _predict(nds, vars_, booster.predict)
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