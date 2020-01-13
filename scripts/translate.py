#!/usr/bin/env python
# coding: utf-8

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


import xarray as xr
import pandas as pd
import logging
from erebos import utils
from erebos.adapters.goes import project_xy_to_latlon, GOESFilename
import numpy as np
import pickle
from pathlib import Path
from pvlib.solarposition import get_solarposition


def translate_pt(ds, lon, lat):
    crs = ds.erebos.crs
    x, y = [
        i / ds.erebos.goes_imager_projection.perspective_point_height
        for i in crs.transform_point(lon, lat, crs.as_geodetic())
    ]
    return x, y


def data_around(ds, lon, lat, km_bnd=10):
    x, y = translate_pt(ds, lon, lat)
    xind = np.argmin(np.abs(ds.x - x)).item()
    yind = np.argmin(np.abs(ds.y - y)).item()
    xmin = xind - km_bnd
    if xmin < 0:
        xmin = 0
    ymin = yind - km_bnd
    if ymin < 0:
        ymin = 0
    xmax = xind + km_bnd
    if xmax > ds.dims["x"]:
        xmax = ds.dims["x"]
    ymax = yind + km_bnd
    if ymax > ds.dims["y"]:
        ymax = ds.dims["y"]
    return ds.isel(x=slice(xmin, xmax), y=slice(ymin, ymax))


def dist(site, other):
    return np.sqrt(
        (site.X - other.X) ** 2 + (site.Y - other.Y) ** 2 + (site.Z - other.Z) ** 2
    )


def find_shifted_index(ds_subset, lat, lon, heights, nans):
    hvals = np.ma.ones(nans.shape)
    hvals.mask = nans
    hvals[~nans] = heights
    hvals[hvals < 1] = 0
    hvals = hvals.reshape(ds_subset.dims["x"], ds_subset.dims["y"]).T

    glon, glat = project_xy_to_latlon(ds_subset.x, ds_subset.y, ds_subset)

    apparent_cloudlocs = utils.RotatedECRPosition.from_geodetic(glat, glon, 0)
    actual_cloudlocs = utils.find_actual_cloud_position(
        ds_subset.erebos.spacecraft_location, apparent_cloudlocs, hvals
    )

    sun_pos = utils.get_solar_ecr_position(
        ds_subset.erebos.t.astype(int).item() / 1e9, lat, lon
    )
    shadowlocs = utils.find_apparent_cloud_position(sun_pos, actual_cloudlocs)

    site_pos = utils.RotatedECRPosition.from_geodetic(lat, lon, 0)

    sdist = dist(site_pos, shadowlocs)
    minargs = sdist.argsort(axis=-1)
    sorted_sdist = sdist.reshape(-1)[minargs]
    limit = (sorted_sdist < np.sqrt(3)).argmin()
    if limit == 0:
        return (10, 10)

    limhvals = hvals.ravel()[minargs[:limit]]
    shifted_loc = minargs[np.argmax(limhvals)]
    shifted_yx = np.unravel_index(shifted_loc, hvals.shape)  # y, x
    return shifted_yx


def process_site(goes_ds, site):
    lat = site.latitude.item()
    lon = site.longitude.item()
    gvals = data_around(goes_ds, lon, lat)
    vars_ = [v for v in gvals.variables.keys() if v.startswith("CMI_C")]
    # assert gvals.sel(x=site_gloc[0], y=site_gloc[1], method='nearest') == gvals.isel(x=10, y=10)
    original_point = gvals.isel(x=10, y=10)
    original_yx = (10, 10)

    arr = gvals[vars_].to_dataframe()
    arr = arr.drop(columns=[c for c in arr.columns if c not in vars_])

    nans = np.isnan(arr.values).any(axis=1)
    cloudy = mask_model.predict(arr.values[~nans])
    cloud_type = type_model.predict(arr.values[~nans])
    arr["cloud_type"] = pd.Series(cloudy * cloud_type, index=arr.index[~nans])
    height = height_model.predict(arr.values[~nans])

    shifted_yx = find_shifted_index(gvals, lat, lon, height, nans)
    shifted_point = gvals.isel(x=shifted_yx[1], y=shifted_yx[0])

    yxs = (original_yx, shifted_yx)

    out_ds = xr.Dataset()
    dims = ["time", "site", "shifted"]
    coords = {
        "shifted": [0, 1],
        "site": [site.site.item()],
        "time": [gvals.erebos.t.data],
    }
    for label, thing in [
        ("height", height),
        ("cloud_type", cloud_type),
        ("cloudy", cloudy),
    ]:
        out_ds[label] = xr.DataArray(
            [[[get_predict_at_pt(thing, yx, gvals) for yx in yxs]]],
            dims=dims,
            coords=coords,
        )
    for var_ in vars_:
        out_ds[var_] = xr.DataArray(
            [[[original_point[var_].item(), shifted_point[var_].item()]]],
            dims=dims,
            coords=coords,
        )

    obs_vals = site.sel(time=slice(*gvals.time_bounds.data)).mean()
    for var in ("ghi", "dni", "dhi"):
        out_ds[var] = xr.DataArray(
            [[obs_vals[var].item()]],
            dims=dims[:-1],
            coords={v: coords[v] for v in ("site", "time")},
        )
    return out_ds


def get_predict_at_pt(pred, yx, gvals):
    return pred.reshape(gvals.dims["x"], gvals.dims["y"]).T[yx]


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level="INFO")

    with open("height.pkl", "rb") as f:
        height_model = pickle.load(f)

    with open("cloud_mask.pkl", "rb") as f:
        mask_model = pickle.load(f)
    with open("cloud_type.pkl", "rb") as f:
        type_model = pickle.load(f)

    site_data = xr.open_dataset("/storage/projects/goes_alg/site_data.nc")
    goes_files = [
        GOESFilename.from_path(f)
        for f in Path("/storage/projects/goes_alg/goes_data/west/CMIP").glob(
            "*MCMIPC*.nc"
        )
    ]

    final_countdown = []
    for gfile in goes_files:
        if gfile.start.hour < 13:
            continue
        logging.info("Processing file from %s", gfile.start)
        with xr.open_dataset(gfile.filename) as goes_ds:
            tomerge = []
            for _, site in site_data.groupby("site"):
                tomerge.append(process_site(goes_ds, site))
            final_countdown.append(xr.merge(tomerge))
    output = xr.merge(final_countdown)

    m = []
    for _, site in site_data.groupby("site"):
        da = (
            get_solarposition(
                output.time.data, site.latitude.data, site.longitude.data
            )[["zenith", "azimuth"]]
            .to_xarray()
            .rename({"index": "time"})
        )
        da.coords["site"] = site.site
        m.append(da)

    solpos = xr.concat(m, "site").transpose("time", "site")
    fullout = xr.merge([output, solpos])
    fullout.to_netcdf("combined_mcmip.nc")
