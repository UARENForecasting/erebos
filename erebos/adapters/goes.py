from dataclasses import dataclass, asdict
from pathlib import Path
import warnings


import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


from erebos.utils import RotatedECRPosition


def project_xy_to_latlon(x, y, goes_file):
    ds = goes_file.erebos
    crs = ds.crs
    X, Y = np.meshgrid(ds.erebos.x, ds.erebos.y)
    lonlat = ccrs.Geodetic(globe=crs.globe).transform_points(crs, X, Y)
    lon = lonlat[:, :, 0].astype("float32")
    lat = lonlat[:, :, 1].astype("float32")
    return lon, lat


def assign_latlon(goes_file):
    sel = {}
    for dim in ("y", "x"):
        if dim not in goes_file.dims:
            goes_file = goes_file.expand_dims(dim)
            sel[dim] = 0

    lon, lat = project_xy_to_latlon(goes_file.x, goes_file.y, goes_file)
    lon_arr = xr.DataArray(lon.astype("float32"), dims=("y", "x"))
    lat_arr = xr.DataArray(lat.astype("float32"), dims=("y", "x"))
    lon_arr.encoding = {"zlib": True, "dtype": "float32", "scale_factor": 0.0001}
    lat_arr.encoding = {"zlib": True, "dtype": "float32", "scale_factor": 0.0001}
    out = goes_file.assign_coords(latitude=lat_arr, longitude=lon_arr)
    if sel:
        out = out.isel(sel)
    return out


def restrict_domain(goes_file, lon_limits, lat_limits):
    ds = goes_file.erebos
    crs = ds.crs
    x = ds.erebos_x
    y = ds.erebos_y
    pts = crs.transform_points(
        crs.as_geodetic(), np.asarray(lon_limits), np.asarray(lat_limits)
    )
    okx = (x > pts[0, 0]) & (x < pts[1, 0])
    oky = (y > pts[0, 1]) & (y < pts[1, 1])
    return goes_file.isel(y=oky, x=okx)


def assign_solarposition_variables(goes_file):
    from pvlib import spa, irradiance

    ds = goes_file
    sel = {}
    for dim in ("t", "y", "x"):
        if dim not in ds.dims:
            ds = ds.expand_dims(dim)
            sel[dim] = 0

    time_data = ds.t.data.astype(int)[:, None, None] / 1e9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # THIS SHOULD SUPPORT MANY t at once
        solpos = spa.solar_position(
            time_data,
            ds.latitude.expand_dims("t").data,
            ds.longitude.expand_dims("t").data,
            0,
            101.325,
            12,
            67,
            0.5667,
        )

    er = irradiance.get_extra_radiation(time_data)
    extra = xr.DataArray(
        np.ones((ds.dims["t"], ds.dims["y"], ds.dims["x"]), dtype="float32") * er,
        dims=("t", "y", "x"),
    )
    extra.encoding = {
        "dtype": "uint8",
        "scale_factor": 0.1,
        "offset": er,
        "zlib": True,
        "_FillValue": 255,
    }
    zen = xr.DataArray(solpos[1].astype("float32"), dims=("t", "y", "x"))
    zen.encoding = {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": -32768,
        "zlib": True,
    }
    az = xr.DataArray(solpos[4].astype("float32"), dims=("t", "y", "x"))
    az.encoding = {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": -32768,
        "zlib": True,
    }
    out = ds.assign(
        {"solar_zenith": zen, "solar_azimuth": az, "solar_extra_radiation": extra}
    )
    if sel:
        out = out.isel(sel)
    return out


def assign_surface_elevation(goes_file):
    # FIX ME

    sel = {}
    for dim in ("y", "x"):
        if dim not in goes_file.dims:
            goes_file = goes_file.expand_dims(dim)
            sel[dim] = 0

    elev = xr.DataArray(
        np.zeros((goes_file.dims["y"], goes_file.dims["x"]), dtype="float32"),
        dims=("y", "x"),
        name="elevation",
    )
    elev.assign_attrs(units="km")
    elev.encoding = {
        "dtype": "uint16",
        "zlib": True,
        "_FillValue": 65535,
        "offset": 1,
        "scale_factor": 0.001,
    }
    out = goes_file.assign(surface_elevation=elev)
    if sel:
        out = out.isel(sel)
    return out


def add_projection(ds):
    proj_info = ds.goes_imager_projection
    pattrs = proj_info.attrs
    for k, v in pattrs.items():
        if isinstance(v, np.ndarray):
            pattrs[k] = v.item()
    globe = ccrs.Globe(
        semimajor_axis=proj_info.semi_major_axis,
        semiminor_axis=proj_info.semi_minor_axis,
        inverse_flattening=proj_info.inverse_flattening,
    )
    crs = ccrs.Geostationary(
        globe=globe,
        satellite_height=proj_info.perspective_point_height,
        central_longitude=proj_info.longitude_of_projection_origin,
        sweep_axis=proj_info.sweep_angle_axis,
    )
    return ds.assign_coords(
        erebos_crs=crs,
        erebos_x=ds.x * proj_info.perspective_point_height,
        erebos_y=ds.y * proj_info.perspective_point_height,
    )


def add_spacecraft_location(ds):
    rep = RotatedECRPosition.from_geodetic(
        0,
        ds.goes_imager_projection.longitude_of_projection_origin,
        ds.goes_imager_projection.perspective_point_height,
    )
    loc = xr.DataArray(np.array(rep)[:, None], dims=("ECR axis", "locations"))
    return ds.assign_coords(erebos_spacecraft_location=loc)


def add_mean_time(ds):
    return ds.assign_attrs(erebos_mean_time=ds.t.values)


def process_goes_dataset(ds):
    return ds.pipe(add_projection).pipe(add_spacecraft_location).pipe(add_mean_time)


@dataclass(frozen=True)
class GOESFilename:
    filename: Path
    processing_level: str = None
    product: str = None
    scan_mode: str = None
    sector: str = None
    channel: int = None
    satellite: str = None
    start: pd.Timestamp = None
    end: pd.Timestamp = None
    creation: pd.Timestamp = None

    @classmethod
    def from_path(cls, filename):
        if not isinstance(filename, Path):
            filename = Path(filename)
        new = {"filename": filename}
        split = filename.name.split("-")
        nextsplit = split[-1].split("_")
        new["processing_level"] = split[1]
        sector = split[2][-1]
        try:
            int(sector)
        except ValueError:
            product = split[2][:-1]
        else:
            sector = split[2][-2:]
            product = split[2][:-2]
        new["product"] = product
        new["sector"] = sector
        new["scan_mode"] = nextsplit[0][:2]
        try:
            channel = int(nextsplit[0][-2:])
        except ValueError:
            channel = 0
        new["channel"] = channel
        new["satellite"] = nextsplit[1]
        timefmt = "%Y%j%H%M%S%f"
        new["start"] = pd.to_datetime(nextsplit[2][1:], format=timefmt).tz_localize(
            "UTC"
        )
        new["end"] = pd.to_datetime(nextsplit[3][1:], format=timefmt).tz_localize("UTC")
        new["creation"] = pd.to_datetime(
            nextsplit[4][1:-3], format=timefmt
        ).tz_localize("UTC")
        return cls(**new)

    def _base_out(self, **kwargs):
        dict_ = asdict(self)
        dict_.update(kwargs)
        for key in ("start", "end", "creation"):
            dict_[key] = dict_[key].strftime("%Y%j%H%M%S%f")[:-5]
        base = ("OR_ABI-{processing_level}-{product}{sector}-{scan_mode}").format(
            **dict_
        )
        if dict_.get("channel", 0):
            base += "C{channel:02d}".format(**dict_)
        base += "_{satellite}_s{start}_".format(**dict_)
        return base, dict_

    def to_path(self, *, glob_ready=False, **kwargs):
        base, dict_ = self._base_out(**kwargs)
        if glob_ready:
            out = base[:-2] + "*.nc"
        else:
            out = base + "e{end}_c{creation}.nc".format(**dict_)
        return Path(out)

    def to_s3_prefix(self, **kwargs):
        base, dict_ = self._base_out(**kwargs)
        start = kwargs.get("start", self.start)
        prefix = "ABI-{processing_level}-{product}{sector}/".format(**dict_)
        prefix += start.strftime("%Y/%j/%H/")
        prefix += base
        return prefix
