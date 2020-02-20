from dataclasses import dataclass, asdict
from pathlib import Path


import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


from erebos.utils import RotatedECRPosition


def project_xy_to_latlon(x, y, goes_file):
    proj_info = goes_file.goes_imager_projection
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis
    f = r_eq / r_pol

    # create meshgrid filled with radian angles
    lat_rad, lon_rad = np.meshgrid(x, y)

    # lat/lon calc routine from satellite radian angle vectors
    lambda_0 = np.radians(lon_origin)

    Vx = np.cos(lat_rad) * np.cos(lon_rad)
    Vy = -1.0 * np.sin(lat_rad)
    Vz = np.cos(lat_rad) * np.sin(lon_rad)

    av = Vx ** 2 + Vy ** 2 + f ** 2 * Vz ** 2
    bv = -2.0 * H * Vx
    cv = H ** 2 - r_eq ** 2

    r_s = (-bv - np.sqrt(bv ** 2 - 4 * av * cv)) / (2 * av)

    lon = np.degrees(lambda_0 - np.arctan2(Vy, H / r_s - Vx))
    lon[lon < -180] += 360
    lat = np.degrees(np.arctan2(f ** 2 * Vz, np.sqrt((H / r_s - Vx) ** 2 + Vy ** 2)))
    return lon, lat


def assign_latlon(goes_file):
    lon, lat = project_xy_to_latlon(goes_file.x, goes_file.y, goes_file)
    lon_arr = xr.DataArray(lon, dims=("y", "x"))
    lat_arr = xr.DataArray(lat, dims=("y", "x"))
    return goes_file.assign_coords(lat=lat_arr, lon=lon_arr)


def add_projection(ds):
    proj_info = ds.goes_imager_projection
    globe = ccrs.Globe(
        semimajor_axis=proj_info.semi_major_axis.item(),
        semiminor_axis=proj_info.semi_minor_axis.item(),
        inverse_flattening=proj_info.inverse_flattening.item(),
    )
    crs = ccrs.Geostationary(
        globe=globe,
        satellite_height=proj_info.perspective_point_height.item(),
        central_longitude=proj_info.longitude_of_projection_origin.item(),
        sweep_axis=proj_info.sweep_angle_axis,
    )
    return ds.assign_coords(crs=crs).update(
        {
            "x": ds.x * proj_info.perspective_point_height.item(),
            "y": ds.y * proj_info.perspective_point_height.item(),
        }
    )


def add_spacecraft_location(ds):
    rep = RotatedECRPosition.from_geodetic(
        0,
        ds.goes_imager_projection.longitude_of_projection_origin.item(),
        ds.goes_imager_projection.perspective_point_height.item(),
    )
    loc = xr.DataArray(np.array(rep)[:, None], dims=("ECR axis", "locations"))
    return ds.assign_coords(spacecraft_location=loc)


def add_mean_time(ds):
    return ds.assign_attrs(mean_time=ds.t.values)


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
