from dataclasses import dataclass
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
    return ds.assign_coords(crs=crs).update(
        {
            "x": ds.x * proj_info.perspective_point_height,
            "y": ds.y * proj_info.perspective_point_height,
        }
    )


def add_spacecraft_location(ds):
    rep = RotatedECRPosition.from_geodetic(
        0,
        ds.goes_imager_projection.longitude_of_projection_origin,
        ds.goes_imager_projection.perspective_point_height,
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

    def __post_init__(self):
        if not isinstance(self.filename, Path):
            object.__setattr__(self, "filename", Path(self.filename))
        split = self.filename.name.split("-")
        nextsplit = split[-1].split("_")
        object.__setattr__(self, "processing_level", split[1])
        sector = split[2][-1]
        try:
            int(sector)
        except ValueError:
            product = split[2][:-1]
        else:
            sector = split[2][-2:]
            product = split[2][:-2]
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "scan_mode", nextsplit[0][:2])
        try:
            channel = int(nextsplit[0][-2:])
        except ValueError:
            channel = 0
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "satellite", nextsplit[1])
        timefmt = "%Y%j%H%M%S"
        object.__setattr__(
            self,
            "start",
            pd.to_datetime(nextsplit[2][1:-1], format=timefmt).tz_localize("UTC"),
        )
        object.__setattr__(
            self,
            "end",
            pd.to_datetime(nextsplit[3][1:-1], format=timefmt).tz_localize("UTC"),
        )
        object.__setattr__(
            self,
            "creation",
            pd.to_datetime(nextsplit[4][1:-4], format=timefmt).tz_localize("UTC"),
        )
