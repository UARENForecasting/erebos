import datetime as dt


import cartopy.crs as ccrs
import numpy as np
import xarray as xr


def name_dimensions(ds):
    record_dims = {}
    for k, v in ds.dims.items():
        if v == len(ds.Latitude):
            record_dims[k] = "record"
        elif v == 10:
            record_dims[k] = "level"
        elif v == 1:
            record_dims[k] = "shot"
    record_dims[ds.Spacecraft_Position.dims[1]] = "ECR axis"
    return ds.rename_dims(record_dims)


def add_proper_time(ds):
    time_arr = (
        ds.Profile_Time
        + np.array(dt.datetime(1993, 1, 1), dtype="datetime64[ns]")
        - np.array(dt.timedelta(seconds=10), dtype="timedelta64[ns]")
    )
    time_arr.name = "time"
    mt = time_arr.min() + (time_arr.max() - time_arr.min()) / 2
    return ds.assign({"time": time_arr}).assign_attrs(mean_time=mt.values)


def add_cloud_vars(ds):
    cloud_mask = (ds.Feature_Classification_Flags & 0b111 != 2).values
    cta = np.ma.array(ds.Layer_Top_Altitude.values, mask=cloud_mask, fill_value=np.nan)
    cta = xr.DataArray(cta, dims=("record", "level"))
    cba = np.ma.array(ds.Layer_Base_Altitude.values, mask=cloud_mask, fill_value=np.nan)
    cba = xr.DataArray(cba, dims=("record", "level"))
    ct = cta - cba

    cloud_type = (
        np.ma.array(
            (ds.Feature_Classification_Flags.values >> 9) & 0b111,
            mask=cloud_mask,
            fill_value=8,
        )
        .filled()
        .astype("uint8")
    )
    cloud_type = xr.DataArray(
        cloud_type,
        dims=("record", "level"),
        attrs={
            "flag_values": np.arange(9, dtype="uint8"),
            "flag_meanings": np.array(
                [
                    "low overcast, transparent",
                    "low overcast, opaque",
                    "transition, stratocumulus",
                    "low, broken cumulus",
                    "altocumulus (transparent)",
                    "altostratus (opaque)",
                    "cirrus (transparent)",
                    "deep convective (opaque)",
                    "no cloud",
                ]
            ),
        },
    )

    return ds.assign(
        {
            "cloud_top_altitude": cta,
            "cloud_type": cloud_type,
            "cloud_base_altitude": cba,
            "cloud_thickness": ct,
        }
    ).rename({"Number_Layers_Found": "cloud_layers"})


def add_other_convenience_vars(ds):
    day = xr.DataArray(
        ds.Day_Night_Flag.values,
        dims=("record", "shot"),
        attrs={
            "flag_values": np.array([0, 1], dtype="uint8"),
            "flag_meanings": np.array(["daytime", "nighttime"]),
        },
    )
    return ds.update({"Day_Night_Flag": day}).rename(
        {
            "Day_Night_Flag": "day_night_flag",
            "DEM_Surface_Elevation": "surface_elevation",
            "Solar_Azimuth_Angle": "solar_azimuth",
            "Solar_Zenith_Angle": "solar_zenith",
        }
    )


def add_projection(ds):
    return ds.assign_coords(erebos_crs=ccrs.PlateCarree())


def add_spacecraft_location(ds):
    x = ds.Spacecraft_Position[:, 1].values
    y = ds.Spacecraft_Position[:, 2].values
    z = ds.Spacecraft_Position[:, 0].values
    loc = xr.DataArray([x, y, z], dims=ds.Spacecraft_Position.dims[::-1])
    return ds.assign_coords(spacecraft_location=loc)


def process_calipso_dataset(ds):
    return (
        ds.pipe(name_dimensions)
        .pipe(add_proper_time)
        .pipe(add_cloud_vars)
        .pipe(add_projection)
        .pipe(add_other_convenience_vars)
        .pipe(add_spacecraft_location)
    )
