from pathlib import Path


import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import pyproj
except ImportError:
    pyproj = None


from erebos.adapters import goes


@pytest.fixture()
def example_file():
    ds = xr.Dataset(
        {
            "x": ("x", np.linspace(-0.1, -0.055, 100, dtype="float32")),
            "y": ("y", np.linspace(0.077, 0.11, 60, dtype="float32")),
        }
    )
    img = xr.DataArray(np.array(-2147483647, dtype="int32")).assign_attrs(
        {
            "long_name": "GOES-R ABI fixed grid projection",
            "grid_mapping_name": "geostationary",
            "perspective_point_height": 35786023.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
            "inverse_flattening": 298.2572221,
            "latitude_of_projection_origin": 0.0,
            "longitude_of_projection_origin": -137.0,
            "sweep_angle_axis": "x",
        }
    )
    return ds.assign({"goes_imager_projection": img})


@pytest.fixture()
def projection(example_file):
    if pyproj is None:
        pytest.skip("Pyproj not installed")
    sat_height = example_file.goes_imager_projection.perspective_point_height
    sat_lon = example_file.goes_imager_projection.longitude_of_projection_origin  # NOQA
    sweep = example_file.goes_imager_projection.sweep_angle_axis
    major_axis = example_file.goes_imager_projection.semi_major_axis
    minor_axis = example_file.goes_imager_projection.semi_minor_axis
    inverse_flattening = example_file.goes_imager_projection.inverse_flattening
    projection = pyproj.Proj(
        proj="geos",
        h=sat_height,
        lon_0=sat_lon,
        sweep=sweep,
        a=major_axis,
        b=minor_axis,
        rf=inverse_flattening,
    )
    return projection


def test_project_xy_to_latlon_against_proj4(projection, example_file):
    sat_height = example_file.goes_imager_projection.perspective_point_height
    XX, YY = np.meshgrid(example_file.x, example_file.y)
    proj4_latlon = projection(XX * sat_height, YY * sat_height, inverse=True)

    pylon, pylat = goes.project_xy_to_latlon(
        example_file.x, example_file.y, example_file
    )
    pylon[pylon < -180] += 360
    np.testing.assert_allclose(proj4_latlon, (pylon, pylat), rtol=1e-5)


@pytest.mark.parametrize(
    "filename,expected",
    [
        (
            "OR_ABI-L2-CMIPC-M3C02_G17_s20190010837189_e20190010839562_c20190010840048.nc",  # NOQA
            {
                "processing_level": "L2",
                "product": "CMIP",
                "scan_mode": "M3",
                "sector": "C",
                "channel": 2,
                "satellite": "G17",
                "start": pd.Timestamp("20190101T083718Z"),
                "end": pd.Timestamp("20190101T083956Z"),
                "creation": pd.Timestamp("20190101T084004Z"),
            },
        ),
        (
            "OR_ABI-L1b-RadM1-M6C03_G16_s20190040837189_e20190040839562_c20190040840048.nc",  # NOQA
            {
                "processing_level": "L1b",
                "product": "Rad",
                "scan_mode": "M6",
                "sector": "M1",
                "channel": 3,
                "satellite": "G16",
                "start": pd.Timestamp("20190104T083718Z"),
                "end": pd.Timestamp("20190104T083956Z"),
                "creation": pd.Timestamp("20190104T084004Z"),
            },
        ),
        (
            Path(
                "/storage/projects/goes_alg/goes_data/southwest_adj"
                "OR_ABI-L2-MCMIPC-M6_G16_s20191162031206_e20191162033579_"
                "c20191162034099.nc"
            ),
            {
                "processing_level": "L2",
                "product": "MCMIP",
                "scan_mode": "M6",
                "sector": "C",
                "channel": 0,
                "satellite": "G16",
                "start": pd.Timestamp("20190426T203120Z"),
                "end": pd.Timestamp("20190426T203357Z"),
                "creation": pd.Timestamp("20190426T203409Z"),
            },
        ),
    ],
)
def test_goesfilename(filename, expected):
    goesfilename = goes.GOESFilename(filename)
    assert goesfilename.filename == Path(filename)
    for k, v in expected.items():
        assert getattr(goesfilename, k) == v
