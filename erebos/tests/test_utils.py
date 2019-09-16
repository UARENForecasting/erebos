import numpy as np
import pandas as pd
import pytest


from erebos import utils


GOES17_POS = (-28648.05623634, 0.0, -30937.08652789)
GOES17_LATLONH = (0.0, -137.2, 35786.0)
APPARENT_POS = (-5157.372532, 3567.386338, -1126.998925)
APPARENT_LATLONH = (34.04925, -102.326614, 0.0)
CORRECTED_POS = (-5167.923661, 3565.784003, -1140.388491)
CORRECTED_LATLONH = (33.971087, -102.443847, 10.0)


@pytest.fixture(
    params=(
        (GOES17_POS, GOES17_LATLONH),
        (APPARENT_POS, APPARENT_LATLONH),
        (CORRECTED_POS, CORRECTED_LATLONH),
    )
)
def known_positions(request):
    return request.param


def test_rotatedecrposition_from_geodetic(known_positions):
    pos = utils.RotatedECRPosition.from_geodetic(*known_positions[1])
    expected = utils.RotatedECRPosition(*known_positions[0])
    np.testing.assert_allclose(pos, expected, rtol=1e-5)


def test_rotatedecrposition_roundtrip_latlon(known_positions):
    exp = known_positions[1][:2]
    pos = utils.RotatedECRPosition.from_geodetic(*known_positions[1])
    back = pos.to_geodetic()
    np.testing.assert_allclose(exp, back)


def test_find_actual_cloud_position_no_height():
    app_cloud = utils.RotatedECRPosition(*APPARENT_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    actual = utils.find_actual_cloud_position(sat_pos, app_cloud, 0)
    np.testing.assert_allclose(actual, app_cloud, rtol=1e-6)


def test_find_actual_cloud_position():
    app_cloud = utils.RotatedECRPosition(*APPARENT_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    actual = utils.find_actual_cloud_position(sat_pos, app_cloud, CORRECTED_LATLONH[-1])
    exp = utils.RotatedECRPosition(*CORRECTED_POS)
    np.testing.assert_allclose(actual, exp)


def test_find_apparent_cloud_position():
    cloud = utils.RotatedECRPosition(*CORRECTED_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    app = utils.find_apparent_cloud_position(sat_pos, cloud)
    exp = utils.RotatedECRPosition(*APPARENT_POS)
    np.testing.assert_allclose(app, exp, rtol=1e-6)


def test_find_apparent_cloud_position_terrain():
    cloud = utils.RotatedECRPosition(*CORRECTED_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    app = utils.find_apparent_cloud_position(sat_pos, cloud, CORRECTED_LATLONH[-1])
    # no shift when terrain is as high as cloud
    np.testing.assert_allclose(app, cloud, rtol=1e-6)


def test_find_apparent_cloud_position_on_surface():
    cloud = utils.RotatedECRPosition(*APPARENT_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    app = utils.find_apparent_cloud_position(sat_pos, cloud)
    np.testing.assert_allclose(app, cloud, rtol=1e-6)


def test_find_position_roundtrip():
    app_cloud = utils.RotatedECRPosition(*APPARENT_POS)
    sat_pos = utils.RotatedECRPosition(*GOES17_POS)
    actual = utils.find_actual_cloud_position(sat_pos, app_cloud, CORRECTED_LATLONH[-1])
    app = utils.find_apparent_cloud_position(sat_pos, actual)
    np.testing.assert_allclose(app, app_cloud, rtol=1e-6)


def test_get_solar_ecr_position():
    time = pd.Timestamp("20190726T125619-06:00")
    cloud = utils.RotatedECRPosition(*CORRECTED_POS)
    sol = utils.get_solar_ecr_position(
        time.value / 1e9, CORRECTED_LATLONH[0], CORRECTED_LATLONH[1]
    )
    pos = utils.find_apparent_cloud_position(sol, cloud)
    # sun should be at same latitude at this time
    np.testing.assert_allclose(pos.to_geodetic()[1], cloud.to_geodetic()[1])
    assert pos.to_geodetic()[0] > cloud.to_geodetic()[0]
