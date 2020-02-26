"""
Satellite
  X
    X
      X
        X
          Cloud
          X X
          X   X
          X     X
          X       X
          Actual   Apparent
        position   position
"""
from dataclasses import dataclass


import boto3
import numpy as np
from pvlib import spa


R_equator = 6378.137  # km
R_polar = 6356.752  # km
R_geo = R_equator + 35786.0  # km


@dataclass(frozen=True)
class RotatedECRPosition:
    X: float
    Y: float
    Z: float

    def __array__(self):
        """For convienence with testing functions"""
        return np.array([self.X, self.Y, self.Z])

    @classmethod
    def from_geodetic(cls, lat, lon, height_above_surface):
        rlat = np.radians(lat)
        rlon = np.radians(lon)
        R_surface = R_equator / np.sqrt(
            np.cos(rlat) ** 2 + np.sin(rlat) ** 2 * (R_equator / R_polar) ** 2
        )
        R = height_above_surface + R_surface
        X = R * np.cos(rlat) * np.sin(rlon)
        Y = R * np.sin(rlat)
        Z = R * np.cos(rlat) * np.cos(rlon)
        return cls(X, Y, Z)

    def to_geodetic(self):
        lat = np.degrees(np.arctan2(self.Y, np.sqrt(self.X ** 2 + self.Z ** 2)))
        lon = np.degrees(np.arctan2(self.X, self.Z))
        return lat, lon


def _parallax_adjustment(satellite_position, cloud_position, cloud_height):
    B = ((R_equator + cloud_height) / (R_polar + cloud_height)) ** 2
    Xd = satellite_position.X - cloud_position.X
    Yd = satellite_position.Y - cloud_position.Y
    Zd = satellite_position.Z - cloud_position.Z
    C = Xd ** 2 + B * Yd ** 2 + Zd ** 2
    D = 2 * (Xd * cloud_position.X + B * Yd * cloud_position.Y + Zd * cloud_position.Z)
    E = (
        cloud_position.X ** 2
        + B * cloud_position.Y ** 2
        + cloud_position.Z ** 2
        - (R_equator + cloud_height) ** 2
    )
    A = (-D + np.sqrt(D ** 2 - 4 * C * E)) / (2 * C)
    Xp = cloud_position.X + A * Xd
    Yp = cloud_position.Y + A * Yd
    Zp = cloud_position.Z + A * Zd
    return RotatedECRPosition(Xp, Yp, Zp)


def find_actual_cloud_position(
    satellite_position, apparent_cloud_position, cloud_height
):
    return _parallax_adjustment(
        satellite_position, apparent_cloud_position, cloud_height
    )


def find_apparent_cloud_position(
    satellite_position, actual_cloud_position, terrain_height=0.0
):
    return _parallax_adjustment(
        satellite_position, actual_cloud_position, terrain_height
    )


def get_solar_ecr_position(unixtime, lat, lon, delta_t=67.0):
    jd = spa.julian_day(unixtime)
    jde = spa.julian_ephemeris_day(jd, delta_t)
    jc = spa.julian_century(jd)
    jce = spa.julian_ephemeris_century(jde)
    jme = spa.julian_ephemeris_millennium(jce)
    R = spa.heliocentric_radius_vector(jme) * 149597870.7
    L = spa.heliocentric_longitude(jme)
    B = spa.heliocentric_latitude(jme)
    Theta = spa.geocentric_longitude(L)
    beta = spa.geocentric_latitude(B)
    x0 = spa.mean_elongation(jce)
    x1 = spa.mean_anomaly_sun(jce)
    x2 = spa.mean_anomaly_moon(jce)
    x3 = spa.moon_argument_latitude(jce)
    x4 = spa.moon_ascending_longitude(jce)
    delta_psi = spa.longitude_nutation(jce, x0, x1, x2, x3, x4)
    delta_epsilon = spa.obliquity_nutation(jce, x0, x1, x2, x3, x4)
    epsilon0 = spa.mean_ecliptic_obliquity(jme)
    epsilon = spa.true_ecliptic_obliquity(epsilon0, delta_epsilon)
    delta_tau = spa.aberration_correction(R)
    lamd = spa.apparent_sun_longitude(Theta, delta_psi, delta_tau)
    v0 = spa.mean_sidereal_time(jd, jc)
    v = spa.apparent_sidereal_time(v0, delta_psi, epsilon)
    alpha = spa.geocentric_sun_right_ascension(lamd, epsilon, beta)
    delta = spa.geocentric_sun_declination(lamd, epsilon, beta)
    H = spa.local_hour_angle(v, lon, alpha)
    lat = np.radians(delta)
    lon = np.radians(H + lon)
    X = R * np.cos(lat) * np.sin(lon)
    Y = R * np.sin(lat)
    Z = R * np.cos(lat) * np.cos(lon)
    return RotatedECRPosition(X, Y, Z)


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
