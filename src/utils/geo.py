from math import radians, sin, cos, sqrt, atan2

import pandas as pd
from numpy import datetime64
from pyproj import Geod
import numpy as np

_EARTH_RADIUS = 6378100  # expressed in meters


def haversine_distance(latitude_1: float, longitude_1: float, latitude_2: float,
                       longitude_2: float) -> float:
    """
    Given the coordinates of two geographical points, expressed with decimal degrees, returns the
    Haversine distance between the two points, in meters.

    See more about decimal degrees: https://en.wikipedia.org/wiki/Decimal_degrees

    :param latitude_1: latitude of the first point
    :param longitude_1: longitude of the first point
    :param latitude_2: latitude of the second point
    :param longitude_2: longitude of the second point
    :return: the Haversine distance (in meters) between the two points
    """

    # override the received coordinates by expressing them in radians
    coordinates = (latitude_1, longitude_1, latitude_2, longitude_2)
    latitude_1, longitude_1, latitude_2, longitude_2 = map(radians, coordinates)

    delta_longitude = longitude_2 - longitude_1
    delta_latitude = latitude_2 - latitude_1

    # compute Haversine distance accordingly to its formula
    a = sin(delta_latitude / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(
        delta_longitude / 2) ** 2
    return _EARTH_RADIUS * 2 * atan2(sqrt(a), sqrt(1 - a))


def interpolate_coordinates(trajectory: list[list[int]], start_time: datetime64,
                            frequency: int = 15) -> pd.DataFrame:
    """
    Interpolate trajectory points with a 1s granularity.

    :param trajectory: the list of coordinates, sampled with constant frequency
    :param start_time: the timestamp related to the first sample
    :param frequency: the number of seconds between each sample
    :return: a dataframe containing the interpolated coordinates for each timestamp
    """
    geoid = Geod(ellps="WGS84")
    num_seconds = len(trajectory) + (len(trajectory) - 1) * (frequency - 1)

    # first column is longitude, second column is latitude
    # each row is a different second
    interp_coord = np.empty((num_seconds, 2))

    interp_coord[0] = trajectory[0]

    for i in range(1, len(trajectory)):
        interpolated = geoid.npts(
            lon1=trajectory[i][0],
            lat1=trajectory[i][1],
            lon2=trajectory[i - 1][0],
            lat2=trajectory[i - 1][1],
            npts=(frequency - 1)
        )
        first_interp_sec = i * frequency - (frequency - 1)
        last_interp_sec = i * frequency  # excluded
        interp_coord[first_interp_sec:last_interp_sec] = interpolated

        interp_coord[i * frequency] = trajectory[i]

    return pd.DataFrame(
        data=interp_coord,
        index=pd.date_range(start_time, periods=num_seconds, freq="1s"),
        columns=("longitude", "latitude")
    )
