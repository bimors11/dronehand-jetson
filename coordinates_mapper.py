from math import cos, pi, sin

from drone_manager import Telemetry

EARTH_RADIUS_M = 6_371_000


def adjust_coordinates(lat_o: float, lng_o: float, east_delta: float, north_delta: float):
    """
    Adjusts the latitude and longitude by the given deltas in meters.

    :param lat_o: The original latitude.
    :param lng_o: The original longitude.
    :param east_delta: The distance to move east (positive) or west (negative) in meters.
    :param north_delta: The distance to move north (positive) or south (negative) in meters.
    """
    latitude = lat_o + (north_delta / EARTH_RADIUS_M) * (180 / pi)
    longitude = lng_o + (east_delta / EARTH_RADIUS_M) * (180 / pi) / cos(lat_o * pi / 180)

    return latitude, longitude


def get_object_coordinates(ground_width: float, ground_height: float, object_cx: float, object_cy: float, telemetry: Telemetry):
    """Calculate the latitude and longitude of an object in the camera's field of view."""
    angle = -telemetry.heading * pi / 180

    object_x = object_cx * 2 - 1
    object_y = 1 - object_cy * 2

    object_x = object_x * ground_width
    object_y = object_y * ground_height

    rotated_object_x = object_x * cos(angle) - object_y * sin(angle)
    rotated_object_y = object_x * sin(angle) + object_y * cos(angle)

    return list(adjust_coordinates(telemetry.latitude, telemetry.longitude, rotated_object_x, rotated_object_y))
