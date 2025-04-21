# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import ephem
from geopy.distance import great_circle  # Keep existing imports

# Import your Satellite class definition (adjust path if necessary)
from src.dynamic_state.topology import Satellite


def distance_m_between_satellites(
    sat1: Satellite, sat2: Satellite, epoch_str: str, date_str: str
) -> float:
    """
    Computes the straight distance between two satellites in meters.

    Accepts custom Satellite wrapper objects.

    :param sat1:       The first Satellite object.
    :param sat2:       The other Satellite object.
    :param epoch_str:  Epoch time string (e.g., "2000-01-01 00:00:00").
    :param date_str:   The time instant string (e.g., "2000-01-01 00:00:00").

    :return: The distance between the satellites in meters (float).
    :raises AttributeError: If satellite objects lack expected position/ephem attributes.
    :raises ValueError: If ephem objects are invalid.
    :raises RuntimeError: For other ephem calculation errors.
    """
    try:
        # 1. Extract underlying ephem.Body objects
        #    Choose manual or direct based on which one is populated/needed
        if not hasattr(sat1, "position") or not hasattr(sat1.position, "ephem_obj_manual"):
            raise AttributeError(
                f"Satellite object for sat_id {getattr(sat1, 'id', 'UNKNOWN')} is missing position or ephem_obj_manual"
            )
        ephem_body1 = sat1.position.ephem_obj_manual  # Or ephem_obj_direct

        if not hasattr(sat2, "position") or not hasattr(sat2.position, "ephem_obj_manual"):
            raise AttributeError(
                f"Satellite object for sat_id {getattr(sat2, 'id', 'UNKNOWN')} is missing position or ephem_obj_manual"
            )
        ephem_body2 = sat2.position.ephem_obj_manual  # Or ephem_obj_direct

        if not isinstance(ephem_body1, ephem.Body) or not isinstance(ephem_body2, ephem.Body):
            raise ValueError("Extracted ephem objects are not valid ephem.Body types.")

        # 2. Create an observer (position doesn't strictly matter for separation angle,
        #    but range calculation depends on it - using 0,0 is fine here)
        observer = ephem.Observer()
        observer.epoch = epoch_str  # TLE epoch
        observer.date = date_str  # Current time
        observer.lat = "0"  # degrees string
        observer.lon = "0"  # degrees string
        observer.elevation = 0.0

        # 3. Calculate the relative location by computing the ephem.Body objects
        ephem_body1.compute(observer)
        ephem_body2.compute(observer)

        # 4. Calculate the separation angle using the computed ephem.Body objects
        # Ensure ephem.separation result is converted correctly (it's often an ephem.Angle)
        angle_radians = float(ephem.separation(ephem_body1, ephem_body2))

        # 5. Get ranges from the *computed* ephem.Body objects
        range1 = ephem_body1.range
        range2 = ephem_body2.range

        # Check for potential issues (e.g., satellite below horizon for the arbitrary observer)
        if range1 is None or range2 is None or range1 <= 0 or range2 <= 0:
            # This might happen if a satellite is below the horizon for the observer at 0,0
            # The geometric approach might be less stable in such cases.
            # Consider alternative (e.g., Cartesian distance) if this occurs frequently.
            # For now, return infinity or raise an error.
            # print(f"Warning: Invalid range detected for ISL distance ({range1}, {range2})") # Use logger
            return float("inf")  # Indicate invalid distance

        # 6. Calculate distance using Law of Cosines
        # c^2 = a^2 + b^2 - 2 * a * b * cos(C)
        cos_term = 2 * range1 * range2 * math.cos(angle_radians)
        distance_sq = (range1**2) + (range2**2) - cos_term

        # Handle potential floating point issues near zero
        if distance_sq < 0:
            # This can happen due to floating point errors if angle is near 0 or pi
            # and ranges are almost equal. Treat distance as near zero.
            # print(f"Warning: Negative value in sqrt for ISL distance ({distance_sq}). Clamping to 0.") # Use logger
            return 0.0

        distance_m = math.sqrt(distance_sq)
        return distance_m

    except (AttributeError, ValueError) as e:
        print(f"[distance_tools] Input Error calculating ISL distance: {e}")  # Use logger
        raise e  # Re-raise configuration/input errors
    except Exception as e:
        print(f"[distance_tools] Runtime Error calculating ISL distance: {e}")  # Use logger
        raise


def distance_m_ground_station_to_satellite(ground_station, satellite, epoch_str, date_str):
    """
    Computes the straight distance between a ground station and a satellite in meters

    :param ground_station:  The ground station
    :param satellite:       The satellite
    :param epoch_str:       Epoch time of the observer (ground station) (string)
    :param date_str:        The time instant when the distance should be measured (string)

    :return: The distance between the ground station and the satellite in meters
    """

    # Create an observer on the planet where the ground station is
    observer = ephem.Observer()
    observer.epoch = epoch_str
    observer.date = date_str
    observer.lat = str(
        ground_station["latitude_degrees_str"]
    )  # Very important: string argument is in degrees.
    observer.lon = str(
        ground_station["longitude_degrees_str"]
    )  # DO NOT pass a float as it is interpreted as radians
    observer.elevation = ground_station["elevation_m_float"]

    # Compute distance from satellite to observer
    satellite.compute(observer)

    # Return distance
    return satellite.range


def geodesic_distance_m_between_ground_stations(ground_station_1, ground_station_2):
    """
    Calculate the geodesic distance between two ground stations.

    :param ground_station_1:         First ground station
    :param ground_station_2:         Another ground station

    :return: Geodesic distance in meters
    """

    # WGS72 value; taken from https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    earth_radius_km = 6378.135  # 6378135.0 meters

    return great_circle(
        (
            float(ground_station_1["latitude_degrees_str"]),
            float(ground_station_1["longitude_degrees_str"]),
        ),
        (
            float(ground_station_2["latitude_degrees_str"]),
            float(ground_station_2["longitude_degrees_str"]),
        ),
        radius=earth_radius_km,
    ).m


def straight_distance_m_between_ground_stations(ground_station_1, ground_station_2):
    """
    Calculate the straight distance between two ground stations (goes through the Earth)

    :param ground_station_1:         First ground station
    :param ground_station_2:         Another ground station

    :return: Straight distance in meters (goes through the Earth)
    """

    # WGS72 value; taken from https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    earth_radius_m = 6378135.0

    # First get the angle between the two ground stations from the Earth's core
    fraction_of_earth_circumference = geodesic_distance_m_between_ground_stations(
        ground_station_1, ground_station_2
    ) / (earth_radius_m * 2.0 * math.pi)
    angle_radians = fraction_of_earth_circumference * 2 * math.pi

    # Now see the Earth as a circle you know the hypotenuse, and half the angle is that of the triangle
    # with the 90 degree corner. Multiply by two to get the straight distance.
    polygon_side_m = 2 * math.sin(angle_radians / 2.0) * earth_radius_m

    return polygon_side_m


def create_basic_ground_station_for_satellite_shadow(satellite, epoch_str, date_str):
    """
    Calculate the (latitude, longitude) of the satellite shadow on the Earth and creates a ground station there.

    :param satellite:   Satellite
    :param epoch_str:   Epoch (string)
    :param date_str:    Time moment (string)

    :return: Basic ground station
    """

    satellite.compute(date_str, epoch=epoch_str)

    return {
        "gid": -1,
        "name": "Shadow of " + satellite.name,
        "latitude_degrees_str": str(math.degrees(satellite.sublat)),
        "longitude_degrees_str": str(math.degrees(satellite.sublong)),
        "elevation_m_float": 0,
    }


def geodetic2cartesian(lat_degrees, lon_degrees, ele_m):
    """
    Compute geodetic coordinates (latitude, longitude, elevation) to Cartesian coordinates.

    :param lat_degrees: Latitude in degrees (float)
    :param lon_degrees: Longitude in degrees (float)
    :param ele_m:  Elevation in meters

    :return: Cartesian coordinate as 3-tuple of (x, y, z)
    """

    #
    # Adapted from: https://github.com/andykee/pygeodesy/blob/master/pygeodesy/transform.py
    #

    # WGS72 value,
    # Source: https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    a = 6378135.0

    # Ellipsoid flattening factor; WGS72 value
    # Taken from https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    f = 1.0 / 298.26

    # First numerical eccentricity of ellipsoid
    e = math.sqrt(2.0 * f - f * f)
    lat = lat_degrees * (math.pi / 180.0)
    lon = lon_degrees * (math.pi / 180.0)

    # Radius of curvature in the prime vertical of the surface of the geodetic ellipsoid
    v = a / math.sqrt(1.0 - e * e * math.sin(lat) * math.sin(lat))

    x = (v + ele_m) * math.cos(lat) * math.cos(lon)
    y = (v + ele_m) * math.cos(lat) * math.sin(lon)
    z = (v * (1.0 - e * e) + ele_m) * math.sin(lat)

    return x, y, z
