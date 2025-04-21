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
import unittest

import ephem
import exputil
from astropy import units as u
from astropy.time import Time

from src.distance_tools import *
from src.dynamic_state.topology import Satellite
from src.distance_tools import (
    create_basic_ground_station_for_satellite_shadow,
    distance_m_between_satellites,
    distance_m_ground_station_to_satellite,
    geodesic_distance_m_between_ground_stations,
    geodetic2cartesian,
    straight_distance_m_between_ground_stations,
)
from src.ground_stations import read_ground_stations_basic


class TestDistanceTools(unittest.TestCase):

    def test_distance_between_satellites(self):
        # --- Create ephem.Body objects ---
        ephem_sat_0 = ephem.readtle(
            "Kuiper-630 0",
            "1 00001U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    04",
            "2 00001  51.9000   0.0000 0000001   0.0000   0.0000 14.80000000    02",
        )
        ephem_sat_1 = ephem.readtle(
            "Kuiper-630 1",
            "1 00002U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00002  51.9000   0.0000 0000001   0.0000  10.5882 14.80000000    07",
        )
        ephem_sat_17 = ephem.readtle(
            "Kuiper-630 17",
            "1 00018U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    02",
            "2 00018  51.9000   0.0000 0000001   0.0000 180.0000 14.80000000    09",
        )
        ephem_sat_18 = ephem.readtle(
            "Kuiper-630 18",
            "1 00019U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    03",
            "2 00019  51.9000   0.0000 0000001   0.0000 190.5882 14.80000000    04",
        )
        ephem_sat_19 = ephem.readtle(
            "Kuiper-630 19",
            "1 00020U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00020  51.9000   0.0000 0000001   0.0000 201.1765 14.80000000    05",
        )

        # --- Wrap ephem.Body objects in Satellite objects ---
        # Assign unique IDs
        sat_obj_0 = Satellite(id=0, ephem_obj_manual=ephem_sat_0, ephem_obj_direct=ephem_sat_0)
        sat_obj_1 = Satellite(id=1, ephem_obj_manual=ephem_sat_1, ephem_obj_direct=ephem_sat_1)
        sat_obj_17 = Satellite(id=17, ephem_obj_manual=ephem_sat_17, ephem_obj_direct=ephem_sat_17)
        sat_obj_18 = Satellite(id=18, ephem_obj_manual=ephem_sat_18, ephem_obj_direct=ephem_sat_18)
        sat_obj_19 = Satellite(id=19, ephem_obj_manual=ephem_sat_19, ephem_obj_direct=ephem_sat_19)

        for extra_time_ns in [
            0,
            1,
            1000,
            1000000,
            1000000000,
            60000000000,
            10 * 60000000000,
            20 * 60000000000,
            30 * 60000000000,
            40 * 60000000000,
            50 * 60000000000,
            60 * 60000000000,
            70 * 60000000000,
            80 * 60000000000,
            90 * 60000000000,
            100 * 60000000000,
        ]:
            epoch = Time("2000-01-01 00:00:00", scale="tdb")
            # Ensure time calculation is correct and compatible with str() conversion expected by distance func
            time_obj = epoch + extra_time_ns * u.ns
            time_str = time_obj.strftime("%Y/%m/%d %H:%M:%S")  # Format as YYYY/MM/DD HH:MM:SS
            epoch_str_for_ephem = str(epoch.strftime("%Y/%m/%d"))

            # --- Pass Satellite WRAPPER objects to the function ---
            # Distance to themselves should always be zero
            self.assertAlmostEqual(
                distance_m_between_satellites(
                    sat_obj_0,
                    sat_obj_0,
                    epoch_str_for_ephem,
                    time_str,
                ),
                0,
                delta=1e-3,
            )
            # ... other assertions using epoch_str_for_ephem and time_str ...
            dist_0_1 = distance_m_between_satellites(
                sat_obj_0, sat_obj_1, epoch_str_for_ephem, time_str
            )
            dist_1_0 = distance_m_between_satellites(
                sat_obj_1, sat_obj_0, epoch_str_for_ephem, time_str
            )
            self.assertAlmostEqual(dist_0_1, dist_1_0, delta=1e-3)

            dist_1_17 = distance_m_between_satellites(
                sat_obj_1, sat_obj_17, epoch_str_for_ephem, time_str
            )
            dist_17_1 = distance_m_between_satellites(
                sat_obj_17, sat_obj_1, epoch_str_for_ephem, time_str
            )
            self.assertAlmostEqual(dist_1_17, dist_17_1, delta=1e-3)

            dist_19_17 = distance_m_between_satellites(
                sat_obj_19, sat_obj_17, epoch_str_for_ephem, time_str
            )
            dist_17_19 = distance_m_between_satellites(
                sat_obj_17, sat_obj_19, epoch_str_for_ephem, time_str
            )
            self.assertAlmostEqual(dist_19_17, dist_17_19, delta=1e-3)

            # Distance between 0 and 1 should be less than between 0 and 18
            dist_0_18 = distance_m_between_satellites(
                sat_obj_0, sat_obj_18, epoch_str_for_ephem, time_str
            )
            # Re-use dist_0_1 calculated above
            self.assertGreater(dist_0_18, dist_0_1)

            # Triangle inequality
            # Re-use dist_17_1 calculated above
            dist_18_19 = distance_m_between_satellites(
                sat_obj_18, sat_obj_19, epoch_str_for_ephem, time_str
            )
            # Re-use dist_17_19 calculated above
            # Add a small tolerance epsilon for floating point comparisons
            epsilon = 1e-3
            self.assertGreater(
                dist_17_1 + dist_18_19 + epsilon,  # Use dist_17_1 instead of recalculating 17->18
                dist_17_19,
                f"Triangle inequality failed: {dist_17_1} + {dist_18_19} <= {dist_17_19}",
            )
            # Note: The original test used dist(17,18)+dist(18,19) > dist(17,19).
            # Need dist(17,18)
            dist_17_18 = distance_m_between_satellites(
                sat_obj_17, sat_obj_18, epoch_str_for_ephem, time_str
            )
            self.assertGreater(
                dist_17_18 + dist_18_19 + epsilon,
                dist_17_19,
                f"Triangle inequality failed: {dist_17_18} + {dist_18_19} <= {dist_17_19}",
            )

            # Polygon side calculation check (verify assumptions)
            # Earth radius = 6378135 m
            # Kuiper altitude = 630 km -> Orbit radius = 7008135 m
            # Assuming 34 satellites per plane (based on old comment? TLEs don't specify)
            num_sats_per_plane = 34  # This is an assumption from the old test logic
            polygon_side_m = 2 * (
                7008135.0 * math.sin(math.radians(360.0 / num_sats_per_plane) / 2.0)
            )

            # Compare calculated distances (allow for some variation from perfect circle)
            dist_17_18 = distance_m_between_satellites(
                sat_obj_17, sat_obj_18, epoch_str_for_ephem, time_str
            )
            self.assertTrue(
                0.9 * polygon_side_m <= dist_17_18 <= 1.1 * polygon_side_m,  # Looser bound?
                f"Dist 17-18 ({dist_17_18}) vs expected polygon side ({polygon_side_m}) out of bounds",
            )

            dist_18_19 = distance_m_between_satellites(
                sat_obj_18, sat_obj_19, epoch_str_for_ephem, time_str
            )
            self.assertTrue(
                0.9 * polygon_side_m <= dist_18_19 <= 1.1 * polygon_side_m,
                f"Dist 18-19 ({dist_18_19}) vs expected polygon side ({polygon_side_m}) out of bounds",
            )

            dist_0_1 = distance_m_between_satellites(
                sat_obj_0, sat_obj_1, epoch_str_for_ephem, time_str
            )
            self.assertTrue(
                0.9 * polygon_side_m <= dist_0_1 <= 1.1 * polygon_side_m,
                f"Dist 0-1 ({dist_0_1}) vs expected polygon side ({polygon_side_m}) out of bounds",
            )

    def test_distance_between_ground_stations(self):
        local_shell = exputil.LocalShell()

        # Create some ground stations
        with open("ground_stations.temp.txt", "w+") as f_out:
            f_out.write("0,Amsterdam,52.379189,4.899431,0\n")
            f_out.write("1,Paris,48.864716,2.349014,0\n")
            f_out.write("2,Rio de Janeiro,-22.970722,-43.182365,0\n")
            f_out.write("3,Manila,14.599512,120.984222,0\n")
            f_out.write("4,Perth,-31.953512,115.857048,0\n")
            f_out.write("5,Some place on Antarctica,-72.927148,33.450844,0\n")
            f_out.write("6,New York,40.730610,-73.935242,0\n")
            f_out.write("7,Some place in Greenland,79.741382,-53.143087,0")
        ground_stations = read_ground_stations_basic("ground_stations.temp.txt")

        # Distance to itself is always 0
        for i in range(8):
            self.assertEqual(
                geodesic_distance_m_between_ground_stations(ground_stations[i], ground_stations[i]),
                0,
            )
            self.assertEqual(
                straight_distance_m_between_ground_stations(ground_stations[i], ground_stations[i]),
                0,
            )

        # Direction does not matter
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(
                    geodesic_distance_m_between_ground_stations(
                        ground_stations[i], ground_stations[j]
                    ),
                    geodesic_distance_m_between_ground_stations(
                        ground_stations[j], ground_stations[i]
                    ),
                    delta=0.00001,
                )
                self.assertAlmostEqual(
                    straight_distance_m_between_ground_stations(
                        ground_stations[i], ground_stations[j]
                    ),
                    straight_distance_m_between_ground_stations(
                        ground_stations[j], ground_stations[i]
                    ),
                    delta=0.00001,
                )

                # Geodesic is always strictly greater than straight
                if i != j:
                    self.assertGreater(
                        geodesic_distance_m_between_ground_stations(
                            ground_stations[i], ground_stations[j]
                        ),
                        straight_distance_m_between_ground_stations(
                            ground_stations[i], ground_stations[j]
                        ),
                    )

        # Amsterdam to Paris
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[0], ground_stations[1]),
            430000,  # 430 km
            delta=1000.0,
        )

        # Amsterdam to New York
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[0], ground_stations[6]),
            5861000,  # 5861 km
            delta=5000.0,
        )

        # New York to Antarctica
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[6], ground_stations[5]),
            14861000,  # 14861 km
            delta=20000.0,
        )

        # Clean up
        local_shell.remove("ground_stations.temp.txt")

    def test_distance_ground_station_to_satellite(self):

        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        time = epoch + 100 * 1000 * 1000 * 1000 * u.ns

        # Two satellites
        telesat_18 = ephem.readtle(
            "Telesat-1015 18",
            "1 00019U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    03",
            "2 00019  98.9800  13.3333 0000001   0.0000 152.3077 13.66000000    04",
        )
        telesat_19 = ephem.readtle(
            "Telesat-1015 19",
            "1 00020U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00020  98.9800  13.3333 0000001   0.0000 180.0000 13.66000000    00",
        )

        # Their shadows
        shadow_18 = create_basic_ground_station_for_satellite_shadow(
            telesat_18, str(epoch), str(time)
        )
        shadow_19 = create_basic_ground_station_for_satellite_shadow(
            telesat_19, str(epoch), str(time)
        )

        # Distance to shadow should be around 1015km
        self.assertAlmostEqual(
            distance_m_ground_station_to_satellite(shadow_18, telesat_18, str(epoch), str(time)),
            1015000,  # 1015km
            delta=5000,  # Accurate within 5km
        )
        distance_shadow_19_to_satellite_19 = distance_m_ground_station_to_satellite(
            shadow_19, telesat_19, str(epoch), str(time)
        )
        self.assertAlmostEqual(
            distance_shadow_19_to_satellite_19,
            1015000,  # 1015km
            delta=5000,  # Accurate within 5km
        )

        # Distance between the two shadows:
        # 21.61890110054602, 96.54190305000301
        # -5.732296878862085, 92.0396062736707
        shadow_distance_m = geodesic_distance_m_between_ground_stations(shadow_18, shadow_19)
        self.assertAlmostEqual(
            shadow_distance_m,
            3080640,  # 3080.64 km, from Google Maps
            delta=5000,  # With an accuracy of 5km
        )

        # The Pythagoras distance must be within 10% assuming the geodesic does not cause to much of an increase
        distance_shadow_18_to_satellite_19 = distance_m_ground_station_to_satellite(
            shadow_18, telesat_19, str(epoch), str(time)
        )
        self.assertAlmostEqual(
            math.sqrt(shadow_distance_m**2 + distance_shadow_19_to_satellite_19**2),
            distance_shadow_18_to_satellite_19,
            delta=0.1 * math.sqrt(shadow_distance_m**2 + distance_shadow_19_to_satellite_19**2),
        )

        # Check that the hypotenuse is not exceeded
        straight_shadow_distance_m = straight_distance_m_between_ground_stations(
            shadow_18, shadow_19
        )
        self.assertGreater(
            distance_shadow_18_to_satellite_19,
            math.sqrt(straight_shadow_distance_m**2 + distance_shadow_19_to_satellite_19**2),
        )

        # Check what happens with cartesian coordinates
        a = geodetic2cartesian(
            float(shadow_18["latitude_degrees_str"]),
            float(shadow_18["longitude_degrees_str"]),
            shadow_18["elevation_m_float"],
        )
        b = geodetic2cartesian(
            float(shadow_19["latitude_degrees_str"]),
            float(shadow_19["longitude_degrees_str"]),
            shadow_19["elevation_m_float"],
        )

        # For now, we will keep a loose bound of 1% here, but it needs to be tightened
        # It mostly has to do with that the great circle does not account for the ellipsoid effect
        self.assertAlmostEqual(
            math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2),
            straight_shadow_distance_m,
            delta=20000,  # 20km
        )
