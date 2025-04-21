# tests/dynamic_state/test_generate_dynamic_state_integration.py

import math
import unittest

import ephem  # For creating real satellite objects from TLEs
from astropy.time import Time

# Modules and classes to test/use
from src.dynamic_state.generate_dynamic_state import generate_dynamic_state_at
from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite

# NOTE: This test relies on the ACTUAL implementation of:
# - generate_dynamic_state_at and its helpers (_build_topologies, _compute_isls, _compute_ground_station_satellites_in_range)
# - algorithm_free_one_only_over_isls
# - calculate_fstate_shortest_path_object_no_gs_relay
# - distance_tools functions
# It is therefore an INTEGRATION test.


class TestDynamicStateIntegration(unittest.TestCase):

    # Use setUp to define common data if multiple scenarios are tested later
    # For now, define directly in the test method

    def test_equator_scenario_t0(self):
        """
        Integration test mimicking the old test_around_equator_connectivity_with_starlink
        by calling generate_dynamic_state_at directly for t=0.
        Checks the calculated fstate and bandwidth objects.
        """
        # --- Inputs ---
        output_dir = None  # Not writing files
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        time_since_epoch_ns = 0  # Test at t=0
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
        prev_output = None

        # Max lengths from old test
        max_gsl_length_m = 1089686.4181956202
        max_isl_length_m = 5016591.2330984278

        # TLE Data (4 satellites)
        tle_data = {
            0: (
                "Starlink-550 0",
                "1 01308U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 01308  53.0000 295.0000 0000001   0.0000 155.4545 15.19000000    04",
            ),
            1: (
                "Starlink-550 1",
                "1 01309U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 01309  53.0000 295.0000 0000001   0.0000 171.8182 15.19000000    04",
            ),
            2: (
                "Starlink-550 2",
                "1 01310U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    08",
                "2 01310  53.0000 295.0000 0000001   0.0000 188.1818 15.19000000    03",
            ),
            3: (
                "Starlink-550 3",
                "1 01311U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    09",
                "2 01311  53.0000 295.0000 0000001   0.0000 204.5455 15.19000000    04",
            ),
        }
        satellites = []
        for sat_id, tle_lines in tle_data.items():
            ephem_obj = ephem.readtle(tle_lines[0], tle_lines[1], tle_lines[2])
            # Create Satellite object - assuming constructor takes id and ephem objects
            satellites.append(
                Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
            )

        # Ground Station Data (4 stations, IDs 4-7)
        # Note: Original test used GS IDs 0-3, but code adds GS after satellites.
        # Need IDs unique from satellites. Assume GS IDs start after last sat ID.
        gs_data = [
            # Old ID 0 -> New ID 4
            {
                "gid": 4,
                "name": "Luanda",
                "lat": "-8.836820",
                "lon": "13.234320",
                "elv": 0.0,
                "x": 6135530.18,
                "y": 1442953.50,
                "z": -973332.34,
            },
            # Old ID 1 -> New ID 5
            {
                "gid": 5,
                "name": "Lagos",
                "lat": "6.453060",
                "lon": "3.395830",
                "elv": 0.0,
                "x": 6326864.17,
                "y": 375422.89,
                "z": 712064.78,
            },
            # Old ID 2 -> New ID 6
            {
                "gid": 6,
                "name": "Kinshasa",
                "lat": "-4.327580",
                "lon": "15.313570",
                "elv": 0.0,
                "x": 6134256.67,
                "y": 1679704.40,
                "z": -478073.16,
            },
            # Old ID 3 -> New ID 7
            {
                "gid": 7,
                "name": "Ar-Riyadh-(Riyadh)",
                "lat": "24.690466",
                "lon": "46.709566",
                "elv": 0.0,
                "x": 3975957.34,
                "y": 4220595.03,
                "z": 2647959.98,
            },
        ]
        ground_stations = [
            GroundStation(
                gid=d["gid"],
                name=d["name"],
                latitude_degrees_str=d["lat"],
                longitude_degrees_str=d["lon"],
                elevation_m_float=d["elv"],
                cartesian_x=d["x"],
                cartesian_y=d["y"],
                cartesian_z=d["z"],
            )
            for d in gs_data
        ]

        # ConstellationData
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",  # Match TLE epoch
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,  # Pass list of Satellite objects
        )

        # Undirected ISLs (using sat IDs 0-3)
        undirected_isls = [(0, 1), (1, 2), (2, 3)]

        # GSL Interface Info (Nodes 0-3 are Sats, 4-7 are GSs)
        list_gsl_interfaces_info = [
            {"id": 0, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 1, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 2, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 3, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 4, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},  # GS 0
            {"id": 5, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},  # GS 1
            {"id": 6, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},  # GS 2
            {"id": 7, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},  # GS 3
        ]

        # --- Execute ---
        # Call generate_dynamic_state_at for the single time step t=0
        result_state = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=prev_output,
            # enable_verbose_logs - removed
        )

        # --- Assertions ---
        self.assertIsNotNone(result_state, "generate_dynamic_state_at returned None")
        self.assertIn("fstate", result_state)
        self.assertIn("bandwidth", result_state)

        # 1. Assert Bandwidth state
        # Refactored algorithm returns {node_id: bandwidth}
        expected_bandwidth = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,  # Sats
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 1.0,  # GSs
        }
        self.assertDictEqual(result_state["bandwidth"], expected_bandwidth)

        # 2. Assert Forwarding state
        # Translate expected results from old test, using new GS IDs (4-7)
        # Format: {(src_id, dst_id): (next_hop_id, my_if, next_hop_if)}
        # Interface calculation needs care! Recalculate based on ISL count.
        # ISL Counts at t=0 (after _compute_isls): Sat0:1, Sat1:2, Sat2:2, Sat3:1
        # Sat GSL IFs: Sat0=1, Sat1=2, Sat2=2, Sat3=1. GS GSL IF = 0.
        # ISL IF map: (0,1):0,(1,0):0; (1,2):1,(2,1):0; (2,3):1,(3,2):0 (based on _setup_scenario logic)
        expected_fstate = {
            # Sat 0 -> GS
            (0, 4): (1, 0, 0),  # Path 0->1->2->4. Hop 1. IFs (0,1)=0, (1,0)=0.
            (0, 5): (1, 0, 0),  # Path 0->1->5. Hop 1. IFs (0,1)=0, (1,0)=0.
            (0, 6): (1, 0, 0),  # Path 0->1->2->6. Hop 1. IFs (0,1)=0, (1,0)=0.
            (0, 7): (-1, -1, -1),  # Riyadh out of range? Check old test. Yes.
            # Sat 1 -> GS
            (1, 4): (2, 1, 0),  # Path 1->2->4. Hop 2. IFs (1,2)=1, (2,1)=0.
            (1, 5): (5, 2, 0),  # Direct 1->5. IFs Sat=2, GS=0.
            (1, 6): (2, 1, 0),  # Path 1->2->6. Hop 2. IFs (1,2)=1, (2,1)=0.
            (1, 7): (-1, -1, -1),
            # Sat 2 -> GS
            (2, 4): (4, 2, 0),  # Direct 2->4. IFs Sat=2, GS=0.
            (2, 5): (1, 0, 1),  # Path 2->1->5. Hop 1. IFs (2,1)=0, (1,2)=1.
            (2, 6): (6, 2, 0),  # Direct 2->6. IFs Sat=2, GS=0.
            (2, 7): (-1, -1, -1),
            # Sat 3 -> GS
            (3, 4): (2, 0, 1),  # Path 3->2->4. Hop 2. IFs (3,2)=0, (2,3)=1.
            (3, 5): (2, 0, 1),  # Path 3->2->1->5. Hop 2. IFs (3,2)=0, (2,3)=1.
            (3, 6): (2, 0, 1),  # Path 3->2->6. Hop 2. IFs (3,2)=0, (2,3)=1.
            (3, 7): (-1, -1, -1),
            # GS 4 -> GS
            (4, 5): (2, 0, 2),  # Path 4->2->1->5. Hop 2. IFs GS=0, Sat=2.
            (4, 6): (2, 0, 2),  # Path 4->2->6. Hop 2. IFs GS=0, Sat=2.
            (4, 7): (-1, -1, -1),
            # GS 5 -> GS
            (5, 4): (1, 0, 2),  # Path 5->1->2->4. Hop 1. IFs GS=0, Sat=2.
            (5, 6): (1, 0, 2),  # Path 5->1->2->6. Hop 1. IFs GS=0, Sat=2.
            (5, 7): (-1, -1, -1),
            # GS 6 -> GS
            (6, 4): (2, 0, 2),  # Path 6->2->4. Hop 2. IFs GS=0, Sat=2.
            (6, 5): (2, 0, 2),  # Path 6->2->1->5. Hop 2. IFs GS=0, Sat=2.
            (6, 7): (-1, -1, -1),
            # GS 7 -> GS (Riyadh - out of range of all sats?)
            (7, 4): (-1, -1, -1),
            (7, 5): (-1, -1, -1),
            (7, 6): (-1, -1, -1),
        }
        self.maxDiff = None  # Show full diff on error
        self.assertDictEqual(result_state["fstate"], expected_fstate)
