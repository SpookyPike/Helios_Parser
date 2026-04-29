from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import electron_debye_length_cm, electron_plasma_frequency_rad_s
from helios.services.derived.selection import (
    WEIGHTING_MASS,
    build_analysis_geometry,
    cylindrical_shell_factor_cm2,
    cylindrical_shell_path_length_cm,
    path_length_cm,
    weight_array,
)
from helios.services.derived.shock_tracking import _smooth_track, track_shock_front
from helios.services.derived.spectroscopy import doppler_width_fraction, doppler_width_fraction_array


def _synthetic_dataset() -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 5
    n_zones = 10
    time_s = np.asarray([0.0, 1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9], dtype=np.float64)
    static_x = np.linspace(1.0e-4, 1.0e-3, n_zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, 1.05e-3, n_zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, n_zones), 1.0e-4, dtype=np.float64)
    density = np.ones((n_snapshots, n_zones), dtype=np.float64)
    for snapshot, lower_zone in enumerate((8, 7, 6, 5, 4)):
        density[snapshot, lower_zone + 1 :] = 4.0
    velocity = np.zeros_like(density)
    temperature_e = np.full_like(density, 100.0)
    temperature_i = np.full_like(density, 80.0)
    electron_density = np.full_like(density, 1.0e21)
    mean_charge = np.full_like(density, 10.0)
    zone_region_id = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    zone_material = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    regions = {
        "region_index": np.asarray([1, 2], dtype=np.int32),
        "min_zone_index": np.asarray([1, 6], dtype=np.int32),
        "max_zone_index": np.asarray([5, 10], dtype=np.int32),
        "atomic_weight": np.asarray([56.0, 56.0], dtype=np.float64),
        "initial_mass_density": np.asarray([1.0, 1.0], dtype=np.float64),
        "initial_temperature": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path("synthetic.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={
            "geometry": "PLANAR",
            "coordinate_model": {"coordinate_name": "x"},
            "input_parameters": {"laser_source": {"origin_zone_index": 11, "propagation_direction": "Rmin"}},
        },
        regions=regions,
        materials={"index": np.asarray([1, 2], dtype=np.int32)},
        time_s=time_s,
        static_x_cm=static_x,
        static_x_edge_cm=static_x_edges,
        zone_width_cm=zone_width,
        density_g_cm3=density,
        velocity_cm_s=velocity,
        temperature_e_ev=temperature_e,
        temperature_i_ev=temperature_i,
        temperature_radiation_ev=None,
        electron_density_cm3=electron_density,
        mean_charge=mean_charge,
        radius_cm=None,
        radius_edge_cm=None,
        zone_region_id=zone_region_id,
        zone_material_index=zone_material,
        zone_atomic_weight=np.full(n_zones, 56.0, dtype=np.float64),
        zone_initial_density_g_cm3=np.full(n_zones, 1.0, dtype=np.float64),
        zone_initial_temperature_ev=np.full(n_zones, 1.0, dtype=np.float64),
        laser_entry={"boundary_kind": "high"},
    )
    context = RunContext(
        path=Path("synthetic.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={},
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=time_s.copy(),
        static_x_values=static_x.copy(),
        zone_region_id=zone_region_id.copy(),
        zone_material_index=zone_material.copy(),
        has_dynamic_radius=False,
        snapshot_index=2,
        map_coordinate="static_x",
        slice_coordinate="zone",
        selected_region_ids=(1, 2),
        selected_material_ids=(1, 2),
    )
    return dataset, context


class DerivedServicesPhase4Tests(unittest.TestCase):
    def test_shock_track_smoothing_matches_previous_monotonic_enforcement(self) -> None:
        track = np.asarray([np.nan, np.nan, 8.0, 7.4, 7.8, 6.1, 6.3, 4.0], dtype=np.float64)

        def reference(values: np.ndarray, direction: str) -> np.ndarray:
            valid = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(valid)
            if np.count_nonzero(finite) < 3:
                return valid
            indices = np.flatnonzero(finite)
            smoothed = valid.copy()
            kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            segment = np.asarray(valid[indices], dtype=np.float64)
            padded = np.pad(segment, (2, 2), mode="edge")
            filtered = np.convolve(padded, kernel, mode="valid")
            smoothed[indices] = filtered
            smoothed[indices[0]] = segment[0]
            smoothed[indices[-1]] = segment[-1]
            if direction == "high_to_low":
                for idx in indices[1:]:
                    previous = smoothed[idx - 1]
                    if np.isfinite(previous):
                        smoothed[idx] = min(smoothed[idx], previous)
            else:
                for idx in indices[1:]:
                    previous = smoothed[idx - 1]
                    if np.isfinite(previous):
                        smoothed[idx] = max(smoothed[idx], previous)
            return smoothed

        np.testing.assert_allclose(_smooth_track(track, direction="high_to_low"), reference(track, "high_to_low"))
        np.testing.assert_allclose(_smooth_track(track[::-1], direction="low_to_high"), reference(track[::-1], "low_to_high"))

    def test_shock_tracker_finds_interface_crossing_on_synthetic_stack(self) -> None:
        dataset, context = _synthetic_dataset()
        result = track_shock_front(dataset, context)
        np.testing.assert_array_equal(result.zone_index, np.asarray([8, 7, 6, 5, 4], dtype=np.int32))
        self.assertEqual(len(result.interface_crossings), 1)
        crossing = result.interface_crossings[0]
        self.assertEqual(crossing.interface_label, "Region 1 -> 2")
        self.assertEqual(crossing.crossing_snapshot, 4)
        self.assertAlmostEqual(float(crossing.crossing_time_s), 4.0e-9, places=15)

    def test_nrl_formula_helpers_return_dimensionally_sane_values(self) -> None:
        te_ev = 100.0
        ne_cm3 = 1.0e21
        ion_mass_mu = 56.0
        self.assertAlmostEqual(
            electron_debye_length_cm(te_ev, ne_cm3),
            7.43e2 * math.sqrt(te_ev / ne_cm3),
            places=18,
        )
        self.assertAlmostEqual(
            electron_plasma_frequency_rad_s(ne_cm3),
            5.64e4 * math.sqrt(ne_cm3),
            places=4,
        )
        width_fraction = doppler_width_fraction(te_ev, ion_mass_mu)
        self.assertGreater(width_fraction, 0.0)
        self.assertLess(width_fraction, 1.0e-3)
        vector_width = doppler_width_fraction_array(
            np.asarray([te_ev, -1.0, te_ev], dtype=np.float64),
            np.asarray([ion_mass_mu, ion_mass_mu, 0.0], dtype=np.float64),
        )
        self.assertAlmostEqual(float(vector_width[0]), width_fraction)
        self.assertTrue(np.isnan(vector_width[1]))
        self.assertTrue(np.isnan(vector_width[2]))

    def test_cylindrical_mass_weighting_uses_edge_based_shell_geometry(self) -> None:
        dataset, context = _synthetic_dataset()
        dataset.metadata["geometry"] = "CYLINDRICAL"
        dataset.metadata["coordinate_model"] = {"coordinate_name": "radius"}
        dataset.radius_cm = np.broadcast_to(dataset.static_x_cm, (dataset.time_s.size, dataset.static_x_cm.size)).copy()
        dataset.radius_edge_cm = np.broadcast_to(dataset.static_x_edge_cm, (dataset.time_s.size, dataset.static_x_edge_cm.size)).copy()
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side="front",
            line_of_sight_angle_deg=0.0,
            profile_coordinate_mode="moving_radius",
        )
        mask = np.ones(dataset.summary["n_zones"], dtype=bool)
        density = np.asarray(dataset.density_g_cm3[0], dtype=np.float64)
        weights = weight_array(
            density,
            dataset,
            0,
            mask,
            mode=WEIGHTING_MASS,
            geometry=geometry,
        )
        shell_factor = cylindrical_shell_factor_cm2(dataset, 0)
        assert shell_factor is not None
        np.testing.assert_allclose(weights, density * shell_factor)

    def test_cylindrical_shell_path_length_helper_handles_centerline_and_off_axis_cases(self) -> None:
        edges = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        centerline = cylindrical_shell_path_length_cm(edges, 0.0)
        off_axis = cylindrical_shell_path_length_cm(edges, 1.0)
        tangent = cylindrical_shell_path_length_cm(edges, 3.0)
        np.testing.assert_allclose(centerline, np.asarray([2.0, 2.0, 2.0], dtype=np.float64))
        self.assertTrue(np.all(np.isfinite(off_axis)))
        self.assertTrue(np.all(off_axis >= 0.0))
        self.assertEqual(off_axis[0], 0.0)
        self.assertAlmostEqual(off_axis[1], 2.0 * np.sqrt(3.0), places=12)
        self.assertAlmostEqual(off_axis[2], 2.0 * (np.sqrt(8.0) - np.sqrt(3.0)), places=12)
        np.testing.assert_allclose(tangent, np.zeros(3, dtype=np.float64))

    def test_path_length_dispatch_keeps_planar_unchanged_and_uses_shells_for_cylindrical(self) -> None:
        dataset, context = _synthetic_dataset()
        planar_geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side="front",
            line_of_sight_angle_deg=30.0,
            profile_coordinate_mode="static_x",
        )
        planar_path = path_length_cm(dataset, 0, planar_geometry)
        np.testing.assert_allclose(planar_path, dataset.zone_width_cm[0] / np.cos(np.deg2rad(30.0)))

        dataset.metadata["geometry"] = "CYLINDRICAL"
        dataset.metadata["coordinate_model"] = {"coordinate_name": "radius"}
        dataset.radius_cm = np.broadcast_to(dataset.static_x_cm, (dataset.time_s.size, dataset.static_x_cm.size)).copy()
        dataset.radius_edge_cm = np.broadcast_to(dataset.static_x_edge_cm, (dataset.time_s.size, dataset.static_x_edge_cm.size)).copy()
        cylindrical_geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side="front",
            line_of_sight_angle_deg=30.0,
            line_of_sight_impact_parameter_cm=0.0,
            profile_coordinate_mode="moving_radius",
        )
        cylindrical_path = path_length_cm(dataset, 0, cylindrical_geometry)
        expected_path = cylindrical_shell_path_length_cm(dataset.radius_edge_cm[0], 0.0)
        np.testing.assert_allclose(cylindrical_path, expected_path)
        self.assertEqual(cylindrical_geometry.path_length_mode, "cylindrical-shell-centerline")
        self.assertFalse(np.allclose(cylindrical_path, planar_path))


if __name__ == "__main__":
    unittest.main()
