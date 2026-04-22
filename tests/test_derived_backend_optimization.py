from __future__ import annotations

import math
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT
from helios.cache import clear_session_raw_data_cache
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import field_capability_summary, load_run_data, publish_field_payload, publish_open_run_payload
from helios.services.derived.transmission import evaluate_transmission
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_geometry,
    build_analysis_mask,
    path_length_cm,
    selection_cache_key,
    weight_array,
    weighted_average,
    weighted_means,
)
from helios_parser import HeliosRun
from helios.services.derived.xrd import estimate_xrd
from helios.services.units.conversions import photon_energy_kev_to_wavelength_angstrom


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int | None = None) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=min(20, max(0, dataset.time_s.size - 1)) if snapshot_index is None else int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class DerivedBackendOptimizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session_raw_data_cache(reason="test_cleanup")

    def test_analysis_state_cache_reuses_masks_paths_and_base_weights(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        parameters = DerivedAnalysisParameters(
            weighting_mode="electron_column",
            exclude_low_density=True,
            min_density_g_cm3=0.01,
            zone_index_upper=250,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()

        mask_a, selection_a, warnings_a = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        mask_b, selection_b, warnings_b = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        self.assertEqual(len(analysis_cache.mask_cache), 1)
        np.testing.assert_array_equal(mask_a, mask_b)
        self.assertEqual(selection_a, selection_b)
        self.assertEqual(warnings_a, warnings_b)

        path_a = path_length_cm(dataset, context.snapshot_index, geometry, analysis_cache=analysis_cache)
        path_b = path_length_cm(dataset, context.snapshot_index, geometry, analysis_cache=analysis_cache)
        self.assertEqual(len(analysis_cache.path_cache), 1)
        np.testing.assert_allclose(path_a, path_b)

        density = np.asarray(dataset.density_g_cm3[int(context.snapshot_index)], dtype=np.float64)
        temperature = np.asarray(dataset.temperature_e_ev[int(context.snapshot_index)], dtype=np.float64)
        weights_a = weight_array(
            density,
            dataset,
            context.snapshot_index,
            mask_a,
            mode=parameters.weighting_mode,
            geometry=geometry,
            selection_key=selection_cache_key(selection_a),
            analysis_cache=analysis_cache,
        )
        weights_b = weight_array(
            temperature,
            dataset,
            context.snapshot_index,
            mask_a,
            mode=parameters.weighting_mode,
            geometry=geometry,
            selection_key=selection_cache_key(selection_a),
            analysis_cache=analysis_cache,
        )
        self.assertEqual(len(analysis_cache.weight_cache), 1)
        np.testing.assert_allclose(weights_a, weights_b)

    def test_weight_cache_uses_selection_identity_without_mask_packbits(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        parameters = DerivedAnalysisParameters(weighting_mode="electron_column", exclude_low_density=True, min_density_g_cm3=0.01)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()
        mask, selection, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        density = np.asarray(dataset.density_g_cm3[int(context.snapshot_index)], dtype=np.float64)
        with mock.patch("helios.services.derived.selection._mask_signature", side_effect=AssertionError("mask signature should not be used")):
            weight_array(
                density,
                dataset,
                context.snapshot_index,
                mask,
                mode=parameters.weighting_mode,
                geometry=geometry,
                selection_key=selection_cache_key(selection),
                analysis_cache=analysis_cache,
            )

    def test_load_run_data_reuses_viewer_published_raw_arrays(self) -> None:
        clear_session_raw_data_cache(reason="preload_viewer_fields")
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        with HeliosRun(path) as run:
            fields = run.list_fields()
            summary = run.summary()
            metadata = run.get_metadata()
            regions = run.get_regions()
            materials = run.get_materials()
            publish_open_run_payload(
                path,
                summary=summary,
                metadata=metadata,
                regions=regions,
                materials=materials,
                fields=fields,
                diagnostics=run.list_diagnostics(),
                time_values=run.get_time(),
                static_x_center=run.get_static_coordinate(location="center"),
                static_x_edge=run.get_static_coordinate(location="edge"),
                zone_region_id=run.get_grid("zone_region_id"),
                zone_material_index=run.get_grid("zone_material_index"),
                has_dynamic_radius="radius" in fields,
                run_status=run.get_run_status(),
                visar_support_metadata={
                    "velocity_field_name": run.get_visar_support_metadata().velocity_field_name,
                    "time_axis_name": run.get_visar_support_metadata().time_axis_name,
                    "static_coordinate_name": run.get_visar_support_metadata().static_coordinate_name,
                    "dynamic_coordinate_field_name": run.get_visar_support_metadata().dynamic_coordinate_field_name,
                    "boundary_indexing_consistent": run.get_visar_support_metadata().boundary_indexing_consistent,
                    "candidate_boundaries": tuple(run.get_visar_support_metadata().candidate_boundaries),
                    "event_timing_source": run.get_visar_support_metadata().event_timing_source,
                    "notes": tuple(run.get_visar_support_metadata().notes),
                },
            )
            for field_name in (
                "density",
                "velocity",
                "temperature_e",
                "temperature_i",
                "temperature_radiation",
                "electron_density",
                "mean_charge",
                "zone_width",
                "radius",
            ):
                if field_name not in fields:
                    continue
                data = run.get_dynamic_coordinate(location="center") if field_name == "radius" and run.has_dynamic_coordinate() else run.get_field(field_name)
                edge_data = run.get_dynamic_coordinate(location="edge") if field_name == "radius" and run.has_dynamic_coordinate() else None
                publish_field_payload(path, field_name=field_name, data=data, edge_data=edge_data)

        with mock.patch("helios.services.derived.common.HeliosRun", side_effect=AssertionError("shared cache should avoid HDF5 reopen")):
            dataset = load_run_data(path)
        self.assertEqual(dataset.density_g_cm3.shape[0], int(dataset.summary["n_snapshots"]))
        self.assertEqual(dataset.static_x_cm.shape[0], int(dataset.summary["n_zones"]))
        self.assertIsInstance(dataset.run_status, dict)
        self.assertIsInstance(dataset.visar_support_metadata, dict)
        self.assertTrue(bool(dataset.field_capabilities.run_status_available))
        self.assertTrue(bool(dataset.field_capabilities.visar_support_available))

    def test_load_run_data_promotes_optional_fields_and_capabilities(self) -> None:
        path = _resolve_existing_path(
            HDF5_ROOT / "Cu1e17_cyl_stabilized.h5",
            Path(__file__).resolve().parents[1] / "Cu1e17_cyl_stabilized.h5",
        )
        dataset = load_run_data(path)
        summary = field_capability_summary(dataset)

        self.assertIsNotNone(dataset.pressure_i_j_cm3)
        self.assertIsNotNone(dataset.pressure_e_j_cm3)
        self.assertIsNotNone(dataset.pressure_radiation_j_cm3)
        self.assertIsNotNone(dataset.pressure_total_j_cm3)
        self.assertIsNotNone(dataset.artificial_viscosity_j_cm3)
        self.assertIsNotNone(dataset.ion_energy_j_g)
        self.assertIsNotNone(dataset.electron_energy_j_g)
        self.assertIsNotNone(dataset.radiation_energy_j_g)
        self.assertIsNotNone(dataset.kinetic_energy_j_g)
        self.assertIsNotNone(dataset.ion_heat_capacity_j_g_ev)
        self.assertIsNotNone(dataset.electron_heat_capacity_j_g_ev)
        self.assertIsNotNone(dataset.radiation_heating_j_g_s)
        self.assertIsNotNone(dataset.radiation_cooling_j_g_s)
        self.assertIsNotNone(dataset.radiation_net_heating_j_g_s)
        self.assertIsNotNone(dataset.laser_source_j_g_s)
        self.assertIsNotNone(dataset.laser_deposition_j_g_s)
        self.assertEqual("completed", dataset.run_status["state"] if dataset.run_status is not None else None)
        self.assertEqual("velocity", dataset.visar_support_metadata.get("velocity_field_name") if dataset.visar_support_metadata is not None else None)
        self.assertTrue(summary["run_status_available"])
        self.assertTrue(summary["visar_support_available"])
        self.assertTrue(summary["pressure_components_available"])
        self.assertTrue(summary["total_pressure_available"])
        self.assertTrue(summary["radiation_components_available"])
        self.assertTrue(summary["radiation_net_heating_available"])
        self.assertTrue(summary["kinetic_energy_available"])
        self.assertIn("pressure", summary["available_fields"])
        self.assertIn("laser_deposition", summary["optional_available_fields"])
        self.assertIsNotNone(summary["total_pressure_matches_components"])
        self.assertIsNotNone(summary["radiation_net_heating_matches_components"])
        self.assertIsNotNone(summary["kinetic_energy_matches_velocity"])

    def test_weighted_means_matches_scalar_weighted_average(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters(weighting_mode="electron_column", exclude_low_density=True, min_density_g_cm3=0.05)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()
        mask, selection, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        stacked = np.stack(
            (
                np.asarray(dataset.temperature_e_ev[int(context.snapshot_index)], dtype=np.float64),
                np.asarray(dataset.temperature_i_ev[int(context.snapshot_index)], dtype=np.float64),
                np.asarray(dataset.electron_density_cm3[int(context.snapshot_index)], dtype=np.float64),
                np.asarray(dataset.mean_charge[int(context.snapshot_index)], dtype=np.float64),
                np.asarray(dataset.zone_atomic_weight, dtype=np.float64),
            ),
            axis=0,
        )
        actual = weighted_means(
            stacked,
            dataset,
            context.snapshot_index,
            mask,
            mode=parameters.weighting_mode,
            geometry=geometry,
            selection_key=selection_cache_key(selection),
            analysis_cache=analysis_cache,
        )
        expected = np.asarray(
            [
                weighted_average(
                    field,
                    dataset,
                    context.snapshot_index,
                    mask,
                    mode=parameters.weighting_mode,
                    geometry=geometry,
                    selection_key=selection_cache_key(selection),
                    analysis_cache=analysis_cache,
                )
                for field in stacked
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_xrd_vectorized_region_aggregation_matches_manual_snapshot_loop(self) -> None:
        path = HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=120)
        parameters = DerivedAnalysisParameters(weighting_mode="mass", exclude_low_density=True, min_density_g_cm3=0.05)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()
        result = estimate_xrd(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.xrd_photon_energy_kev,
            initial_bragg_angle_deg=parameters.xrd_initial_bragg_angle_deg,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=analysis_cache,
        )
        mask, selection, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        density = np.asarray(dataset.density_g_cm3[int(context.snapshot_index)], dtype=np.float64)
        zone_width = np.asarray(dataset.zone_width_cm[int(context.snapshot_index)], dtype=np.float64)
        initial_density = np.asarray(dataset.regions["initial_mass_density"], dtype=np.float64)
        region_ids = np.asarray(dataset.regions["region_index"], dtype=np.int32)
        wavelength_angstrom = photon_energy_kev_to_wavelength_angstrom(parameters.xrd_photon_energy_kev)
        theta0_rad = math.radians(parameters.xrd_initial_bragg_angle_deg)
        q0 = 4.0 * math.pi * math.sin(theta0_rad) / wavelength_angstrom

        manual_layers: list[tuple[int, int, float, float, float, float, float | None, float | None, float]] = []
        for region_offset, region_id in enumerate(region_ids):
            region_mask = mask & (np.asarray(dataset.zone_region_id, dtype=np.int32) == int(region_id))
            if not np.any(region_mask):
                continue
            weights = weight_array(
                density,
                dataset,
                context.snapshot_index,
                region_mask,
                mode=parameters.weighting_mode,
                geometry=geometry,
                analysis_cache=analysis_cache,
            )
            weight_sum = float(np.sum(weights))
            if not math.isfinite(weight_sum) or weight_sum <= 0.0:
                continue
            rho_avg = float(np.sum(weights * density) / weight_sum)
            compression = rho_avg / float(initial_density[region_offset]) if float(initial_density[region_offset]) > 0.0 else float("nan")
            d_over_d0 = compression ** (-1.0 / 3.0) if math.isfinite(compression) and compression > 0.0 else float("nan")
            shifted_theta_deg: float | None = None
            q_compressed = float("nan")
            bragg_shift_deg: float | None = None
            if math.isfinite(d_over_d0) and d_over_d0 > 0.0:
                sin_theta_shifted = math.sin(theta0_rad) / d_over_d0
                if sin_theta_shifted <= 1.0:
                    shifted_theta_rad = math.asin(sin_theta_shifted)
                    shifted_theta_deg = math.degrees(shifted_theta_rad)
                    q_compressed = 4.0 * math.pi * math.sin(shifted_theta_rad) / wavelength_angstrom
                    bragg_shift_deg = shifted_theta_deg - parameters.xrd_initial_bragg_angle_deg
            manual_layers.append(
                (
                    int(region_id),
                    int(np.count_nonzero(region_mask)),
                    rho_avg,
                    compression,
                    d_over_d0,
                    q0,
                    q_compressed,
                    bragg_shift_deg,
                    float(np.sum(zone_width[region_mask])),
                )
            )

        self.assertEqual(len(result.layers), len(manual_layers))
        for layer, expected in zip(result.layers, manual_layers):
            self.assertEqual(layer.region_id, expected[0])
            self.assertEqual(layer.zone_count, expected[1])
            self.assertAlmostEqual(layer.compressed_density_g_cm3, expected[2], places=10)
            self.assertAlmostEqual(layer.compression_ratio, expected[3], places=10)
            self.assertAlmostEqual(layer.d_over_d0, expected[4], places=10)
            self.assertAlmostEqual(layer.q0_inv_angstrom, expected[5], places=10)
            if math.isfinite(float(expected[6])):
                self.assertAlmostEqual(layer.q_compressed_inv_angstrom, float(expected[6]), places=10)
                self.assertAlmostEqual(layer.bragg_shift_deg, float(expected[7]), places=10)
            else:
                self.assertTrue(math.isnan(layer.q_compressed_inv_angstrom))
                self.assertIsNone(layer.bragg_shift_deg)
            self.assertAlmostEqual(layer.compressed_thickness_cm, expected[8], places=10)

    def test_transmission_region_budgets_match_manual_region_loop(self) -> None:
        path = HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=150)
        parameters = DerivedAnalysisParameters(exclude_low_density=True, min_density_g_cm3=0.01)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()
        result = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=analysis_cache,
        )
        mask, _, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode="path_integrated",
            analysis_cache=analysis_cache,
        )
        density = np.asarray(dataset.density_g_cm3[int(context.snapshot_index)], dtype=np.float64)
        electron_density = np.asarray(dataset.electron_density_cm3[int(context.snapshot_index)], dtype=np.float64)
        path_length = path_length_cm(dataset, context.snapshot_index, geometry, analysis_cache=analysis_cache)
        expected = []
        for region_id in np.asarray(dataset.regions["region_index"], dtype=np.int32):
            region_mask = mask & (np.asarray(dataset.zone_region_id, dtype=np.int32) == int(region_id))
            if not np.any(region_mask):
                continue
            region_areal = float(np.sum(density[region_mask] * path_length[region_mask]))
            region_column = float(np.sum(electron_density[region_mask] * path_length[region_mask]))
            expected.append((int(region_id), region_areal, region_column))
        self.assertEqual(len(result.region_budgets), len(expected))
        for budget, expected_values in zip(result.region_budgets, expected):
            self.assertEqual(budget.region_id, expected_values[0])
            self.assertTrue(np.isclose(budget.areal_density_g_cm2, expected_values[1], rtol=1e-12, atol=1e-12))
            self.assertTrue(np.isclose(budget.electron_column_cm2, expected_values[2], rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
