from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

try:
    from PySide6 import QtCore, QtWidgets  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional GUI dependency in CI/container
    QtCore = None  # type: ignore
    QtWidgets = None  # type: ignore

import _test_bootstrap  # noqa: F401

if QtWidgets is not None:
    from _viewer_test_utils import get_app, process_events, reset_test_settings
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.plasmon_config import (
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_OBSERVABLE_MODE_XRTS,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
)
from helios.services.derived.models import DerivedRunData, DerivedWarning
from helios.services.derived.plasmon_validation import q_to_angle_deg, uniform_al_dataset

if QtWidgets is not None:
    from helios_analysis.workspace import HeliosDerivedWorkspace


def _synthetic_dataset() -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 3
    n_zones = 6
    time_s = np.asarray([0.0, 1.0e-9, 2.0e-9], dtype=np.float64)
    static_x = np.linspace(1.0e-4, 6.0e-4, n_zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, 6.5e-4, n_zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, n_zones), 1.0e-4, dtype=np.float64)
    density = np.ones((n_snapshots, n_zones), dtype=np.float64)
    velocity = np.zeros_like(density)
    temperature_e = np.full_like(density, 120.0)
    temperature_i = np.full_like(density, 80.0)
    electron_density = np.full_like(density, 8.0e20)
    mean_charge = np.full_like(density, 6.0)
    zone_region_id = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    zone_material = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    regions = {
        "region_index": np.asarray([1, 2], dtype=np.int32),
        "min_zone_index": np.asarray([1, 4], dtype=np.int32),
        "max_zone_index": np.asarray([3, 6], dtype=np.int32),
        "atomic_weight": np.asarray([27.0, 63.5], dtype=np.float64),
        "initial_mass_density": np.asarray([1.0, 1.0], dtype=np.float64),
        "initial_temperature": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={"geometry": "PLANAR", "coordinate_model": {"coordinate_name": "x"}},
        regions=regions,
        materials={
            "index": np.asarray([1, 2], dtype=np.int32),
            "eos_file_path": np.asarray(["Al.prp", "Cu.prp"], dtype=object),
            "opacity_file_path": np.asarray(["Al.prp", "Cu.prp"], dtype=object),
            "eos_model": np.asarray(["EOSOPA", "EOSOPA"], dtype=object),
            "opacity_model": np.asarray(["EOSOPA", "EOSOPA"], dtype=object),
        },
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
        zone_atomic_weight=np.asarray([27.0, 27.0, 27.0, 63.5, 63.5, 63.5], dtype=np.float64),
        zone_initial_density_g_cm3=np.full(n_zones, 1.0, dtype=np.float64),
        zone_initial_temperature_ev=np.full(n_zones, 1.0, dtype=np.float64),
        laser_entry=None,
    )
    context = RunContext(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={},
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=time_s.copy(),
        static_x_values=static_x.copy(),
        zone_region_id=zone_region_id.copy(),
        zone_material_index=zone_material.copy(),
        has_dynamic_radius=False,
        snapshot_index=1,
        map_coordinate="static_x",
        slice_coordinate="zone",
        selected_region_ids=(1, 2),
        selected_material_ids=(1, 2),
    )
    return dataset, context


def _combo_keys(combo: "QtWidgets.QComboBox") -> tuple[str, ...]:
    return tuple(str(combo.itemData(index)) for index in range(combo.count()))


def _combo_item_enabled(combo: "QtWidgets.QComboBox", key: str) -> bool:
    index = combo.findData(key)
    if index < 0:
        return False
    model_index = combo.model().index(index, 0)
    return bool(combo.model().flags(model_index) & QtCore.Qt.ItemFlag.ItemIsEnabled)


@unittest.skipIf(QtWidgets is None, "PySide6 is not available in this environment")
class PlasmonUiPhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_workspace_prefers_spectrum_profile_and_supports_manual_recompute(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "spectrum_observed")
        self.assertIn(
            workspace.plasmon_plot_panel.profile_combo.currentData(),
            {"spectrum_observed", "dispersion_selected_model", "local_k_lambda_profile"},
        )
        rpa_index = workspace.plasmon_model_combo.findData("rpa")
        self.assertGreaterEqual(rpa_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(rpa_index)
        process_events(20)
        workspace.plasmon_auto_recompute_checkbox.setChecked(False)
        process_events(20)
        workspace.plasmon_window_spin.setValue(55.0)
        process_events(20)
        self.assertIn("pending", workspace.plasmon_recompute_button.text().lower())
        workspace.close()

    def test_quicklook_prefers_plasma_frequency_time_plot_over_legacy_temperature_trace(self) -> None:
        dataset, context = _synthetic_dataset()
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(plasmon_model="quicklook"),
            context_key=("ui-phase2-quicklook", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "plasma_frequency")
        workspace.close()

    def test_workspace_uses_compact_sidebar_and_prefers_dispersion_compare_plot(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_study_mode="dispersion",
            plasmon_scan_axis="k_angstrom_inv",
            plasmon_scan_start=0.4,
            plasmon_scan_stop=1.2,
            plasmon_scan_points=7,
            plasmon_compare_models=True,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-dispersion", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "dispersion_compare_models")
        self.assertEqual(workspace.plasmon_plot_panel.profile_combo.currentData(), "dispersion_compare_width_models")
        self.assertLessEqual(workspace.plasmon_sidebar.width(), 320)
        self.assertGreater(workspace.plasmon_tab.width(), workspace.plasmon_sidebar.width())
        workspace.close()

    def test_workspace_prefers_policy_comparison_plot_when_enabled(self) -> None:
        dataset, context = uniform_al_dataset(2.7, 0.3)
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=q_to_angle_deg(1.28, 8.307),
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_study_mode="spectrum",
            plasmon_compare_policies=True,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-policy-compare", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "spectrum_compare_policies")
        workspace.close()

    def test_plasmon_heavy_spinboxes_disable_keyboard_tracking(self) -> None:
        workspace = HeliosDerivedWorkspace()
        self.assertFalse(workspace.plasmon_energy_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_scan_points_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_gamma_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_collision_scale_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_manual_collision_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_window_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_points_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_fwhm_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_cluster_log_ne_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_cluster_log_te_spin.keyboardTracking())
        self.assertFalse(workspace.plasmon_cluster_z_spin.keyboardTracking())
        workspace.close()

    def test_plasmon_observable_mode_flows_into_parameters_without_prior_result(self) -> None:
        workspace = HeliosDerivedWorkspace()
        index = workspace.plasmon_observable_mode_combo.findData(PLASMON_OBSERVABLE_MODE_XRTS)
        self.assertGreaterEqual(index, 0)
        workspace.plasmon_observable_mode_combo.setCurrentIndex(index)
        process_events(20)
        self.assertEqual(workspace.parameters().plasmon_observable_mode, PLASMON_OBSERVABLE_MODE_XRTS)
        workspace.close()

    def test_article_native_observable_mode_is_available_without_prior_result(self) -> None:
        workspace = HeliosDerivedWorkspace()
        index = workspace.plasmon_observable_mode_combo.findData(PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE)
        self.assertGreaterEqual(index, 0)
        workspace.plasmon_observable_mode_combo.setCurrentIndex(index)
        process_events(20)
        self.assertEqual(workspace.parameters().plasmon_observable_mode, PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE)
        workspace.close()

    def test_plasmon_result_sync_preserves_observable_mode_and_matches_request(self) -> None:
        dataset, context = uniform_al_dataset(2.7, 0.3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(
                plasmon_model="quantum_hydrodynamic",
                plasmon_execution_mode="benchmark",
                plasmon_observable_mode=PLASMON_OBSERVABLE_MODE_XRTS,
                plasmon_photon_energy_kev=8.307,
                plasmon_scattering_angle_deg=q_to_angle_deg(1.28, 8.307),
                plasmon_energy_window_ev=45.0,
                plasmon_energy_points=1201,
                plasmon_instrument_fwhm_ev=3.5,
            ),
            context_key=("ui-phase2-observable-mode", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_observable_mode_combo.currentData(), PLASMON_OBSERVABLE_MODE_XRTS)
        self.assertTrue(workspace._plasmon_request_matches_result())
        workspace.close()

    def test_stale_plasmon_result_does_not_overwrite_current_controls(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-stale", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        lindhard_index = workspace.plasmon_model_combo.findData("lindhard")
        self.assertGreaterEqual(lindhard_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(lindhard_index)
        process_events(20)
        workspace._populate_plasmon(result.plasmon)
        self.assertEqual(workspace.plasmon_model_combo.currentData(), "lindhard")
        workspace.close()

    def test_active_probe_energy_edit_is_not_overwritten_by_result_sync(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-active-edit", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        workspace.plasmon_energy_spin.setFocus()
        process_events(20)
        line_edit = workspace.plasmon_energy_spin.lineEdit()
        self.assertIsNotNone(line_edit)
        assert line_edit is not None
        original_text = line_edit.text()
        line_edit.selectAll()
        line_edit.setText("9.5")
        process_events(20)
        workspace._populate_plasmon(result.plasmon)
        self.assertTrue(line_edit.hasFocus())
        self.assertNotEqual(line_edit.text(), original_text)
        self.assertTrue(line_edit.text().startswith("9"))
        workspace.close()

    def test_compare_model_list_is_available_and_propagates_to_parameters(self) -> None:
        workspace = HeliosDerivedWorkspace()
        workspace.show()
        rpa_index = workspace.plasmon_model_combo.findData("rpa")
        self.assertGreaterEqual(rpa_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(rpa_index)
        workspace.plasmon_compare_models_checkbox.setChecked(True)
        workspace._set_checked_plasmon_compare_models(("rpa_static_lfc", "lindhard"))
        process_events(20)
        params = workspace.parameters()
        self.assertTrue(workspace.plasmon_compare_model_list.isEnabled())
        self.assertEqual(params.plasmon_compare_model_names, ("rpa_static_lfc", "lindhard"))
        self.assertGreaterEqual(workspace.plasmon_compare_model_list.count(), 3)
        workspace.close()

    def test_quantum_hydrodynamic_model_is_available_in_ui_selection(self) -> None:
        workspace = HeliosDerivedWorkspace()
        workspace.show()
        model_index = workspace.plasmon_model_combo.findData(PLASMON_MODEL_QUANTUM_HYDRODYNAMIC)
        self.assertGreaterEqual(model_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(model_index)
        process_events(20)
        self.assertEqual(workspace.plasmon_model_combo.currentData(), PLASMON_MODEL_QUANTUM_HYDRODYNAMIC)
        workspace.close()

    def test_finite_t_stls_model_is_available_in_ui_selection(self) -> None:
        workspace = HeliosDerivedWorkspace()
        workspace.show()
        model_index = workspace.plasmon_model_combo.findData(PLASMON_MODEL_FINITE_T_STLS)
        self.assertGreaterEqual(model_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(model_index)
        process_events(20)
        self.assertEqual(workspace.plasmon_model_combo.currentData(), PLASMON_MODEL_FINITE_T_STLS)
        workspace.close()

    def test_fresh_session_plasmon_options_are_config_driven_without_prior_benchmark_run(self) -> None:
        _dataset, context = _synthetic_dataset()
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        rpa_index = workspace.plasmon_model_combo.findData("rpa")
        benchmark_index = workspace.plasmon_execution_mode_combo.findData("benchmark")
        dispersion_index = workspace.plasmon_study_mode_combo.findData("dispersion")
        self.assertGreaterEqual(rpa_index, 0)
        self.assertGreaterEqual(benchmark_index, 0)
        self.assertGreaterEqual(dispersion_index, 0)
        workspace.plasmon_model_combo.setCurrentIndex(rpa_index)
        workspace.plasmon_execution_mode_combo.setCurrentIndex(benchmark_index)
        workspace.plasmon_study_mode_combo.setCurrentIndex(dispersion_index)
        workspace.plasmon_compare_models_checkbox.setChecked(True)
        process_events(50)
        self.assertIn("dispersion_selected_model", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_selected_width", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        self.assertIn("dispersion_compare_width_models", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        self.assertFalse(_combo_item_enabled(workspace.plasmon_plot_panel.time_combo, "dispersion_selected_model"))
        self.assertFalse(_combo_item_enabled(workspace.plasmon_plot_panel.profile_combo, "dispersion_selected_width"))
        workspace.close()

    def test_quicklook_result_does_not_gate_benchmark_capability_options(self) -> None:
        dataset, context = _synthetic_dataset()
        quicklook_result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(plasmon_model="quicklook"),
            context_key=("ui-phase2-capability-stale", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(quicklook_result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "plasma_frequency")
        workspace.plasmon_model_combo.setCurrentIndex(workspace.plasmon_model_combo.findData("rpa"))
        workspace.plasmon_execution_mode_combo.setCurrentIndex(workspace.plasmon_execution_mode_combo.findData("benchmark"))
        workspace.plasmon_study_mode_combo.setCurrentIndex(workspace.plasmon_study_mode_combo.findData("dispersion"))
        workspace.plasmon_compare_models_checkbox.setChecked(True)
        process_events(50)
        self.assertIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_compare_width_models", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "dispersion_compare_models")
        self.assertFalse(_combo_item_enabled(workspace.plasmon_plot_panel.time_combo, "dispersion_compare_models"))
        workspace.plasmon_compare_models_checkbox.setChecked(False)
        workspace.plasmon_model_combo.setCurrentIndex(workspace.plasmon_model_combo.findData("quicklook"))
        process_events(50)
        self.assertNotIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("plasma_frequency", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        workspace.close()

    def test_benchmark_preset_exposes_dispersion_options_before_recompute(self) -> None:
        dataset, context = _synthetic_dataset()
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        ambient_index = workspace.plasmon_benchmark_preset_combo.findData("al_ambient_article")
        self.assertGreaterEqual(ambient_index, 0)
        workspace.plasmon_benchmark_preset_combo.setCurrentIndex(ambient_index)
        with mock.patch("helios_analysis.workspace.load_run_data", return_value=dataset):
            workspace._apply_plasmon_benchmark_preset()
        process_events(50)
        self.assertIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_compare_width_models", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "dispersion_compare_models")
        workspace.close()

    def test_plasmon_options_stay_available_across_mode_switching_without_benchmark_unlock(self) -> None:
        dataset, context = _synthetic_dataset()
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        workspace.plasmon_model_combo.setCurrentIndex(workspace.plasmon_model_combo.findData("rpa"))
        workspace.plasmon_execution_mode_combo.setCurrentIndex(workspace.plasmon_execution_mode_combo.findData("benchmark"))
        workspace.plasmon_study_mode_combo.setCurrentIndex(workspace.plasmon_study_mode_combo.findData("dispersion"))
        workspace.plasmon_compare_models_checkbox.setChecked(True)
        process_events(50)
        self.assertIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_compare_width_models", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        workspace.plasmon_study_mode_combo.setCurrentIndex(workspace.plasmon_study_mode_combo.findData("spectrum"))
        process_events(30)
        self.assertIn("spectrum_observed", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        workspace.plasmon_execution_mode_combo.setCurrentIndex(workspace.plasmon_execution_mode_combo.findData("quicklook"))
        process_events(30)
        self.assertIn("plasma_frequency", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        workspace.plasmon_execution_mode_combo.setCurrentIndex(workspace.plasmon_execution_mode_combo.findData("benchmark"))
        workspace.plasmon_study_mode_combo.setCurrentIndex(workspace.plasmon_study_mode_combo.findData("dispersion"))
        with mock.patch("helios_analysis.workspace.load_run_data", return_value=dataset):
            workspace.plasmon_benchmark_preset_combo.setCurrentIndex(workspace.plasmon_benchmark_preset_combo.findData("al_ambient_article"))
            workspace._apply_plasmon_benchmark_preset()
        process_events(50)
        self.assertIn("dispersion_compare_models", _combo_keys(workspace.plasmon_plot_panel.time_combo))
        self.assertIn("dispersion_compare_width_models", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        self.assertIn("spectrum_observed", _combo_keys(workspace.plasmon_plot_panel.profile_combo))
        workspace.close()

    def test_plasmon_splitter_can_expand_plot_area_beyond_old_sidebar_clamp(self) -> None:
        workspace = HeliosDerivedWorkspace()
        workspace.resize(1800, 1100)
        workspace.show()
        process_events(50)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(20)
        workspace.plasmon_splitter.setSizes([220, 1480])
        process_events(50)
        self.assertLess(workspace.plasmon_splitter.sizes()[0], 280)
        self.assertLess(workspace.plasmon_sidebar.width(), 280)
        workspace.close()

    def test_benchmark_preset_button_loads_article_al_defaults(self) -> None:
        dataset, context = _synthetic_dataset()
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        ambient_index = workspace.plasmon_benchmark_preset_combo.findData("al_ambient_article")
        self.assertGreaterEqual(ambient_index, 0)
        workspace.plasmon_benchmark_preset_combo.setCurrentIndex(ambient_index)
        with mock.patch("helios_analysis.workspace.load_run_data", return_value=dataset):
            workspace._apply_plasmon_benchmark_preset()
        process_events(20)
        self.assertEqual(workspace.plasmon_execution_mode_combo.currentData(), "benchmark")
        self.assertEqual(workspace.plasmon_study_mode_combo.currentData(), "dispersion")
        self.assertEqual(workspace.plasmon_electron_policy_combo.currentData(), "article_al_benchmark")
        self.assertEqual(workspace.plasmon_collision_model_combo.currentData(), "benchmark_dense")
        self.assertTrue(workspace.plasmon_compare_models_checkbox.isChecked())
        self.assertFalse(workspace.plasmon_compare_policies_checkbox.isChecked())
        selected_material_ids = tuple(
            sorted(
                int(workspace.material_list.item(row).data(QtCore.Qt.ItemDataRole.UserRole))
                for row in range(workspace.material_list.count())
                if workspace.material_list.item(row).checkState() == QtCore.Qt.CheckState.Checked
            )
        )
        self.assertEqual(selected_material_ids, (1,))
        workspace.close()

    def test_driven_benchmark_preset_uses_article_driven_increment_policy(self) -> None:
        dataset, context = uniform_al_dataset(4.2, 0.6)
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        driven_index = workspace.plasmon_benchmark_preset_combo.findData("al_driven_article")
        self.assertGreaterEqual(driven_index, 0)
        workspace.plasmon_benchmark_preset_combo.setCurrentIndex(driven_index)
        with mock.patch("helios_analysis.workspace.load_run_data", return_value=dataset):
            workspace._apply_plasmon_benchmark_preset()
        process_events(20)
        self.assertEqual(workspace.plasmon_execution_mode_combo.currentData(), "benchmark")
        self.assertEqual(workspace.plasmon_study_mode_combo.currentData(), "dispersion")
        self.assertEqual(workspace.plasmon_electron_policy_combo.currentData(), "article_al_driven_increment")
        self.assertEqual(workspace.plasmon_collision_model_combo.currentData(), "benchmark_dense")
        workspace.close()

    def test_plasmon_request_matching_includes_preset_and_electron_policy(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_electron_policy="article_al_benchmark",
            plasmon_benchmark_preset="al_ambient_article",
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-policy-match", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertTrue(workspace._plasmon_request_matches_result(result.plasmon))
        raw_index = workspace.plasmon_electron_policy_combo.findData("raw_helios")
        self.assertGreaterEqual(raw_index, 0)
        workspace.plasmon_electron_policy_combo.setCurrentIndex(raw_index)
        process_events(20)
        self.assertFalse(workspace._plasmon_request_matches_result(result.plasmon))
        workspace.close()

    def test_plasmon_request_matching_includes_policy_compare_toggle(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_electron_policy="article_al_benchmark",
            plasmon_benchmark_preset="al_ambient_article",
            plasmon_compare_policies=True,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-policy-toggle", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertTrue(workspace._plasmon_request_matches_result(result.plasmon))
        workspace.plasmon_compare_policies_checkbox.setChecked(False)
        process_events(20)
        self.assertFalse(workspace._plasmon_request_matches_result(result.plasmon))
        workspace.close()

    def test_plasmon_metrics_show_runtime_breakdown(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase2-runtime", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        metrics = workspace.plasmon_metrics.toPlainText()
        self.assertIn("Runtime total", metrics)
        self.assertIn("Runtime spec", metrics)
        self.assertIn("Runtime disp", metrics)
        workspace.close()

    def test_warning_tree_keeps_message_text_readable(self) -> None:
        dataset, context = _synthetic_dataset()
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("ui-phase2-warnings", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        warning_result = replace(
            result,
            warnings=(DerivedWarning("plasmon", "Synthetic warning visibility check", severity="warning"),),
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(warning_result)
        process_events(20)
        parent = workspace.warnings_tree.topLevelItem(0)
        self.assertIsNotNone(parent)
        item = parent.child(0)
        self.assertIsNotNone(item)
        self.assertEqual(item.foreground(0).color().name().lower(), item.foreground(1).color().name().lower())
        workspace.close()

    def test_cancel_button_emits_cancel_requested_signal(self) -> None:
        workspace = HeliosDerivedWorkspace()
        emitted: list[str] = []
        workspace.cancel_requested.connect(lambda: emitted.append("cancel"))
        workspace.show()
        workspace.set_busy(True, "Busy test")
        process_events(20)
        self.assertTrue(workspace.cancel_button.isVisible())
        workspace.cancel_button.click()
        process_events(20)
        self.assertEqual(emitted, ["cancel"])
        workspace.close()


if __name__ == "__main__":
    unittest.main()
