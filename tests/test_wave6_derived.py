from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings, wait_until
from helios.runtime import RunContext
from helios.services.derived.analysis import (
    DerivedAnalysisParameters,
    analysis_result_time_plot_modules,
    compute_analysis_result,
    refresh_analysis_result_for_snapshot,
    registered_module_contracts,
)
from helios.services.derived.common import load_run_data
from helios.services.derived.module_contract import DerivedModuleContract
from helios.services.derived.models import (
    DerivedPlotBundle,
    InterfaceEventRecord,
    InterfaceEventsResult,
    TransmissionPartitionSummary,
    TransmissionColdRefinement,
    TransmissionRegionBudget,
    TransmissionRegimeSummary,
    PreheatBudgetRow,
    PreheatOnsetMarker,
    PreheatProfileField,
    PreheatStateMetric,
    PreheatSummary,
    PreheatThresholds,
    WaveBranchSummary,
    WaveTrackingResult,
)
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from helios.services.derived.shock_tracking import build_shock_tracking_compatibility_result, track_shock_front, track_wave_branches
from helios_analysis.controller import DerivedController
from helios_analysis.workspace import DerivedPlotPanel, HeliosDerivedWorkspace
from helios_viewer.main_window import HeliosViewerMainWindow


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
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(abs(value)) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


@dataclass(frozen=True, slots=True)
class _TemplateResult:
    scalar: float
    time_plots: tuple[object, ...]
    profile_plots: tuple[object, ...]


def _synthetic_wavefront_result(base_result):
    assert base_result.wave_tracking is not None
    time_axis = np.asarray(base_result.shock.time_s, dtype=np.float64)
    tracked = WaveBranchSummary(
        branch_id="tracked-1",
        family="shock_like",
        branch_type="compressive_shock",
        snapshot_indices=np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
        interface_index=np.asarray([10.0, 10.2, 10.4, 10.7, 11.0], dtype=np.float64),
        position_cm=np.asarray([1.0e-3, 1.2e-3, 1.45e-3, 1.7e-3, 2.0e-3], dtype=np.float64),
        velocity_cm_s=np.asarray([2.0e6, 2.2e6, 2.4e6, 2.6e6, 2.8e6], dtype=np.float64),
        score=np.asarray([0.8, 0.9, 1.0, 1.05, 1.1], dtype=np.float64),
        width_cm=np.asarray([6.0e-5, 6.2e-5, 6.3e-5, 6.4e-5, 6.5e-5], dtype=np.float64),
        confidence=0.82,
        ambiguous=False,
        propagation_direction="low_to_high",
        breakout_time_s=float(time_axis[5]),
        support_class="tracked",
        sample_count=5,
        duration_s=float(time_axis[5] - time_axis[1]),
        integrated_score=4.85,
        position_span_cm=1.0e-3,
        significance=8.5,
        continuity_fraction=1.0,
        primary=True,
    )
    provisional = WaveBranchSummary(
        branch_id="prov-1",
        family="contact_like",
        branch_type="contact_transition",
        snapshot_indices=np.asarray([3, 4], dtype=np.int32),
        interface_index=np.asarray([13.0, 13.2], dtype=np.float64),
        position_cm=np.asarray([2.4e-3, 2.55e-3], dtype=np.float64),
        velocity_cm_s=np.asarray([1.0e6, 1.1e6], dtype=np.float64),
        score=np.asarray([1.25, 1.10], dtype=np.float64),
        width_cm=np.asarray([4.0e-5, 4.4e-5], dtype=np.float64),
        confidence=0.94,
        ambiguous=False,
        propagation_direction="low_to_high",
        breakout_time_s=None,
        support_class="provisional",
        sample_count=2,
        duration_s=float(time_axis[4] - time_axis[3]),
        integrated_score=2.35,
        position_span_cm=1.5e-4,
        significance=0.7,
        continuity_fraction=1.0,
        notes=("Support classification: provisional candidate (fewer than 3 supporting samples).",),
    )
    reflected = WaveBranchSummary(
        branch_id="weak-1",
        family="shock_like",
        branch_type="reflected_shock",
        snapshot_indices=np.asarray([2, 3, 4, 5], dtype=np.int32),
        interface_index=np.asarray([11.2, 10.9, 10.6, 10.3], dtype=np.float64),
        position_cm=np.asarray([2.2e-3, 2.0e-3, 1.82e-3, 1.65e-3], dtype=np.float64),
        velocity_cm_s=np.asarray([-1.7e6, -1.8e6, -1.9e6, -2.0e6], dtype=np.float64),
        score=np.asarray([0.55, 0.68, 0.71, 0.74], dtype=np.float64),
        width_cm=np.asarray([5.0e-5, 5.1e-5, 5.2e-5, 5.3e-5], dtype=np.float64),
        confidence=0.71,
        ambiguous=False,
        propagation_direction="high_to_low",
        breakout_time_s=None,
        support_class="short_weak",
        sample_count=4,
        duration_s=float(time_axis[5] - time_axis[2]),
        integrated_score=2.68,
        position_span_cm=5.5e-4,
        significance=3.2,
        continuity_fraction=1.0,
    )
    wave_tracking = WaveTrackingResult(
        method="synthetic-wavefront",
        coordinate_label=base_result.wave_tracking.coordinate_label,
        supported_formula_hooks=base_result.wave_tracking.supported_formula_hooks,
        evidence_maps=base_result.wave_tracking.evidence_maps,
        candidates=base_result.wave_tracking.candidates,
        branches=(tracked, reflected, provisional),
        primary_branch_id="tracked-1",
        candidate_count=7,
        tracked_branch_count=1,
        short_branch_count=1,
        provisional_branch_count=1,
        suppressed_branch_count=3,
        warnings=base_result.wave_tracking.warnings,
    )
    interface_events = InterfaceEventsResult(
        available=True,
        supported=True,
        events=(
            InterfaceEventRecord(
                event_kind="wave_interface_event",
                interface_label="Region 1 -> 2",
                boundary_zone=32,
                snapshot_index=4,
                time_s=float(time_axis[4]),
                position_cm=2.55e-3,
                branch_id="tracked-1",
                event_classification="transmitted_shock",
                support_class="tracked",
                significance=7.2,
                confidence=0.88,
                ambiguous=False,
                incident_branch_type="compressive_shock",
                incident_arrival_time_s=float(time_axis[4]),
                transmitted_branch_id="tracked-1",
                transmitted_branch_type="transmitted_shock",
                transmitted_time_s=float(time_axis[4]),
                incident_peak_pressure_j_cm3=2.5,
                transmitted_peak_pressure_j_cm3=1.8,
                incident_compression_ratio=1.6,
                transmitted_compression_ratio=1.3,
                incident_speed_cm_s=2.6e6,
                transmitted_speed_cm_s=2.2e6,
                pressure_impulse_upstream_j_s_cm3=3.2e-9,
                pressure_impulse_downstream_j_s_cm3=2.4e-9,
                incident_energy_j_cm2=1.5e-2,
                transmitted_energy_j_cm2=8.5e-3,
                reflected_energy_j_cm2=1.2e-3,
                transfer_fraction=0.567,
                reflection_fraction=0.080,
                dominant_transfer_channel="mostly pressure-work",
                channel_fraction_internal=0.22,
                channel_fraction_kinetic=0.14,
                channel_fraction_pressure_work=0.64,
                impedance_preview_supported=True,
                impedance_upstream=12.5,
                impedance_downstream=16.8,
                impedance_reflection_preview=0.021,
                impedance_transmission_preview=0.979,
                legal_behavior=True,
                notes=("Synthetic event for UI regression coverage.",),
            ),
        ),
        tracked_event_count=1,
        weak_event_count=0,
        suppressed_event_count=2,
        classification_counts=(("transmitted_shock", 1),),
        available_metrics=("classification", "pressure_impulse", "support", "timing", "transfer_fractions"),
        notes=("Synthetic interface-event summary for WaveFront UI coverage.",),
        warnings=(),
    )
    return replace(base_result, wave_tracking=wave_tracking, interface_events=interface_events)


def _synthetic_preheat_summary(base_result) -> PreheatSummary:
    time_axis = np.asarray(base_result.shock.time_s, dtype=np.float64)
    n_zones = 8
    zone_axis = np.arange(n_zones, dtype=np.float64)
    delta_te = np.full(time_axis.shape, np.nan, dtype=np.float64)
    extent = np.full(time_axis.shape, np.nan, dtype=np.float64)
    budget = np.full(time_axis.shape, np.nan, dtype=np.float64)
    limit = min(5, time_axis.size)
    delta_te[:limit] = np.linspace(0.05, 0.12, limit)
    extent[:limit] = np.linspace(0.01, 0.08, limit)
    budget[:limit] = np.linspace(0.0, 1.1e-2, limit)
    profile_delta_te = np.tile(np.linspace(0.0, 0.12, n_zones, dtype=np.float64), (time_axis.size, 1))
    profile_density_ratio = np.tile(np.linspace(1.0, 1.08, n_zones, dtype=np.float64), (time_axis.size, 1))
    profile_mask = np.zeros((time_axis.size, n_zones), dtype=np.float64)
    if limit:
        profile_mask[:limit, : max(1, n_zones // 3)] = 1.0
    return PreheatSummary(
        available=True,
        supported=True,
        method="synthetic-preheat",
        candidate_metric_names=("temperature_e", "radiation_net_heating", "laser_deposition"),
        scalar_summaries={
            "target_entry_time_s": float(time_axis[min(limit, time_axis.size - 1)]),
            "affected_depth_cm": 2.5e-4,
            "affected_thickness_fraction": 0.08,
            "affected_areal_mass_fraction": 0.07,
            "max_temperature_e_ev": 0.12,
            "preheat_penalty_ratio": 0.06,
        },
        target_selection_mode="user_selected",
        target_region_id=1,
        auto_target_region_id=2,
        incident_region_id=3,
        deepest_reached_region_id=4,
        target_material_index=1,
        target_label="Region 1 (Al)",
        auto_target_label="Region 2 (Glue)",
        incident_region_label="Region 3 (Ablator)",
        deepest_reached_label="Region 4 (Window)",
        primary_branch_id="tracked-1",
        primary_branch_support_class="tracked",
        primary_branch_significance=8.5,
        target_entry_interface_label="Region 1 -> 2",
        target_entry_boundary_zone=32,
        target_entry_time_s=float(time_axis[min(limit, time_axis.size - 1)]),
        preheat_window_end_time_s=float(time_axis[min(limit, time_axis.size - 1)]),
        target_zone_count=120,
        available_fields=("temperature_e", "temperature_i", "radiation_net_heating", "laser_deposition"),
        missing_fields=("laser_source",),
        thresholds=PreheatThresholds(
            max_density_ratio=1.10,
            max_relative_pressure=0.25,
            min_delta_temperature_e_ev=0.05,
            min_delta_mean_charge=0.05,
            min_delta_electron_energy_j_g=1.0e7,
            min_radiation_net_heating_j_g_s=1.0e10,
            min_laser_deposition_j_g_s=1.0e10,
        ),
        state_metrics=(
            PreheatStateMetric("temperature_e", "Electron temperature", "eV", 0.08, 0.12),
            PreheatStateMetric("pressure_total", "Total pressure", "J/cm^3", 2.0, 3.5),
            PreheatStateMetric("delta_internal_energy", "Internal energy change", "J/g", 1.2e6, 2.4e6),
        ),
        budget_rows=(
            PreheatBudgetRow("laser_deposition", "Laser deposition", "J/cm^2", 3.0e-3, 0.25),
            PreheatBudgetRow("radiation_net_heating", "Net radiation heating", "J/cm^2", 6.5e-3, 0.54),
            PreheatBudgetRow("observed_internal_delta", "Observed internal-energy change", "J/cm^2", 1.2e-2, 1.0),
            PreheatBudgetRow("residual", "Residual", "J/cm^2", 2.5e-3, 0.21, notes=("Residual remains explicit by design.",)),
        ),
        onset_markers=(
            PreheatOnsetMarker("temperature_e", "Electron-temperature onset", 0.05, float(time_axis[0]), 0.05, "eV"),
            PreheatOnsetMarker("radiation", "Radiation-heating onset", 1.0e10, float(time_axis[1]), 2.5e10, "J/g/s"),
        ),
        time_plots=(
            DerivedPlotBundle(
                key="preheat_temperature",
                title="Target preheat state before primary-shock entry",
                x_label="Time [s]",
                y_label="Temperature rise [eV]",
                x_values=time_axis,
                y_series=(delta_te, np.asarray(delta_te * 0.5, dtype=np.float64), np.asarray(delta_te * 0.2, dtype=np.float64)),
                curve_names=("Delta Te", "Delta Ti", "Delta Tr"),
            ),
            DerivedPlotBundle(
                key="preheat_extent",
                title="Preheat extent before target entry",
                x_label="Time [s]",
                y_label="Affected fraction",
                x_values=time_axis,
                y_series=(extent, np.asarray(extent * 0.9, dtype=np.float64)),
                curve_names=("Thickness fraction", "Areal-mass fraction"),
            ),
            DerivedPlotBundle(
                key="preheat_budget",
                title="Target-integrated preheat budget before shock entry",
                x_label="Time [s]",
                y_label="Areal budget [J/cm^2]",
                x_values=time_axis,
                y_series=(budget, np.asarray(budget * 0.7, dtype=np.float64), np.asarray(budget * 1.1, dtype=np.float64)),
                curve_names=("Laser deposition", "Net radiation heating", "Observed internal delta"),
            ),
        ),
        snapshot_indices=np.arange(time_axis.size, dtype=np.int32),
        latest_pre_entry_snapshot_index=(None if limit <= 0 else int(limit - 1)),
        peak_snapshot_index=(None if limit <= 0 else int(limit - 1)),
        target_zone_indices=zone_axis,
        target_static_x_cm=np.linspace(0.0, 7.0e-4, n_zones, dtype=np.float64),
        profile_fields=(
            PreheatProfileField(
                key="delta_temperature_e",
                label="Delta Te",
                unit="eV",
                values=profile_delta_te,
                notes=("Synthetic target electron-temperature rise profile.",),
            ),
            PreheatProfileField(
                key="density_ratio",
                label="rho / rho0",
                unit="",
                values=profile_density_ratio,
                notes=("Synthetic compression proxy profile.",),
            ),
            PreheatProfileField(
                key="preheat_mask",
                label="Preheat mask",
                unit="fraction",
                values=profile_mask,
                notes=("Synthetic preheated-but-unshocked mask.",),
            ),
        ),
        snapshot_scalar_series={
            "affected_depth_cm": np.asarray(extent * 2.5e-3, dtype=np.float64),
            "affected_thickness_fraction": np.asarray(extent, dtype=np.float64),
            "affected_areal_mass_fraction": np.asarray(extent * 0.9, dtype=np.float64),
            "delta_temperature_e_mean": np.asarray(delta_te, dtype=np.float64),
            "delta_temperature_e_peak": np.asarray(delta_te * 1.2, dtype=np.float64),
            "delta_mean_charge_peak": np.asarray(delta_te * 0.5, dtype=np.float64),
        },
        affected_depth_cm=2.5e-4,
        affected_thickness_fraction=0.08,
        affected_areal_mass_fraction=0.07,
        severity_label="mild",
        preheat_penalty_ratio=0.06,
        dominant_source="net radiation heating",
        notes=("Synthetic preheat summary for WaveFront UI coverage.",),
        warnings=(),
    )


def _bundle(key: str, title: str, values: tuple[float, ...]) -> DerivedPlotBundle:
    axis = np.arange(len(values), dtype=np.float64)
    return DerivedPlotBundle(
        key=key,
        title=title,
        x_label="t [ns]",
        y_label="value",
        x_values=axis,
        y_series=(np.asarray(values, dtype=np.float64),),
        curve_names=(title,),
        boundary_positions=(),
        value_scale_mode="linear",
    )


class Wave6DerivedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_lazy_time_plots_can_be_filled_incrementally(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()

        partial = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "partial"),
            requested_time_plot_modules=frozenset(),
        )
        self.assertEqual(analysis_result_time_plot_modules(partial), frozenset())

        xrd_only = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "xrd"),
            requested_time_plot_modules=frozenset({"xrd"}),
            base_result=partial,
        )
        self.assertEqual(analysis_result_time_plot_modules(xrd_only), frozenset({"xrd"}))
        self.assertGreater(len(xrd_only.xrd.time_plots), 0)
        self.assertEqual(len(xrd_only.plasmon.time_plots), 0)
        self.assertEqual(len(xrd_only.transmission.time_plots), 0)
        self.assertEqual(len(xrd_only.spectroscopy.time_plots), 0)

        full = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "full"),
        )
        np.testing.assert_allclose(
            np.asarray(xrd_only.xrd.time_plots[0].y_series[0], dtype=np.float64),
            np.asarray(full.xrd.time_plots[0].y_series[0], dtype=np.float64),
            equal_nan=True,
        )

    def test_legacy_path_can_skip_wavefront_until_requested(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()

        legacy = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "legacy-only"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=False,
        )
        self.assertIsNone(legacy.wave_tracking)
        self.assertIsNone(legacy.interface_events)
        self.assertIsNone(legacy.preheat)

        wavefront = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "legacy-then-wavefront"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=True,
            base_result=legacy,
        )
        self.assertEqual(legacy.shock, wavefront.shock)
        self.assertIsNotNone(wavefront.wave_tracking)
        self.assertIsNotNone(wavefront.interface_events)
        self.assertIsNotNone(wavefront.preheat)

    def test_plasmon_reuses_cached_time_series_moments_on_repeat(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=40)
        parameters = DerivedAnalysisParameters()
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()

        first = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=True,
            analysis_cache=analysis_cache,
        )
        stats_after_first = analysis_cache.stats()
        second = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=True,
            analysis_cache=analysis_cache,
        )
        stats_after_second = analysis_cache.stats()

        self.assertGreater(stats_after_first["time_series_misses"], 0)
        self.assertGreater(stats_after_second["time_series_hits"], stats_after_first["time_series_hits"])
        np.testing.assert_allclose(
            np.asarray(first.time_plots[0].y_series[0], dtype=np.float64),
            np.asarray(second.time_plots[0].y_series[0], dtype=np.float64),
            equal_nan=True,
        )

    def test_workspace_requests_missing_time_plots_when_module_tab_becomes_active(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "workspace-partial"),
            requested_time_plot_modules=frozenset(),
        )

        workspace = HeliosDerivedWorkspace()
        triggered = {"count": 0}
        try:
            workspace.refresh_requested.connect(lambda: triggered.__setitem__("count", triggered["count"] + 1))
            workspace.set_context(context)
            workspace.set_result(partial)
            process_events(20)
            self.assertEqual(triggered["count"], 0)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.xrd_tab))
            process_events(20)
            self.assertGreaterEqual(triggered["count"], 1)
        finally:
            workspace.close()

    def test_workspace_requests_wavefront_when_tab_becomes_active(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "workspace-wavefront-partial"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=False,
        )

        workspace = HeliosDerivedWorkspace()
        triggered = {"count": 0}
        try:
            workspace.refresh_requested.connect(lambda: triggered.__setitem__("count", triggered["count"] + 1))
            workspace.set_context(context)
            workspace.set_result(partial)
            process_events(20)
            self.assertEqual(triggered["count"], 0)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)
            self.assertGreaterEqual(triggered["count"], 1)
        finally:
            workspace.close()

    def test_workspace_requests_preheat_when_tab_becomes_active(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "workspace-preheat-partial"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=False,
        )

        workspace = HeliosDerivedWorkspace()
        triggered = {"count": 0}
        try:
            workspace.refresh_requested.connect(lambda: triggered.__setitem__("count", triggered["count"] + 1))
            workspace.set_context(context)
            workspace.set_result(partial)
            process_events(20)
            self.assertEqual(triggered["count"], 0)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.preheat_tab))
            process_events(20)
            self.assertGreaterEqual(triggered["count"], 1)
        finally:
            workspace.close()

    def test_workspace_resets_to_shock_tab_on_run_change(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        other_path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        other_dataset = load_run_data(other_path)
        other_context = _context_from_dataset(other_path, other_dataset, snapshot_index=3)

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)
            self.assertTrue(workspace.wavefront_requested())

            workspace.set_context(other_context)
            process_events(20)
            self.assertEqual(workspace.result_tabs.currentWidget(), workspace.shock_tab)
            self.assertFalse(workspace.wavefront_requested())
        finally:
            workspace.close()

    def test_workspace_preserves_dedicated_wavefront_performance_summary(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_performance_summary("Source: cache | WaveFront 22.1 s | tracker 22.1 s", wavefront=True)
            workspace.set_performance_summary("Source: background compute | snapshot refresh 1.6 ms")
            self.assertEqual(
                "Source: background compute | snapshot refresh 1.6 ms",
                workspace.performance_summary_label.text(),
            )
            self.assertEqual(
                "Source: cache | WaveFront 22.1 s | tracker 22.1 s",
                workspace.wavefront_metrics_label.text(),
            )
        finally:
            workspace.close()

    def test_workspace_transmission_tab_surfaces_xcom_status(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-xcom-ui"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        transmission = replace(
            result.transmission,
            model_type="xcom",
            selected_mode="xcom",
            photon_energy_kev=8.0,
            selected_tau=0.198,
            selected_transmission=0.82,
            source="cache",
            status_message="XCOM refinement active at 8 keV (cache).",
            cold_refinement=TransmissionColdRefinement(
                backend_status="refined",
                applicability="recommended",
                message="XCOM refinement active at 8 keV (cache).",
                backend_name="XCOM",
                backend_available=True,
                backend_fingerprint="fake-backend-fingerprint",
                source="cache",
                photon_energies_kev=(8.0,),
                transmission=(0.82,),
                optical_depth=(0.198,),
                attenuation_mode="total_with_coherent",
                resolved_materials=("Fe",),
            ),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            workspace.set_result(replace(result, transmission=transmission))
            process_events(20)
            self.assertIn("XCOM", workspace.transmission_model_label.text())
            self.assertIn("Requested:", workspace.transmission_model_label.text())
            self.assertIn("Applied:", workspace.transmission_model_label.text())
            self.assertIn("cache", workspace.transmission_model_label.text().lower())
            self.assertIn("recommended", workspace.transmission_applicability_label.text().lower())
            self.assertIn("Requested=XCOM", workspace.transmission_summary_label.text())
            self.assertIn("Requested mode: XCOM", workspace.transmission_status_pane.toPlainText())
            self.assertTrue(workspace.transmission_refine_button.isEnabled())
        finally:
            workspace.close()

    def test_workspace_transmission_panel_preserves_requested_selection_and_marks_pending_apply(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-requested-vs-applied"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("free_free_thomson")))
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("keV")))
            workspace.transmission_energy_spin.setValue(9.5)
            process_events(20)
            workspace.set_result(result)
            process_events(20)
            self.assertEqual("free_free_thomson", workspace.transmission_mode_combo.currentData())
            self.assertEqual("keV", workspace.transmission_energy_unit_combo.currentData())
            self.assertAlmostEqual(9.5, workspace.transmission_energy_spin.value(), places=6)
            self.assertIn("Requested: Free-free + Thomson", workspace.transmission_model_label.text())
            self.assertIn("Applied: Thomson", workspace.transmission_model_label.text())
            self.assertIn("Pending apply", workspace.transmission_status_pane.toPlainText())
            self.assertIn("Pending apply", workspace.transmission_refinement_label.text())
        finally:
            workspace.close()

    def test_workspace_transmission_energy_unit_selector_converts_value_and_clears_pending_apply(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-energy-units"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        transmission = replace(
            result.transmission,
            model_type="xcom",
            selected_mode="xcom",
            photon_energy_kev=0.1,
            selected_tau=0.25,
            selected_transmission=0.78,
            source="cache",
            status_message="XCOM refinement active at 100 eV (cache).",
            cold_refinement=TransmissionColdRefinement(
                backend_status="refined",
                applicability="recommended",
                message="XCOM refinement active at 100 eV (cache).",
                backend_name="XCOM",
                backend_available=True,
                backend_fingerprint="fake-backend-fingerprint",
                source="cache",
                photon_energies_kev=(0.1,),
                transmission=(0.78,),
                optical_depth=(0.25,),
                attenuation_mode="total_with_coherent",
                resolved_materials=("Al",),
            ),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("eV")))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            workspace.transmission_energy_spin.setValue(100.0)
            self.assertAlmostEqual(0.1, workspace.parameters().transmission_photon_energy_kev, places=6)
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("Angstrom")))
            process_events(20)
            self.assertAlmostEqual(123.98419843320028, workspace.transmission_energy_spin.value(), places=3)
            self.assertAlmostEqual(0.1, workspace.parameters().transmission_photon_energy_kev, places=6)
            workspace.set_result(replace(result, transmission=transmission))
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            self.assertNotIn("Pending apply", workspace.transmission_status_pane.toPlainText())
            self.assertIn("Angstrom", workspace.transmission_summary_label.text())
        finally:
            workspace.close()

    def test_workspace_non_thomson_transmission_shows_selected_time_traces(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-no-time-reference"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        transmission = replace(
            result.transmission,
            model_type="free_free_thomson",
            selected_mode="free_free_thomson",
            photon_energy_kev=8.0,
            selected_tau=0.1,
            selected_transmission=0.9,
            source="baseline",
            status_message="Free-free + Thomson quick look.",
            time_plots=(
                DerivedPlotBundle(
                    key="selected_tau",
                    title="Free-free + Thomson tau vs time",
                    x_label="Time [ns]",
                    y_label="Tau",
                    x_values=np.asarray(result.transmission.time_plots[0].x_values, dtype=np.float64),
                    y_series=(np.asarray(result.transmission.time_plots[0].y_series[0], dtype=np.float64),),
                    curve_names=("Selected tau",),
                ),
                DerivedPlotBundle(
                    key="selected_transmission",
                    title="Free-free + Thomson transmission vs time",
                    x_label="Time [ns]",
                    y_label="Transmission",
                    x_values=np.asarray(result.transmission.time_plots[0].x_values, dtype=np.float64),
                    y_series=(np.asarray(result.transmission.time_plots[1].y_series[0], dtype=np.float64),),
                    curve_names=("Selected transmission",),
                ),
            ),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("free_free_thomson")))
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("keV")))
            workspace.transmission_energy_spin.setValue(8.0)
            workspace.set_result(replace(result, transmission=transmission))
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            self.assertGreaterEqual(workspace.transmission_plot_panel.time_combo.count(), 2)
            self.assertTrue(workspace.transmission_plot_panel.time_combo.isEnabled())
            self.assertIn("transmission vs time", workspace.transmission_plot_panel.time_combo.itemText(1).lower())
            self.assertGreaterEqual(workspace.transmission_plot_panel.profile_combo.count(), 1)
            self.assertIn("Photon energy: 8 keV.", workspace.transmission_status_pane.toPlainText())
            self.assertNotIn("hidden", workspace.transmission_status_pane.toPlainText().lower())
        finally:
            workspace.close()

    def test_workspace_transmission_tab_requests_refresh_when_energy_differs(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-energy-refresh"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        transmission = replace(
            result.transmission,
            model_type="xcom",
            selected_mode="xcom",
            photon_energy_kev=8.0,
            selected_tau=0.198,
            selected_transmission=0.82,
            source="cache",
            status_message="XCOM refinement active at 8 keV (cache).",
        )
        workspace = HeliosDerivedWorkspace()
        triggered = {"count": 0}
        try:
            workspace.set_context(context)
            workspace.refresh_requested.connect(lambda: triggered.__setitem__("count", triggered["count"] + 1))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("keV")))
            workspace.set_result(replace(result, transmission=transmission))
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.xrd_tab))
            process_events(20)
            triggered["count"] = 0
            workspace.transmission_energy_spin.setValue(9.5)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            self.assertGreaterEqual(triggered["count"], 1)
        finally:
            workspace.close()

    def test_workspace_transmission_request_matching_requires_snapshot_identity(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        result_context = _context_from_dataset(path, dataset, snapshot_index=0)
        request_context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            result_context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-snapshot-match"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(request_context)
            workspace.set_result(result)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            self.assertFalse(workspace._transmission_request_matches_result(result.transmission))
            self.assertIn("Pending apply", workspace.transmission_status_pane.toPlainText())
        finally:
            workspace.close()

    def test_transmission_status_pane_scroll_position_persists_across_updates(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-scroll"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        partition = TransmissionPartitionSummary(
            mode="auto_hybrid",
            photon_energy_kev=8.0,
            zone_count=20,
            regime_summaries=(
                TransmissionRegimeSummary("xcom", 8, 0.4, 0.5, 0.7),
                TransmissionRegimeSummary("free_free_thomson", 6, 0.35, 0.3, 0.2),
                TransmissionRegimeSummary("thomson_fallback", 6, 0.25, 0.2, 0.1),
            ),
            notes=tuple(f"Transmission note {index}" for index in range(120)),
        )
        updated = replace(
            result.transmission,
            selected_mode="auto_hybrid",
            model_type="auto_hybrid",
            photon_energy_kev=8.0,
            partition=partition,
        )
        updated_second = replace(
            updated,
            partition=replace(partition, notes=tuple(f"Transmission note {index} updated" for index in range(120))),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(replace(result, transmission=updated))
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            scroll_bar = workspace.transmission_status_pane.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), 0)
            target_value = max(1, scroll_bar.maximum() // 2)
            scroll_bar.setSliderPosition(target_value)
            process_events(20)
            self.assertGreater(scroll_bar.value(), 0)
            workspace.set_result(replace(result, transmission=updated_second))
            process_events(20)
            self.assertGreater(scroll_bar.value(), 0)
            self.assertGreaterEqual(scroll_bar.value(), max(1, target_value - 5))
        finally:
            workspace.close()

    def test_workspace_transmission_table_marks_mixed_auto_regions_explicitly(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "transmission-mixed-table"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )
        mixed_budget = TransmissionRegionBudget(
            region_id=1,
            areal_density_g_cm2=0.01,
            electron_column_cm2=1.0,
            thomson_tau=1.0e-4,
            free_free_tau=2.0e-4,
            xcom_tau=2.0e-2,
            total_tau=2.03e-2,
            xcom_path_fraction=0.01,
            free_free_thomson_path_fraction=0.98,
            thomson_fallback_path_fraction=0.01,
            xcom_tau_fraction=0.97,
            free_free_thomson_tau_fraction=0.02,
            thomson_fallback_tau_fraction=0.01,
            dominant_regime="xcom",
            notes=("Mixed region: path is mostly Free-free + Thomson (98.0%), but selected tau is dominated by XCOM (97.0%).",),
        )
        partition = TransmissionPartitionSummary(
            mode="auto_hybrid",
            photon_energy_kev=8.0,
            zone_count=10,
            regime_summaries=(
                TransmissionRegimeSummary("xcom", 2, 0.01, 0.5, 0.97),
                TransmissionRegimeSummary("free_free_thomson", 7, 0.98, 0.49, 0.02),
                TransmissionRegimeSummary("thomson_fallback", 1, 0.01, 0.01, 0.01),
            ),
            notes=("Selected mode: Auto hybrid.",),
        )
        updated = replace(
            result.transmission,
            selected_mode="auto_hybrid",
            model_type="auto_hybrid",
            source="precomputed_xcom_table",
            photon_energy_kev=8.0,
            partition=partition,
            region_budgets=(mixed_budget,),
        )
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(replace(result, transmission=updated))
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(20)
            self.assertEqual("Region mixture", workspace.transmission_table.horizontalHeaderItem(14).text())
            item = workspace.transmission_table.item(0, 14)
            self.assertIsNotNone(item)
            assert item is not None
            self.assertEqual("Mixed (XCOM-dominant)", item.text())
            self.assertIn("Mixed region:", item.toolTip())
        finally:
            workspace.close()

    def test_controller_promotes_partial_core_result_to_time_plot_request(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "controller-partial"),
            requested_time_plot_modules=frozenset(),
        )

        controller = DerivedController()
        captured: dict[str, object] = {}
        try:
            workspace = controller.widget()
            controller._active = True
            controller._context = context.copy()
            workspace.set_context(context)
            workspace.set_result(partial)
            core_key = controller._build_core_request_key(workspace.parameters())
            controller._analysis_core_cache[core_key] = partial
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.xrd_tab))
            process_events(20)

            original_launch = controller._launch_request

            def _capture(request) -> None:
                captured["request"] = request

            controller._launch_request = _capture  # type: ignore[method-assign]
            controller._start_recompute()
            request = captured.get("request")
            self.assertIsNotNone(request)
            self.assertEqual("time_plots", request.update_kind)
            self.assertEqual(frozenset({"xrd"}), request.requested_time_plot_modules)
        finally:
            controller.shutdown()

    def test_controller_schedules_transmission_model_request(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        parameters = DerivedAnalysisParameters()
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "controller-transmission-model"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )

        controller = DerivedController()
        captured: dict[str, object] = {}
        try:
            workspace = controller.widget()
            controller._active = True
            controller._context = context.copy()
            workspace.set_display_settings(SimpleNamespace(photon_unit="keV"))
            workspace.set_context(context)
            workspace.set_result(result)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            workspace.transmission_energy_unit_combo.setCurrentIndex(max(0, workspace.transmission_energy_unit_combo.findData("keV")))
            workspace.transmission_energy_spin.setValue(9.5)
            process_events(20)

            def _capture(request) -> None:
                captured["request"] = request

            controller._launch_request = _capture  # type: ignore[method-assign]
            workspace.transmission_refine_requested.emit()
            process_events(20)

            request = captured["request"]
            self.assertEqual("transmission_model", request.update_kind)
            self.assertAlmostEqual(9.5, request.transmission_model_energy_kev, places=6)
            self.assertEqual("xcom", request.transmission_model_mode)
            self.assertIsNotNone(request.base_request_key)
        finally:
            controller.shutdown()

    def test_controller_transmission_refine_uses_baseline_request_key_when_current_result_is_overlay(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=3)
        parameters = DerivedAnalysisParameters()
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "controller-transmission-overlay-base"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )

        controller = DerivedController()
        captured: dict[str, object] = {}
        try:
            workspace = controller.widget()
            controller._active = True
            controller._context = context.copy()
            workspace.set_context(context)
            workspace.set_result(result)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            process_events(20)

            base_key = controller._build_request_key(workspace.parameters())
            controller._displayed_result_key = (*base_key, "transmission_model", "free_free", 8.0)

            def _capture(request) -> None:
                captured["request"] = request

            controller._launch_request = _capture  # type: ignore[method-assign]
            workspace.transmission_refine_requested.emit()
            process_events(20)

            request = captured["request"]
            self.assertEqual(base_key, request.base_request_key)
            self.assertEqual("transmission_model", request.update_kind)
        finally:
            controller.shutdown()

    def test_controller_core_request_key_changes_with_snapshot(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context0 = _context_from_dataset(path, dataset, snapshot_index=0)
        context3 = _context_from_dataset(path, dataset, snapshot_index=3)
        controller = DerivedController()
        try:
            parameters = DerivedAnalysisParameters()
            controller._context = context0.copy()
            key0 = controller._build_core_request_key(parameters)
            controller._context = context3.copy()
            key3 = controller._build_core_request_key(parameters)
            self.assertNotEqual(key0, key3)
            self.assertEqual(0, key0[-1])
            self.assertEqual(3, key3[-1])
        finally:
            controller.shutdown()

    def test_controller_transmission_refine_uses_current_snapshot_request_not_stale_displayed_snapshot(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context0 = _context_from_dataset(path, dataset, snapshot_index=0)
        context3 = _context_from_dataset(path, dataset, snapshot_index=3)
        parameters = DerivedAnalysisParameters()
        result3 = compute_analysis_result(
            dataset,
            context3,
            parameters=parameters,
            context_key=("wave6", "controller-transmission-current-snapshot"),
            requested_time_plot_modules=frozenset({"transmission"}),
        )

        controller = DerivedController()
        captured: dict[str, object] = {}
        try:
            workspace = controller.widget()
            controller._active = True
            controller._context = context3.copy()
            workspace.set_context(context3)
            workspace.set_result(result3)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            workspace.transmission_mode_combo.setCurrentIndex(max(0, workspace.transmission_mode_combo.findData("xcom")))
            process_events(20)

            controller._context = context0.copy()
            stale_request_key = controller._build_request_key(workspace.parameters())
            controller._context = context3.copy()
            current_request_key = controller._build_request_key(workspace.parameters())
            controller._displayed_result_key = stale_request_key

            def _capture(request) -> None:
                captured["request"] = request

            controller._launch_request = _capture  # type: ignore[method-assign]
            workspace.transmission_refine_requested.emit()
            process_events(20)

            request = captured["request"]
            self.assertEqual(current_request_key, request.base_request_key)
            self.assertNotEqual(stale_request_key, request.base_request_key)
            self.assertEqual(3, request.context.snapshot_index)
        finally:
            controller.shutdown()

    def test_viewer_material_list_uses_resolved_metadata_labels(self) -> None:
        window = HeliosViewerMainWindow()
        try:
            window.load_file(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(lambda: window.run_payload is not None and window.material_list.count() > 0, timeout_s=30.0)
            item_text = window.material_list.item(0).text()
            self.assertIn("Material 1", item_text)
            self.assertIn("Fe", item_text)
            self.assertIn("zones", item_text)
        finally:
            window.close()

    def test_controller_promotes_partial_core_result_to_wavefront_request(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "controller-wavefront-partial"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=False,
        )

        controller = DerivedController()
        captured: dict[str, object] = {}
        try:
            workspace = controller.widget()
            controller._active = True
            controller._context = context.copy()
            workspace.set_context(context)
            workspace.set_result(partial)
            core_key = controller._build_core_request_key(workspace.parameters())
            controller._analysis_core_cache[core_key] = partial
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)

            def _capture(request) -> None:
                captured["request"] = request

            controller._launch_request = _capture  # type: ignore[method-assign]
            controller._start_recompute()
            request = captured.get("request")
            self.assertIsNotNone(request)
            self.assertEqual("wavefront", request.update_kind)
            self.assertTrue(request.include_wavefront)
        finally:
            controller.shutdown()

    def test_lazy_time_plot_fill_reuses_existing_shock_result(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()
        partial = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("wave6", "shock-partial"),
            requested_time_plot_modules=frozenset(),
        )

        with mock.patch(
            "helios.services.derived.analysis.track_shock_front",
            side_effect=AssertionError("shock should be reused from the base result"),
        ):
            updated = compute_analysis_result(
                dataset,
                context,
                parameters=parameters,
                context_key=("wave6", "shock-reused"),
                requested_time_plot_modules=frozenset({"plasmon"}),
                base_result=partial,
            )
        self.assertEqual(partial.shock, updated.shock)
        self.assertGreater(len(updated.plasmon.time_plots), 0)

    def test_preheat_region_override_reuses_cached_wave_tracking(self) -> None:
        path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        base = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-reuse-base"),
            requested_time_plot_modules=frozenset(),
        )
        assert base.wave_tracking is not None

        with mock.patch(
            "helios.services.derived.analysis.track_wave_branches",
            side_effect=AssertionError("wave tracking should be reused when only the preheat ROI changes"),
        ):
            updated = compute_analysis_result(
                dataset,
                context,
                parameters=DerivedAnalysisParameters(preheat_target_region_id=2),
                context_key=("wave6", "preheat-reuse-updated"),
                requested_time_plot_modules=frozenset(),
                base_result=base,
            )
        self.assertIs(base.wave_tracking, updated.wave_tracking)
        assert updated.preheat is not None
        self.assertEqual("user_selected", updated.preheat.target_selection_mode)
        self.assertEqual(2, updated.preheat.target_region_id)

    def test_shock_tracking_reuses_cached_detector_series_on_repeat(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        parameters = DerivedAnalysisParameters()
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()

        first = track_shock_front(
            dataset,
            context,
            parameters=parameters,
            geometry=geometry,
            analysis_cache=analysis_cache,
        )
        stats_after_first = analysis_cache.stats()
        second = track_shock_front(
            dataset,
            context,
            parameters=parameters,
            geometry=geometry,
            analysis_cache=analysis_cache,
        )
        stats_after_second = analysis_cache.stats()

        self.assertGreater(stats_after_first["time_series_misses"], 0)
        self.assertGreater(stats_after_second["time_series_hits"], stats_after_first["time_series_hits"])
        np.testing.assert_allclose(first.smoothed_position_cm, second.smoothed_position_cm, equal_nan=True)

    def test_workspace_formats_tiny_times_without_fake_zero_seconds(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            formatted = workspace._format_time(2.5e-12)
            self.assertIn("ps", formatted)
            self.assertNotEqual("0.000 s", formatted)
        finally:
            workspace.close()

    def test_workspace_busy_indicator_tracks_background_state(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            process_events(20)
            workspace.set_busy(True, "Tracking WaveFront branches...")
            process_events(300)
            self.assertTrue(workspace.activity_progress.isVisible())
            self.assertIn("elapsed", workspace.result_status_label.text())
            workspace.set_busy(False, "WaveFront branches updated.")
            process_events(20)
            self.assertFalse(workspace.activity_progress.isVisible())
            self.assertEqual("WaveFront branches updated.", workspace.result_status_label.text())
        finally:
            workspace.close()

    def test_wave_tracking_exposes_branch_support_and_significance(self) -> None:
        path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-support"),
            requested_time_plot_modules=frozenset(),
        )
        assert result.wave_tracking is not None
        self.assertGreaterEqual(result.wave_tracking.candidate_count, 1)
        self.assertGreaterEqual(result.wave_tracking.tracked_branch_count, 1)
        self.assertTrue(all(branch.support_class in {"tracked", "short_weak", "provisional"} for branch in result.wave_tracking.branches))
        self.assertTrue(all(branch.sample_count == int(np.asarray(branch.snapshot_indices, dtype=np.int32).size) for branch in result.wave_tracking.branches))
        self.assertTrue(all(branch.significance is None or float(branch.significance) >= 0.0 for branch in result.wave_tracking.branches))
        if len(result.wave_tracking.branches) >= 2:
            self.assertGreaterEqual(
                float(result.wave_tracking.branches[0].significance or 0.0),
                float(result.wave_tracking.branches[1].significance or 0.0),
            )

    def test_wavefront_default_view_suppresses_provisional_branches_and_supports_selector_modes(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-ui"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic = _synthetic_wavefront_result(result)
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)
            process_events(20)
            self.assertEqual(1, workspace.wavefront_branch_table.rowCount())
            self.assertEqual("Tracked branch", workspace.wavefront_branch_table.item(0, 2).text())
            self.assertGreaterEqual(workspace.wavefront_position_plot.current_curve_count, 1)
            self.assertIn("Top branch", workspace.wavefront_overview_label.text())

            workspace.wavefront_scope_combo.setCurrentIndex(workspace.wavefront_scope_combo.findData("all"))
            process_events(20)
            self.assertEqual(3, workspace.wavefront_branch_table.rowCount())
            self.assertGreaterEqual(workspace.wavefront_position_plot.current_curve_count, 1)
            self.assertIn("Provisional detections", workspace.wavefront_overview_label.text())

            workspace.wavefront_display_combo.setCurrentIndex(workspace.wavefront_display_combo.findData("events"))
            process_events(20)
            self.assertTrue(workspace.wavefront_plot_empty_label.isVisible())
            self.assertTrue(workspace.wavefront_plot_splitter.isHidden())
            self.assertEqual(1, workspace.wavefront_event_table.rowCount())
            self.assertEqual("transmitted shock", workspace.wavefront_event_table.item(0, 3).text())
            self.assertEqual("Tracked branch", workspace.wavefront_event_table.item(0, 4).text())
            self.assertIn("pressure-work", workspace.wavefront_event_table.item(0, 12).text())
            self.assertIn("Top interface events", workspace.wavefront_notes.toPlainText())
        finally:
            workspace.close()

    def test_wavefront_selectors_rerender_without_triggering_refresh(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-selector"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic = _synthetic_wavefront_result(result)
        workspace = HeliosDerivedWorkspace()
        triggered = {"count": 0}
        try:
            workspace.show()
            workspace.refresh_requested.connect(lambda: triggered.__setitem__("count", triggered["count"] + 1))
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)
            triggered["count"] = 0
            workspace.wavefront_display_combo.setCurrentIndex(workspace.wavefront_display_combo.findData("warnings"))
            workspace.wavefront_scope_combo.setCurrentIndex(workspace.wavefront_scope_combo.findData("all"))
            workspace.wavefront_direction_combo.setCurrentIndex(workspace.wavefront_direction_combo.findData("high_to_low"))
            process_events(20)
            self.assertEqual(0, triggered["count"])
        finally:
            workspace.close()

    def test_wavefront_primary_and_direction_controls_focus_expected_branches(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-primary-controls"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic = _synthetic_wavefront_result(result)
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.wavefront_tab))
            process_events(20)

            workspace.wavefront_scope_combo.setCurrentIndex(workspace.wavefront_scope_combo.findData("primary"))
            workspace.wavefront_display_combo.setCurrentIndex(workspace.wavefront_display_combo.findData("primary_position"))
            process_events(20)
            self.assertEqual(1, workspace.wavefront_branch_table.rowCount())
            self.assertEqual("tracked-1", workspace.wavefront_branch_table.item(0, 0).text())
            self.assertIn("Primary compressive", workspace.wavefront_overview_label.text())
            self.assertGreaterEqual(workspace.wavefront_position_plot.current_curve_count, 1)

            workspace.wavefront_scope_combo.setCurrentIndex(workspace.wavefront_scope_combo.findData("tracked_weak"))
            workspace.wavefront_direction_combo.setCurrentIndex(workspace.wavefront_direction_combo.findData("high_to_low"))
            process_events(20)
            self.assertEqual(1, workspace.wavefront_branch_table.rowCount())
            self.assertEqual("weak-1", workspace.wavefront_branch_table.item(0, 0).text())
        finally:
            workspace.close()

    def test_wavefront_primary_compressive_branch_matches_tracker_primary_on_clear_shock_case(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        legacy = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-primary-legacy"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=False,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "wavefront-primary-real"),
            requested_time_plot_modules=frozenset(),
            include_wavefront=True,
            base_result=legacy,
        )
        assert result.wave_tracking is not None
        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            primary = workspace._primary_compressive_branch(tuple(result.wave_tracking.branches))
            self.assertIsNotNone(primary)
            assert primary is not None
            self.assertEqual(str(result.wave_tracking.primary_branch_id), str(primary.branch_id))
            self.assertEqual("compressive_shock", str(primary.branch_type))
            self.assertEqual(str(legacy.shock.propagation_direction), str(primary.propagation_direction))
        finally:
            workspace.close()

    def test_module_contract_merges_existing_time_plots(self) -> None:
        contract = DerivedModuleContract(
            name="template",
            compute=lambda **kwargs: _TemplateResult(1.0, (), ()),
            validate=lambda result: None,
        )
        base = _TemplateResult(1.0, ("time",), ("profile",))
        updated = _TemplateResult(2.0, (), ("profile",))
        merged = contract.merge_time_plots(base, updated)
        self.assertEqual(("time",), merged.time_plots)
        self.assertEqual(("profile",), merged.profile_plots)
        self.assertEqual(2.0, merged.scalar)

    def test_analysis_result_exposes_forward_compatible_wave_seams(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "future-seams"),
            requested_time_plot_modules=frozenset(),
        )

        self.assertIsNotNone(result.wave_tracking)
        self.assertIsNotNone(result.interface_events)
        self.assertIsNotNone(result.preheat)
        assert result.wave_tracking is not None
        assert result.interface_events is not None
        assert result.preheat is not None
        self.assertIsNotNone(result.wave_tracking.primary_branch_id)
        self.assertIsNone(result.wave_tracking.compatibility_source)
        self.assertGreaterEqual(len(result.wave_tracking.branches), 1)
        self.assertEqual(
            ("shock_like", "release_like", "contact_like"),
            tuple(hook.family for hook in result.wave_tracking.supported_formula_hooks),
        )
        self.assertEqual(
            ("shock_like", "release_like", "contact_like"),
            tuple(item.family for item in result.wave_tracking.evidence_maps),
        )
        self.assertTrue(all(branch.branch_type for branch in result.wave_tracking.branches))
        self.assertGreaterEqual(result.wave_tracking.tracked_branch_count, 1)
        self.assertTrue(
            all(branch.support_class in {"tracked", "short_weak", "provisional"} for branch in result.wave_tracking.branches)
        )
        self.assertTrue(result.interface_events.supported)
        self.assertTrue(result.preheat.supported)
        self.assertIn("temperature_e", result.preheat.candidate_metric_names)

    def test_wave_tracking_compatibility_adapter_produces_legacy_shock_shape(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        wave = track_wave_branches(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            geometry=build_analysis_geometry(
                dataset,
                context,
                observation_side="front",
                line_of_sight_angle_deg=0.0,
                line_of_sight_impact_parameter_cm=0.0,
                profile_coordinate_mode="viewer",
            ),
            analysis_cache=AnalysisStateCache(),
        )
        adapted = build_shock_tracking_compatibility_result(dataset, wave)
        self.assertEqual(dataset.time_s.size, adapted.time_s.size)
        self.assertEqual(dataset.time_s.size, adapted.position_cm.size)
        self.assertEqual(dataset.time_s.size, adapted.detector_score.size)
        self.assertEqual(dataset.summary["n_snapshots"], adapted.zone_index.size)

    def test_wave_tracking_finds_multiple_branch_types_on_layered_run(self) -> None:
        path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "layered-wave"),
            requested_time_plot_modules=frozenset(),
        )
        assert result.wave_tracking is not None
        branch_types = {branch.branch_type for branch in result.wave_tracking.branches}
        self.assertIn("compressive_shock", branch_types)
        self.assertGreaterEqual(len(branch_types), 2)

    def test_interface_events_on_layered_run_expose_classification_and_support_filtering(self) -> None:
        path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "interface-events-layered"),
            requested_time_plot_modules=frozenset(),
        )
        assert result.interface_events is not None
        self.assertTrue(result.interface_events.supported)
        self.assertGreaterEqual(result.interface_events.suppressed_event_count, 0)
        self.assertGreaterEqual(len(result.interface_events.events), 1)
        self.assertTrue(all(event.support_class in {"tracked", "short_weak"} for event in result.interface_events.events))
        self.assertTrue(
            all(
                (event.event_classification or "") in {
                    "transmitted_shock",
                    "reflected_shock",
                    "reflected_release",
                    "contact_continuation",
                    "unresolved_ambiguous_split",
                }
                for event in result.interface_events.events
            )
        )
        self.assertIn("classification", result.interface_events.available_metrics)
        lead = result.interface_events.events[0]
        self.assertIsNotNone(lead.branch_id)
        self.assertIsNotNone(lead.event_classification)
        if lead.transfer_fraction is not None:
            self.assertGreaterEqual(float(lead.transfer_fraction), 0.0)
        else:
            self.assertTrue(
                lead.ambiguous
                or lead.support_class != "tracked"
                or any("Transfer fractions were suppressed" in note for note in lead.notes)
            )

    def test_preheat_summary_on_layered_run_is_target_anchored_and_budgeted(self) -> None:
        path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-layered"),
            requested_time_plot_modules=frozenset(),
        )
        assert result.preheat is not None
        self.assertTrue(result.preheat.supported)
        self.assertEqual("branch-anchored-target-preheat", result.preheat.method)
        self.assertEqual("auto", result.preheat.target_selection_mode)
        self.assertEqual(1, result.preheat.target_region_id)
        self.assertEqual(1, result.preheat.auto_target_region_id)
        self.assertEqual(3, result.preheat.incident_region_id)
        self.assertIn("Region 3", str(result.preheat.incident_region_label))
        self.assertIn("Region 1", str(result.preheat.target_label))
        self.assertIsNotNone(result.preheat.target_entry_time_s)
        self.assertEqual("tracked", result.preheat.primary_branch_support_class)
        self.assertGreaterEqual(len(result.preheat.state_metrics), 3)
        budget_keys = {row.key for row in result.preheat.budget_rows}
        self.assertIn("radiation_net_heating", budget_keys)
        self.assertIn("observed_internal_delta", budget_keys)
        self.assertIn("residual", budget_keys)
        self.assertGreaterEqual(len(result.preheat.time_plots), 2)
        self.assertIn("radiation", {marker.key for marker in result.preheat.onset_markers if marker.first_time_s is not None})

        refreshed = refresh_analysis_result_for_snapshot(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-layered-refresh"),
            base_result=result,
        )
        self.assertIs(result.preheat, refreshed.preheat)

        manual = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(preheat_target_region_id=2),
            context_key=("wave6", "preheat-layered-manual"),
            requested_time_plot_modules=frozenset(),
        )
        assert manual.preheat is not None
        self.assertEqual("user_selected", manual.preheat.target_selection_mode)
        self.assertEqual(2, manual.preheat.target_region_id)
        self.assertEqual(1, manual.preheat.auto_target_region_id)
        self.assertIn("Region 2", str(manual.preheat.target_label))
        self.assertNotEqual(manual.preheat.target_region_id, manual.preheat.auto_target_region_id)

    def test_preheat_tab_surfaces_summary_and_region_override_controls(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=88)
        base = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-ui-base"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic = _synthetic_wavefront_result(base)
        synthetic = replace(synthetic, preheat=_synthetic_preheat_summary(base))

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.preheat_tab))
            process_events(20)
            self.assertLess(workspace.wavefront_display_combo.findData("preheat"), 0)
            self.assertGreater(workspace.preheat_summary_table.rowCount(), 0)
            self.assertGreater(workspace.preheat_budget_table.rowCount(), 0)
            self.assertGreater(workspace.preheat_onset_table.rowCount(), 0)
            self.assertGreaterEqual(workspace.preheat_plot_panel.time_combo.count(), 1)
            self.assertGreaterEqual(workspace.preheat_plot_panel.profile_combo.count(), 1)
            self.assertFalse(workspace.preheat_snapshot_spin.isEnabled())
            self.assertIn("preheat", workspace.preheat_summary_label.text().lower() + workspace.preheat_overview_label.text().lower())
            self.assertEqual(1, workspace.preheat_target_combo.currentData())
            self.assertIn("manual", workspace.preheat_summary_table.item(0, 3).text().lower())
            self.assertIn("thresholds", workspace.preheat_notes.toPlainText().lower())
        finally:
            workspace.close()

    def test_preheat_time_modes_keep_snapshot_controls_and_profiles_in_sync(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=12)
        base = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-ui-sync-base"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic = replace(_synthetic_wavefront_result(base), preheat=_synthetic_preheat_summary(base))

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.preheat_tab))
            process_events(20)

            self.assertEqual("shock_relative", workspace.preheat_time_mode_combo.currentData())
            self.assertTrue(workspace.preheat_offset_combo.isEnabled())
            self.assertFalse(workspace.preheat_snapshot_slider.isEnabled())
            self.assertIn("pre-entry", workspace.preheat_snapshot_label.text())
            self.assertGreaterEqual(workspace.preheat_plot_panel.profile_combo.count(), 2)

            workspace.preheat_offset_combo.setCurrentIndex(workspace.preheat_offset_combo.findData(2))
            process_events(20)
            self.assertEqual("2", workspace.preheat_snapshot_table.item(0, 1).text())

            workspace.preheat_time_mode_combo.setCurrentIndex(workspace.preheat_time_mode_combo.findData("manual"))
            process_events(20)
            self.assertTrue(workspace.preheat_snapshot_slider.isEnabled())
            self.assertTrue(workspace.preheat_snapshot_spin.isEnabled())
            self.assertTrue(workspace.preheat_time_spin.isEnabled())

            workspace.preheat_snapshot_slider.setValue(6)
            process_events(20)
            self.assertEqual(6, workspace.preheat_snapshot_spin.value())
            self.assertEqual("6", workspace.preheat_snapshot_table.item(0, 1).text())
            self.assertIn("mode=manual", workspace.preheat_snapshot_label.text())

            workspace.preheat_time_spin.setValue(float(workspace.preheat_time_spin.value()))
            process_events(20)
            self.assertEqual(6, workspace.preheat_snapshot_spin.value())
        finally:
            workspace.close()

    def test_module_specific_controls_live_in_their_own_tabs(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            self.assertTrue(workspace.xrd_tab.isAncestorOf(workspace.xrd_energy_spin))
            self.assertTrue(workspace.xrd_tab.isAncestorOf(workspace.xrd_display_combo))
            self.assertTrue(workspace.plasmon_tab.isAncestorOf(workspace.plasmon_energy_spin))
            self.assertTrue(workspace.plasmon_tab.isAncestorOf(workspace.plasmon_gamma_spin))
            self.assertTrue(workspace.spectroscopy_tab.isAncestorOf(workspace.spectroscopy_wavelength_spin))
            self.assertTrue(workspace.spectroscopy_tab.isAncestorOf(workspace.spectroscopy_shift_unit_combo))
        finally:
            workspace.close()

    def test_derived_plot_panel_defers_redraw_while_popup_navigation_is_active(self) -> None:
        panel = DerivedPlotPanel()
        try:
            bundles = (_bundle("a", "A", (1.0, 2.0, 3.0)), _bundle("b", "B", (2.0, 3.0, 4.0)))
            panel.set_bundles(bundles, bundles, view_scope="test")

            with mock.patch.object(panel, "_combo_popup_open", side_effect=lambda combo: combo is panel.time_combo):
                with mock.patch.object(panel, "_render_time_bundle_for_index") as render_time:
                    panel._on_time_combo_index_changed(1)
                    self.assertEqual(1, panel._pending_time_render_index)
                    render_time.assert_not_called()

                    panel._on_time_combo_activated(1)
                    process_events(20)
                    render_time.assert_not_called()

            with mock.patch.object(panel, "_combo_popup_open", return_value=False):
                with mock.patch.object(panel, "_render_time_bundle_for_index") as render_time:
                    panel._flush_pending_time_render()
                    render_time.assert_called_once_with(1)
        finally:
            panel.close()

    def test_derived_plot_panel_programmatic_index_change_still_renders(self) -> None:
        panel = DerivedPlotPanel()
        try:
            bundles = (_bundle("a", "A", (1.0, 2.0, 3.0)), _bundle("b", "B", (2.0, 3.0, 4.0)))
            panel.set_bundles(bundles, bundles, view_scope="test")

            with mock.patch.object(panel, "_combo_popup_open", return_value=False):
                with mock.patch.object(panel, "_render_time_bundle_for_index") as render_time:
                    panel._on_time_combo_index_changed(1)
                    render_time.assert_called_once_with(1)
        finally:
            panel.close()

    def test_derived_plot_panel_preserves_selection_when_bundle_choices_are_unchanged(self) -> None:
        panel = DerivedPlotPanel()
        try:
            bundles = (_bundle("a", "A", (1.0, 2.0, 3.0)), _bundle("b", "B", (2.0, 3.0, 4.0)))
            panel.set_bundles(bundles, bundles, view_scope="test")
            panel.time_combo.setCurrentIndex(1)
            panel.profile_combo.setCurrentIndex(1)

            panel.set_bundles(bundles, bundles, view_scope="test")

            self.assertEqual(1, panel.time_combo.currentIndex())
            self.assertEqual(1, panel.profile_combo.currentIndex())
            self.assertEqual("b", panel.time_combo.currentData())
            self.assertEqual("b", panel.profile_combo.currentData())
        finally:
            panel.close()

    def test_combo_boxes_use_explicit_popup_views_for_stable_selection(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            for combo in (
                workspace.preheat_target_combo,
                workspace.preheat_time_mode_combo,
                workspace.preheat_offset_combo,
                workspace.wavefront_display_combo,
                workspace.wavefront_scope_combo,
                workspace.wavefront_direction_combo,
            ):
                self.assertIsInstance(combo.view(), QtWidgets.QListView)
                self.assertEqual(QtCore.Qt.StrongFocus, combo.focusPolicy())
                self.assertEqual(QtCore.Qt.StrongFocus, combo.view().focusPolicy())
        finally:
            workspace.close()

    def test_preheat_notes_panel_scroll_position_persists(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=12)
        base = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("wave6", "preheat-ui-notes"),
            requested_time_plot_modules=frozenset(),
        )
        synthetic_preheat = replace(
            _synthetic_preheat_summary(base),
            notes=tuple(f"Scrollable synthetic note {index}" for index in range(80)),
        )
        synthetic = replace(_synthetic_wavefront_result(base), preheat=synthetic_preheat)

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.show()
            workspace.set_context(context)
            workspace.set_result(synthetic)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.preheat_tab))
            process_events(20)
            scroll_bar = workspace.preheat_notes.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), 0)
            scroll_bar.setSliderPosition(scroll_bar.maximum())
            process_events(20)
            self.assertGreater(scroll_bar.value(), 0)
        finally:
            workspace.close()

    def test_registered_module_contracts_include_prepared_future_modules(self) -> None:
        names = {contract.name for contract in registered_module_contracts()}
        self.assertTrue({"xrd", "plasmon", "transmission", "spectroscopy", "wave_tracking", "interface_events", "preheat"}.issubset(names))
        dataset = load_run_data(HDF5_ROOT / "Cu_0166_stabilized.h5")
        contracts = {contract.name: contract for contract in registered_module_contracts()}
        self.assertTrue(contracts["wave_tracking"].capabilities_met(dataset))
        self.assertTrue(contracts["interface_events"].capabilities_met(dataset))
        self.assertTrue(contracts["preheat"].capabilities_met(dataset))
        self.assertFalse(contracts["interface_events"].supports_lazy_time_plots)


if __name__ == "__main__":
    unittest.main()
