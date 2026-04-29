"""High-level Derived / Analysis orchestration for HELIOS Analyzer.

This module composes the first-wave experiment-facing services into one
structured analysis result. It keeps the UI out of the scientific logic: the
workspace/controller asks for a single `DerivedAnalysisResult`, and the
individual service modules remain independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Callable

import numpy as np

from helios.feature_flags import production_feature_visible
from helios.instrumentation import increment_counter, timed_block
from helios.runtime import RunContext
from helios.services.derived.common import aggregate_warnings
from helios.services.derived.module_contract import DerivedModuleContract
from helios.services.derived.models import (
    DerivedAnalysisResult,
    DerivedRunData,
    DerivedWarning,
    PlasmonResult,
    PreheatSummary,
    TransmissionResult,
)
from helios.services.derived.physical_sanity import (
    validate_shock_result,
    validate_spectroscopy_result,
    validate_xrd_result,
)
from helios.services.derived.plasmon_config import (
    PLASMON_AXIS_ANGLE_DEG,
    PLASMON_BENCHMARK_PRESET_NONE,
    PLASMON_COLLISION_MODEL_NRL_CONSTANT,
    PLASMON_EXECUTION_MODE_QUICKLOOK,
    PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE,
    PLASMON_LFC_MODEL_NONE,
    PLASMON_MODEL_QUICKLOOK,
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_NORMALIZATION_PEAK,
    PLASMON_STUDY_MODE_SPECTRUM,
)
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.preheat import evaluate_preheat
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry, build_analysis_mask
from helios.services.derived.shock_tracking import (
    build_interface_events_from_wave_tracking,
    track_shock_front,
    track_wave_branches,
)
from helios.services.derived.spectroscopy import evaluate_spectroscopy
from helios.services.derived.transmission import evaluate_transmission
from helios.services.derived.xrd import estimate_xrd

LOGGER = logging.getLogger(__name__)
TIME_PLOT_MODULES = frozenset({"xrd", "plasmon", "transmission", "spectroscopy"})


def _validate_module_result(result: object) -> None:
    if not hasattr(result, "time_plots") or not hasattr(result, "profile_plots"):
        raise TypeError("Derived module result does not provide the expected plot-bundle fields.")


@dataclass(frozen=True, slots=True)
class DerivedAnalysisParameters:
    """User-configurable quick-look analysis settings for the MVP."""

    xrd_photon_energy_kev: float = 8.0
    xrd_initial_bragg_angle_deg: float = 20.0
    plasmon_photon_energy_kev: float = 8.0
    plasmon_scattering_angle_deg: float = 45.0
    plasmon_adiabatic_index: float = 1.0
    plasmon_model: str = PLASMON_MODEL_QUICKLOOK
    plasmon_execution_mode: str = PLASMON_EXECUTION_MODE_QUICKLOOK
    plasmon_energy_window_ev: float = 80.0
    plasmon_energy_points: int = 1201
    plasmon_instrument_fwhm_ev: float = 0.0
    plasmon_collision_model: str = PLASMON_COLLISION_MODEL_NRL_CONSTANT
    plasmon_collision_scale: float = 1.0
    plasmon_manual_collision_rate_s: float = 0.0
    plasmon_lfc_model: str = PLASMON_LFC_MODEL_NONE
    plasmon_integration_mode: str = PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE
    plasmon_normalization: str = PLASMON_NORMALIZATION_PEAK
    plasmon_observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC
    plasmon_cluster_log_ne_tol: float = 0.02
    plasmon_cluster_log_te_tol: float = 0.02
    plasmon_cluster_z_tol: float = 0.1
    plasmon_electron_policy: str = "raw_helios"
    plasmon_driven_response_model: str = ""
    plasmon_benchmark_preset: str = PLASMON_BENCHMARK_PRESET_NONE
    plasmon_study_mode: str = PLASMON_STUDY_MODE_SPECTRUM
    plasmon_coordinate_axis: str = PLASMON_AXIS_ANGLE_DEG
    plasmon_coordinate_value: float = 45.0
    plasmon_scan_axis: str = PLASMON_AXIS_ANGLE_DEG
    plasmon_scan_start: float = 10.0
    plasmon_scan_stop: float = 140.0
    plasmon_scan_points: int = 61
    plasmon_compare_models: bool = False
    plasmon_compare_model_names: tuple[str, ...] | None = None
    plasmon_compare_policies: bool = False
    spectroscopy_line_wavelength_nm: float = 500.0
    transmission_mode: str = "thomson"
    transmission_photon_energy_kev: float = 8.0
    observation_side: str = "front"
    line_of_sight_angle_deg: float = 0.0
    line_of_sight_impact_parameter_cm: float = 0.0
    profile_coordinate_mode: str = "viewer"
    reuse_viewer_subset: bool = True
    derived_region_ids: tuple[int, ...] | None = None
    derived_material_ids: tuple[int, ...] | None = None
    exclude_entry_region: bool = False
    exclude_low_density: bool = False
    min_density_g_cm3: float = 0.0
    exclude_opposite_velocity: bool = False
    zone_index_lower: int | None = None
    zone_index_upper: int | None = None
    weighting_mode: str = "auto"
    preheat_target_region_id: int | None = None

    def key(self) -> tuple[object, ...]:
        """Stable cache/task key for the baseline parameter set.

        Transmission mode/energy stay out of the baseline derived cache key so
        the Transmission tab can request model-aware snapshot overlays without
        forcing unrelated modules onto an energy-dependent path.
        """

        return (
            float(self.xrd_photon_energy_kev),
            float(self.xrd_initial_bragg_angle_deg),
            float(self.plasmon_photon_energy_kev),
            float(self.plasmon_scattering_angle_deg),
            float(self.plasmon_adiabatic_index),
            str(self.plasmon_model),
            str(self.plasmon_execution_mode),
            float(self.plasmon_energy_window_ev),
            int(self.plasmon_energy_points),
            float(self.plasmon_instrument_fwhm_ev),
            str(self.plasmon_collision_model),
            float(self.plasmon_collision_scale),
            float(self.plasmon_manual_collision_rate_s),
            str(self.plasmon_lfc_model),
            str(self.plasmon_integration_mode),
            str(self.plasmon_normalization),
            str(self.plasmon_observable_mode),
            float(self.plasmon_cluster_log_ne_tol),
            float(self.plasmon_cluster_log_te_tol),
            float(self.plasmon_cluster_z_tol),
            str(self.plasmon_electron_policy),
            str(self.plasmon_driven_response_model),
            str(self.plasmon_benchmark_preset),
            str(self.plasmon_study_mode),
            str(self.plasmon_coordinate_axis),
            float(self.plasmon_coordinate_value),
            str(self.plasmon_scan_axis),
            float(self.plasmon_scan_start),
            float(self.plasmon_scan_stop),
            int(self.plasmon_scan_points),
            bool(self.plasmon_compare_models),
            None if self.plasmon_compare_model_names is None else tuple(str(value) for value in self.plasmon_compare_model_names),
            bool(self.plasmon_compare_policies),
            float(self.spectroscopy_line_wavelength_nm),
            str(self.observation_side),
            float(self.line_of_sight_angle_deg),
            float(self.line_of_sight_impact_parameter_cm),
            str(self.profile_coordinate_mode),
            bool(self.reuse_viewer_subset),
            None if self.derived_region_ids is None else tuple(int(value) for value in self.derived_region_ids),
            None if self.derived_material_ids is None else tuple(int(value) for value in self.derived_material_ids),
            bool(self.exclude_entry_region),
            bool(self.exclude_low_density),
            float(self.min_density_g_cm3),
            bool(self.exclude_opposite_velocity),
            self.zone_index_lower,
            self.zone_index_upper,
            str(self.weighting_mode),
            self.preheat_target_region_id,
        )

    def core_key(self) -> tuple[object, ...]:
        """Cache key for geometry/selection/science state excluding tab-local overlays."""

        return self.key()[:-1]


def _validate_xrd_module_result(result: object) -> None:
    _validate_module_result(result)
    validate_xrd_result(result)


def _validate_spectroscopy_module_result(result: object) -> None:
    _validate_module_result(result)
    validate_spectroscopy_result(result)


_MODULE_CONTRACTS: tuple[DerivedModuleContract[object], ...] = (
    DerivedModuleContract(name="xrd", compute=estimate_xrd, validate=_validate_xrd_module_result),
    DerivedModuleContract(name="plasmon", compute=evaluate_plasmon_regime, validate=_validate_module_result),
    DerivedModuleContract(name="transmission", compute=evaluate_transmission, validate=_validate_module_result),
    DerivedModuleContract(name="spectroscopy", compute=evaluate_spectroscopy, validate=_validate_spectroscopy_module_result),
)
_PREPARED_MODULE_CONTRACTS: tuple[DerivedModuleContract[object], ...] = (
    DerivedModuleContract(
        name="wave_tracking",
        compute=lambda **kwargs: None,
        validate=lambda result: None,
        required_capabilities=("density", "velocity"),
        supports_lazy_time_plots=True,
    ),
    DerivedModuleContract(
        name="interface_events",
        compute=lambda **kwargs: None,
        validate=lambda result: None,
        required_capabilities=("run_status",),
        supports_lazy_time_plots=False,
    ),
    DerivedModuleContract(
        name="preheat",
        compute=lambda **kwargs: None,
        validate=lambda result: None,
        required_capabilities=("temperature_e",),
        supports_lazy_time_plots=True,
    ),
)


def registered_module_contracts() -> tuple[DerivedModuleContract[object], ...]:
    """Return active plus prepared module contracts for future extensions."""

    return (*_MODULE_CONTRACTS, *_PREPARED_MODULE_CONTRACTS)


def normalize_time_plot_modules(requested: frozenset[str] | set[str] | tuple[str, ...] | list[str] | None) -> frozenset[str]:
    """Normalize the requested lazy time-plot module set."""

    if requested is None:
        requested_modules = TIME_PLOT_MODULES
    else:
        requested_modules = frozenset(str(value) for value in requested if str(value) in TIME_PLOT_MODULES)
    normalized = frozenset(value for value in requested_modules if production_feature_visible(value))
    return normalized


def analysis_result_time_plot_modules(result: DerivedAnalysisResult) -> frozenset[str]:
    """Return which modules currently carry time plots in a cached result."""

    loaded: set[str] = set()
    for contract in _MODULE_CONTRACTS:
        if not production_feature_visible(contract.name):
            continue
        module_result = getattr(result, contract.name)
        if contract.time_plots_loaded(module_result):
            loaded.add(contract.name)
    return frozenset(loaded)


def _hidden_plasmon_result(*, snapshot_index: int, parameters: DerivedAnalysisParameters) -> PlasmonResult:
    return PlasmonResult(
        snapshot_index=int(snapshot_index),
        weighting_mode=str(parameters.weighting_mode),
        geometry_summary="Plasmon/XRTS is disabled in the production backend.",
        photon_energy_kev=float(parameters.plasmon_photon_energy_kev),
        scattering_angle_deg=float(parameters.plasmon_scattering_angle_deg),
        adiabatic_index=float(parameters.plasmon_adiabatic_index),
        electron_density_cm3=float("nan"),
        electron_temperature_ev=float("nan"),
        ion_temperature_ev=float("nan"),
        mean_charge=float("nan"),
        ion_mass_mu=float("nan"),
        debye_length_cm=float("nan"),
        plasma_frequency_rad_s=float("nan"),
        plasma_frequency_ev=float("nan"),
        electron_collision_rate_s=float("nan"),
        coulomb_logarithm=float("nan"),
        ion_sound_speed_cm_s=float("nan"),
        probe_wavelength_angstrom=float("nan"),
        scattering_wavevector_cm_inv=float("nan"),
        regime_label="disabled",
        model_name="disabled",
        requested_model_name=str(parameters.plasmon_model),
        execution_mode="disabled",
        model_executed_fully=False,
        advanced_model_available=False,
        benchmark_status="disabled",
    )


def _hidden_transmission_result(*, snapshot_index: int, parameters: DerivedAnalysisParameters) -> TransmissionResult:
    return TransmissionResult(
        snapshot_index=int(snapshot_index),
        weighting_mode=str(parameters.weighting_mode),
        geometry_summary="Transmission is disabled in the production backend.",
        areal_density_g_cm2=float("nan"),
        electron_column_cm2=float("nan"),
        thomson_tau=float("nan"),
        thomson_transmission=float("nan"),
        time_plots=(),
        profile_plots=(),
        region_budgets=(),
        model_type="disabled",
        selected_mode=str(parameters.transmission_mode),
        photon_energy_kev=float(parameters.transmission_photon_energy_kev),
        source="hidden",
        status_message="Transmission is available only with HELIOS_DEV_MODE=1 or HELIOS_ENABLE_EXPERIMENTAL=1.",
    )


def _hidden_module_result(
    name: str,
    *,
    snapshot_index: int,
    parameters: DerivedAnalysisParameters,
) -> object:
    if name == "plasmon":
        return _hidden_plasmon_result(snapshot_index=snapshot_index, parameters=parameters)
    if name == "transmission":
        return _hidden_transmission_result(snapshot_index=snapshot_index, parameters=parameters)
    raise ValueError(f"No hidden production placeholder is defined for derived module {name!r}.")


def compute_analysis_result(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: DerivedAnalysisParameters,
    context_key: tuple[object, ...],
    requested_time_plot_modules: frozenset[str] | set[str] | tuple[str, ...] | list[str] | None = None,
    base_result: DerivedAnalysisResult | None = None,
    include_wavefront: bool = True,
    progress_check: Callable[[], None] | None = None,
) -> DerivedAnalysisResult:
    """Run the current first-wave derived analyses for a context snapshot."""

    analysis_cache = AnalysisStateCache()
    requested_modules = normalize_time_plot_modules(requested_time_plot_modules)
    with timed_block("derived.compute.full", logger=LOGGER):
        if progress_check is not None:
            progress_check()
        snapshot_index, geometry, zone_mask, selection, geometry_warnings, selection_warnings = _resolve_analysis_state(
            dataset,
            context,
            parameters=parameters,
            analysis_cache=analysis_cache,
        )
        if _can_reuse_shock(base_result, dataset=dataset, geometry=geometry, selection=selection):
            shock = base_result.shock
        else:
            with timed_block("derived.compute.shock", logger=LOGGER):
                shock = track_shock_front(
                    dataset,
                    context,
                    parameters=parameters,
                    geometry=geometry,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
            if np.count_nonzero(zone_mask) > 0:
                validate_shock_result(shock)
        xrd, plasmon, transmission, spectroscopy = _compute_module_results(
            dataset,
            context,
            parameters=parameters,
            snapshot_index=snapshot_index,
            geometry=geometry,
            requested_time_plot_modules=requested_modules,
            base_result=base_result,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        wave_tracking = None
        interface_events = None
        preheat = None
        reusable_wavefront = _can_reuse_wave_tracking(base_result, dataset=dataset, geometry=geometry, selection=selection)
        if include_wavefront:
            with timed_block("derived.compute.wavefront", logger=LOGGER):
                if reusable_wavefront:
                    increment_counter("derived.cache.wavefront.reuse")
                    wave_tracking = base_result.wave_tracking
                    interface_events = base_result.interface_events
                    if _can_reuse_preheat(base_result.preheat, target_region_id=parameters.preheat_target_region_id):
                        increment_counter("derived.cache.preheat.reuse")
                        preheat = base_result.preheat
                    else:
                        preheat = None
                else:
                    with timed_block("derived.compute.wave_tracking", logger=LOGGER):
                        wave_tracking = track_wave_branches(
                            dataset,
                            context,
                            parameters=parameters,
                            geometry=geometry,
                            analysis_cache=analysis_cache,
                            progress_check=progress_check,
                        )
                    with timed_block("derived.compute.interface_events", logger=LOGGER):
                        interface_events = build_interface_events_from_wave_tracking(dataset, wave_tracking)
                    with timed_block("derived.compute.preheat", logger=LOGGER):
                        preheat = evaluate_preheat(
                            dataset,
                            wave_tracking=wave_tracking,
                            interface_events=interface_events,
                            target_region_id=parameters.preheat_target_region_id,
                        )
                if preheat is None:
                    with timed_block("derived.compute.preheat", logger=LOGGER):
                        preheat = evaluate_preheat(
                            dataset,
                            wave_tracking=wave_tracking,
                            interface_events=interface_events,
                            target_region_id=parameters.preheat_target_region_id,
                        )
        elif reusable_wavefront:
            wave_tracking = base_result.wave_tracking
            interface_events = base_result.interface_events
            preheat = base_result.preheat
        warnings = aggregate_warnings(
            geometry_warnings,
            selection_warnings,
            shock.warnings,
            xrd.warnings,
            plasmon.warnings,
            transmission.warnings,
            spectroscopy.warnings,
            (() if preheat is None else preheat.warnings),
            (() if interface_events is None else interface_events.warnings),
            (() if wave_tracking is None else wave_tracking.warnings),
        )
        result = _assemble_analysis_result(
            dataset,
            snapshot_index=snapshot_index,
            context_key=context_key,
            geometry=geometry,
            selection=selection,
            zone_mask=zone_mask,
            shock=shock,
            xrd=xrd,
            plasmon=plasmon,
            transmission=transmission,
            spectroscopy=spectroscopy,
            wave_tracking=wave_tracking,
            interface_events=interface_events,
            preheat=preheat,
            warnings=warnings,
        )
    LOGGER.debug("Derived analysis cache stats: %s", analysis_cache.stats())
    return result


def refresh_analysis_result_for_snapshot(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: DerivedAnalysisParameters,
    context_key: tuple[object, ...],
    base_result: DerivedAnalysisResult,
    progress_check: Callable[[], None] | None = None,
) -> DerivedAnalysisResult:
    """Refresh only snapshot-local derived products while reusing full-run traces."""

    analysis_cache = AnalysisStateCache()
    with timed_block("derived.compute.snapshot_refresh", logger=LOGGER):
        if progress_check is not None:
            progress_check()
        snapshot_index, geometry, zone_mask, selection, geometry_warnings, selection_warnings = _resolve_analysis_state(
            dataset,
            context,
            parameters=parameters,
            analysis_cache=analysis_cache,
        )
        xrd, plasmon, transmission, spectroscopy = _compute_module_results(
            dataset,
            context,
            parameters=parameters,
            snapshot_index=snapshot_index,
            geometry=geometry,
            requested_time_plot_modules=frozenset(),
            base_result=base_result,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        warnings = aggregate_warnings(
            geometry_warnings,
            selection_warnings,
            base_result.shock.warnings,
            xrd.warnings,
            plasmon.warnings,
            transmission.warnings,
            spectroscopy.warnings,
            (() if base_result.preheat is None else base_result.preheat.warnings),
            (() if base_result.interface_events is None else base_result.interface_events.warnings),
            (() if base_result.wave_tracking is None else base_result.wave_tracking.warnings),
        )
        result = _assemble_analysis_result(
            dataset,
            snapshot_index=snapshot_index,
            context_key=context_key,
            geometry=geometry,
            selection=selection,
            zone_mask=zone_mask,
            shock=base_result.shock,
            xrd=xrd,
            plasmon=plasmon,
            transmission=transmission,
            spectroscopy=spectroscopy,
            wave_tracking=base_result.wave_tracking,
            interface_events=base_result.interface_events,
            preheat=base_result.preheat,
            warnings=warnings,
        )
    LOGGER.debug("Derived snapshot-refresh cache stats: %s", analysis_cache.stats())
    return result


def _compute_module_results(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: DerivedAnalysisParameters,
    snapshot_index: int,
    geometry,
    requested_time_plot_modules: frozenset[str],
    base_result: DerivedAnalysisResult | None,
    analysis_cache: AnalysisStateCache | None,
    progress_check: Callable[[], None] | None,
) -> tuple[object, object, object, object]:
    computed: dict[str, object] = {}
    for contract in _MODULE_CONTRACTS:
        if not production_feature_visible(contract.name):
            computed[contract.name] = _hidden_module_result(
                contract.name,
                snapshot_index=snapshot_index,
                parameters=parameters,
            )
            increment_counter(f"derived.compute.{contract.name}.hidden")
            continue
        existing = getattr(base_result, contract.name) if base_result is not None else None
        include_time_plots = contract.name in requested_time_plot_modules and not (existing is not None and contract.time_plots_loaded(existing))
        if existing is not None and not include_time_plots and int(getattr(base_result, "snapshot_index", snapshot_index)) == int(snapshot_index):
            computed[contract.name] = existing
            continue
        if progress_check is not None:
            progress_check()
        with timed_block(f"derived.compute.{contract.name}", logger=LOGGER):
            if contract.name == "xrd":
                module_result = contract.compute(
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    photon_energy_kev=parameters.xrd_photon_energy_kev,
                    initial_bragg_angle_deg=parameters.xrd_initial_bragg_angle_deg,
                    parameters=parameters,
                    geometry=geometry,
                    include_time_plots=include_time_plots,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
            elif contract.name == "plasmon":
                module_result = contract.compute(
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    photon_energy_kev=parameters.plasmon_photon_energy_kev,
                    scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
                    adiabatic_index=parameters.plasmon_adiabatic_index,
                    parameters=parameters,
                    geometry=geometry,
                    include_time_plots=include_time_plots,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
            elif contract.name == "transmission":
                module_result = contract.compute(
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    parameters=parameters,
                    geometry=geometry,
                    include_time_plots=include_time_plots,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
            else:
                module_result = contract.compute(
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    line_wavelength_nm=parameters.spectroscopy_line_wavelength_nm,
                    parameters=parameters,
                    geometry=geometry,
                    include_time_plots=include_time_plots,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
        contract.validate(module_result)
        if existing is not None:
            module_result = contract.merge_time_plots(existing, module_result)
        computed[contract.name] = module_result
    return (
        computed["xrd"],
        computed["plasmon"],
        computed["transmission"],
        computed["spectroscopy"],
    )


def _resolve_analysis_state(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: DerivedAnalysisParameters,
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[int, object, np.ndarray, object, tuple[DerivedWarning, ...], tuple[DerivedWarning, ...]]:
    snapshot_index = max(0, min(int(context.snapshot_index), max(0, dataset.time_s.size - 1)))
    geometry = build_analysis_geometry(
        dataset,
        context,
        observation_side=parameters.observation_side,
        line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
        line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
        profile_coordinate_mode=parameters.profile_coordinate_mode,
    )
    geometry_warnings: list[DerivedWarning] = []
    if geometry.path_length_mode in {"normal-incidence", "oblique-sec(theta)", "cylindrical-shell-unavailable-fallback-slab"} and (
        not math.isfinite(float(geometry.line_of_sight_cosine)) or abs(float(geometry.line_of_sight_cosine)) <= 1.0e-6
    ):
        geometry_warnings.append(
            DerivedWarning(
                "geometry",
                "Degenerate LOS geometry produced an ill-conditioned path-length scale.",
                severity="error",
            )
        )
    elif geometry.path_length_mode in {"oblique-sec(theta)", "cylindrical-shell-unavailable-fallback-slab"} and abs(float(geometry.line_of_sight_cosine)) < 0.1:
        geometry_warnings.append(
            DerivedWarning(
                "geometry",
                "LOS angle is close to grazing incidence; path-integrated quantities are highly geometry-sensitive.",
                severity="warning",
            )
        )
    zone_mask, selection, selection_warnings = build_analysis_mask(
        dataset,
        context,
        snapshot_index=snapshot_index,
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
    return snapshot_index, geometry, zone_mask, selection, tuple(geometry_warnings), tuple(selection_warnings)


def _assemble_analysis_result(
    dataset: DerivedRunData,
    *,
    snapshot_index: int,
    context_key: tuple[object, ...],
    geometry,
    selection,
    zone_mask: np.ndarray,
    shock,
    xrd,
    plasmon,
    transmission,
    spectroscopy,
    wave_tracking,
    interface_events,
    preheat,
    warnings: tuple[DerivedWarning, ...],
) -> DerivedAnalysisResult:
    return DerivedAnalysisResult(
        context_key=context_key,
        dataset_path=dataset.path,
        snapshot_index=snapshot_index,
        snapshot_time_s=float(dataset.time_s[snapshot_index]) if dataset.time_s.size else float("nan"),
        selected_zone_count=int(np.count_nonzero(zone_mask)),
        geometry=geometry,
        selection=selection,
        shock=shock,
        xrd=xrd,
        plasmon=plasmon,
        transmission=transmission,
        spectroscopy=spectroscopy,
        warnings=warnings,
        wave_tracking=wave_tracking,
        interface_events=interface_events,
        preheat=preheat,
    )


def _can_reuse_shock(
    base_result: DerivedAnalysisResult | None,
    *,
    dataset: DerivedRunData,
    geometry: object,
    selection: object,
) -> bool:
    if base_result is None:
        return False
    return (
        base_result.dataset_path == dataset.path
        and base_result.geometry == geometry
        and base_result.selection == selection
    )


def _can_reuse_wave_tracking(
    base_result: DerivedAnalysisResult | None,
    *,
    dataset: DerivedRunData,
    geometry: object,
    selection: object,
) -> bool:
    if base_result is None or base_result.wave_tracking is None or base_result.interface_events is None:
        return False
    return (
        base_result.dataset_path == dataset.path
        and base_result.geometry == geometry
        and base_result.selection == selection
    )


def _can_reuse_preheat(preheat: PreheatSummary | None, *, target_region_id: int | None) -> bool:
    if preheat is None:
        return False
    requested_region_id = None if target_region_id is None else int(target_region_id)
    existing_region_id = None
    if str(preheat.target_selection_mode or "auto") == "user_selected":
        existing_region_id = None if preheat.target_region_id is None else int(preheat.target_region_id)
    return existing_region_id == requested_region_id
