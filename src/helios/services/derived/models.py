"""Structured result objects for HELIOS Derived / Analysis services.

The Phase 4.1 data model keeps the GUI thin by carrying:

- explicit geometry / selection metadata
- weighting semantics
- compact scalar summaries
- reusable time-trace and snapshot-profile plot bundles
- warning severities

UI widgets consume these objects directly; scientific calculations stay in the
derived-service backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DerivedWarning:
    source: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True, slots=True)
class DerivedPlotBundle:
    """A ready-to-render bundle of one or more curves that share an x-axis."""

    key: str
    title: str
    x_label: str
    y_label: str
    x_values: np.ndarray
    y_series: tuple[np.ndarray, ...]
    curve_names: tuple[str, ...] = ()
    boundary_positions: tuple[float, ...] = ()
    value_scale_mode: str = "linear"


@dataclass(frozen=True, slots=True)
class AnalysisGeometryMetadata:
    """Geometry/projection settings used by a derived analysis run."""

    observation_side: str
    observation_boundary: str
    line_of_sight_angle_deg: float
    line_of_sight_cosine: float
    profile_coordinate_mode: str
    path_length_mode: str
    propagation_direction: str
    impact_parameter_cm: float = 0.0


@dataclass(frozen=True, slots=True)
class AnalysisSelectionMetadata:
    """Subset/filter semantics used to construct the active analysis zones."""

    reuse_viewer_subset: bool
    viewer_region_ids: tuple[int, ...]
    viewer_material_ids: tuple[int, ...]
    derived_region_ids: tuple[int, ...]
    derived_material_ids: tuple[int, ...]
    exclude_entry_region: bool
    exclude_low_density: bool
    min_density_g_cm3: float | None
    exclude_opposite_velocity: bool
    zone_index_lower: int | None
    zone_index_upper: int | None
    weighting_mode: str
    selected_zone_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivedFieldConsistency:
    total_pressure_matches_components: bool | None = None
    radiation_net_heating_matches_components: bool | None = None
    kinetic_energy_matches_velocity: bool | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivedFieldCapabilities:
    available_fields: tuple[str, ...] = ()
    optional_available_fields: tuple[str, ...] = ()
    missing_optional_fields: tuple[str, ...] = ()
    dynamic_radius_available: bool = False
    run_status_available: bool = False
    visar_support_available: bool = False
    pressure_components_available: bool = False
    total_pressure_available: bool = False
    radiation_components_available: bool = False
    radiation_net_heating_available: bool = False
    kinetic_energy_available: bool = False
    consistency: DerivedFieldConsistency = field(default_factory=DerivedFieldConsistency)


@dataclass(frozen=True, slots=True)
class WavePhysicsCapabilities:
    shock_evidence_supported: bool = False
    release_evidence_supported: bool = False
    contact_evidence_supported: bool = False
    interface_event_supported: bool = False
    preheat_supported: bool = False
    pressure_support_level: str = "unavailable"
    viscosity_support_level: str = "unavailable"
    radiation_support_level: str = "unavailable"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveEvidenceFormulaHook:
    family: str
    expression_label: str
    configurable_terms: tuple[str, ...]
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class DerivedRunData:
    path: Path
    summary: dict[str, Any]
    metadata: dict[str, Any]
    regions: dict[str, Any]
    materials: dict[str, Any]
    time_s: np.ndarray
    # Legacy x-named fields remain for API stability, but they now always use
    # explicit center/edge semantics from the runtime layer.
    static_x_cm: np.ndarray
    static_x_edge_cm: np.ndarray
    zone_width_cm: np.ndarray
    density_g_cm3: np.ndarray
    velocity_cm_s: np.ndarray
    temperature_e_ev: np.ndarray
    temperature_i_ev: np.ndarray
    temperature_radiation_ev: np.ndarray | None
    electron_density_cm3: np.ndarray
    mean_charge: np.ndarray
    pressure_i_j_cm3: np.ndarray | None = None
    pressure_e_j_cm3: np.ndarray | None = None
    pressure_radiation_j_cm3: np.ndarray | None = None
    pressure_total_j_cm3: np.ndarray | None = None
    artificial_viscosity_j_cm3: np.ndarray | None = None
    ion_energy_j_g: np.ndarray | None = None
    electron_energy_j_g: np.ndarray | None = None
    radiation_energy_j_g: np.ndarray | None = None
    kinetic_energy_j_g: np.ndarray | None = None
    ion_heat_capacity_j_g_ev: np.ndarray | None = None
    electron_heat_capacity_j_g_ev: np.ndarray | None = None
    radiation_heating_j_g_s: np.ndarray | None = None
    radiation_cooling_j_g_s: np.ndarray | None = None
    radiation_sink_j_g_s: np.ndarray | None = None
    radiation_net_heating_j_g_s: np.ndarray | None = None
    laser_source_j_g_s: np.ndarray | None = None
    laser_deposition_j_g_s: np.ndarray | None = None
    radius_cm: np.ndarray | None = None
    radius_edge_cm: np.ndarray | None = None
    zone_region_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    zone_material_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    zone_atomic_weight: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    zone_initial_density_g_cm3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    zone_initial_temperature_ev: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    laser_entry: dict[str, Any] | None = None
    run_status: dict[str, Any] | None = None
    visar_support_metadata: dict[str, Any] | None = None
    field_capabilities: DerivedFieldCapabilities = field(default_factory=DerivedFieldCapabilities)
    wave_physics_capabilities: WavePhysicsCapabilities = field(default_factory=WavePhysicsCapabilities)


@dataclass(frozen=True, slots=True)
class ShockInterfaceCrossing:
    interface_label: str
    boundary_zone: int
    crossing_snapshot: int | None
    crossing_time_s: float | None
    crossing_position_cm: float | None


@dataclass(frozen=True, slots=True)
class ShockTrackingResult:
    method: str
    coordinate_label: str
    time_s: np.ndarray
    position_cm: np.ndarray
    zone_index: np.ndarray
    velocity_cm_s: np.ndarray
    speed_magnitude_cm_s: np.ndarray
    detector_score: np.ndarray
    smoothed_position_cm: np.ndarray
    smoothed_zone_index: np.ndarray
    activation_snapshot_index: int | None
    propagation_direction: str
    breakout_time_s: float | None
    interface_crossings: tuple[ShockInterfaceCrossing, ...]
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveFrontFitSeed:
    model_name: str
    front_position_cm: float | None
    effective_width_cm: float | None
    fit_quality: float | None = None
    confidence: float | None = None
    fitted_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveEvidenceMap:
    family: str
    coordinate_label: str
    time_s: np.ndarray
    interface_position_cm: np.ndarray
    score: np.ndarray
    formula_hook: WaveEvidenceFormulaHook
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveLocalStateSummary:
    density_g_cm3: float | None = None
    velocity_cm_s: float | None = None
    pressure_total_j_cm3: float | None = None
    temperature_e_ev: float | None = None
    temperature_i_ev: float | None = None
    temperature_radiation_ev: float | None = None
    mean_charge: float | None = None
    material_index: int | None = None
    region_id: int | None = None


@dataclass(frozen=True, slots=True)
class WaveFrontCandidate:
    snapshot_index: int
    family: str
    candidate_type: str
    coordinate_label: str
    interface_index: float
    position_cm: float | None
    width_cm: float | None
    score: float
    propagation_direction: str | None = None
    direction_sign: float | None = None
    fit_quality: float | None = None
    confidence: float | None = None
    ambiguous: bool = False
    branch_hint: str | None = None
    upstream_state: WaveLocalStateSummary | None = None
    downstream_state: WaveLocalStateSummary | None = None
    fit_seed: WaveFrontFitSeed | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveBranchSummary:
    branch_id: str
    family: str
    branch_type: str
    snapshot_indices: np.ndarray
    interface_index: np.ndarray | None
    position_cm: np.ndarray
    velocity_cm_s: np.ndarray
    score: np.ndarray
    width_cm: np.ndarray | None = None
    confidence: float | None = None
    ambiguous: bool = False
    propagation_direction: str | None = None
    breakout_time_s: float | None = None
    support_class: str = "tracked"
    sample_count: int = 0
    duration_s: float | None = None
    integrated_score: float | None = None
    position_span_cm: float | None = None
    significance: float | None = None
    continuity_fraction: float | None = None
    upstream_state: WaveLocalStateSummary | None = None
    downstream_state: WaveLocalStateSummary | None = None
    primary: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WaveTrackingResult:
    method: str
    coordinate_label: str
    supported_formula_hooks: tuple[WaveEvidenceFormulaHook, ...]
    evidence_maps: tuple[WaveEvidenceMap, ...]
    candidates: tuple[WaveFrontCandidate, ...]
    branches: tuple[WaveBranchSummary, ...]
    primary_branch_id: str | None
    candidate_count: int = 0
    tracked_branch_count: int = 0
    short_branch_count: int = 0
    provisional_branch_count: int = 0
    suppressed_branch_count: int = 0
    compatibility_source: str | None = None
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class InterfaceEventRecord:
    event_kind: str
    interface_label: str
    boundary_zone: int
    snapshot_index: int | None
    time_s: float | None
    position_cm: float | None
    branch_id: str | None = None
    event_classification: str | None = None
    support_class: str = "tracked"
    significance: float | None = None
    confidence: float | None = None
    ambiguous: bool = False
    incident_branch_type: str | None = None
    incident_arrival_time_s: float | None = None
    transmitted_branch_id: str | None = None
    transmitted_branch_type: str | None = None
    transmitted_time_s: float | None = None
    reflected_branch_id: str | None = None
    reflected_branch_type: str | None = None
    reflected_time_s: float | None = None
    incident_peak_pressure_j_cm3: float | None = None
    transmitted_peak_pressure_j_cm3: float | None = None
    reflected_peak_pressure_j_cm3: float | None = None
    incident_compression_ratio: float | None = None
    transmitted_compression_ratio: float | None = None
    reflected_compression_ratio: float | None = None
    incident_speed_cm_s: float | None = None
    transmitted_speed_cm_s: float | None = None
    reflected_speed_cm_s: float | None = None
    pressure_impulse_upstream_j_s_cm3: float | None = None
    pressure_impulse_downstream_j_s_cm3: float | None = None
    incident_energy_j_cm2: float | None = None
    transmitted_energy_j_cm2: float | None = None
    reflected_energy_j_cm2: float | None = None
    transfer_fraction: float | None = None
    reflection_fraction: float | None = None
    dominant_transfer_channel: str | None = None
    channel_fraction_internal: float | None = None
    channel_fraction_kinetic: float | None = None
    channel_fraction_pressure_work: float | None = None
    impedance_preview_supported: bool | None = None
    impedance_upstream: float | None = None
    impedance_downstream: float | None = None
    impedance_reflection_preview: float | None = None
    impedance_transmission_preview: float | None = None
    upstream_state: WaveLocalStateSummary | None = None
    downstream_state: WaveLocalStateSummary | None = None
    legal_behavior: bool | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InterfaceEventsResult:
    available: bool
    supported: bool
    events: tuple[InterfaceEventRecord, ...]
    tracked_event_count: int = 0
    weak_event_count: int = 0
    suppressed_event_count: int = 0
    classification_counts: tuple[tuple[str, int], ...] = ()
    available_metrics: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatThresholds:
    max_density_ratio: float
    max_relative_pressure: float
    min_delta_temperature_e_ev: float
    min_delta_mean_charge: float
    min_delta_electron_energy_j_g: float
    min_radiation_net_heating_j_g_s: float
    min_laser_deposition_j_g_s: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatStateMetric:
    key: str
    label: str
    unit: str
    representative_value: float | None = None
    max_value: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatBudgetRow:
    key: str
    label: str
    unit: str
    integrated_value: float | None = None
    fraction_of_observed: float | None = None
    available: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatOnsetMarker:
    key: str
    label: str
    threshold_value: float | None
    first_time_s: float | None = None
    observed_value: float | None = None
    unit: str = ""
    available: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatProfileField:
    key: str
    label: str
    unit: str
    values: np.ndarray
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreheatSummary:
    available: bool
    supported: bool
    method: str
    candidate_metric_names: tuple[str, ...]
    scalar_summaries: dict[str, float | None]
    target_selection_mode: str = "auto"
    target_region_id: int | None = None
    auto_target_region_id: int | None = None
    incident_region_id: int | None = None
    deepest_reached_region_id: int | None = None
    target_material_index: int | None = None
    target_label: str | None = None
    auto_target_label: str | None = None
    incident_region_label: str | None = None
    deepest_reached_label: str | None = None
    primary_branch_id: str | None = None
    primary_branch_support_class: str | None = None
    primary_branch_significance: float | None = None
    target_entry_interface_label: str | None = None
    target_entry_boundary_zone: int | None = None
    target_entry_time_s: float | None = None
    preheat_window_end_time_s: float | None = None
    target_zone_count: int = 0
    available_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    thresholds: PreheatThresholds | None = None
    state_metrics: tuple[PreheatStateMetric, ...] = ()
    budget_rows: tuple[PreheatBudgetRow, ...] = ()
    onset_markers: tuple[PreheatOnsetMarker, ...] = ()
    time_plots: tuple[DerivedPlotBundle, ...] = ()
    profile_plots: tuple[DerivedPlotBundle, ...] = ()
    snapshot_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    latest_pre_entry_snapshot_index: int | None = None
    peak_snapshot_index: int | None = None
    target_zone_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    target_static_x_cm: np.ndarray | None = None
    target_dynamic_coordinate_cm: np.ndarray | None = None
    profile_fields: tuple[PreheatProfileField, ...] = ()
    snapshot_scalar_series: dict[str, np.ndarray] = field(default_factory=dict)
    affected_depth_cm: float | None = None
    affected_thickness_fraction: float | None = None
    affected_areal_mass_fraction: float | None = None
    severity_label: str | None = None
    preheat_penalty_ratio: float | None = None
    dominant_source: str | None = None
    notes: tuple[str, ...] = ()
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class XrdLayerEstimate:
    region_id: int
    zone_count: int
    initial_density_g_cm3: float
    compressed_density_g_cm3: float
    compression_ratio: float
    d_over_d0: float
    q0_inv_angstrom: float
    q_compressed_inv_angstrom: float
    initial_bragg_angle_deg: float
    shifted_bragg_angle_deg: float | None
    bragg_shift_deg: float | None
    compressed_thickness_cm: float


@dataclass(frozen=True, slots=True)
class XrdResult:
    snapshot_index: int
    photon_energy_kev: float
    wavelength_angstrom: float
    initial_bragg_angle_deg: float
    weighting_mode: str
    geometry_summary: str
    profile_coordinate_label: str
    profile_coordinate_values: np.ndarray
    time_plots: tuple[DerivedPlotBundle, ...]
    profile_plots: tuple[DerivedPlotBundle, ...]
    layers: tuple[XrdLayerEstimate, ...]
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class PlasmonResult:
    snapshot_index: int
    weighting_mode: str
    geometry_summary: str
    photon_energy_kev: float
    scattering_angle_deg: float
    adiabatic_index: float
    electron_density_cm3: float
    electron_temperature_ev: float
    ion_temperature_ev: float
    mean_charge: float
    ion_mass_mu: float
    debye_length_cm: float
    plasma_frequency_rad_s: float
    plasma_frequency_ev: float
    electron_collision_rate_s: float
    coulomb_logarithm: float
    ion_sound_speed_cm_s: float
    probe_wavelength_angstrom: float
    scattering_wavevector_cm_inv: float
    scattering_wavevector_m_inv: float = float("nan")
    k_lambda_debye: float = float("nan")
    collectivity_parameter: float = float("nan")
    regime_label: str = "non-collective"
    fermi_energy_ev: float = float("nan")
    theta_degeneracy: float = float("nan")
    wigner_seitz_rs: float = float("nan")
    model_name: str = "quicklook"
    requested_model_name: str = "quicklook"
    execution_mode: str = "quicklook"
    integration_mode: str = "effective_state"
    collision_model: str = "nrl_constant"
    collision_scale: float = 1.0
    manual_collision_rate_s: float = 0.0
    lfc_model: str = "none"
    normalization: str = "peak"
    observable_mode: str = "dielectric"
    observable_summary: str = ""
    observable_decomposition_mode: str = ""
    observable_peak_extraction_mode: str = ""
    observable_elastic_exclusion_ev: float = 0.0
    observable_free_fraction: float = float("nan")
    observable_bound_fraction: float = float("nan")
    observable_elastic_fraction: float = float("nan")
    observable_comparison_mode: str = ""
    observable_subtraction_mode: str = ""
    observable_normalization_mode: str = ""
    observable_peak_discrete_energy_ev: float = float("nan")
    observable_peak_fit_energy_ev: float = float("nan")
    observable_peak_fit_status: str = ""
    observable_peak_edge_dominated: bool = False
    observable_elastic_form_factor_total: float = float("nan")
    observable_elastic_form_factor_core: float = float("nan")
    observable_elastic_screening_form_factor: float = float("nan")
    observable_ion_structure_factor: float = float("nan")
    observable_bound_core_mode: str = ""
    observable_bound_shell_summary: str = ""
    spectrum_window_ev: float = 80.0
    spectrum_points: int = 1201
    instrument_fwhm_ev: float = 0.0
    spectral_imag_shift_ev: float = 0.0
    peak_fit_method: str = "discrete"
    peak_energy_ev: float = float("nan")
    peak_fwhm_ev: float = float("nan")
    static_lfc_value: float = float("nan")
    q_over_qf: float = float("nan")
    response_backend: str = "classical_maxwellian"
    backend_summary: str = ""
    stls_converged: bool = False
    stls_iteration_count: int = 0
    stls_convergence_residual: float = float("nan")
    stls_convergence_relative_residual: float = float("nan")
    stls_closure_name: str = ""
    stls_local_field_value: float = float("nan")
    stls_q_over_qf: float = float("nan")
    auto_model_summary: str = ""
    benchmark_preset: str = "none"
    requested_electron_policy: str = "raw_helios"
    electron_policy: str = "raw_helios"
    driven_response_model: str = "none"
    driven_response_summary: str = ""
    driven_response_ensemble_mode: str = ""
    electron_density_source: str = "raw HELIOS ne/zbar"
    material_policy_summary: str = ""
    resolved_materials: tuple[str, ...] = ()
    unresolved_materials: tuple[str, ...] = ()
    raw_kept_materials: tuple[str, ...] = ()
    collision_source: str = "nrl_constant"
    collision_summary: str = ""
    cluster_log_ne_tol: float = 0.02
    cluster_log_te_tol: float = 0.02
    cluster_z_tol: float = 0.1
    study_mode: str = "spectrum"
    coordinate_axis: str = "angle_deg"
    coordinate_value: float = 45.0
    scan_axis: str = "angle_deg"
    scan_start: float = 10.0
    scan_stop: float = 140.0
    scan_points: int = 61
    compare_models: bool = False
    comparison_models: tuple[str, ...] = ()
    compare_policies: bool = False
    policy_comparison_policies: tuple[str, ...] = ()
    zone_count_used: int = 0
    cluster_count_used: int = 0
    benchmark_status: str = "not_applicable"
    model_executed_fully: bool = True
    fallback_fraction: float = 0.0
    domain_failure_fraction: float = 0.0
    degenerate_zone_count: int = 0
    noncollective_zone_count: int = 0
    weak_coupling_zone_count: int = 0
    lfc_out_of_domain_zone_count: int = 0
    invalid_collision_zone_count: int = 0
    degenerate_cluster_count: int = 0
    noncollective_cluster_count: int = 0
    weak_coupling_cluster_count: int = 0
    lfc_out_of_domain_cluster_count: int = 0
    invalid_collision_cluster_count: int = 0
    advanced_model_available: bool = False
    total_runtime_s: float = 0.0
    spectrum_runtime_s: float = 0.0
    comparison_runtime_s: float = 0.0
    dispersion_runtime_s: float = 0.0
    time_series_runtime_s: float = 0.0
    spectrum_energy_ev: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    spectrum_intensity: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    spectrum_free_component: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    spectrum_bound_component: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    spectrum_elastic_component: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    dielectric_real: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    dielectric_imag: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    loss_function: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    time_plots: tuple[DerivedPlotBundle, ...] = ()
    profile_plots: tuple[DerivedPlotBundle, ...] = ()
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class TransmissionRegionBudget:
    region_id: int
    areal_density_g_cm2: float
    electron_column_cm2: float
    thomson_tau: float
    free_free_tau: float = 0.0
    xcom_tau: float = 0.0
    total_tau: float = 0.0
    xcom_path_fraction: float | None = None
    free_free_thomson_path_fraction: float | None = None
    thomson_fallback_path_fraction: float | None = None
    xcom_tau_fraction: float | None = None
    free_free_thomson_tau_fraction: float | None = None
    thomson_fallback_tau_fraction: float | None = None
    dominant_regime: str = "thomson"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransmissionRegimeSummary:
    regime: str
    zone_count: int
    path_fraction: float | None = None
    areal_density_fraction: float | None = None
    tau_fraction: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransmissionPartitionSummary:
    mode: str
    photon_energy_kev: float | None = None
    zone_count: int = 0
    backend_status: str | None = None
    approximate: bool = False
    cached: bool = False
    regime_summaries: tuple[TransmissionRegimeSummary, ...] = ()
    unresolved_materials: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransmissionColdMaterialBudget:
    label: str
    areal_density_g_cm2: float
    optical_depth: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TransmissionColdRefinement:
    backend_status: str
    applicability: str
    message: str
    backend_name: str | None = None
    backend_available: bool = False
    backend_fingerprint: str | None = None
    source: str = "baseline"
    photon_energies_kev: tuple[float, ...] = ()
    optical_depth: tuple[float, ...] = ()
    transmission: tuple[float, ...] = ()
    attenuation_mode: str | None = None
    resolved_materials: tuple[str, ...] = ()
    unresolved_materials: tuple[str, ...] = ()
    material_budgets: tuple[TransmissionColdMaterialBudget, ...] = ()
    cold_fraction: float | None = None
    path_weighted_temperature_e_ev: float | None = None
    path_weighted_mean_charge: float | None = None


@dataclass(frozen=True, slots=True)
class TransmissionResult:
    snapshot_index: int
    weighting_mode: str
    geometry_summary: str
    areal_density_g_cm2: float
    electron_column_cm2: float
    thomson_tau: float
    thomson_transmission: float
    time_plots: tuple[DerivedPlotBundle, ...]
    profile_plots: tuple[DerivedPlotBundle, ...]
    region_budgets: tuple[TransmissionRegionBudget, ...]
    model_type: str = "thomson"
    selected_mode: str = "thomson"
    photon_energy_kev: float | None = None
    selected_tau: float | None = None
    selected_transmission: float | None = None
    source: str = "baseline"
    status_message: str = "Thomson quick-look estimate."
    backend_status: str | None = None
    partition: TransmissionPartitionSummary | None = None
    cold_refinement: TransmissionColdRefinement | None = None
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class SpectroscopyResult:
    snapshot_index: int
    weighting_mode: str
    geometry_summary: str
    line_wavelength_nm: float
    line_of_sight_cosine: float
    bulk_velocity_cm_s: float
    los_velocity_cm_s: float
    doppler_shift_nm: float
    thermal_width_fraction: float
    thermal_width_nm: float
    ion_temperature_ev: float
    ion_mass_mu: float
    time_plots: tuple[DerivedPlotBundle, ...]
    profile_plots: tuple[DerivedPlotBundle, ...]
    warnings: tuple[DerivedWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivedAnalysisResult:
    context_key: tuple[object, ...]
    dataset_path: Path
    snapshot_index: int
    snapshot_time_s: float
    selected_zone_count: int
    geometry: AnalysisGeometryMetadata
    selection: AnalysisSelectionMetadata
    shock: ShockTrackingResult
    xrd: XrdResult
    plasmon: PlasmonResult
    transmission: TransmissionResult
    spectroscopy: SpectroscopyResult
    warnings: tuple[DerivedWarning, ...] = ()
    wave_tracking: WaveTrackingResult | None = None
    interface_events: InterfaceEventsResult | None = None
    preheat: PreheatSummary | None = None
