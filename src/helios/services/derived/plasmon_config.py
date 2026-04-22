"""Shared plasmon/XRTS configuration enums and defaults."""

from __future__ import annotations

from dataclasses import dataclass

PLASMON_MODEL_QUICKLOOK = "quicklook"
PLASMON_EXECUTION_MODE_QUICKLOOK = "quicklook"
PLASMON_EXECUTION_MODE_BENCHMARK = "benchmark"
PLASMON_EXECUTION_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Quicklook", PLASMON_EXECUTION_MODE_QUICKLOOK),
    ("Benchmark", PLASMON_EXECUTION_MODE_BENCHMARK),
)

PLASMON_MODEL_RPA = "rpa"
PLASMON_MODEL_MERMIN = "mermin"
PLASMON_MODEL_RPA_STATIC_LFC = "rpa_static_lfc"
PLASMON_MODEL_MERMIN_STATIC_LFC = "mermin_static_lfc"
PLASMON_MODEL_LINDHARD = "lindhard"
PLASMON_MODEL_LINDHARD_MERMIN = "lindhard_mermin"
PLASMON_MODEL_LINDHARD_STATIC_LFC = "lindhard_static_lfc"
PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC = "lindhard_mermin_static_lfc"
PLASMON_MODEL_FINITE_T_STLS = "finite_t_stls"
PLASMON_MODEL_QUANTUM_HYDRODYNAMIC = "quantum_hydrodynamic"
PLASMON_MODEL_AUTO_BEST = "auto_best"
PLASMON_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("Quick look", PLASMON_MODEL_QUICKLOOK),
    ("RPA", PLASMON_MODEL_RPA),
    ("Mermin", PLASMON_MODEL_MERMIN),
    ("RPA + static LFC", PLASMON_MODEL_RPA_STATIC_LFC),
    ("Mermin + static LFC", PLASMON_MODEL_MERMIN_STATIC_LFC),
    ("Finite-T Lindhard", PLASMON_MODEL_LINDHARD),
    ("Finite-T Lindhard + Mermin", PLASMON_MODEL_LINDHARD_MERMIN),
    ("Finite-T Lindhard + static LFC", PLASMON_MODEL_LINDHARD_STATIC_LFC),
    ("Finite-T Lindhard + Mermin + static LFC", PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC),
    ("Finite-T STLS (experimental)", PLASMON_MODEL_FINITE_T_STLS),
    ("Quantum hydrodynamic (experimental)", PLASMON_MODEL_QUANTUM_HYDRODYNAMIC),
    ("Auto best per state", PLASMON_MODEL_AUTO_BEST),
)
PLASMON_COMPARISON_MODEL_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (label, value)
    for label, value in PLASMON_MODEL_CHOICES
    if str(value) != PLASMON_MODEL_QUICKLOOK
)

PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE = "effective_state"
PLASMON_INTEGRATION_MODE_LOS_INTEGRATED = "los_integrated"
PLASMON_INTEGRATION_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Effective state", PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE),
    ("LOS integrated", PLASMON_INTEGRATION_MODE_LOS_INTEGRATED),
)

PLASMON_COLLISION_MODEL_NRL_CONSTANT = "nrl_constant"
PLASMON_COLLISION_MODEL_BENCHMARK_DENSE = "benchmark_dense"
PLASMON_COLLISION_MODEL_MANUAL_CONSTANT = "manual_constant"
PLASMON_COLLISION_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("NRL constant nu", PLASMON_COLLISION_MODEL_NRL_CONSTANT),
    ("Benchmark dense nu", PLASMON_COLLISION_MODEL_BENCHMARK_DENSE),
    ("Manual constant nu", PLASMON_COLLISION_MODEL_MANUAL_CONSTANT),
)

PLASMON_LFC_MODEL_NONE = "none"
PLASMON_LFC_MODEL_ESA_STATIC = "esa_static"
PLASMON_LFC_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("None", PLASMON_LFC_MODEL_NONE),
    ("ESA static", PLASMON_LFC_MODEL_ESA_STATIC),
)

PLASMON_NORMALIZATION_PEAK = "peak"
PLASMON_NORMALIZATION_AREA = "area"
PLASMON_NORMALIZATION_NONE = "none"
PLASMON_NORMALIZATION_CHOICES: tuple[tuple[str, str], ...] = (
    ("Peak", PLASMON_NORMALIZATION_PEAK),
    ("Area", PLASMON_NORMALIZATION_AREA),
    ("None", PLASMON_NORMALIZATION_NONE),
)

PLASMON_OBSERVABLE_MODE_DIELECTRIC = "dielectric"
PLASMON_OBSERVABLE_MODE_XRTS = "xrts_observable"
PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE = "xrts_article_native_al"
PLASMON_OBSERVABLE_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Dielectric-only spectrum", PLASMON_OBSERVABLE_MODE_DIELECTRIC),
    ("XRTS observable (experimental)", PLASMON_OBSERVABLE_MODE_XRTS),
    ("XRTS article-native Al (experimental)", PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE),
)

PLASMON_STUDY_MODE_SPECTRUM = "spectrum"
PLASMON_STUDY_MODE_DISPERSION = "dispersion"
PLASMON_STUDY_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Spectrum at fixed angle / k", PLASMON_STUDY_MODE_SPECTRUM),
    ("Dispersion scan", PLASMON_STUDY_MODE_DISPERSION),
)

PLASMON_AXIS_ANGLE_DEG = "angle_deg"
PLASMON_AXIS_K_ANGSTROM_INV = "k_angstrom_inv"
PLASMON_AXIS_CHOICES: tuple[tuple[str, str], ...] = (
    ("Angle", PLASMON_AXIS_ANGLE_DEG),
    ("k", PLASMON_AXIS_K_ANGSTROM_INV),
)

PLASMON_BENCHMARK_PRESET_NONE = "none"
PLASMON_BENCHMARK_PRESET_AL_AMBIENT_ARTICLE = "al_ambient_article"
PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE = "al_driven_article"
PLASMON_BENCHMARK_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    ("None", PLASMON_BENCHMARK_PRESET_NONE),
    ("Al article: ambient", PLASMON_BENCHMARK_PRESET_AL_AMBIENT_ARTICLE),
    ("Al article: driven", PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE),
)


@dataclass(frozen=True, slots=True)
class PlasmonPlotCapabilityOption:
    key: str
    label: str


@dataclass(frozen=True, slots=True)
class PlasmonUiCapabilities:
    primary_label: str
    secondary_label: str
    time_options: tuple[PlasmonPlotCapabilityOption, ...]
    profile_options: tuple[PlasmonPlotCapabilityOption, ...]
    preferred_time_key: str | None = None
    preferred_profile_key: str | None = None
    advanced_model_requested: bool = False
    compare_models_available: bool = False
    compare_models_reason: str = ""
    compare_policies_available: bool = False
    compare_policies_reason: str = ""


def plasmon_ui_capabilities(
    *,
    model: str,
    execution_mode: str,
    study_mode: str,
    compare_models: bool,
    compare_policies: bool,
) -> PlasmonUiCapabilities:
    """Return config-driven UI capabilities for the plasmon panel.

    This is intentionally independent of the last computed result. The
    workspace uses it as the source of truth for which studies/overlays should
    be visible now, while result payloads only determine which of those views
    are currently populated.
    """

    normalized_model = str(model or PLASMON_MODEL_QUICKLOOK)
    normalized_execution = str(execution_mode or PLASMON_EXECUTION_MODE_QUICKLOOK)
    normalized_study = str(study_mode or PLASMON_STUDY_MODE_SPECTRUM)
    selected_model_advanced = normalized_model != PLASMON_MODEL_QUICKLOOK
    compare_models_available = True
    policies_available = selected_model_advanced and normalized_model != PLASMON_MODEL_AUTO_BEST and normalized_execution == PLASMON_EXECUTION_MODE_BENCHMARK
    compare_policies_enabled = bool(compare_policies and policies_available)
    compare_models_enabled = bool(compare_models and compare_models_available)
    advanced = selected_model_advanced or compare_models_enabled or compare_policies_enabled

    diagnostics_time = (
        PlasmonPlotCapabilityOption("plasma_frequency", "Plasma frequency vs time"),
        PlasmonPlotCapabilityOption("k_lambda", "k*lambda_D vs time"),
        PlasmonPlotCapabilityOption("collision_rate", "Collision rate vs time"),
        PlasmonPlotCapabilityOption("electron_density", "Electron density vs time"),
        PlasmonPlotCapabilityOption("electron_temperature", "Electron temperature vs time"),
        PlasmonPlotCapabilityOption("ion_temperature", "Ion temperature vs time"),
        PlasmonPlotCapabilityOption("mean_charge", "Mean charge vs time"),
        PlasmonPlotCapabilityOption("debye_length", "Debye length vs time"),
    )
    diagnostics_profile = (
        PlasmonPlotCapabilityOption("local_k_lambda_profile", "Local k*lambda_D profile"),
        PlasmonPlotCapabilityOption("electron_temperature_profile", "Electron temperature profile"),
        PlasmonPlotCapabilityOption("ion_temperature_profile", "Ion temperature profile"),
        PlasmonPlotCapabilityOption("temperature_profile", "Electron and ion temperature profile"),
        PlasmonPlotCapabilityOption("electron_density_profile", "Electron density profile"),
        PlasmonPlotCapabilityOption("mean_charge_profile", "Mean charge profile"),
        PlasmonPlotCapabilityOption("angle_scan", "k*lambda_D scan vs scattering angle"),
    )

    if not advanced:
        return PlasmonUiCapabilities(
            primary_label="State diagnostics",
            secondary_label="Profiles / scan context",
            time_options=diagnostics_time,
            profile_options=diagnostics_profile,
            preferred_time_key="plasma_frequency",
            preferred_profile_key="local_k_lambda_profile",
        advanced_model_requested=False,
        compare_models_available=True,
        compare_models_reason="",
        compare_policies_available=False,
        compare_policies_reason="Benchmark electron-policy comparison is only available for non-Quicklook spectral models.",
    )

    compare_models_reason = ""
    if normalized_model == PLASMON_MODEL_AUTO_BEST:
        compare_policies_reason = "Policy comparison is disabled for Auto best because each policy can switch to a different backend family."
    elif normalized_execution != PLASMON_EXECUTION_MODE_BENCHMARK:
        compare_policies_reason = "Policy comparison is benchmark-only; switch spectral mode to Benchmark to enable it."
    else:
        compare_policies_reason = ""

    if normalized_study == PLASMON_STUDY_MODE_DISPERSION:
        time_options: list[PlasmonPlotCapabilityOption] = []
        if selected_model_advanced:
            time_options.append(PlasmonPlotCapabilityOption("dispersion_selected_model", "Peak shift vs scan axis"))
        if compare_models_enabled:
            time_options.append(PlasmonPlotCapabilityOption("dispersion_compare_models", "Peak-shift comparison across models"))
        if compare_policies_enabled:
            time_options.append(PlasmonPlotCapabilityOption("dispersion_compare_policies", "Peak-shift comparison across e- policies"))
        time_options.extend(diagnostics_time)

        profile_options: list[PlasmonPlotCapabilityOption] = []
        if selected_model_advanced:
            profile_options.append(PlasmonPlotCapabilityOption("dispersion_selected_width", "FWHM vs scan axis"))
        if compare_models_enabled:
            profile_options.append(PlasmonPlotCapabilityOption("dispersion_compare_width_models", "FWHM comparison across models"))
        if selected_model_advanced:
            profile_options.append(PlasmonPlotCapabilityOption("spectrum_observed", "Representative spectrum"))
        if compare_models_enabled:
            profile_options.append(PlasmonPlotCapabilityOption("spectrum_compare_models", "Representative spectra across models"))
        if compare_policies_enabled:
            profile_options.append(PlasmonPlotCapabilityOption("spectrum_compare_policies", "Representative spectra across e- policies"))
        profile_options.extend(diagnostics_profile)

        return PlasmonUiCapabilities(
            primary_label="Peak shift / comparison",
            secondary_label="FWHM / representative spectra",
            time_options=tuple(time_options),
            profile_options=tuple(profile_options),
            preferred_time_key=(
                "dispersion_compare_policies"
                if compare_policies_enabled
                else (
                    "dispersion_compare_models"
                    if compare_models_enabled
                    else ("dispersion_selected_model" if selected_model_advanced else "plasma_frequency")
                )
            ),
            preferred_profile_key=(
                "dispersion_compare_width_models"
                if compare_models_enabled
                else ("dispersion_selected_width" if selected_model_advanced else "local_k_lambda_profile")
            ),
            advanced_model_requested=advanced,
            compare_models_available=compare_models_available,
            compare_models_reason=compare_models_reason,
            compare_policies_available=policies_available,
            compare_policies_reason=compare_policies_reason,
        )

    time_options: list[PlasmonPlotCapabilityOption] = []
    if selected_model_advanced:
        time_options.append(PlasmonPlotCapabilityOption("spectrum_observed", "Representative spectrum"))
    if compare_models_enabled:
        time_options.append(PlasmonPlotCapabilityOption("spectrum_compare_models", "Representative spectra across models"))
    if compare_policies_enabled:
        time_options.append(PlasmonPlotCapabilityOption("spectrum_compare_policies", "Representative spectra across e- policies"))
    time_options.extend(diagnostics_time)

    profile_options: list[PlasmonPlotCapabilityOption] = []
    if selected_model_advanced:
        profile_options.append(PlasmonPlotCapabilityOption("dispersion_selected_model", "Peak shift vs scan axis"))
        profile_options.append(PlasmonPlotCapabilityOption("dispersion_selected_width", "FWHM vs scan axis"))
    if compare_models_enabled:
        profile_options.extend(
            (
                PlasmonPlotCapabilityOption("dispersion_compare_models", "Peak-shift comparison across models"),
                PlasmonPlotCapabilityOption("dispersion_compare_width_models", "FWHM comparison across models"),
            )
        )
    if compare_policies_enabled:
        profile_options.append(PlasmonPlotCapabilityOption("dispersion_compare_policies", "Peak-shift comparison across e- policies"))
    profile_options.extend(diagnostics_profile)

    return PlasmonUiCapabilities(
        primary_label="Representative spectra",
        secondary_label="Dispersion / local state",
        time_options=tuple(time_options),
        profile_options=tuple(profile_options),
        preferred_time_key=(
            "spectrum_compare_policies"
            if compare_policies_enabled
            else (
                "spectrum_compare_models"
                if compare_models_enabled
                else ("spectrum_observed" if selected_model_advanced else "plasma_frequency")
            )
        ),
        preferred_profile_key=(
            "dispersion_compare_policies"
            if compare_policies_enabled
            else (
                "dispersion_compare_models"
                if compare_models_enabled
                else ("dispersion_selected_model" if selected_model_advanced else "local_k_lambda_profile")
            )
        ),
        advanced_model_requested=advanced,
        compare_models_available=compare_models_available,
        compare_models_reason=compare_models_reason,
        compare_policies_available=policies_available,
        compare_policies_reason=compare_policies_reason,
    )
