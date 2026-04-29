"""Quick spectroscopy estimates for HELIOS Derived mode."""

from __future__ import annotations

from typing import TYPE_CHECKING
import math
from typing import Callable

import numpy as np

from helios.runtime import RunContext
from helios.services.constants import C_LIGHT_CM_S
from helios.services.derived.common import picosecond_drive_warning
from helios.services.derived.models import DerivedPlotBundle, DerivedRunData, DerivedWarning, SpectroscopyResult
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_mask,
    cached_time_series_payload,
    cylindrical_path_note,
    observation_axis_cosine,
    path_geometry_summary,
    profile_boundary_positions,
    profile_coordinate_values,
    resolve_weighting_mode,
    selection_cache_key,
    shared_time_series_weighted_means,
    weighted_means,
)

if TYPE_CHECKING:
    from helios.services.derived.analysis import DerivedAnalysisParameters
    from helios.services.derived.models import AnalysisGeometryMetadata


def doppler_width_fraction(temperature_ev: float, ion_mass_mu: float) -> float:
    """NRL 2023 p. 58 Eq. (25), fractional Doppler width ``Delta(lambda) / lambda``."""

    if temperature_ev <= 0.0 or ion_mass_mu <= 0.0:
        return float("nan")
    return 7.7e-5 * math.sqrt(float(temperature_ev) / float(ion_mass_mu))


def doppler_width_fraction_array(temperature_ev: np.ndarray, ion_mass_mu: np.ndarray) -> np.ndarray:
    """Vectorized fractional Doppler width with the scalar helper's domain rules."""

    temperature = np.asarray(temperature_ev, dtype=np.float64)
    mass = np.asarray(ion_mass_mu, dtype=np.float64)
    result = np.full(np.broadcast_shapes(temperature.shape, mass.shape), np.nan, dtype=np.float64)
    temperature_b = np.broadcast_to(temperature, result.shape)
    mass_b = np.broadcast_to(mass, result.shape)
    valid = (temperature_b > 0.0) & (mass_b > 0.0) & np.isfinite(temperature_b) & np.isfinite(mass_b)
    result[valid] = 7.7e-5 * np.sqrt(temperature_b[valid] / mass_b[valid])
    return result


def evaluate_spectroscopy(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    line_wavelength_nm: float,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    include_time_plots: bool = True,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> SpectroscopyResult:
    """Estimate bulk Doppler shift and thermal broadening for a chosen line."""

    weighting_mode = resolve_weighting_mode(parameters.weighting_mode, module_name="spectroscopy")
    warnings: list[DerivedWarning] = [
        DerivedWarning(
            "spectroscopy",
            "Spectroscopy is a line-of-sight quick look: no line formation, emissivity, or opacity model is included.",
            severity="info",
        )
    ]
    ps_warning = picosecond_drive_warning(
        "spectroscopy",
        dataset,
        "The spectroscopy quick look is less validated for ps-scale drives, where transient nonequilibrium structure can weaken a subset-averaged Doppler interpretation.",
    )
    if ps_warning is not None:
        warnings.append(ps_warning)
    cylindrical_warning = cylindrical_path_note("spectroscopy", dataset, geometry)
    if cylindrical_warning is not None:
        warnings.append(cylindrical_warning)
    mask, selection, selection_warnings = build_analysis_mask(
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
        weighting_mode=weighting_mode,
        analysis_cache=analysis_cache,
    )
    warnings.extend(selection_warnings)

    axis_cosine = observation_axis_cosine(
        dataset,
        observation_side=geometry.observation_side,
        los_cosine=geometry.line_of_sight_cosine,
        propagation_direction=geometry.propagation_direction,
    )
    summary_fields = np.stack(
        (
            np.asarray(dataset.velocity_cm_s[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.zone_atomic_weight, dtype=np.float64),
        ),
        axis=0,
    )
    bulk_velocity, ion_temperature, ion_mass_mu = weighted_means(
        summary_fields,
        dataset,
        snapshot_index,
        mask,
        mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_cache_key(selection),
        analysis_cache=analysis_cache,
    )
    los_velocity_cm_s = float(bulk_velocity) * axis_cosine
    shift_nm = float(line_wavelength_nm) * los_velocity_cm_s / C_LIGHT_CM_S
    width_fraction = doppler_width_fraction(ion_temperature, ion_mass_mu)
    thermal_width_nm = float(line_wavelength_nm) * width_fraction if math.isfinite(width_fraction) else float("nan")

    if not np.any(mask):
        warnings.append(DerivedWarning("spectroscopy", "No active zones are selected for the spectroscopy estimate.", severity="error"))
    elif not all(math.isfinite(float(value)) for value in (bulk_velocity, ion_temperature, ion_mass_mu)):
        warnings.append(
            DerivedWarning(
                "spectroscopy",
                "The active weighting/filter selection did not produce a finite effective spectroscopy state for this snapshot.",
                severity="warning",
            )
        )
    if abs(float(axis_cosine)) < 0.1:
        warnings.append(
            DerivedWarning(
                "spectroscopy",
                "LOS cosine is very small; projected Doppler shifts are geometry-sensitive near transverse viewing.",
                severity="warning",
            )
        )
    if math.isfinite(los_velocity_cm_s) and abs(los_velocity_cm_s) / C_LIGHT_CM_S > 0.05:
        warnings.append(DerivedWarning("spectroscopy", "Bulk LOS velocity exceeds 5% of c; the non-relativistic Doppler estimate is only approximate.", severity="warning"))
    if not math.isfinite(width_fraction):
        warnings.append(DerivedWarning("spectroscopy", "Thermal Doppler width could not be estimated from the active subset.", severity="warning"))

    time_plots: tuple[DerivedPlotBundle, ...] = ()
    if include_time_plots:
        time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
        mean_series = shared_time_series_weighted_means(
            dataset,
            context,
            geometry=geometry,
            weighting_mode=weighting_mode,
            field_series=(
                ("bulk_velocity_cm_s", np.asarray(dataset.velocity_cm_s, dtype=np.float64)),
                ("ion_temperature_ev", np.asarray(dataset.temperature_i_ev, dtype=np.float64)),
                ("ion_mass_mu", np.asarray(dataset.zone_atomic_weight, dtype=np.float64)),
            ),
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        series_cache_key = (
            "spectroscopy.derived_series",
            geometry.observation_side,
            round(float(geometry.line_of_sight_angle_deg), 12),
            round(float(line_wavelength_nm), 12),
            weighting_mode,
            parameters.reuse_viewer_subset,
            tuple(parameters.derived_region_ids or ()),
            tuple(parameters.derived_material_ids or ()),
            bool(parameters.exclude_entry_region),
            bool(parameters.exclude_low_density),
            round(float(parameters.min_density_g_cm3), 12),
            bool(parameters.exclude_opposite_velocity),
            parameters.zone_index_lower,
            parameters.zone_index_upper,
        )

        def _build_derived_series() -> dict[str, np.ndarray]:
            bulk_series = np.asarray(mean_series["bulk_velocity_cm_s"], dtype=np.float64)
            ti_series = np.asarray(mean_series["ion_temperature_ev"], dtype=np.float64)
            ion_mass_series = np.asarray(mean_series["ion_mass_mu"], dtype=np.float64)
            los_series = bulk_series * float(axis_cosine)
            shift_series = float(line_wavelength_nm) * los_series / C_LIGHT_CM_S
            if progress_check is not None:
                progress_check()
            width_series = float(line_wavelength_nm) * doppler_width_fraction_array(ti_series, ion_mass_series)
            return {
                "los_velocity_cm_s": los_series,
                "doppler_shift_nm": shift_series,
                "thermal_width_nm": width_series,
            }

        derived_series = cached_time_series_payload(
            series_cache_key,
            analysis_cache=analysis_cache,
            builder=_build_derived_series,
        )
        bulk_velocity_series = np.asarray(mean_series["bulk_velocity_cm_s"], dtype=np.float64) * 1.0e-5
        los_velocity_series = np.asarray(derived_series["los_velocity_cm_s"], dtype=np.float64) * 1.0e-5
        shift_series = np.asarray(derived_series["doppler_shift_nm"], dtype=np.float64)
        width_series = np.asarray(derived_series["thermal_width_nm"], dtype=np.float64)
        time_plots = (
            DerivedPlotBundle(
                key="bulk_velocity",
                title="Bulk velocity vs time",
                x_label="Time [ns]",
                y_label="Velocity [km/s]",
                x_values=time_ns,
                y_series=(bulk_velocity_series,),
                curve_names=("Bulk velocity",),
            ),
            DerivedPlotBundle(
                key="los_velocity",
                title="LOS velocity vs time",
                x_label="Time [ns]",
                y_label="Velocity [km/s]",
                x_values=time_ns,
                y_series=(los_velocity_series,),
                curve_names=("LOS velocity",),
            ),
            DerivedPlotBundle(
                key="velocity",
                title="Bulk and LOS velocity vs time",
                x_label="Time [ns]",
                y_label="Velocity [km/s]",
                x_values=time_ns,
                y_series=(bulk_velocity_series, los_velocity_series),
                curve_names=("Bulk velocity", "LOS velocity"),
            ),
            DerivedPlotBundle(
                key="doppler_shift",
                title="Doppler shift vs time",
                x_label="Time [ns]",
                y_label="Doppler shift [nm]",
                x_values=time_ns,
                y_series=(shift_series,),
                curve_names=("Shift",),
            ),
            DerivedPlotBundle(
                key="thermal_width",
                title="Thermal broadening vs time",
                x_label="Time [ns]",
                y_label="Thermal width [nm]",
                x_values=time_ns,
                y_series=(width_series,),
                curve_names=("Thermal width",),
            ),
        )

    coordinate_values, coordinate_label = profile_coordinate_values(dataset, snapshot_index, geometry.profile_coordinate_mode)
    boundary_positions = profile_boundary_positions(dataset, snapshot_index, geometry.profile_coordinate_mode)
    velocity_profile = np.where(mask, np.asarray(dataset.velocity_cm_s[int(snapshot_index)], dtype=np.float64) * 1.0e-5, np.nan)
    ion_temperature_profile = np.where(mask, np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64), np.nan)
    local_shift_profile = np.where(mask, np.asarray(dataset.velocity_cm_s[int(snapshot_index)], dtype=np.float64) * axis_cosine * float(line_wavelength_nm) / C_LIGHT_CM_S, np.nan)

    geometry_summary = (
        f"{geometry.observation_side} side ({geometry.observation_boundary}-index boundary) | "
        f"LOS cos={axis_cosine:.3f} | {path_geometry_summary(dataset, geometry)}"
    )

    return SpectroscopyResult(
        snapshot_index=int(snapshot_index),
        weighting_mode=weighting_mode,
        geometry_summary=geometry_summary,
        line_wavelength_nm=float(line_wavelength_nm),
        line_of_sight_cosine=float(axis_cosine),
        bulk_velocity_cm_s=float(bulk_velocity),
        los_velocity_cm_s=float(los_velocity_cm_s),
        doppler_shift_nm=float(shift_nm),
        thermal_width_fraction=float(width_fraction),
        thermal_width_nm=float(thermal_width_nm),
        ion_temperature_ev=float(ion_temperature),
        ion_mass_mu=float(ion_mass_mu),
        time_plots=time_plots,
        profile_plots=(
            DerivedPlotBundle(
                key="velocity_profile",
                title="Velocity profile",
                x_label=coordinate_label,
                y_label="Velocity [km/s]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(velocity_profile,),
                curve_names=("Velocity",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="ion_temperature_profile",
                title="Ion temperature profile",
                x_label=coordinate_label,
                y_label="Ion temperature [eV]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(ion_temperature_profile,),
                curve_names=("Ti",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="doppler_proxy_profile",
                title="Local Doppler-shift proxy profile",
                x_label=coordinate_label,
                y_label="LOS Doppler proxy [nm]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(local_shift_profile,),
                curve_names=("LOS Doppler proxy",),
                boundary_positions=boundary_positions,
            ),
        ),
        warnings=tuple(warnings),
    )
