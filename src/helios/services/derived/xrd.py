"""Quick-look XRD compressed-state estimates for HELIOS Derived mode."""

from __future__ import annotations

from typing import TYPE_CHECKING
import math
from typing import Callable

import numpy as np

from helios.runtime import RunContext
from helios.services.derived.common import picosecond_drive_warning
from helios.services.derived.models import DerivedPlotBundle, DerivedRunData, DerivedWarning, XrdLayerEstimate, XrdResult
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_mask,
    cached_time_series_payload,
    profile_boundary_positions,
    profile_coordinate_values,
    resolve_weighting_mode,
    selection_cache_key,
    shared_time_series_selection_state,
    weight_array,
)
from helios.services.units.conversions import photon_energy_kev_to_wavelength_angstrom

if TYPE_CHECKING:
    from helios.services.derived.analysis import DerivedAnalysisParameters
    from helios.services.derived.models import AnalysisGeometryMetadata


def _zone_region_dense_index(region_ids: np.ndarray, zone_region_id: np.ndarray) -> np.ndarray:
    dense = np.searchsorted(region_ids, zone_region_id)
    valid = (dense >= 0) & (dense < region_ids.size)
    if np.any(valid):
        valid &= region_ids[dense] == zone_region_id
    if not np.all(valid):
        raise ValueError("Zone-to-region mapping contains region identifiers not present in the region table.")
    return dense.astype(np.int32, copy=False)


def _region_metrics_for_snapshot(
    dataset: DerivedRunData,
    mask: np.ndarray,
    snapshot_index: int,
    *,
    photon_energy_kev: float,
    initial_bragg_angle_deg: float,
    weighting_mode: str,
    geometry: "AnalysisGeometryMetadata",
    selection_key: tuple[object, ...] | None,
    region_ids: np.ndarray,
    initial_densities: np.ndarray,
    zone_region_dense_index: np.ndarray,
    analysis_cache: AnalysisStateCache | None = None,
    emit_warnings: bool = True,
) -> tuple[dict[str, np.ndarray | float], list[DerivedWarning]]:
    warnings: list[DerivedWarning] = []
    wavelength_angstrom = photon_energy_kev_to_wavelength_angstrom(float(photon_energy_kev))
    theta0_rad = math.radians(float(initial_bragg_angle_deg))
    q0 = 4.0 * math.pi * math.sin(theta0_rad) / wavelength_angstrom
    n_regions = int(region_ids.size)
    zone_width = np.asarray(dataset.zone_width_cm[int(snapshot_index)], dtype=np.float64)
    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    zone_count = np.bincount(zone_region_dense_index[mask_array], minlength=n_regions).astype(np.int32, copy=False)
    thickness_cm = np.bincount(zone_region_dense_index[mask_array], weights=zone_width[mask_array], minlength=n_regions).astype(np.float64, copy=False)
    weights = weight_array(
        density,
        dataset,
        snapshot_index,
        mask_array,
        mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_key,
        analysis_cache=analysis_cache,
    )
    valid_weights = weights > 0.0
    weight_sum = np.bincount(zone_region_dense_index[valid_weights], weights=weights[valid_weights], minlength=n_regions).astype(np.float64, copy=False)
    density_sum = np.bincount(
        zone_region_dense_index[valid_weights],
        weights=(weights[valid_weights] * density[valid_weights]),
        minlength=n_regions,
    ).astype(np.float64, copy=False)
    current_density = np.divide(
        density_sum,
        weight_sum,
        out=np.full(n_regions, np.nan, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    compression_ratio = np.divide(
        current_density,
        initial_densities,
        out=np.full(n_regions, np.nan, dtype=np.float64),
        where=initial_densities > 0.0,
    )
    d_over_d0 = np.full(n_regions, np.nan, dtype=np.float64)
    valid_compression = np.isfinite(compression_ratio) & (compression_ratio > 0.0)
    d_over_d0[valid_compression] = np.power(compression_ratio[valid_compression], -1.0 / 3.0)

    sin_theta_shifted = np.full(n_regions, np.nan, dtype=np.float64)
    valid_d_ratio = np.isfinite(d_over_d0) & (d_over_d0 > 0.0)
    sin_theta_shifted[valid_d_ratio] = math.sin(theta0_rad) / d_over_d0[valid_d_ratio]
    valid_bragg = valid_d_ratio & (sin_theta_shifted <= 1.0)
    shifted_theta_rad = np.full(n_regions, np.nan, dtype=np.float64)
    shifted_theta_rad[valid_bragg] = np.arcsin(sin_theta_shifted[valid_bragg])
    shifted_theta_deg = np.full(n_regions, np.nan, dtype=np.float64)
    shifted_theta_deg[valid_bragg] = np.degrees(shifted_theta_rad[valid_bragg])
    bragg_shift_deg = np.full(n_regions, np.nan, dtype=np.float64)
    bragg_shift_deg[valid_bragg] = shifted_theta_deg[valid_bragg] - float(initial_bragg_angle_deg)
    q_compressed = np.full(n_regions, np.nan, dtype=np.float64)
    q_compressed[valid_bragg] = 4.0 * math.pi * np.sin(shifted_theta_rad[valid_bragg]) / wavelength_angstrom

    if emit_warnings:
        invalid_density_regions = np.flatnonzero((zone_count > 0) & ~np.isfinite(current_density))
        for region_offset in invalid_density_regions:
            warnings.append(
                DerivedWarning(
                    "xrd",
                    f"Region {int(region_ids[region_offset])} has no finite positive {weighting_mode.replace('_', ' ')} weights after filtering.",
                    severity="warning",
                )
            )
        invalid_bragg_regions = np.flatnonzero(valid_d_ratio & (sin_theta_shifted > 1.0))
        for region_offset in invalid_bragg_regions:
            warnings.append(
                DerivedWarning(
                    "xrd",
                    f"Region {int(region_ids[region_offset])} exceeds the chosen Bragg-angle approximation (sin(theta') > 1).",
                    severity="caution",
                )
            )

    return (
        {
            "q0": float(q0),
            "zone_count": zone_count,
            "current_density": current_density,
            "compression_ratio": compression_ratio,
            "d_over_d0": d_over_d0,
            "shifted_theta_deg": shifted_theta_deg,
            "bragg_shift_deg": bragg_shift_deg,
            "q_compressed": q_compressed,
            "thickness_cm": thickness_cm,
        },
        warnings,
    )


def _layer_estimates_for_snapshot(
    dataset: DerivedRunData,
    mask: np.ndarray,
    snapshot_index: int,
    *,
    photon_energy_kev: float,
    initial_bragg_angle_deg: float,
    weighting_mode: str,
    geometry: "AnalysisGeometryMetadata",
    selection_key: tuple[object, ...] | None,
    region_ids: np.ndarray,
    initial_densities: np.ndarray,
    zone_region_dense_index: np.ndarray,
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[tuple[XrdLayerEstimate, ...], list[DerivedWarning]]:
    metrics, warnings = _region_metrics_for_snapshot(
        dataset,
        mask,
        snapshot_index,
        photon_energy_kev=photon_energy_kev,
        initial_bragg_angle_deg=initial_bragg_angle_deg,
        weighting_mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_key,
        region_ids=region_ids,
        initial_densities=initial_densities,
        zone_region_dense_index=zone_region_dense_index,
        analysis_cache=analysis_cache,
        emit_warnings=True,
    )
    zone_count = np.asarray(metrics["zone_count"], dtype=np.int32)
    current_density = np.asarray(metrics["current_density"], dtype=np.float64)
    active_regions = np.flatnonzero((zone_count > 0) & np.isfinite(current_density))
    layers = tuple(
        XrdLayerEstimate(
            region_id=int(region_ids[region_offset]),
            zone_count=int(zone_count[region_offset]),
            initial_density_g_cm3=float(initial_densities[region_offset]),
            compressed_density_g_cm3=float(current_density[region_offset]),
            compression_ratio=float(metrics["compression_ratio"][region_offset]),
            d_over_d0=float(metrics["d_over_d0"][region_offset]),
            q0_inv_angstrom=float(metrics["q0"]),
            q_compressed_inv_angstrom=float(metrics["q_compressed"][region_offset]),
            initial_bragg_angle_deg=float(initial_bragg_angle_deg),
            shifted_bragg_angle_deg=(
                float(metrics["shifted_theta_deg"][region_offset])
                if np.isfinite(metrics["shifted_theta_deg"][region_offset])
                else None
            ),
            bragg_shift_deg=(
                float(metrics["bragg_shift_deg"][region_offset])
                if np.isfinite(metrics["bragg_shift_deg"][region_offset])
                else None
            ),
            compressed_thickness_cm=float(metrics["thickness_cm"][region_offset]),
        )
        for region_offset in active_regions
    )
    return layers, warnings


def estimate_xrd(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    photon_energy_kev: float,
    initial_bragg_angle_deg: float,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    include_time_plots: bool = True,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> XrdResult:
    """Estimate compressed-state Bragg shifts using isotropic compression."""

    warnings = [
        DerivedWarning(
            "xrd",
            "XRD uses isotropic compression only; crystallographic texture, preferred orientation, and phase changes are not modeled.",
            severity="info",
        )
    ]
    ps_warning = picosecond_drive_warning(
        "xrd",
        dataset,
        "This XRD quick look has only been validated on slower hydrodynamic runs; ps-scale drives should be treated as approximate.",
    )
    if ps_warning is not None:
        warnings.append(ps_warning)
    weighting_mode = resolve_weighting_mode(parameters.weighting_mode, module_name="xrd")
    region_ids_all = np.asarray(dataset.regions["region_index"], dtype=np.int32)
    initial_densities = np.asarray(dataset.regions["initial_mass_density"], dtype=np.float64)
    zone_region_dense_index = _zone_region_dense_index(region_ids_all, np.asarray(dataset.zone_region_id, dtype=np.int32))
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

    layers, layer_warnings = _layer_estimates_for_snapshot(
        dataset,
        mask,
        snapshot_index,
        photon_energy_kev=photon_energy_kev,
        initial_bragg_angle_deg=initial_bragg_angle_deg,
        weighting_mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_cache_key(selection),
        region_ids=region_ids_all,
        initial_densities=initial_densities,
        zone_region_dense_index=zone_region_dense_index,
        analysis_cache=analysis_cache,
    )
    warnings.extend(layer_warnings)

    region_ids = [int(layer.region_id) for layer in layers]
    if not layers:
        warnings.append(DerivedWarning("xrd", "No active crystalline layers were available in the current subset.", severity="error"))

    time_plots: tuple[DerivedPlotBundle, ...] = ()
    if include_time_plots:
        time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
        curve_names: list[str] = []
        selected_region_offsets = [int(np.flatnonzero(region_ids_all == int(region_id))[0]) for region_id in region_ids]
        series_cache_key = (
            "xrd.time_series",
            geometry.observation_side,
            round(float(geometry.line_of_sight_angle_deg), 12),
            round(float(photon_energy_kev), 12),
            round(float(initial_bragg_angle_deg), 12),
            weighting_mode,
            tuple(int(value) for value in region_ids),
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

        def _build_time_series() -> dict[str, np.ndarray]:
            density_matrix = np.full((len(selected_region_offsets), dataset.time_s.size), np.nan, dtype=np.float64)
            compression_matrix = np.full_like(density_matrix, np.nan)
            d_ratio_matrix = np.full_like(density_matrix, np.nan)
            bragg_matrix = np.full_like(density_matrix, np.nan)
            q_matrix = np.full_like(density_matrix, np.nan)
            thickness_matrix = np.full_like(density_matrix, np.nan)
            mask_matrix, selection_keys = shared_time_series_selection_state(
                dataset,
                context,
                geometry=geometry,
                weighting_mode=weighting_mode,
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
            for time_index in range(dataset.time_s.size):
                if progress_check is not None and (time_index % 8 == 0):
                    progress_check()
                metrics, _ = _region_metrics_for_snapshot(
                    dataset,
                    np.asarray(mask_matrix[int(time_index)], dtype=bool),
                    time_index,
                    photon_energy_kev=photon_energy_kev,
                    initial_bragg_angle_deg=initial_bragg_angle_deg,
                    weighting_mode=weighting_mode,
                    geometry=geometry,
                    selection_key=selection_keys[int(time_index)],
                    region_ids=region_ids_all,
                    initial_densities=initial_densities,
                    zone_region_dense_index=zone_region_dense_index,
                    analysis_cache=analysis_cache,
                    emit_warnings=False,
                )
                density_matrix[:, time_index] = np.asarray(metrics["current_density"], dtype=np.float64)[selected_region_offsets]
                compression_matrix[:, time_index] = np.asarray(metrics["compression_ratio"], dtype=np.float64)[selected_region_offsets]
                d_ratio_matrix[:, time_index] = np.asarray(metrics["d_over_d0"], dtype=np.float64)[selected_region_offsets]
                bragg_matrix[:, time_index] = np.asarray(metrics["bragg_shift_deg"], dtype=np.float64)[selected_region_offsets]
                q_matrix[:, time_index] = np.asarray(metrics["q_compressed"], dtype=np.float64)[selected_region_offsets]
                thickness_matrix[:, time_index] = np.asarray(metrics["thickness_cm"], dtype=np.float64)[selected_region_offsets] * 1.0e4
            return {
                "density": density_matrix,
                "compression_ratio": compression_matrix,
                "d_ratio": d_ratio_matrix,
                "bragg_shift": bragg_matrix,
                "q_compressed": q_matrix,
                "thickness": thickness_matrix,
            }

        time_series = cached_time_series_payload(
            series_cache_key,
            analysis_cache=analysis_cache,
            builder=_build_time_series,
        )
        density_matrix = np.asarray(time_series["density"], dtype=np.float64)
        compression_matrix = np.asarray(time_series["compression_ratio"], dtype=np.float64)
        d_ratio_matrix = np.asarray(time_series["d_ratio"], dtype=np.float64)
        bragg_matrix = np.asarray(time_series["bragg_shift"], dtype=np.float64)
        q_matrix = np.asarray(time_series["q_compressed"], dtype=np.float64)
        thickness_matrix = np.asarray(time_series["thickness"], dtype=np.float64)

        for region_id in region_ids:
            curve_names.append(f"Region {region_id}")
        time_plots = (
            DerivedPlotBundle(
                key="density",
                title="Compressed density vs time",
                x_label="Time [ns]",
                y_label="Density [g/cm3]",
                x_values=time_ns,
                y_series=tuple(np.asarray(density_matrix[index], dtype=np.float64) for index in range(density_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
            DerivedPlotBundle(
                key="compression_ratio",
                title="Compression ratio vs time",
                x_label="Time [ns]",
                y_label="Compression ratio rho/rho0",
                x_values=time_ns,
                y_series=tuple(np.asarray(compression_matrix[index], dtype=np.float64) for index in range(compression_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
            DerivedPlotBundle(
                key="d_ratio",
                title="d/d0 vs time",
                x_label="Time [ns]",
                y_label="d/d0",
                x_values=time_ns,
                y_series=tuple(np.asarray(d_ratio_matrix[index], dtype=np.float64) for index in range(d_ratio_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
            DerivedPlotBundle(
                key="bragg_shift",
                title="Bragg shift vs time",
                x_label="Time [ns]",
                y_label="Bragg shift [deg]",
                x_values=time_ns,
                y_series=tuple(np.asarray(bragg_matrix[index], dtype=np.float64) for index in range(bragg_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
            DerivedPlotBundle(
                key="q_compressed",
                title="Compressed scattering vector vs time",
                x_label="Time [ns]",
                y_label="Q [1/A]",
                x_values=time_ns,
                y_series=tuple(np.asarray(q_matrix[index], dtype=np.float64) for index in range(q_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
            DerivedPlotBundle(
                key="thickness",
                title="Compressed layer thickness vs time",
                x_label="Time [ns]",
                y_label="Thickness [um]",
                x_values=time_ns,
                y_series=tuple(np.asarray(thickness_matrix[index], dtype=np.float64) for index in range(thickness_matrix.shape[0])),
                curve_names=tuple(curve_names),
            ),
        )

    coordinate_values, coordinate_label = profile_coordinate_values(dataset, snapshot_index, geometry.profile_coordinate_mode)
    boundary_positions = profile_boundary_positions(dataset, snapshot_index, geometry.profile_coordinate_mode)
    density_profile = np.where(mask, np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64), np.nan)
    compression_profile = np.where(mask, density_profile / dataset.zone_initial_density_g_cm3, np.nan)
    geometry_summary = (
        f"{geometry.observation_side} side ({geometry.observation_boundary}-index boundary) | "
        f"profile={geometry.profile_coordinate_mode}"
    )

    return XrdResult(
        snapshot_index=int(snapshot_index),
        photon_energy_kev=float(photon_energy_kev),
        wavelength_angstrom=float(photon_energy_kev_to_wavelength_angstrom(float(photon_energy_kev))),
        initial_bragg_angle_deg=float(initial_bragg_angle_deg),
        weighting_mode=weighting_mode,
        geometry_summary=geometry_summary,
        profile_coordinate_label=coordinate_label,
        profile_coordinate_values=np.asarray(coordinate_values, dtype=np.float64),
        time_plots=time_plots,
        profile_plots=(
            DerivedPlotBundle(
                key="density_profile",
                title="Zone-resolved density profile",
                x_label=coordinate_label,
                y_label="Density [g/cm3]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(density_profile,),
                curve_names=("Active subset",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="compression_profile",
                title="Zone-resolved compression ratio profile",
                x_label=coordinate_label,
                y_label="Compression ratio rho/rho0",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(compression_profile,),
                curve_names=("Active subset",),
                boundary_positions=boundary_positions,
            ),
        ),
        layers=tuple(layers),
        warnings=tuple(warnings),
    )
