from __future__ import annotations

"""Reusable helpers for plasmon validation scripts.

These functions intentionally keep benchmark plumbing separate from the main
service path so validation scripts can compare hydro-derived states, simplified
uniform states, and literature digitizations without mutating the production UI
or dispatch layer.
"""

from dataclasses import asdict
from pathlib import Path
import math
from typing import Any

import numpy as np

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.selection import build_analysis_geometry

NA = 6.02214076e23
HBARC_EV_A = 1973.269804
AL_A = 26.9815
AL_Z = 3.0


def q_to_angle_deg(q_ang_inv: float, photon_energy_kev: float) -> float:
    argument = float(q_ang_inv) * HBARC_EV_A / (2.0 * float(photon_energy_kev) * 1000.0)
    argument = min(max(argument, -1.0), 1.0)
    return math.degrees(2.0 * math.asin(argument))


def make_run_context(dataset: DerivedRunData, path: Path, *, snapshot_index: int) -> RunContext:
    return RunContext(
        path=Path(path),
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


def uniform_al_dataset(rho_g_cm3: float, te_ev: float, *, zones: int = 12) -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 1
    time_s = np.asarray([0.0], dtype=np.float64)
    static_x = np.linspace(1.0e-4, float(zones) * 1.0e-4, zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, float(zones) * 1.0e-4 + 5.0e-5, zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, zones), 1.0e-4, dtype=np.float64)
    density = np.full((n_snapshots, zones), float(rho_g_cm3), dtype=np.float64)
    velocity = np.zeros_like(density)
    ne_cm3 = float(rho_g_cm3) / AL_A * NA * AL_Z
    temperature_e = np.full_like(density, float(te_ev))
    temperature_i = np.full_like(density, float(te_ev))
    electron_density = np.full_like(density, ne_cm3)
    mean_charge = np.full_like(density, AL_Z)
    zone_region_id = np.ones(zones, dtype=np.int32)
    zone_material = np.ones(zones, dtype=np.int32)
    regions = {
        "region_index": np.asarray([1], dtype=np.int32),
        "min_zone_index": np.asarray([1], dtype=np.int32),
        "max_zone_index": np.asarray([zones], dtype=np.int32),
        "atomic_weight": np.asarray([AL_A], dtype=np.float64),
        "initial_mass_density": np.asarray([rho_g_cm3], dtype=np.float64),
        "initial_temperature": np.asarray([te_ev], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path("uniform_al.h5"),
        summary={"n_zones": zones, "n_snapshots": n_snapshots},
        metadata={"geometry": "PLANAR", "coordinate_model": {"coordinate_name": "x"}},
        regions=regions,
        materials={
            "index": np.asarray([1], dtype=np.int32),
            "eos_file_path": np.asarray(["Al.prp"], dtype=object),
            "opacity_file_path": np.asarray(["Al.prp"], dtype=object),
            "eos_model": np.asarray(["EOSOPA"], dtype=object),
            "opacity_model": np.asarray(["EOSOPA"], dtype=object),
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
        zone_atomic_weight=np.full(zones, AL_A, dtype=np.float64),
        zone_initial_density_g_cm3=np.full(zones, rho_g_cm3, dtype=np.float64),
        zone_initial_temperature_ev=np.full(zones, te_ev, dtype=np.float64),
        laser_entry=None,
    )
    context = RunContext(
        path=Path("uniform_al.h5"),
        summary={"n_zones": zones, "n_snapshots": n_snapshots},
        metadata={},
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=time_s.copy(),
        static_x_values=static_x.copy(),
        zone_region_id=zone_region_id.copy(),
        zone_material_index=zone_material.copy(),
        has_dynamic_radius=False,
        snapshot_index=0,
        map_coordinate="static_x",
        slice_coordinate="zone",
        selected_region_ids=(1,),
        selected_material_ids=(1,),
    )
    return dataset, context


def compute_plasmon(dataset: DerivedRunData, context: RunContext, *, analysis_cache=None, **kwargs):
    params = DerivedAnalysisParameters(**kwargs)
    geometry = build_analysis_geometry(
        dataset,
        context,
        observation_side=params.observation_side,
        line_of_sight_angle_deg=params.line_of_sight_angle_deg,
        line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
        profile_coordinate_mode=params.profile_coordinate_mode,
    )
    return evaluate_plasmon_regime(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        photon_energy_kev=params.plasmon_photon_energy_kev,
        scattering_angle_deg=params.plasmon_scattering_angle_deg,
        adiabatic_index=params.plasmon_adiabatic_index,
        parameters=params,
        geometry=geometry,
        include_time_plots=False,
        analysis_cache=analysis_cache,
    )


def shocked_al_slab_summary(
    dataset: DerivedRunData,
    *,
    snapshot_index: int,
    density_floor_g_cm3: float = 3.5,
    material_id: int = 1,
) -> dict[str, Any]:
    """Return a contiguous shocked-Al slab summary for one snapshot.

    The selection rule is intentionally transparent: keep only zones with the
    requested material ID and then retain the contiguous span between the first
    and last zone whose mass density exceeds ``density_floor_g_cm3``.
    """

    snapshot_index = int(snapshot_index)
    zone_material = np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))
    rho = np.asarray(dataset.density_g_cm3[snapshot_index], dtype=np.float64)
    te = np.asarray(dataset.temperature_e_ev[snapshot_index], dtype=np.float64)
    ti = np.asarray(dataset.temperature_i_ev[snapshot_index], dtype=np.float64)
    ne = np.asarray(dataset.electron_density_cm3[snapshot_index], dtype=np.float64)
    zbar = np.asarray(dataset.mean_charge[snapshot_index], dtype=np.float64)
    width = np.asarray(dataset.zone_width_cm[snapshot_index], dtype=np.float64)

    base_mask = zone_material == int(abs(material_id))
    threshold_mask = base_mask & np.isfinite(rho) & (rho >= float(density_floor_g_cm3))
    indices = np.flatnonzero(threshold_mask)
    if indices.size == 0:
        raise ValueError(
            f"No zones satisfy material={material_id} and rho>={density_floor_g_cm3:.3f} g/cm^3 at snapshot {snapshot_index}."
        )
    zone_index_lower = int(indices[0] + 1)
    zone_index_upper = int(indices[-1] + 1)
    slab_mask = np.zeros_like(base_mask, dtype=bool)
    slab_mask[zone_index_lower - 1 : zone_index_upper] = True
    slab_mask &= base_mask

    slab_width = width[slab_mask]
    weight_sum = float(np.sum(slab_width))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Selected slab has non-positive total path length.")

    def _wavg(values: np.ndarray) -> float:
        local = np.asarray(values[slab_mask], dtype=np.float64)
        return float(np.average(local, weights=slab_width))

    return {
        "snapshot_index": snapshot_index,
        "time_ns": float(dataset.time_s[snapshot_index] * 1.0e9),
        "material_id": int(abs(material_id)),
        "density_floor_g_cm3": float(density_floor_g_cm3),
        "zone_index_lower": zone_index_lower,
        "zone_index_upper": zone_index_upper,
        "zone_count": int(np.count_nonzero(slab_mask)),
        "rho_min_g_cm3": float(np.min(rho[slab_mask])),
        "rho_max_g_cm3": float(np.max(rho[slab_mask])),
        "rho_weighted_g_cm3": _wavg(rho),
        "te_weighted_ev": _wavg(te),
        "ti_weighted_ev": _wavg(ti),
        "ne_weighted_cm3": _wavg(ne),
        "zbar_weighted": _wavg(zbar),
        "path_length_total_cm": weight_sum,
    }
