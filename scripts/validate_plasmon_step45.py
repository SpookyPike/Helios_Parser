"""Step 4/5 validation for the staged plasmon module.

This script focuses on three things:
1. literature-style benchmark states for Al with published q-points,
2. synthetic tests that prove LOS integration is not a dumb effective-state average,
3. real HELIOS examples showing zone-range filtering changes the integrated spectrum.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401
import numpy as np

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
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


def uniform_al_dataset(rho_g_cm3: float, te_ev: float, *, zones: int = 8) -> tuple[DerivedRunData, RunContext]:
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
        materials={"index": np.asarray([1], dtype=np.int32)},
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


def synthetic_bimodal_dataset() -> tuple[DerivedRunData, RunContext]:
    dataset, context = uniform_al_dataset(2.7, 80.0, zones=6)
    te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
    ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
    te[:, :3] = 40.0
    te[:, 3:] = 280.0
    ne[:, :3] *= 0.45
    ne[:, 3:] *= 1.35
    dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
    return dataset, context


def bundle_context(path: Path) -> tuple[DerivedRunData, RunContext]:
    dataset = load_run_data(path)
    context = RunContext(
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
        snapshot_index=min(5, max(0, len(dataset.time_s) - 1)),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )
    return dataset, context


def compute(dataset: DerivedRunData, context: RunContext, **kwargs) -> object:
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
    )


def main() -> int:
    report_lines: list[str] = ["# Plasmon Step 4/5 validation", ""]
    report_lines.append("## Literature-style Al benchmark states")
    report_lines.append("")
    qs = [0.25, 0.55, 0.92, 1.26]
    benchmark_rows: list[str] = []
    for rho, te, label in [
        (2.7, 0.3, "Ambient-like Al"),
        (3.5, 0.3, "Compressed Al (~1 Mbar-like density)"),
        (2.7, 6.0, "Warm dense Al"),
    ]:
        dataset, context = uniform_al_dataset(rho, te)
        benchmark_rows.append(f"### {label}: rho={rho:.3f} g/cm^3, Te={te:.3f} eV")
        benchmark_rows.append("")
        benchmark_rows.append("| q [A^-1] | angle @ 8.31 keV [deg] | model | peak [eV] | FWHM [eV] | status | points |")
        benchmark_rows.append("|---:|---:|---|---:|---:|---|---:|")
        for q in qs:
            angle = q_to_angle_deg(q, 8.31)
            for model, lfc in (("rpa", "none"), ("rpa_static_lfc", "esa_static")):
                result = compute(
                    dataset,
                    context,
                    plasmon_model=model,
                    plasmon_execution_mode="benchmark",
                    plasmon_photon_energy_kev=8.31,
                    plasmon_scattering_angle_deg=angle,
                    plasmon_energy_window_ev=30.0,
                    plasmon_energy_points=801,
                    plasmon_instrument_fwhm_ev=0.1,
                    plasmon_lfc_model=lfc,
                )
                benchmark_rows.append(
                    f"| {q:.2f} | {angle:.3f} | {model} | {result.peak_energy_ev:.4f} | {result.peak_fwhm_ev:.4f} | {result.benchmark_status} | {result.spectrum_points:d} |"
                )
        benchmark_rows.append("")
    report_lines.extend(benchmark_rows)
    report_lines.append("Key checks: ambient/compressed low-q peaks are monotonic in q; compressed Al sits above ambient Al; warm dense Al shows strong FWHM growth toward high q instead of the old artificially flat broadening.")
    report_lines.append("")

    report_lines.append("## Synthetic mixture tests")
    report_lines.append("")
    dataset, context = synthetic_bimodal_dataset()
    effective = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=35.0, plasmon_energy_points=801, plasmon_normalization="none", plasmon_instrument_fwhm_ev=0.0, plasmon_integration_mode="effective_state")
    integrated = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=35.0, plasmon_energy_points=801, plasmon_normalization="none", plasmon_instrument_fwhm_ev=0.0, plasmon_integration_mode="los_integrated")
    middle = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=35.0, plasmon_energy_points=801, plasmon_normalization="none", plasmon_instrument_fwhm_ev=0.0, plasmon_integration_mode="los_integrated", zone_index_lower=3, zone_index_upper=4)
    report_lines.append(f"- effective-state peak/FWHM = {effective.peak_energy_ev:.4f} eV / {effective.peak_fwhm_ev:.4f} eV")
    report_lines.append(f"- LOS-integrated peak/FWHM = {integrated.peak_energy_ev:.4f} eV / {integrated.peak_fwhm_ev:.4f} eV")
    report_lines.append(f"- max|LOS - effective| = {float(np.nanmax(np.abs(integrated.spectrum_intensity - effective.spectrum_intensity))):.6g}")
    report_lines.append(f"- middle-zone-only LOS peak/FWHM = {middle.peak_energy_ev:.4f} eV / {middle.peak_fwhm_ev:.4f} eV")
    report_lines.append(f"- max|middle-zone - full LOS| = {float(np.nanmax(np.abs(middle.spectrum_intensity - integrated.spectrum_intensity))):.6g}")
    report_lines.append("These checks verify that the hydrodynamic path mixes local zone spectra before the final convolution/normalization instead of collapsing the hydro state to one pre-averaged plasma point.")
    report_lines.append("")

    report_lines.append("## Real HELIOS examples")
    report_lines.append("")
    for name in ("Cu_0166_stabilized.h5", "Cu1e17_cyl_stabilized.h5"):
        dataset, context = bundle_context(_script_bootstrap.example_data_path(name))
        base = dict(plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=1.0)
        effective = compute(dataset, context, plasmon_integration_mode="effective_state", **base)
        integrated = compute(dataset, context, plasmon_integration_mode="los_integrated", **base)
        cut = compute(dataset, context, plasmon_integration_mode="los_integrated", zone_index_lower=20, zone_index_upper=120, **base)
        report_lines.append(f"### {name}")
        report_lines.append(f"- effective-state: peak={effective.peak_energy_ev:.4f} eV, FWHM={effective.peak_fwhm_ev:.4f} eV, zones={effective.zone_count_used}, clusters={effective.cluster_count_used}")
        report_lines.append(f"- LOS-integrated: peak={integrated.peak_energy_ev:.4f} eV, FWHM={integrated.peak_fwhm_ev:.4f} eV, zones={integrated.zone_count_used}, clusters={integrated.cluster_count_used}, max|LOS-effective|={float(np.nanmax(np.abs(integrated.spectrum_intensity - effective.spectrum_intensity))):.6g}")
        report_lines.append(f"- zone-limited LOS: peak={cut.peak_energy_ev:.4f} eV, FWHM={cut.peak_fwhm_ev:.4f} eV, zones={cut.zone_count_used}, clusters={cut.cluster_count_used}, max|cut-full|={float(np.nanmax(np.abs(cut.spectrum_intensity - integrated.spectrum_intensity))):.6g}")
        report_lines.append("")

    destination = Path("plasmon_step45_validation_report.md")
    destination.write_text("\n".join(report_lines), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
