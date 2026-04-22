"""Step 6 plasmon validation: benchmark hardening and literature overlays."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_reference_data import GAWNE_2024_AMBIENT_AL_REFERENCE
from helios.services.derived.selection import build_analysis_geometry

NA = 6.02214076e23
HBARC_EV_A = 1973.269804
AL_A = 26.9815
AL_Z = 3.0


def q_to_angle_deg(q_ang_inv: float, photon_energy_kev: float) -> float:
    argument = float(q_ang_inv) * HBARC_EV_A / (2.0 * float(photon_energy_kev) * 1000.0)
    argument = min(max(argument, -1.0), 1.0)
    return math.degrees(2.0 * math.asin(argument))


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
    regions = {"region_index": np.asarray([1], dtype=np.int32), "min_zone_index": np.asarray([1], dtype=np.int32), "max_zone_index": np.asarray([zones], dtype=np.int32), "atomic_weight": np.asarray([AL_A], dtype=np.float64), "initial_mass_density": np.asarray([rho_g_cm3], dtype=np.float64), "initial_temperature": np.asarray([te_ev], dtype=np.float64)}
    dataset = DerivedRunData(path=Path("uniform_al.h5"), summary={"n_zones": zones, "n_snapshots": n_snapshots}, metadata={"geometry": "PLANAR", "coordinate_model": {"coordinate_name": "x"}}, regions=regions, materials={"index": np.asarray([1], dtype=np.int32)}, time_s=time_s, static_x_cm=static_x, static_x_edge_cm=static_x_edges, zone_width_cm=zone_width, density_g_cm3=density, velocity_cm_s=velocity, temperature_e_ev=temperature_e, temperature_i_ev=temperature_i, temperature_radiation_ev=None, electron_density_cm3=electron_density, mean_charge=mean_charge, radius_cm=None, radius_edge_cm=None, zone_region_id=zone_region_id, zone_material_index=zone_material, zone_atomic_weight=np.full(zones, AL_A, dtype=np.float64), zone_initial_density_g_cm3=np.full(zones, rho_g_cm3, dtype=np.float64), zone_initial_temperature_ev=np.full(zones, te_ev, dtype=np.float64), laser_entry=None)
    context = RunContext(path=Path("uniform_al.h5"), summary={"n_zones": zones, "n_snapshots": n_snapshots}, metadata={}, fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"), diagnostics=(), time_values=time_s.copy(), static_x_values=static_x.copy(), zone_region_id=zone_region_id.copy(), zone_material_index=zone_material.copy(), has_dynamic_radius=False, snapshot_index=0, map_coordinate="static_x", slice_coordinate="zone", selected_region_ids=(1,), selected_material_ids=(1,))
    return dataset, context


def hardening_dataset() -> tuple[DerivedRunData, RunContext]:
    dataset, context = uniform_al_dataset(2.7, 0.3, zones=10)
    density = np.asarray(dataset.density_g_cm3, dtype=np.float64).copy()
    temperature_e = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
    temperature_i = np.asarray(dataset.temperature_i_ev, dtype=np.float64).copy()
    electron_density = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
    density[:, :2] = 0.01
    temperature_e[:, :2] = 200.0
    temperature_i[:, :2] = 200.0
    electron_density[:, :2] = electron_density[:, :2] * (0.01 / 2.7)
    return replace(dataset, density_g_cm3=density, temperature_e_ev=temperature_e, temperature_i_ev=temperature_i, electron_density_cm3=electron_density), context


def bundle_context(path: Path) -> tuple[DerivedRunData, RunContext]:
    dataset = load_run_data(path)
    context = RunContext(path=path, summary=dict(dataset.summary), metadata=dict(dataset.metadata), fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"), diagnostics=(), time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(), static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(), zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(), zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(), has_dynamic_radius=dataset.radius_cm is not None, snapshot_index=min(5, max(0, len(dataset.time_s) - 1)), map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x", slice_coordinate="zone", selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))), selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))))
    return dataset, context


def compute(dataset: DerivedRunData, context: RunContext, **kwargs):
    params = DerivedAnalysisParameters(**kwargs)
    geometry = build_analysis_geometry(dataset, context, observation_side=params.observation_side, line_of_sight_angle_deg=params.line_of_sight_angle_deg, line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm, profile_coordinate_mode=params.profile_coordinate_mode)
    return evaluate_plasmon_regime(dataset, context, snapshot_index=context.snapshot_index, photon_energy_kev=params.plasmon_photon_energy_kev, scattering_angle_deg=params.plasmon_scattering_angle_deg, adiabatic_index=params.plasmon_adiabatic_index, parameters=params, geometry=geometry, include_time_plots=False)


def normalize_peak(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    peak = float(np.nanmax(arr)) if arr.size else float("nan")
    if not np.isfinite(peak) or peak <= 0.0:
        return arr
    return arr / peak


def interp_reference(ref_energy: np.ndarray, ref_intensity: np.ndarray, model_energy: np.ndarray, model_intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(model_energy) & np.isfinite(model_intensity)
    x = model_energy[mask]
    y = model_intensity[mask]
    if x.size < 2:
        return ref_energy, ref_intensity, np.full_like(ref_energy, np.nan)
    y_interp = np.interp(ref_energy, x, y, left=np.nan, right=np.nan)
    return ref_energy, ref_intensity, y_interp


def main() -> int:
    out_dir = Path("plasmon_step6_overlay")
    out_dir.mkdir(exist_ok=True)
    lines: list[str] = ["# Plasmon Step 6 validation", "", "## 1. Hard benchmark domain policing", ""]
    dataset, context = hardening_dataset()
    full = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=0.1)
    filtered = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=0.1, zone_index_lower=3, zone_index_upper=10)
    lines.append(f"- Full synthetic LOS selection: status={full.benchmark_status}, noncollective z/c={full.noncollective_zone_count}/{full.noncollective_cluster_count}, domain-fail={100.0*full.domain_failure_fraction:.1f}%")
    lines.append(f"- Filtered inner-zone LOS selection: status={filtered.benchmark_status}, noncollective z/c={filtered.noncollective_zone_count}/{filtered.noncollective_cluster_count}, domain-fail={100.0*filtered.domain_failure_fraction:.1f}%")
    lines.append("This demonstrates that benchmark mode now polices the active zone subset before cluster mixing, and that excluding bad edge zones in the left-panel range controls can recover a valid benchmark path.")
    lines.append("")
    lines.append("## 2. Overlay against coarse digitized published ambient-Al curves")
    lines.append("")
    lines.append(f"Reference source: {GAWNE_2024_AMBIENT_AL_REFERENCE['source']}")
    lines.append("")
    lines.append("| q [A^-1] | model | peak [eV] | reference peak [eV] | peak offset [eV] | RMSE | overlay |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    dataset, context = uniform_al_dataset(float(GAWNE_2024_AMBIENT_AL_REFERENCE['rho_g_cm3']), float(GAWNE_2024_AMBIENT_AL_REFERENCE['te_ev']))
    photon_energy_kev = float(GAWNE_2024_AMBIENT_AL_REFERENCE['photon_energy_kev'])
    for q, curve in GAWNE_2024_AMBIENT_AL_REFERENCE['curves'].items():
        ref_energy = np.asarray(curve['energy_ev'], dtype=np.float64)
        ref_intensity = normalize_peak(np.asarray(curve['intensity'], dtype=np.float64))
        angle = q_to_angle_deg(float(q), photon_energy_kev)
        for model, lfc in (("rpa", "none"), ("rpa_static_lfc", "esa_static")):
            result = compute(dataset, context, plasmon_model=model, plasmon_execution_mode="benchmark", plasmon_photon_energy_kev=photon_energy_kev, plasmon_scattering_angle_deg=angle, plasmon_energy_window_ev=30.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=0.1, plasmon_lfc_model=lfc)
            model_intensity = normalize_peak(np.asarray(result.spectrum_intensity, dtype=np.float64))
            _, ref_y, model_y = interp_reference(ref_energy, ref_intensity, np.asarray(result.spectrum_energy_ev, dtype=np.float64), model_intensity)
            mask = np.isfinite(ref_y) & np.isfinite(model_y)
            rmse = float(np.sqrt(np.mean((ref_y[mask] - model_y[mask]) ** 2))) if np.any(mask) else float("nan")
            ref_peak = float(ref_energy[int(np.nanargmax(ref_intensity))])
            peak_offset = float(result.peak_energy_ev - ref_peak) if np.isfinite(result.peak_energy_ev) else float("nan")
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(ref_energy, ref_intensity, marker="o", label="Digitized reference")
            ax.plot(result.spectrum_energy_ev, model_intensity, label=f"{model} ({result.benchmark_status})")
            ax.set_title(f"Ambient Al overlay | q={q:.2f} A^-1 | {model}")
            ax.set_xlabel("Energy loss [eV]")
            ax.set_ylabel("Normalized intensity [arb. u.]")
            ax.set_xlim(5.0, 25.0)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fname = f"overlay_q{str(q).replace('.', 'p')}_{model}.png"
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=160)
            plt.close(fig)
            lines.append(f"| {q:.2f} | {model} | {result.peak_energy_ev:.3f} | {ref_peak:.3f} | {peak_offset:.3f} | {rmse:.4f} | `{fname}` |")
    lines.append("")
    lines.append("The overlay uses coarse hand-digitized points from the published supplemental figure. It is suitable for staged paper-facing comparisons, but not as a substitute for a full source-data benchmark or a finite-T Lindhard forward model.")
    lines.append("")
    lines.append("## 3. Real HELIOS hydro sanity checks")
    lines.append("")
    lines.append("| dataset | benchmark angle [deg] | benchmark status | noncollective z/c | degenerate z/c | quicklook max|LOS-effective| | quicklook max|filtered-full LOS| |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for name, benchmark_angle in (("Cu_0166_stabilized.h5", 2.0), ("Cu1e17_cyl_stabilized.h5", 3.0)):
        dataset, context = bundle_context(_script_bootstrap.example_data_path(name))
        benchmark = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=benchmark_angle, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0)
        benchmark_bad = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0)
        effective = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="quicklook", plasmon_integration_mode="effective_state", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0)
        los_full = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0)
        zmax = max(10, int(np.asarray(dataset.static_x_cm, dtype=np.float64).size // 2))
        los_cut = compute(dataset, context, plasmon_model="rpa", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0, zone_index_lower=10, zone_index_upper=zmax)
        n = min(effective.spectrum_intensity.size, los_full.spectrum_intensity.size)
        los_vs_eff = float(np.nanmax(np.abs(los_full.spectrum_intensity[:n] - effective.spectrum_intensity[:n]))) if n > 0 else float("nan")
        n2 = min(los_full.spectrum_intensity.size, los_cut.spectrum_intensity.size)
        los_cut_vs_full = float(np.nanmax(np.abs(los_cut.spectrum_intensity[:n2] - los_full.spectrum_intensity[:n2]))) if n2 > 0 else float("nan")
        lines.append(f"| {name} | {benchmark_angle:.1f} | {benchmark.benchmark_status} (20° -> {benchmark_bad.benchmark_status}) | {benchmark.noncollective_zone_count}/{benchmark.noncollective_cluster_count} | {benchmark.degenerate_zone_count}/{benchmark.degenerate_cluster_count} | {los_vs_eff:.6f} | {los_cut_vs_full:.6f} |")
    lines.append("")
    lines.append("The quicklook hydro comparisons above are not benchmark claims; they are plumbing checks showing that LOS mixing does not collapse to a single pre-averaged state and that left-panel zone exclusion changes the integrated spectrum on real datasets.")
    lines.append("")
    lines.append("For publication-grade forward modelling in dense/degenerate metals, the next real physics step is still a finite-T Lindhard / stronger response backend. Step 6 hardening makes the current benchmark path honest; it does not magically promote the classical baseline to a final-reference model.")
    Path("plasmon_step6_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(Path("plasmon_step6_validation_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
