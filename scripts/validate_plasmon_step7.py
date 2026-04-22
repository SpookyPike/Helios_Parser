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
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result, refresh_analysis_result_for_snapshot
from helios.services.derived.common import load_run_data
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_reference_data import GAWNE_2024_AMBIENT_AL_REFERENCE
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry

NA = 6.02214076e23
HBARC_EV_A = 1973.269804
AL_A = 26.9815
AL_Z = 3.0
Q_POINTS = (0.25, 0.55, 0.92, 1.26)


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


def mixed_los_dataset() -> tuple[DerivedRunData, RunContext]:
    dataset, context = uniform_al_dataset(2.7, 1.0, zones=8)
    te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
    ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
    te[:, :4] = 0.3
    ne[:, :4] = 1.8e23
    te[:, 4:] = 30.0
    ne[:, 4:] = 1.0e21
    dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
    return dataset, context


def bundle_context(path: Path, *, snapshot_index: int | None = None) -> tuple[DerivedRunData, RunContext]:
    dataset = load_run_data(path)
    if snapshot_index is None:
        snapshot_index = min(20, max(0, len(dataset.time_s) - 1))
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
        snapshot_index=int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )
    return dataset, context


def compute(dataset: DerivedRunData, context: RunContext, **kwargs):
    analysis_cache = kwargs.pop("analysis_cache", None)
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


def save_overlay_plot(path: Path, title: str, ref_energy: np.ndarray, ref_intensity: np.ndarray, curves: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ref_energy, ref_intensity, marker="o", label="Digitized reference")
    for label, energy, intensity in curves:
        ax.plot(energy, intensity, label=label)
    ax.set_title(title)
    ax.set_xlabel("Energy loss [eV]")
    ax.set_ylabel("Normalized intensity [arb. u.]")
    ax.set_xlim(float(np.nanmin(ref_energy)), float(np.nanmax(ref_energy)))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_trend_plot(path: Path, title: str, xlabel: str, ylabel: str, x_values: list[float], series: list[tuple[str, list[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, values in series:
        ax.plot(x_values, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    out_dir = Path("plasmon_step7_overlay")
    out_dir.mkdir(exist_ok=True)
    lines: list[str] = [
        "# Plasmon Step 7 / 0.9.2.12 validation report",
        "",
        "## 1. Ambient Al overlays against digitized published curves",
        "",
        f"Reference source: {GAWNE_2024_AMBIENT_AL_REFERENCE['source']}",
        "",
        "The overlay below uses the staged service path with a moderate spectral grid so manual finite-T Lindhard branches remain computationally tractable in CI-like validation runs.",
        "",
        "| q [A^-1] | model | backend | peak [eV] | ref peak [eV] | peak offset [eV] | RMSE | overlay |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    photon_energy_kev = float(GAWNE_2024_AMBIENT_AL_REFERENCE["photon_energy_kev"])
    ambient_dataset, ambient_context = uniform_al_dataset(float(GAWNE_2024_AMBIENT_AL_REFERENCE["rho_g_cm3"]), float(GAWNE_2024_AMBIENT_AL_REFERENCE["te_ev"]))
    overlay_models = (
        ("rpa", "none"),
        ("rpa_static_lfc", "esa_static"),
        ("lindhard", "none"),
        ("lindhard_static_lfc", "esa_static"),
    )
    aggregate_rmse: dict[str, list[float]] = {name: [] for name, _ in overlay_models}
    aggregate_peak_offset: dict[str, list[float]] = {name: [] for name, _ in overlay_models}
    for q in Q_POINTS:
        curve = GAWNE_2024_AMBIENT_AL_REFERENCE["curves"][q]
        ref_energy = np.asarray(curve["energy_ev"], dtype=np.float64)
        ref_intensity = normalize_peak(np.asarray(curve["intensity"], dtype=np.float64))
        angle = q_to_angle_deg(float(q), photon_energy_kev)
        curves_for_plot: list[tuple[str, np.ndarray, np.ndarray]] = []
        for model, lfc in overlay_models:
            result = compute(
                ambient_dataset,
                ambient_context,
                plasmon_model=model,
                plasmon_execution_mode="quicklook",
                plasmon_photon_energy_kev=photon_energy_kev,
                plasmon_scattering_angle_deg=angle,
                plasmon_energy_window_ev=30.0,
                plasmon_energy_points=801,
                plasmon_instrument_fwhm_ev=0.1,
                plasmon_lfc_model=lfc,
            )
            model_intensity = normalize_peak(np.asarray(result.spectrum_intensity, dtype=np.float64))
            _, ref_y, model_y = interp_reference(ref_energy, ref_intensity, np.asarray(result.spectrum_energy_ev, dtype=np.float64), model_intensity)
            mask = np.isfinite(ref_y) & np.isfinite(model_y)
            rmse = float(np.sqrt(np.mean((ref_y[mask] - model_y[mask]) ** 2))) if np.any(mask) else float("nan")
            ref_peak = float(ref_energy[int(np.nanargmax(ref_intensity))])
            peak_offset = float(result.peak_energy_ev - ref_peak) if np.isfinite(result.peak_energy_ev) else float("nan")
            aggregate_rmse[model].append(rmse)
            aggregate_peak_offset[model].append(peak_offset)
            curves_for_plot.append((f"{model} [{result.response_backend}]", np.asarray(result.spectrum_energy_ev, dtype=np.float64), model_intensity))
            lines.append(f"| {q:.2f} | {model} | {result.response_backend} | {result.peak_energy_ev:.3f} | {ref_peak:.3f} | {peak_offset:.3f} | {rmse:.4f} | `ambient_q{str(q).replace('.', 'p')}.png` |")
        save_overlay_plot(out_dir / f"ambient_q{str(q).replace('.', 'p')}.png", f"Ambient Al overlay | q={q:.2f} A^-1", ref_energy, ref_intensity, curves_for_plot)
    lines.extend([
        "",
        "### Ambient overlay summary",
        "",
        "| model | mean RMSE | mean peak offset [eV] |",
        "|---|---:|---:|",
    ])
    for model in aggregate_rmse:
        lines.append(f"| {model} | {float(np.nanmean(np.asarray(aggregate_rmse[model], dtype=np.float64))):.4f} | {float(np.nanmean(np.asarray(aggregate_peak_offset[model], dtype=np.float64))):.3f} |")

    lines.extend([
        "",
        "## 2. Ambient vs compressed Al peak-shift trend",
        "",
        "Compressed state uses rho=3.5 g/cm^3, Te=0.3 eV as in the compressed-Al literature context. The table checks whether the service preserves the expected higher plasmon energy for the denser state.",
        "",
        "| q [A^-1] | model | ambient peak [eV] | compressed peak [eV] | compressed-ambient [eV] |",
        "|---:|---|---:|---:|---:|",
    ])
    trend_data: dict[str, list[float]] = {"ambient_rpa": [], "compressed_rpa": [], "ambient_lindhard": [], "compressed_lindhard": []}
    compressed_dataset, compressed_context = uniform_al_dataset(3.5, 0.3)
    for q in Q_POINTS:
        angle = q_to_angle_deg(float(q), photon_energy_kev)
        for model, label in (("rpa", "rpa"), ("lindhard", "lindhard")):
            ambient = compute(ambient_dataset, ambient_context, plasmon_model=model, plasmon_execution_mode="quicklook", plasmon_photon_energy_kev=photon_energy_kev, plasmon_scattering_angle_deg=angle, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=0.1)
            compressed = compute(compressed_dataset, compressed_context, plasmon_model=model, plasmon_execution_mode="quicklook", plasmon_photon_energy_kev=photon_energy_kev, plasmon_scattering_angle_deg=angle, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=0.1)
            trend_data[f"ambient_{label}"].append(float(ambient.peak_energy_ev))
            trend_data[f"compressed_{label}"].append(float(compressed.peak_energy_ev))
            lines.append(f"| {q:.2f} | {model} | {ambient.peak_energy_ev:.3f} | {compressed.peak_energy_ev:.3f} | {compressed.peak_energy_ev - ambient.peak_energy_ev:.3f} |")
    save_trend_plot(out_dir / "ambient_vs_compressed_peak_shift.png", "Ambient vs compressed Al peak shift", "q [A^-1]", "Peak energy [eV]", list(Q_POINTS), [("RPA ambient", trend_data["ambient_rpa"]), ("RPA compressed", trend_data["compressed_rpa"]), ("Lindhard ambient", trend_data["ambient_lindhard"]), ("Lindhard compressed", trend_data["compressed_lindhard"])])

    lines.extend([
        "",
        "## 3. Warm Al linewidth / damping trend",
        "",
        "Warm state uses rho=2.7 g/cm^3, Te=6 eV. This table checks whether linewidth broadening grows toward larger q where Landau/continuum damping should become more important.",
        "",
        "| q [A^-1] | model | peak [eV] | FWHM [eV] |",
        "|---:|---|---:|---:|",
    ])
    warm_dataset, warm_context = uniform_al_dataset(2.7, 6.0)
    warm_rpa_fwhm: list[float] = []
    warm_lindhard_fwhm: list[float] = []
    for q in Q_POINTS:
        angle = q_to_angle_deg(float(q), photon_energy_kev)
        for model in ("rpa", "lindhard"):
            result = compute(warm_dataset, warm_context, plasmon_model=model, plasmon_execution_mode="quicklook", plasmon_photon_energy_kev=photon_energy_kev, plasmon_scattering_angle_deg=angle, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=0.1)
            lines.append(f"| {q:.2f} | {model} | {result.peak_energy_ev:.3f} | {result.peak_fwhm_ev:.3f} |")
            if model == "rpa":
                warm_rpa_fwhm.append(float(result.peak_fwhm_ev))
            else:
                warm_lindhard_fwhm.append(float(result.peak_fwhm_ev))
    save_trend_plot(out_dir / "warm_al_fwhm_vs_q.png", "Warm Al linewidth trend", "q [A^-1]", "FWHM [eV]", list(Q_POINTS), [("RPA", warm_rpa_fwhm), ("Lindhard", warm_lindhard_fwhm)])

    lines.extend([
        "",
        "## 4. Real-hydrodynamic LOS semantics and zone filtering",
        "",
        "| dataset | auto-best quicklook backend | auto summary | max|LOS-effective| | max|filtered-full LOS| | benchmark status @ small angle | benchmark status @ 20 deg |",
        "|---|---|---|---:|---:|---|---|",
    ])
    for name, good_angle in (("Cu_0166_stabilized.h5", 2.0), ("Cu1e17_cyl_stabilized.h5", 3.0)):
        dataset, context = bundle_context(_script_bootstrap.example_data_path(name), snapshot_index=20)
        effective = compute(dataset, context, plasmon_model="auto_best", plasmon_execution_mode="quicklook", plasmon_integration_mode="effective_state", plasmon_energy_window_ev=60.0, plasmon_energy_points=601, plasmon_instrument_fwhm_ev=1.0)
        los_full = compute(dataset, context, plasmon_model="auto_best", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_energy_window_ev=60.0, plasmon_energy_points=601, plasmon_instrument_fwhm_ev=1.0)
        zmax = max(12, int(np.asarray(dataset.static_x_cm, dtype=np.float64).size // 2))
        los_cut = compute(dataset, context, plasmon_model="auto_best", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_energy_window_ev=60.0, plasmon_energy_points=601, plasmon_instrument_fwhm_ev=1.0, zone_index_lower=10, zone_index_upper=zmax)
        benchmark_good = compute(dataset, context, plasmon_model="auto_best", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=good_angle, plasmon_energy_window_ev=30.0, plasmon_energy_points=401, plasmon_instrument_fwhm_ev=0.8)
        benchmark_bad = compute(dataset, context, plasmon_model="auto_best", plasmon_execution_mode="benchmark", plasmon_integration_mode="los_integrated", plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=401, plasmon_instrument_fwhm_ev=0.8)
        n = min(effective.spectrum_intensity.size, los_full.spectrum_intensity.size)
        los_vs_eff = float(np.nanmax(np.abs(los_full.spectrum_intensity[:n] - effective.spectrum_intensity[:n]))) if n > 0 else float("nan")
        n2 = min(los_full.spectrum_intensity.size, los_cut.spectrum_intensity.size)
        los_cut_vs_full = float(np.nanmax(np.abs(los_cut.spectrum_intensity[:n2] - los_full.spectrum_intensity[:n2]))) if n2 > 0 else float("nan")
        lines.append(f"| {name} | {los_full.response_backend} | {los_full.auto_model_summary or '-'} | {los_vs_eff:.6f} | {los_cut_vs_full:.6f} | {benchmark_good.benchmark_status} | {benchmark_bad.benchmark_status} |")

    lines.extend([
        "",
        "## 5. Snapshot refresh, lazy service reuse, and cache buckets",
        "",
    ])
    dataset, context = bundle_context(_script_bootstrap.example_data_path("Cu_0166_stabilized.h5"), snapshot_index=20)
    params = DerivedAnalysisParameters(plasmon_model="auto_best", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_energy_window_ev=40.0, plasmon_energy_points=401, plasmon_instrument_fwhm_ev=0.8)
    base = compute_analysis_result(dataset, context, parameters=params, context_key=("step7", "base"), requested_time_plot_modules=frozenset())
    updated_context = context.copy()
    updated_context.set_snapshot_index(40)
    refreshed = refresh_analysis_result_for_snapshot(dataset, updated_context, parameters=params, context_key=("step7", "refresh"), base_result=base)
    base_profile = np.asarray(base.plasmon.profile_plots[1].y_series[0], dtype=np.float64)
    refreshed_profile = np.asarray(refreshed.plasmon.profile_plots[1].y_series[0], dtype=np.float64)
    finite = np.isfinite(base_profile) & np.isfinite(refreshed_profile)
    profile_delta = float(np.nanmax(np.abs(base_profile[finite] - refreshed_profile[finite]))) if np.any(finite) else float("nan")
    cache = AnalysisStateCache()
    first = compute(dataset, context, plasmon_model="lindhard", plasmon_execution_mode="quicklook", plasmon_integration_mode="effective_state", plasmon_energy_window_ev=20.0, plasmon_energy_points=301, plasmon_instrument_fwhm_ev=0.5, analysis_cache=cache)
    second = compute(dataset, context, plasmon_model="lindhard", plasmon_execution_mode="quicklook", plasmon_integration_mode="effective_state", plasmon_energy_window_ev=20.0, plasmon_energy_points=301, plasmon_instrument_fwhm_ev=0.5, analysis_cache=cache)
    stats = cache.stats()
    auto_dataset, auto_context = mixed_los_dataset()
    auto_mix = compute(auto_dataset, auto_context, plasmon_model="auto_best", plasmon_execution_mode="quicklook", plasmon_integration_mode="los_integrated", plasmon_energy_window_ev=24.0, plasmon_energy_points=401, plasmon_instrument_fwhm_ev=0.4, plasmon_cluster_log_ne_tol=0.25, plasmon_cluster_log_te_tol=0.25, plasmon_cluster_z_tol=0.25, plasmon_lfc_model="esa_static", plasmon_collision_model="manual_constant", plasmon_manual_collision_rate_s=1.0e15)
    lines.extend([
        f"- Snapshot refresh check: base snapshot={base.snapshot_index}, refreshed snapshot={refreshed.snapshot_index}, reused shock object={refreshed.shock is base.shock}, profile delta={profile_delta:.6f}.",
        f"- Lindhard cache bucket check: identical repeated request produced time_series_hits={int(stats['time_series_hits'])}, time_series_misses={int(stats['time_series_misses'])}, spectra_equal={bool(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))}.",
        f"- Auto-best mixed LOS check: backend={auto_mix.response_backend}, auto summary=`{auto_mix.auto_model_summary}`.",
        "",
        "## 6. Interpretation",
        "",
        "- Manual finite-T Lindhard branches are now wired into the same service/UI/cache stack as the older models, but the ambient-Al overlay shows they are not yet closer to the digitized low-q reference curves than the older staged RPA baseline.",
        "- Auto Best therefore stays conservative and uses the strongest validated local classical branch per state/cluster instead of silently auto-promoting the still-experimental Lindhard backend.",
        "- Real-hydrodynamic checks still show that LOS spectra do not collapse to one pre-averaged state and that left-panel zone exclusion changes the integrated spectrum.",
    ])
    report_path = Path("plasmon_step7_validation_report.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
