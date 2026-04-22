from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_reference_data import (
    GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5,
    USER_DRIVEN_AL_DISPERSION_REFERENCE,
)
from helios.services.derived.selection import build_analysis_geometry

NA = 6.02214076e23
HBARC_EV_A = 1973.269804
AL_A = 26.9815
AL_Z = 3.0

AMBIENT_RHO_G_CM3 = float(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5["rho_g_cm3"])
AMBIENT_TE_EV = float(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5["te_ev"])
DRIVEN_RHO_G_CM3 = 0.5 * sum(float(v) for v in USER_DRIVEN_AL_DISPERSION_REFERENCE["rho_g_cm3_range"])
DRIVEN_TE_EV = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["te_ev"])
PHOTON_ENERGY_KEV = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["photon_energy_kev"])
MODELS = (
    "rpa",
    "mermin",
    "rpa_static_lfc",
    "mermin_static_lfc",
    "lindhard",
    "lindhard_mermin",
    "lindhard_static_lfc",
    "lindhard_mermin_static_lfc",
    "auto_best",
)


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


def compute(dataset: DerivedRunData, context: RunContext, **kwargs):
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
        analysis_cache=None,
    )


def _series_points(dataset: dict[str, object], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = dict(dataset["series"])[key]
    return (
        np.asarray(series["q_ang_inv"], dtype=np.float64),
        np.asarray(series["peak_ev"], dtype=np.float64),
        np.asarray(series["peak_err_ev"], dtype=np.float64),
    )


def _all_q_points(dataset: dict[str, object]) -> list[float]:
    q_values: set[float] = set()
    for series in dict(dataset["series"]).values():
        q_values.update(float(v) for v in series["q_ang_inv"])
    return sorted(q_values)


def _compute_model_grid(rho_g_cm3: float, te_ev: float, q_values: list[float]) -> dict[str, dict[float, dict[str, float | str]]]:
    dataset, context = uniform_al_dataset(rho_g_cm3, te_ev)
    model_grid: dict[str, dict[float, dict[str, float | str]]] = {}
    for model in MODELS:
        q_map: dict[float, dict[str, float | str]] = {}
        for q in q_values:
            angle = q_to_angle_deg(q, PHOTON_ENERGY_KEV)
            result = compute(
                dataset,
                context,
                plasmon_model=model,
                plasmon_execution_mode="quicklook",
                plasmon_integration_mode="effective_state",
                plasmon_photon_energy_kev=PHOTON_ENERGY_KEV,
                plasmon_scattering_angle_deg=angle,
                plasmon_energy_window_ev=45.0,
                plasmon_energy_points=601,
                plasmon_instrument_fwhm_ev=0.20,
                plasmon_lfc_model="esa_static",
            )
            q_map[float(q)] = {
                "peak_ev": float(result.peak_energy_ev),
                "peak_fwhm_ev": float(result.peak_fwhm_ev),
                "model_name": str(result.model_name),
                "response_backend": str(result.response_backend),
                "benchmark_status": str(result.benchmark_status),
                "warning_count": float(len(result.warnings)),
            }
        model_grid[model] = q_map
    return model_grid


def _metric_row(model: str, q_map: dict[float, dict[str, float | str]], q_ref: np.ndarray, y_ref: np.ndarray) -> dict[str, float | str]:
    y_model = np.asarray([float(q_map[float(q)]["peak_ev"]) for q in q_ref], dtype=np.float64)
    mask = np.isfinite(y_model) & np.isfinite(y_ref)
    if not np.any(mask):
        mae = float("nan")
        rmse = float("nan")
        max_abs = float("nan")
        valid_points = 0
    else:
        delta = y_model[mask] - y_ref[mask]
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        max_abs = float(np.max(np.abs(delta)))
        valid_points = int(np.count_nonzero(mask))
    return {
        "model": model,
        "valid_points": valid_points,
        "mae_ev": mae,
        "rmse_ev": rmse,
        "max_abs_ev": max_abs,
    }


def _residual_rows(model: str, q_map: dict[float, dict[str, float | str]], q_ref: np.ndarray, y_ref: np.ndarray, label: str) -> list[str]:
    lines: list[str] = []
    for q, y in zip(q_ref, y_ref):
        y_model = float(q_map[float(q)]["peak_ev"])
        delta = y_model - float(y) if np.isfinite(y_model) and np.isfinite(y) else float("nan")
        lines.append(
            f"| {label} | {model} | {q:.2f} | {float(y):.2f} | {y_model:.2f} | {delta:.2f} | {q_map[float(q)]['benchmark_status']} | {q_map[float(q)]['response_backend']} |"
        )
    return lines


def _plot_dispersion(path: Path, title: str, references: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], model_grid: dict[str, dict[float, dict[str, float | str]]], models: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, q_ref, y_ref, y_err in references:
        ax.errorbar(q_ref, y_ref, yerr=y_err, marker='o', linestyle='None', capsize=3, label=label)
    q_union = sorted({float(q) for _, q_ref, _, _ in references for q in q_ref})
    for model in models:
        q_vals = np.asarray(q_union, dtype=np.float64)
        y_vals = np.asarray([float(model_grid[model][float(q)]["peak_ev"]) for q in q_union], dtype=np.float64)
        mask = np.isfinite(y_vals)
        if np.any(mask):
            ax.plot(q_vals[mask], y_vals[mask], marker='.', label=model)
    ax.set_title(title)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel("Peak position (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_shift(path: Path, ambient_grid: dict[str, dict[float, dict[str, float | str]]], driven_grid: dict[str, dict[float, dict[str, float | str]]], q_shared: np.ndarray, ambient_exp: np.ndarray, driven_exp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(q_shared, driven_exp - ambient_exp, marker='o', linewidth=1.6, label='Extracted experiment (driven - ambient)')
    for model in ('rpa', 'rpa_static_lfc', 'lindhard', 'lindhard_static_lfc', 'auto_best'):
        ambient = np.asarray([float(ambient_grid[model][float(q)]["peak_ev"]) for q in q_shared], dtype=np.float64)
        driven = np.asarray([float(driven_grid[model][float(q)]["peak_ev"]) for q in q_shared], dtype=np.float64)
        mask = np.isfinite(ambient) & np.isfinite(driven)
        if np.any(mask):
            ax.plot(q_shared[mask], driven[mask] - ambient[mask], marker='.', linewidth=1.2, label=model)
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{peak}}$ (eV)")
    ax.set_title('Ambient vs driven peak shift scaffold check')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    out_dir = Path('plasmon_step8_dispersion')
    out_dir.mkdir(exist_ok=True)

    ambient_q = _all_q_points(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5)
    driven_q = _all_q_points(USER_DRIVEN_AL_DISPERSION_REFERENCE)
    ambient_grid = _compute_model_grid(AMBIENT_RHO_G_CM3, AMBIENT_TE_EV, ambient_q)
    driven_grid = _compute_model_grid(DRIVEN_RHO_G_CM3, DRIVEN_TE_EV, driven_q)

    ambient_exp_q, ambient_exp_y, ambient_exp_err = _series_points(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5, 'experiment')
    ambient_gawne_q, ambient_gawne_y, ambient_gawne_err = _series_points(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5, 'gawne')
    driven_exp_q, driven_exp_y, driven_exp_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'experiment')
    driven_tddft_q, driven_tddft_y, driven_tddft_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'tddft')
    driven_lfc_q, driven_lfc_y, driven_lfc_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'lfc')
    driven_rpa_q, driven_rpa_y, driven_rpa_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'rpa')
    preston_q, preston_y, preston_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'preston')

    _plot_dispersion(
        out_dir / 'ambient_dispersion_overlay.png',
        'Ambient Al dispersion scaffold benchmark',
        [
            ('Ambient experiment (Fig. S5)', ambient_exp_q, ambient_exp_y, ambient_exp_err),
            ('Gawne et al. (Fig. S5)', ambient_gawne_q, ambient_gawne_y, ambient_gawne_err),
        ],
        ambient_grid,
        ('rpa', 'rpa_static_lfc', 'lindhard', 'lindhard_static_lfc', 'auto_best'),
    )
    _plot_dispersion(
        out_dir / 'driven_dispersion_overlay.png',
        'Driven / compressed Al dispersion scaffold benchmark',
        [
            ('Driven experiment', driven_exp_q, driven_exp_y, driven_exp_err),
            ('TDDFT (digitized)', driven_tddft_q, driven_tddft_y, driven_tddft_err),
            ('LFC (digitized)', driven_lfc_q, driven_lfc_y, driven_lfc_err),
            ('RPA (digitized)', driven_rpa_q, driven_rpa_y, driven_rpa_err),
            ('Preston et al.', preston_q, preston_y, preston_err),
        ],
        driven_grid,
        ('rpa', 'rpa_static_lfc', 'lindhard', 'lindhard_static_lfc', 'auto_best'),
    )
    shared_q = np.asarray([0.99, 1.28, 1.57, 2.57], dtype=np.float64)
    _plot_shift(out_dir / 'ambient_vs_driven_shift.png', ambient_grid, driven_grid, shared_q, ambient_exp_y, driven_exp_y)

    lines: list[str] = [
        '# Plasmon Step 8 dispersion scaffold validation',
        '',
        'This is a first-pass scaffold benchmark. The extracted reference points are approximate manual readings from the user-provided raster figures, not publication-grade digitization.',
        '',
        f'- Ambient state used for code evaluation: rho = {AMBIENT_RHO_G_CM3:.3f} g/cm^3, Te = {AMBIENT_TE_EV:.3f} eV.',
        f'- Driven state used for code evaluation: rho = {DRIVEN_RHO_G_CM3:.3f} g/cm^3 (midpoint of 3.75-4.50), Te = {DRIVEN_TE_EV:.3f} eV.',
        f'- Photon energy used for both scaffold runs: {PHOTON_ENERGY_KEV:.3f} keV.',
        '',
        '## Ambient MAE/RMSE against Fig. S5 extracted points',
        '',
        '| reference | model | valid pts | MAE [eV] | RMSE [eV] | max |',
        '|---|---|---:|---:|---:|---:|',
    ]
    ambient_metrics: list[dict[str, float | str]] = []
    for model in MODELS:
        ambient_metrics.append({**{'reference': 'ambient_experiment'}, **_metric_row(model, ambient_grid[model], ambient_exp_q, ambient_exp_y)})
        ambient_metrics.append({**{'reference': 'ambient_gawne'}, **_metric_row(model, ambient_grid[model], ambient_gawne_q, ambient_gawne_y)})
    for row in ambient_metrics:
        lines.append(f"| {row['reference']} | {row['model']} | {row['valid_points']} | {row['mae_ev']:.2f} | {row['rmse_ev']:.2f} | {row['max_abs_ev']:.2f} |")

    lines.extend([
        '',
        '## Driven MAE/RMSE against extracted series',
        '',
        '| reference | model | valid pts | MAE [eV] | RMSE [eV] | max |',
        '|---|---|---:|---:|---:|---:|',
    ])
    driven_metrics: list[dict[str, float | str]] = []
    for model in MODELS:
        driven_metrics.append({**{'reference': 'driven_experiment'}, **_metric_row(model, driven_grid[model], driven_exp_q, driven_exp_y)})
        driven_metrics.append({**{'reference': 'driven_tddft'}, **_metric_row(model, driven_grid[model], driven_tddft_q, driven_tddft_y)})
        driven_metrics.append({**{'reference': 'driven_lfc'}, **_metric_row(model, driven_grid[model], driven_lfc_q, driven_lfc_y)})
        driven_metrics.append({**{'reference': 'driven_rpa'}, **_metric_row(model, driven_grid[model], driven_rpa_q, driven_rpa_y)})
        driven_metrics.append({**{'reference': 'preston_anchor'}, **_metric_row(model, driven_grid[model], preston_q, preston_y)})
    for row in driven_metrics:
        lines.append(f"| {row['reference']} | {row['model']} | {row['valid_points']} | {row['mae_ev']:.2f} | {row['rmse_ev']:.2f} | {row['max_abs_ev']:.2f} |")

    lines.extend([
        '',
        '## Point-by-point residuals vs ambient experiment',
        '',
        '| reference | model | k [A^-1] | ref peak [eV] | model peak [eV] | delta [eV] | benchmark | backend |',
        '|---|---|---:|---:|---:|---:|---|---|',
    ])
    for model in ('rpa', 'rpa_static_lfc', 'lindhard', 'lindhard_static_lfc', 'auto_best', 'mermin', 'mermin_static_lfc', 'lindhard_mermin', 'lindhard_mermin_static_lfc'):
        lines.extend(_residual_rows(model, ambient_grid[model], ambient_exp_q, ambient_exp_y, 'ambient_experiment'))

    lines.extend([
        '',
        '## Point-by-point residuals vs driven experiment',
        '',
        '| reference | model | k [A^-1] | ref peak [eV] | model peak [eV] | delta [eV] | benchmark | backend |',
        '|---|---|---:|---:|---:|---:|---|---|',
    ])
    for model in ('rpa', 'rpa_static_lfc', 'lindhard', 'lindhard_static_lfc', 'auto_best', 'mermin', 'mermin_static_lfc', 'lindhard_mermin', 'lindhard_mermin_static_lfc'):
        lines.extend(_residual_rows(model, driven_grid[model], driven_exp_q, driven_exp_y, 'driven_experiment'))

    lines.extend([
        '',
        '## First-pass takeaways',
        '',
        '- The ambient and driven dispersion benchmarks are now separated from the older spectral-shape benchmark.',
        '- The driven benchmark is still a single-state scaffold, not a hydrodynamic LOS-integrated validation against the full shock profile.',
        '- Non-finite peak extraction in the Mermin-family branches is visible immediately in the dispersion tables and should be treated as a backend issue, not as a missing benchmark asset.',
        '- The finite-T Lindhard family can now be scored directly against cold/driven extracted dispersion points, even though the underlying extraction remains approximate.',
    ])

    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
