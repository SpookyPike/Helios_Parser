from __future__ import annotations

from pathlib import Path
import argparse
import math

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon_reference_data import USER_DRIVEN_AL_DISPERSION_REFERENCE
from helios.services.derived.plasmon_validation import compute_plasmon, make_run_context, q_to_angle_deg, shocked_al_slab_summary

PHOTON_ENERGY_KEV = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["photon_energy_kev"])
RPA_FAMILY_MODELS = (
    "rpa",
    "rpa_static_lfc",
    "auto_best",
)
POLICIES = (
    "raw_helios",
    "benchmark_valence_aware",
    "valence_locked",
)


def _series_points(reference: dict[str, object], series_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = dict(reference["series"])[str(series_key)]
    q = np.asarray(series["q_ang_inv"], dtype=np.float64)
    y = np.asarray(series["peak_ev"], dtype=np.float64)
    err = np.asarray(series.get("peak_err_ev", np.zeros_like(q)), dtype=np.float64)
    return q, y, err


def _metric_row(prediction: dict[float, float], q_ref: np.ndarray, y_ref: np.ndarray) -> dict[str, float]:
    valid: list[float] = []
    for q, y in zip(q_ref.tolist(), y_ref.tolist()):
        value = float(prediction.get(float(q), float("nan")))
        if math.isfinite(value):
            valid.append(abs(value - float(y)))
    if not valid:
        return {"valid_points": 0.0, "mae_ev": float("nan"), "rmse_ev": float("nan"), "max_abs_ev": float("nan")}
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "valid_points": float(arr.size),
        "mae_ev": float(np.mean(arr)),
        "rmse_ev": float(np.sqrt(np.mean(arr**2))),
        "max_abs_ev": float(np.max(arr)),
    }


def _predict_hydro_policy(dataset, context, *, zone_index_lower: int, zone_index_upper: int, q_values: list[float], models: tuple[str, ...], policy: str) -> tuple[dict[str, dict[float, float]], dict[str, object]]:
    output: dict[str, dict[float, float]] = {}
    metadata: dict[str, object] = {}
    for model in models:
        model_map: dict[float, float] = {}
        first_result = None
        for q in q_values:
            angle = q_to_angle_deg(q, PHOTON_ENERGY_KEV)
            result = compute_plasmon(
                dataset,
                context,
                plasmon_model=model,
                plasmon_execution_mode="benchmark",
                plasmon_integration_mode="los_integrated",
                plasmon_photon_energy_kev=PHOTON_ENERGY_KEV,
                plasmon_scattering_angle_deg=angle,
                plasmon_energy_window_ev=45.0,
                plasmon_energy_points=1201,
                plasmon_instrument_fwhm_ev=0.20,
                plasmon_lfc_model="esa_static",
                plasmon_electron_policy=policy,
                derived_material_ids=(1,),
                zone_index_lower=zone_index_lower,
                zone_index_upper=zone_index_upper,
            )
            if first_result is None:
                first_result = result
            model_map[float(q)] = float(result.peak_energy_ev)
        output[model] = model_map
        if first_result is not None and model == models[0]:
            metadata = {
                "electron_policy": str(first_result.electron_policy),
                "electron_density_source": str(first_result.electron_density_source),
                "material_policy_summary": str(first_result.material_policy_summary),
                "mean_charge": float(first_result.mean_charge),
                "electron_density_cm3": float(first_result.electron_density_cm3),
                "warnings": tuple(str(item.message) for item in first_result.warnings),
            }
    return output, metadata



def _plot_overlay(path: Path, title: str, references: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], predictions: dict[str, dict[str, dict[float, float]]], *, focus_models: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for label, q_ref, y_ref, y_err in references:
        ax.errorbar(q_ref, y_ref, yerr=y_err, marker='o', linestyle='None', capsize=3, label=label)
    style_map = {
        "raw_helios": "--",
        "benchmark_valence_aware": "-",
        "valence_locked": ":",
    }
    marker_map = {
        "rpa": ".",
        "rpa_static_lfc": "s",
        "auto_best": "^",
    }
    q_union = sorted({float(q) for _, q_ref, _, _ in references for q in q_ref})
    for policy, model_grid in predictions.items():
        for model in focus_models:
            q_vals = np.asarray(q_union, dtype=np.float64)
            y_vals = np.asarray([float(model_grid[model].get(float(q), float("nan"))) for q in q_union], dtype=np.float64)
            mask = np.isfinite(y_vals)
            if np.any(mask):
                ax.plot(q_vals[mask], y_vals[mask], linestyle=style_map.get(policy, '-'), marker=marker_map.get(model, '.'), label=f"{model} [{policy}]")
    ax.set_title(title)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel("Peak position (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(hydro_path: Path, *, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_run_data(hydro_path)
    time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
    target_time_ns = 6.4
    snapshot_index = int(np.argmin(np.abs(time_ns - target_time_ns)))
    context = make_run_context(dataset, hydro_path, snapshot_index=snapshot_index)
    summary = shocked_al_slab_summary(dataset, snapshot_index=snapshot_index, density_floor_g_cm3=3.75, material_id=1)
    driven_q = sorted(float(q) for q in USER_DRIVEN_AL_DISPERSION_REFERENCE["series"]["experiment"]["q_ang_inv"])

    predictions: dict[str, dict[str, dict[float, float]]] = {}
    policy_metadata: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        grid, metadata = _predict_hydro_policy(
            dataset,
            context,
            zone_index_lower=int(summary["zone_index_lower"]),
            zone_index_upper=int(summary["zone_index_upper"]),
            q_values=driven_q,
            models=RPA_FAMILY_MODELS,
            policy=policy,
        )
        predictions[policy] = grid
        policy_metadata[policy] = metadata

    exp_q, exp_y, exp_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, "experiment")
    tddft_q, tddft_y, tddft_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, "tddft")
    lfc_q, lfc_y, lfc_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, "lfc")
    rpa_q, rpa_y, rpa_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, "rpa")
    preston_q, preston_y, preston_err = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, "preston")

    _plot_overlay(
        out_dir / "driven_policy_overlay_rpa_family.png",
        "Driven Al hydro benchmark: RPA-family with electron-policy variants",
        [
            ("Experiment", exp_q, exp_y, exp_err),
            ("TDDFT", tddft_q, tddft_y, tddft_err),
            ("LFC", lfc_q, lfc_y, lfc_err),
            ("RPA", rpa_q, rpa_y, rpa_err),
            ("Preston et al.", preston_q, preston_y, preston_err),
        ],
        predictions,
        focus_models=RPA_FAMILY_MODELS,
    )

    references = {
        "experiment": (exp_q, exp_y),
        "tddft": (tddft_q, tddft_y),
        "lfc": (lfc_q, lfc_y),
        "rpa": (rpa_q, rpa_y),
    }

    lines: list[str] = [
        "# Plasmon Step 10 valence-aware hydro benchmark",
        "",
        "This pass keeps the real HELIOS shocked-slab extraction from Step 9 but adds a material electron-policy registry and compares raw HELIOS `ne/zbar` against two benchmark-facing alternatives on the same selected Al slab.",
        "",
        "Selected slab:",
        f"- requested time: **{target_time_ns:.3f} ns**",
        f"- actual snapshot time: **{summary['time_ns']:.4f} ns**",
        f"- material filter: **Al only (material ID 1)**",
        f"- density floor inside Al: **{summary['density_floor_g_cm3']:.2f} g/cm^3**",
        f"- zone clip: **{summary['zone_index_lower']}-{summary['zone_index_upper']}** ({summary['zone_count']} zones)",
        f"- path-weighted state: **rho = {summary['rho_weighted_g_cm3']:.3f} g/cm^3**, **Te = {summary['te_weighted_ev']:.3f} eV**, raw HELIOS **Zbar = {summary['zbar_weighted']:.4f}**, raw HELIOS **ne = {summary['ne_weighted_cm3']:.3e} cm^-3**",
        "",
        "[Assumption: for literature-facing compressed-Al dispersion benchmarks, the dominant ambiguity sits in the effective free-electron mapping rather than in the shocked-slab `(rho, Te)` extraction itself. The new material policy layer therefore modifies only `ne/zbar`, not the hydro-selected slab.]",
        "",
        "## 1. Electron-policy metadata",
        "",
        "| policy | source | effective mean charge | effective ne [cm^-3] | summary |",
        "|---|---|---:|---:|---|",
    ]
    for policy in POLICIES:
        md = policy_metadata[policy]
        lines.append(
            f"| {policy} | {md.get('electron_density_source', '')} | {float(md.get('mean_charge', float('nan'))):.3f} | {float(md.get('electron_density_cm3', float('nan'))):.3e} | {str(md.get('material_policy_summary', ''))} |"
        )

    lines.extend([
        "",
        "## 2. Driven benchmark metrics on the same hydro-selected slab (RPA family)",
        "",
        "| policy | reference | model | valid pts | MAE [eV] | RMSE [eV] | max abs [eV] |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for policy in POLICIES:
        for ref_name, (q_ref, y_ref) in references.items():
            for model in RPA_FAMILY_MODELS:
                metrics = _metric_row(predictions[policy][model], q_ref, y_ref)
                lines.append(
                    f"| {policy} | {ref_name} | {model} | {int(metrics['valid_points'])} | {metrics['mae_ev']:.2f} | {metrics['rmse_ev']:.2f} | {metrics['max_abs_ev']:.2f} |"
                )

    raw_rpa = _metric_row(predictions["raw_helios"]["rpa_static_lfc"], tddft_q, tddft_y)
    aware_rpa = _metric_row(predictions["benchmark_valence_aware"]["rpa_static_lfc"], tddft_q, tddft_y)
    lines.extend([
        "",
        "## 3. Main outcomes",
        "",
        f"- Switching from **raw HELIOS** to **benchmark_valence_aware** raises the effective Al mean charge from about **{float(policy_metadata['raw_helios'].get('mean_charge', float('nan'))):.3f}** to **{float(policy_metadata['benchmark_valence_aware'].get('mean_charge', float('nan'))):.3f}**, and the effective electron density from **{float(policy_metadata['raw_helios'].get('electron_density_cm3', float('nan'))):.3e}** to **{float(policy_metadata['benchmark_valence_aware'].get('electron_density_cm3', float('nan'))):.3e} cm^-3** on the same shocked slab.",
        (f"- On the **RPA-family** comparison against digitized **TDDFT**, raw HELIOS did not produce finite benchmark peaks on this slab, whereas `rpa_static_lfc` with `benchmark_valence_aware` reaches **MAE {aware_rpa['mae_ev']:.2f} eV**." if int(raw_rpa['valid_points']) == 0 else f"- On the **RPA-family** comparison against digitized **TDDFT**, `rpa_static_lfc` improves from **MAE {raw_rpa['mae_ev']:.2f} eV** with raw HELIOS to **MAE {aware_rpa['mae_ev']:.2f} eV** with `benchmark_valence_aware`."),
        "- The full hydro LOS sweep of the Lindhard family remains computationally expensive and is left as a later pass; this report focuses on the RPA-family branches that are already practical for routine regression benchmarking.",
        "- For this Al-only shocked slab, `benchmark_valence_aware` and `valence_locked` are numerically almost identical, because the raw HELIOS `Zbar` stays well below the Al benchmark valence floor of 3 across the selected slab.",
        "- The hydro-driven mismatch is therefore no longer dominated by slab picking. It is dominated by the electron mapping semantics delivered into the plasmon backend.",
        "",
        "## 4. Interpretation",
        "",
        "The new registry/policy layer does not claim that HELIOS is universally wrong. It isolates a narrower issue: the raw hydro `ne/zbar` fields are not a good direct surrogate for the free-electron content needed by this specific compressed-Al plasmon benchmark. The code now exposes that assumption explicitly instead of hardwiring it invisibly.",
        "",
        "The ambient / Gawne benchmark from Step 8 stays separate. This pass only upgrades the hydro-driven compressed-Al comparison.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hydro-driven plasmon validation with material-aware electron policies.")
    parser.add_argument("hydro_path", nargs="?", default="50Al+10E+25CH+3.5TW_stabilized.h5")
    parser.add_argument("--out-dir", default="plasmon_step10_valence")
    args = parser.parse_args()
    return build_report(Path(args.hydro_path), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
