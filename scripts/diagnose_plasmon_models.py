"""Generate temporary diagnostic plots for all plasmon models on bundled examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.selection import build_analysis_geometry

MODELS = ('rpa', 'mermin', 'rpa_static_lfc', 'mermin_static_lfc')
ANGLES = (10.0, 20.0, 30.0, 40.0)


def bundle_context(path: Path):
    dataset = load_run_data(path)
    context = RunContext(
        path=path, summary=dict(dataset.summary), metadata=dict(dataset.metadata),
        fields=('density','velocity','temperature_e','temperature_i','electron_density','mean_charge'), diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(), static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(), zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None, snapshot_index=min(5, max(0, len(dataset.time_s)-1)), map_coordinate='moving_radius' if dataset.radius_cm is not None else 'static_x', slice_coordinate='zone',
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )
    return dataset, context


def compute(dataset, context, model: str, angle_deg: float, *, integration_mode: str, points: int):
    params = DerivedAnalysisParameters(
        plasmon_model=model,
        plasmon_photon_energy_kev=7.5,
        plasmon_scattering_angle_deg=angle_deg,
        plasmon_energy_window_ev=100.0,
        plasmon_energy_points=int(points),
        plasmon_instrument_fwhm_ev=1.0,
        plasmon_collision_model='nrl_constant',
        plasmon_lfc_model='esa_static',
        plasmon_integration_mode=str(integration_mode),
    )
    geometry = build_analysis_geometry(dataset, context, observation_side=params.observation_side, line_of_sight_angle_deg=params.line_of_sight_angle_deg, line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm, profile_coordinate_mode=params.profile_coordinate_mode)
    return evaluate_plasmon_regime(dataset, context, snapshot_index=context.snapshot_index, photon_energy_kev=params.plasmon_photon_energy_kev, scattering_angle_deg=params.plasmon_scattering_angle_deg, adiabatic_index=params.plasmon_adiabatic_index, parameters=params, geometry=geometry, include_time_plots=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--outdir', type=Path, default=Path('plasmon_phase8_diagnostics'))
    parser.add_argument('--integration-mode', choices=('effective_state','los_integrated'), default='effective_state')
    parser.add_argument('--points', type=int, default=201)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    examples = [_script_bootstrap.example_data_path(name, root=root) for name in ('5Fe+4.9TW+light_stabilized.h5','Cu1e17_cyl_stabilized.h5','Cu_0166_stabilized.h5')]
    summary_rows = []
    for example in examples:
        dataset, context = bundle_context(example)
        peak_rows = []
        width_rows = []
        for angle in ANGLES:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model in MODELS:
                result = compute(dataset, context, model, angle, integration_mode=args.integration_mode, points=args.points)
                ax.plot(result.spectrum_energy_ev, result.spectrum_intensity, label=f'{model} -> {result.model_name} | {result.benchmark_status}')
                summary_rows.append({
                    'example': example.name, 'angle_deg': angle, 'requested_model': model, 'applied_model': result.model_name,
                    'peak_energy_ev': float(result.peak_energy_ev), 'peak_fwhm_ev': float(result.peak_fwhm_ev),
                    'zones': int(result.zone_count_used), 'clusters': int(result.cluster_count_used),
                    'benchmark_status': result.benchmark_status, 'model_executed_fully': bool(result.model_executed_fully), 'fallback_fraction': float(result.fallback_fraction),
                    'warning_count': len(result.warnings),
                    'warnings': ' | '.join(str(w.message) for w in result.warnings[:4]),
                })
                peak_rows.append((model, float(result.peak_energy_ev), result.model_name))
                width_rows.append((model, float(result.peak_fwhm_ev), result.model_name))
            ax.set_title(f'{example.name} | 7.5 keV | {angle:.0f} deg')
            ax.set_xlabel('Energy transfer [eV]')
            ax.set_ylabel('Observed intensity [arb. u.]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f'{example.stem}_angle_{int(angle):02d}_spectra.png', dpi=160)
            plt.close(fig)
        # peak vs angle
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in MODELS:
            rows = [r for r in summary_rows if r['example'] == example.name and r['requested_model'] == model]
            ax.plot([r['angle_deg'] for r in rows], [r['peak_energy_ev'] for r in rows], marker='o', label=model)
        ax.set_title(f'{example.name} | peak shift vs angle')
        ax.set_xlabel('Scattering angle [deg]')
        ax.set_ylabel('Peak energy [eV]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f'{example.stem}_peak_vs_angle.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in MODELS:
            rows = [r for r in summary_rows if r['example'] == example.name and r['requested_model'] == model]
            ax.plot([r['angle_deg'] for r in rows], [r['peak_fwhm_ev'] for r in rows], marker='o', label=model)
        ax.set_title(f'{example.name} | FWHM vs angle')
        ax.set_xlabel('Scattering angle [deg]')
        ax.set_ylabel('Peak FWHM [eV]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f'{example.stem}_fwhm_vs_angle.png', dpi=160)
        plt.close(fig)
    csv_path = outdir / 'plasmon_phase8_summary.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    md_path = outdir / 'README.md'
    lines = ['# Plasmon phase 8 diagnostics', '', 'These diagnostics are temporary and are not intended to ship in the release bundle.', '', '## Generated files', '- one spectrum overlay plot per example/angle', '- peak-vs-angle plots', '- FWHM-vs-angle plots', '- summary CSV']
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
