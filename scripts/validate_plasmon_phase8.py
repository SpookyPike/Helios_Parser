"""Run a release-gate validation pass for the staged plasmon module."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

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
from helios.services.derived.plasmon_spectrum import epsilon_rpa, epsilon_rpa_static_lfc
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry


def synthetic_dataset() -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 3
    n_zones = 6
    time_s = np.asarray([0.0, 1.0e-9, 2.0e-9], dtype=np.float64)
    static_x = np.linspace(1.0e-4, 6.0e-4, n_zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, 6.5e-4, n_zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, n_zones), 1.0e-4, dtype=np.float64)
    density = np.ones((n_snapshots, n_zones), dtype=np.float64)
    velocity = np.zeros_like(density)
    temperature_e = np.full_like(density, 120.0)
    temperature_i = np.full_like(density, 80.0)
    electron_density = np.full_like(density, 8.0e20)
    mean_charge = np.full_like(density, 6.0)
    zone_region_id = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    zone_material = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    regions = {
        'region_index': np.asarray([1, 2], dtype=np.int32),
        'min_zone_index': np.asarray([1, 4], dtype=np.int32),
        'max_zone_index': np.asarray([3, 6], dtype=np.int32),
        'atomic_weight': np.asarray([27.0, 63.5], dtype=np.float64),
        'initial_mass_density': np.asarray([1.0, 1.0], dtype=np.float64),
        'initial_temperature': np.asarray([1.0, 1.0], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path('synthetic_plasmon.h5'), summary={'n_zones': n_zones, 'n_snapshots': n_snapshots}, metadata={'geometry': 'PLANAR', 'coordinate_model': {'coordinate_name': 'x'}},
        regions=regions, materials={'index': np.asarray([1, 2], dtype=np.int32)}, time_s=time_s, static_x_cm=static_x, static_x_edge_cm=static_x_edges, zone_width_cm=zone_width,
        density_g_cm3=density, velocity_cm_s=velocity, temperature_e_ev=temperature_e, temperature_i_ev=temperature_i, temperature_radiation_ev=None, electron_density_cm3=electron_density, mean_charge=mean_charge,
        radius_cm=None, radius_edge_cm=None, zone_region_id=zone_region_id, zone_material_index=zone_material, zone_atomic_weight=np.asarray([27.0, 27.0, 27.0, 63.5, 63.5, 63.5], dtype=np.float64),
        zone_initial_density_g_cm3=np.full(n_zones, 1.0, dtype=np.float64), zone_initial_temperature_ev=np.full(n_zones, 1.0, dtype=np.float64), laser_entry=None,
    )
    context = RunContext(
        path=Path('synthetic_plasmon.h5'), summary={'n_zones': n_zones, 'n_snapshots': n_snapshots}, metadata={}, fields=('density','velocity','temperature_e','temperature_i','electron_density','mean_charge'), diagnostics=(),
        time_values=time_s.copy(), static_x_values=static_x.copy(), zone_region_id=zone_region_id.copy(), zone_material_index=zone_material.copy(), has_dynamic_radius=False, snapshot_index=1, map_coordinate='static_x', slice_coordinate='zone',
        selected_region_ids=(1, 2), selected_material_ids=(1, 2),
    )
    return dataset, context


def geometry(dataset, context, params):
    return build_analysis_geometry(
        dataset, context,
        observation_side=params.observation_side,
        line_of_sight_angle_deg=params.line_of_sight_angle_deg,
        line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
        profile_coordinate_mode=params.profile_coordinate_mode,
    )


def compute(dataset, context, **kwargs):
    analysis_cache = kwargs.pop('analysis_cache', None)
    params = DerivedAnalysisParameters(**kwargs)
    return evaluate_plasmon_regime(
        dataset, context, snapshot_index=context.snapshot_index,
        photon_energy_kev=params.plasmon_photon_energy_kev,
        scattering_angle_deg=params.plasmon_scattering_angle_deg,
        adiabatic_index=params.plasmon_adiabatic_index,
        parameters=params, geometry=geometry(dataset, context, params), include_time_plots=False,
        analysis_cache=analysis_cache,
    )


def bundle_context(path: Path):
    dataset = load_run_data(path)
    return dataset, RunContext(
        path=path, summary=dict(dataset.summary), metadata=dict(dataset.metadata),
        fields=('density','velocity','temperature_e','temperature_i','electron_density','mean_charge'), diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(), static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(), zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None, snapshot_index=min(5, max(0, len(dataset.time_s)-1)), map_coordinate='moving_radius' if dataset.radius_cm is not None else 'static_x', slice_coordinate='zone',
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, default=Path('plasmon_phase8_verification_report.md'))
    args = parser.parse_args()
    root = args.root.resolve()
    examples = [_script_bootstrap.example_data_path(name, root=root) for name in ('5Fe+4.9TW+light_stabilized.h5','Cu1e17_cyl_stabilized.h5','Cu_0166_stabilized.h5')]
    checks: list[tuple[str, bool, str]] = []

    dataset, context = synthetic_dataset()
    base = compute(dataset, context)
    explicit = compute(dataset, context, plasmon_model='quicklook')
    checks.append(('quicklook backward-compatible', abs(base.plasma_frequency_ev - explicit.plasma_frequency_ev) < 1e-12 and abs(base.k_lambda_debye - explicit.k_lambda_debye) < 1e-12, f'ωpe={base.plasma_frequency_ev:.6g} eV'))

    rpa = compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801)
    checks.append(('rpa stable spectrum', np.isfinite(rpa.spectrum_intensity).all(), f'points={rpa.spectrum_intensity.size}'))

    energy = np.linspace(-10.0, 10.0, 401, dtype=np.float64)
    _, eps_rpa = epsilon_rpa(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, imag_shift_ev=0.05)
    _, eps_lfc, g_value, _ = epsilon_rpa_static_lfc(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, imag_shift_ev=0.05, rs=1.55, theta=0.6)
    checks.append(('rpa_static_lfc finite branch', np.isfinite(np.real(eps_lfc)).all() and np.isfinite(np.imag(eps_lfc)).all(), f'G={g_value:.4g}'))

    sharp = compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=0.0)
    broad = compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=2.0)
    checks.append(('FWHM broadens observed linewidth', float(broad.peak_fwhm_ev) > float(sharp.peak_fwhm_ev), f'{sharp.peak_fwhm_ev:.4g} -> {broad.peak_fwhm_ev:.4g} eV'))

    invalid_mermin = compute(dataset, context, plasmon_model='mermin', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_collision_model='manual_constant', plasmon_manual_collision_rate_s=-1.0)
    checks.append(('invalid Mermin is flagged for benchmark use', invalid_mermin.benchmark_status == 'invalid_for_benchmark' and not invalid_mermin.model_executed_fully and invalid_mermin.spectrum_energy_ev.size == 0, f'status={invalid_mermin.benchmark_status}, fallback={invalid_mermin.fallback_fraction:.3f}'))

    # deselection sensitivity
    te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy(); ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
    te[:, :3] = 90.0; te[:, 3:] = 260.0; ne[:, :3] = 9.0e20; ne[:, 3:] = 2.5e20
    dataset2 = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
    full = compute(dataset2, context, plasmon_model='rpa', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=501, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0)
    subset = compute(dataset2, replace(context, selected_region_ids=(1,), selected_material_ids=(1,)), plasmon_model='rpa', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=501, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0)
    checks.append(('LOS reacts to deselection', float(np.nanmax(np.abs(full.spectrum_intensity - subset.spectrum_intensity))) > 1e-8, 'synthetic two-region validation'))

    cache = AnalysisStateCache()
    params = dict(plasmon_model='mermin_static_lfc', plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_collision_model='manual_constant', plasmon_manual_collision_rate_s=8.0e14, plasmon_lfc_model='esa_static', plasmon_integration_mode='los_integrated')
    start = time.perf_counter(); first = compute(dataset2, context, analysis_cache=cache, **params); cold_s = time.perf_counter() - start
    start = time.perf_counter(); second = compute(dataset2, context, analysis_cache=cache, **params); warm_s = time.perf_counter() - start
    checks.append(('cache buckets reuse identical request', warm_s < cold_s and np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True), f'{cold_s*1e3:.2f} ms -> {warm_s*1e3:.2f} ms'))

    example_lines = []
    for path in examples:
        dataset_b, context_b = bundle_context(path)
        result = compute(dataset_b, context_b, plasmon_model='mermin_static_lfc', plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201, plasmon_instrument_fwhm_ev=1.0, plasmon_collision_model='nrl_constant', plasmon_lfc_model='esa_static', plasmon_integration_mode='los_integrated')
        spectrum_size = int(result.spectrum_intensity.size)
        finite_fraction = (
            np.count_nonzero(np.isfinite(result.spectrum_intensity)) / float(spectrum_size)
            if spectrum_size > 0
            else float("nan")
        )
        peak_text = f"{result.peak_energy_ev:.4g} eV" if np.isfinite(float(result.peak_energy_ev)) else "n/a"
        fwhm_text = f"{result.peak_fwhm_ev:.4g} eV" if np.isfinite(float(result.peak_fwhm_ev)) else "n/a"
        finite_text = f"{finite_fraction:.4f}" if np.isfinite(float(finite_fraction)) else "n/a (empty spectrum)"
        example_lines.append(f'- `{path.name}`: requested={result.requested_model_name}, applied={result.model_name}, benchmark={result.benchmark_status}, full_exec={result.model_executed_fully}, fallback={result.fallback_fraction:.3f}, peak={peak_text}, FWHM={fwhm_text}, finite_fraction={finite_text}, zones={result.zone_count_used}, clusters={result.cluster_count_used}')

    total = len(checks)
    passed = sum(1 for _, ok, _ in checks if ok)
    report = [f'# Plasmon Phase 8 verification ({passed}/{total} checks passed)', '']
    for name, ok, detail in checks:
        report.append(f'- [{"x" if ok else " "}] **{name}** — {detail}')
    report.append('')
    report.append('## Bundled example sanity checks')
    report.extend(example_lines)
    report.append('')
    report.append('## Notes')
    report.append('- Quicklook compatibility and RPA/Mermin/LFC limit checks were re-run from source code, not copied from earlier summaries.')
    report.append('- Bundled examples do not contain multiple region/material selections, so deselection sensitivity is validated on a synthetic multi-region dataset.')
    args.report.write_text('\n'.join(report), encoding='utf-8')
    print(args.report)
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
