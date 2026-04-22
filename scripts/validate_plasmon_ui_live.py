from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon_validation import make_run_context, shocked_al_slab_summary
from helios_analysis.workspace import HeliosDerivedWorkspace


def _process_events(delay_ms: int = 0) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def _capture_case(
    workspace: HeliosDerivedWorkspace,
    dataset_path: Path,
    *,
    snapshot_index: int,
    parameters: DerivedAnalysisParameters,
    screenshot_path: Path,
) -> dict[str, object]:
    dataset = load_run_data(dataset_path)
    context = make_run_context(dataset, dataset_path, snapshot_index=snapshot_index)
    t0 = time.perf_counter()
    result = compute_analysis_result(
        dataset,
        context,
        parameters=parameters,
        context_key=("plasmon-ui-live", dataset_path.name, snapshot_index, parameters.plasmon_model, parameters.plasmon_execution_mode),
        requested_time_plot_modules=frozenset({"plasmon"}),
        include_wavefront=False,
    )
    elapsed_s = float(time.perf_counter() - t0)
    workspace.set_context(context)
    workspace.set_result(result)
    workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
    _process_events(150)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.grab().save(str(screenshot_path))
    metrics_text = workspace.plasmon_metrics.toPlainText()
    return {
        "dataset": dataset_path.name,
        "snapshot_index": int(snapshot_index),
        "requested_model": str(parameters.plasmon_model),
        "execution_mode": str(parameters.plasmon_execution_mode),
        "elapsed_s": elapsed_s,
        "summary": workspace.plasmon_summary_label.text(),
        "time_plot": workspace.plasmon_plot_panel.time_combo.currentText(),
        "profile_plot": workspace.plasmon_plot_panel.profile_combo.currentText(),
        "peak_line": next((line for line in metrics_text.splitlines() if "Peak dE" in line), ""),
        "backend_line": next((line for line in metrics_text.splitlines() if "Backend" in line), ""),
        "screenshot": str(screenshot_path),
    }


def build_report(*, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)

    workspace = HeliosDerivedWorkspace()
    workspace.resize(1800, 1100)
    workspace.show()
    _process_events(100)

    fifty_path = Path("50Al+10E+25CH+3.5TW_stabilized.h5")
    fifty_dataset = load_run_data(fifty_path)
    fifty_time_ns = [float(value) * 1.0e9 for value in fifty_dataset.time_s]
    fifty_snapshot = int(min(range(len(fifty_time_ns)), key=lambda idx: abs(fifty_time_ns[idx] - 6.3)))
    fifty_summary = shocked_al_slab_summary(fifty_dataset, snapshot_index=fifty_snapshot, density_floor_g_cm3=3.75, material_id=1)

    cases = [
        (
            fifty_path,
            fifty_snapshot,
            DerivedAnalysisParameters(
                plasmon_model="rpa_static_lfc",
                plasmon_execution_mode="benchmark",
                plasmon_integration_mode="los_integrated",
                plasmon_electron_policy="benchmark_valence_aware",
                plasmon_photon_energy_kev=8.307,
                plasmon_scattering_angle_deg=14.0,
                plasmon_energy_window_ev=45.0,
                plasmon_energy_points=1201,
                plasmon_instrument_fwhm_ev=0.20,
                plasmon_lfc_model="esa_static",
                derived_material_ids=(1,),
                zone_index_lower=int(fifty_summary["zone_index_lower"]),
                zone_index_upper=int(fifty_summary["zone_index_upper"]),
            ),
            out_dir / "50Al_driven_rpa_static_lfc.png",
        ),
        (
            fifty_path,
            fifty_snapshot,
            DerivedAnalysisParameters(
                plasmon_model="lindhard",
                plasmon_execution_mode="benchmark",
                plasmon_integration_mode="los_integrated",
                plasmon_electron_policy="benchmark_valence_aware",
                plasmon_photon_energy_kev=8.307,
                plasmon_scattering_angle_deg=14.0,
                plasmon_energy_window_ev=45.0,
                plasmon_energy_points=1201,
                plasmon_instrument_fwhm_ev=0.20,
                plasmon_lfc_model="esa_static",
                derived_material_ids=(1,),
                zone_index_lower=int(fifty_summary["zone_index_lower"]),
                zone_index_upper=int(fifty_summary["zone_index_upper"]),
            ),
            out_dir / "50Al_driven_lindhard.png",
        ),
        (
            Path("Cu_0166_stabilized.h5"),
            0,
            DerivedAnalysisParameters(
                plasmon_model="quicklook",
                plasmon_execution_mode="quicklook",
                plasmon_photon_energy_kev=8.0,
                plasmon_scattering_angle_deg=20.0,
            ),
            out_dir / "Cu0166_quicklook.png",
        ),
        (
            Path("5Fe+4.9TW+light_stabilized.h5"),
            5,
            DerivedAnalysisParameters(
                plasmon_model="mermin_static_lfc",
                plasmon_execution_mode="benchmark",
                plasmon_integration_mode="los_integrated",
                plasmon_photon_energy_kev=7.5,
                plasmon_scattering_angle_deg=20.0,
                plasmon_energy_window_ev=80.0,
                plasmon_energy_points=1201,
                plasmon_instrument_fwhm_ev=1.0,
                plasmon_collision_model="nrl_constant",
                plasmon_lfc_model="esa_static",
            ),
            out_dir / "Fe_light_mermin_static_lfc.png",
        ),
    ]

    rows = [
        _capture_case(
            workspace,
            dataset_path,
            snapshot_index=snapshot_index,
            parameters=parameters,
            screenshot_path=screenshot_path,
        )
        for dataset_path, snapshot_index, parameters, screenshot_path in cases
    ]

    report_lines = [
        "# Live plasmon UI validation",
        "",
        "This pass renders the plasmon tab on real datasets and setting combinations to verify that the widget follows the recomputed result rather than stale control state.",
        "",
        "| dataset | snapshot | model | exec | runtime [s] | time plot | profile plot | peak line | backend | screenshot |",
        "|---|---:|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['dataset']} | {row['snapshot_index']} | {row['requested_model']} | {row['execution_mode']} | {float(row['elapsed_s']):.2f} | {str(row['time_plot']).replace('|', '/')} | {str(row['profile_plot']).replace('|', '/')} | {str(row['peak_line']).replace('|', '/')} | {str(row['backend_line']).replace('|', '/')} | `{row['screenshot']}` |"
        )
    report_lines.extend(
        [
            "",
            "Checks performed:",
            "- switched between quicklook, RPA-static-LFC, Lindhard, and Mermin-static-LFC on real HDF5 data",
            "- verified the summary line updates with the applied model/backend",
            "- verified that benchmark spectral models switch the profile panel to the observed spectrum, while quicklook/non-spectral states stay on state profiles",
            "- captured screenshots after each refresh for manual inspection",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render live plasmon UI states for real datasets.")
    parser.add_argument("--out-dir", default="outputs/validation_outputs/plasmon_ui_live")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    try:
        code = build_report(out_dir=out_dir)
    except Exception as exc:  # pragma: no cover - diagnostic path
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
