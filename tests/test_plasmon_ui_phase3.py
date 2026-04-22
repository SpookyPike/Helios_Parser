from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

try:
    from PySide6 import QtWidgets  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    QtWidgets = None  # type: ignore

import _test_bootstrap  # noqa: F401

if QtWidgets is not None:
    from _viewer_test_utils import get_app, process_events, reset_test_settings
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.models import DerivedRunData

if QtWidgets is not None:
    from helios_analysis.workspace import HeliosDerivedWorkspace


def _synthetic_dataset() -> tuple[DerivedRunData, RunContext]:
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
        "region_index": np.asarray([1, 2], dtype=np.int32),
        "min_zone_index": np.asarray([1, 4], dtype=np.int32),
        "max_zone_index": np.asarray([3, 6], dtype=np.int32),
        "atomic_weight": np.asarray([27.0, 63.5], dtype=np.float64),
        "initial_mass_density": np.asarray([1.0, 1.0], dtype=np.float64),
        "initial_temperature": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={"geometry": "PLANAR", "coordinate_model": {"coordinate_name": "x"}},
        regions=regions,
        materials={"index": np.asarray([1, 2], dtype=np.int32)},
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
        zone_atomic_weight=np.asarray([27.0, 27.0, 27.0, 63.5, 63.5, 63.5], dtype=np.float64),
        zone_initial_density_g_cm3=np.full(n_zones, 1.0, dtype=np.float64),
        zone_initial_temperature_ev=np.full(n_zones, 1.0, dtype=np.float64),
        laser_entry=None,
    )
    context = RunContext(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={},
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=time_s.copy(),
        static_x_values=static_x.copy(),
        zone_region_id=zone_region_id.copy(),
        zone_material_index=zone_material.copy(),
        has_dynamic_radius=False,
        snapshot_index=1,
        map_coordinate="static_x",
        slice_coordinate="zone",
        selected_region_ids=(1, 2),
        selected_material_ids=(1, 2),
    )
    return dataset, context


@unittest.skipIf(QtWidgets is None, "PySide6 is not available in this environment")
class PlasmonUiPhase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_workspace_surfaces_mermin_model_and_peak_metrics(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="mermin",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=1.5e15,
        )
        result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("ui-phase3", 1),
            requested_time_plot_modules=frozenset({"plasmon"}),
            include_wavefront=False,
        )
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertEqual(workspace.plasmon_plot_panel.time_combo.currentData(), "spectrum_observed")
        self.assertIn("model=mermin", workspace.plasmon_summary_label.text().lower())
        self.assertIn("peak fwhm", workspace.plasmon_metrics.toPlainText().lower())
        self.assertIn("nu_e", workspace.plasmon_metrics.toPlainText().lower())
        workspace.close()


if __name__ == "__main__":
    unittest.main()
