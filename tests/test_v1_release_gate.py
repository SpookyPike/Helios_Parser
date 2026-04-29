from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np

try:
    from PySide6 import QtWidgets  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    QtWidgets = None  # type: ignore

import _test_bootstrap  # noqa: F401

from helios.services.derived import analysis as analysis_module
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.module_contract import DerivedModuleContract
from helios.services.derived.physical_sanity import validate_spectroscopy_result, validate_xrd_result

if QtWidgets is not None:
    from _viewer_test_utils import get_app, process_events, reset_test_settings
    from helios_analysis.workspace import HeliosDerivedWorkspace


def _tab_labels(workspace: "HeliosDerivedWorkspace") -> list[str]:
    return [workspace.result_tabs.tabText(index) for index in range(workspace.result_tabs.count())]


@unittest.skipIf(QtWidgets is None, "PySide6 is not available in this environment")
class V1FeatureGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_production_gui_hides_experimental_physics_tabs(self) -> None:
        with mock.patch.dict("os.environ", {"HELIOS_DEV_MODE": "", "HELIOS_ENABLE_EXPERIMENTAL": ""}, clear=False):
            workspace = HeliosDerivedWorkspace()
            try:
                labels = _tab_labels(workspace)
                self.assertIn("Shock", labels)
                self.assertIn("XRD", labels)
                self.assertIn("Spectroscopy", labels)
                self.assertNotIn("Plasmon", labels)
                self.assertNotIn("Transmission", labels)
            finally:
                workspace.close()
                workspace.deleteLater()
                process_events(20)

    def test_experimental_flag_restores_hidden_physics_tabs(self) -> None:
        with mock.patch.dict("os.environ", {"HELIOS_ENABLE_EXPERIMENTAL": "1"}, clear=False):
            workspace = HeliosDerivedWorkspace()
            try:
                labels = _tab_labels(workspace)
                self.assertIn("Plasmon", labels)
                self.assertIn("Transmission", labels)
            finally:
                workspace.close()
                workspace.deleteLater()
                process_events(20)


class V1PhysicalSanityTests(unittest.TestCase):
    def test_xrd_sanity_rejects_impossible_compression(self) -> None:
        result = SimpleNamespace(
            wavelength_angstrom=1.5,
            layers=(
                SimpleNamespace(
                    compressed_density_g_cm3=2.7,
                    compression_ratio=1.0e6,
                    d_over_d0=1.0,
                    q0_inv_angstrom=2.0,
                    q_compressed_inv_angstrom=2.0,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "compression ratio"):
            validate_xrd_result(result)

    def test_spectroscopy_sanity_rejects_superluminal_scale_velocity(self) -> None:
        result = SimpleNamespace(
            line_wavelength_nm=500.0,
            bulk_velocity_cm_s=2.0e10,
            los_velocity_cm_s=2.0e10,
            thermal_width_fraction=1.0e-6,
            ion_temperature_ev=10.0,
            ion_mass_mu=27.0,
        )
        with self.assertRaisesRegex(ValueError, "velocity"):
            validate_spectroscopy_result(result)


class V1BackendFeatureGateTests(unittest.TestCase):
    def _compute_with_fake_contracts(self) -> list[str]:
        calls: list[str] = []

        def _fake_compute(name: str):
            def _compute(*_args, **_kwargs):
                calls.append(name)
                return SimpleNamespace(time_plots=(), profile_plots=(), warnings=())

            return _compute

        contracts = tuple(
            DerivedModuleContract(name=name, compute=_fake_compute(name), validate=lambda _result: None)
            for name in ("xrd", "plasmon", "transmission", "spectroscopy")
        )
        dataset = SimpleNamespace(path=Path("dummy.h5"), time_s=np.asarray([0.0], dtype=np.float64))
        context = SimpleNamespace()
        geometry = SimpleNamespace()
        selection = SimpleNamespace()
        shock = SimpleNamespace(warnings=())
        with (
            mock.patch.object(analysis_module, "_MODULE_CONTRACTS", contracts),
            mock.patch.object(
                analysis_module,
                "_resolve_analysis_state",
                return_value=(0, geometry, np.asarray([True], dtype=bool), selection, (), ()),
            ),
            mock.patch.object(analysis_module, "track_shock_front", return_value=shock),
            mock.patch.object(analysis_module, "validate_shock_result"),
        ):
            analysis_module.compute_analysis_result(
                dataset,
                context,
                parameters=DerivedAnalysisParameters(),
                context_key=("v1", "backend-gate"),
                requested_time_plot_modules=None,
                include_wavefront=False,
            )
        return calls

    def test_production_compute_skips_hidden_physics_modules(self) -> None:
        with mock.patch.dict("os.environ", {"HELIOS_DEV_MODE": "", "HELIOS_ENABLE_EXPERIMENTAL": ""}, clear=False):
            calls = self._compute_with_fake_contracts()
        self.assertEqual(calls, ["xrd", "spectroscopy"])

    def test_experimental_flag_computes_hidden_physics_modules(self) -> None:
        with mock.patch.dict("os.environ", {"HELIOS_ENABLE_EXPERIMENTAL": "1"}, clear=False):
            calls = self._compute_with_fake_contracts()
        self.assertEqual(calls, ["xrd", "plasmon", "transmission", "spectroscopy"])


if __name__ == "__main__":
    unittest.main()
