"""Native integration checks; these deliberately compile and exercise XCOM."""
import shutil
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helios.services.derived.xcom_native import (
    NativeXcomClient,
    compute_multizone_transmission,
    resolve_provider,
)
from helios.services.derived.xcom_hook import (
    ColdAttenuationRequest,
    ColdAttenuationZone,
    HeliosXcomBackend,
    describe_optional_cold_backend,
)
from helios.cancellation import AnalysisCancelled

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)
from build_xcom_native import build


@unittest.skipUnless(shutil.which("gfortran"), "native XCOM integration requires gfortran")
class XcomNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._root = None
        if os.environ.get("HELIOS_XCOM_BUNDLE_DIR"):
            cls._provider_dir = Path(os.environ["HELIOS_XCOM_BUNDLE_DIR"])
            cls.client = NativeXcomClient(resolve_provider())
        else:
            cls._root = tempfile.TemporaryDirectory(prefix="helios-xcom-test-")
            cls._provider_dir = Path(cls._root.name) / "provider"
            build(cls._provider_dir)
            cls.client = NativeXcomClient(resolve_provider_for(cls._provider_dir))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        if cls._root is not None:
            cls._root.cleanup()

    def test_native_element_coefficients_cover_35_kev(self):
        rows = self.client.query_element("Cu", (8.0, 35.0))
        self.assertAlmostEqual(52.55, float(rows[8.0]["mu_rho_total_cm2_g"]), delta=0.02)
        self.assertGreater(float(rows[35.0]["mu_rho_total_cm2_g"]), 0.0)

    def test_real_native_engine_flows_through_existing_helios_backend(self):
        module = type(
            "NativeModule",
            (),
            {"compute_multizone_transmission": staticmethod(compute_multizone_transmission)},
        )
        backend = HeliosXcomBackend(module=module, client=self.client)
        request = ColdAttenuationRequest(
            0,
            "front",
            1.0,
            (8.0, 35.0),
            (ColdAttenuationZone(1, 1, 1, "Cu", 8.96, 1.0e-4),),
        )
        result = backend.compute_transmission(request)
        self.assertEqual("total_with_coherent", result.metadata["attenuation_mode"])
        self.assertEqual((8.0, 35.0), tuple(result.energies_kev.tolist()))
        self.assertAlmostEqual(52.55 * 8.96e-4, result.metadata["optical_depth"][0], delta=2e-5)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.transmission))

    def test_element_coefficients_are_reused_across_compounds(self):
        original = self.client.query_element
        with mock.patch.object(self.client, "query_element", wraps=original) as query:
            self.client.query_total("CH", (8.0, 35.0))
            first_calls = query.call_count
            self.client.query_total("CH2", (8.0, 35.0))
        self.assertEqual(2, first_calls)
        self.assertEqual(first_calls, query.call_count)

    def test_native_energy_bounds_are_checked_before_running(self):
        for energies in ((0.9,), (35.01,), (float("nan"),), (36.0,)):
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                self.client.query_element("Cu", energies)

    def test_cancellation_terminates_and_reaps_native_process(self):
        process = mock.Mock()
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("xcom", 0.1),
            ("", ""),
        ]
        checks = 0

        def cancel_after_first_poll():
            nonlocal checks
            checks += 1
            if checks > 1:
                raise AnalysisCancelled("cancelled")

        with mock.patch(
            "helios.services.derived.xcom_native.subprocess.Popen",
            return_value=process,
        ), self.assertRaises(AnalysisCancelled):
            self.client.query_element("Cu", (8.0,), progress_check=cancel_after_first_poll)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once()
        self.assertEqual(2, process.communicate.call_count)

    def test_packaged_provider_manifest_validates_all_element_data(self):
        self.assertTrue((self._provider_dir / "manifest.json").is_file())
        notice = (self._provider_dir / "XCOM_NOTICE.txt").read_text(encoding="utf-8")
        self.assertIn("non-commercial purposes", notice)
        self.assertEqual(100, len(list(self._provider_dir.glob("MDATX3.*"))))
        with mock.patch.dict(
            os.environ,
            {
                "HELIOS_XCOM_EXECUTABLE": str(self._provider_dir / ("xcom.exe" if os.name == "nt" else "xcom")),
                "HELIOS_XCOM_DATA_DIR": str(self._provider_dir),
            },
        ):
            self.assertEqual(self.client.provider.fingerprint, resolve_provider().fingerprint)

    def test_native_provider_status_is_explicit_without_claiming_wrapper_presence(self):
        with mock.patch.dict(
            os.environ,
            {
                "HELIOS_XCOM_EXECUTABLE": str(self._provider_dir / ("xcom.exe" if os.name == "nt" else "xcom")),
                "HELIOS_XCOM_DATA_DIR": str(self._provider_dir),
            },
        ):
            status = describe_optional_cold_backend(probe_compute=True, force=True)
        self.assertTrue(status.available)
        self.assertEqual("ready", status.status)
        self.assertFalse(status.wrapper_present)


def resolve_provider_for(directory: Path):
    with mock.patch.dict(
        os.environ,
        {
            "HELIOS_XCOM_EXECUTABLE": str(directory / ("xcom.exe" if os.name == "nt" else "xcom")),
            "HELIOS_XCOM_DATA_DIR": str(directory),
        },
    ):
        return resolve_provider()


if __name__ == "__main__":
    unittest.main()
