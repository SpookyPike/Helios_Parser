"""A raw deck and its conversion must answer the same questions.

Every `helios_diag` command now accepts an unconverted `.log`, `.bpf` or HDF5
`.exo`. That is only safe if the raw path reads the same numbers the writer
stores, so this test converts the tracked prefix log once and compares the two
payloads command by command.

The comparison drops the fields that are about the file rather than the physics:
`file`, `notes`, `generated_at`, `request`, and the path-derived
`result.context_key` / `result.dataset_path`. Everything else must match exactly,
including float values, because the writer stores float64 without resampling.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401

from helios_diag import analyse as analyse_module
from helios_diag import query as query_module
from helios_diag import stats as stats_module
from helios_parser import write_hdf5

ROOT = Path(__file__).resolve().parents[1]
LOG_SAMPLE = ROOT / "tests" / "fixtures" / "v12" / "cu_v12_prefix.log"
BPF_SAMPLE = ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.bpf"

_VOLATILE_TOP = {"file", "notes", "generated_at", "request"}
_VOLATILE_RESULT = {"context_key", "dataset_path"}


def _stats_args(path: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        path=path,
        field="density",
        over=None,
        at=None,
        at_time=None,
        global_stats=False,
        quantiles="5,50,95",
        region=None,
        material=None,
        by_region=False,
        by_material=False,
        flat_atol=1e-12,
        flat_rtol=1e-9,
    )


def _query_args(path: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        path=path,
        field="temperature_e",
        snapshot=5,
        zone=None,
        time=None,
        region=None,
        material=None,
    )


def _analyse_args(path: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        path=path,
        module=["shock"],
        snapshot=None,
        include_experimental=False,
        breakout_fraction=0.5,
        max_array=64,
    )


def _normalize(payload: dict) -> dict:
    normalized = {key: value for key, value in payload.items() if key not in _VOLATILE_TOP}
    result = normalized.get("result")
    if isinstance(result, dict):
        normalized["result"] = {
            key: value for key, value in result.items() if key not in _VOLATILE_RESULT
        }
    return normalized


class RawInputEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.run_h5 = Path(cls._tmpdir.name) / "prefix.h5"
        write_hdf5(LOG_SAMPLE, cls.run_h5, overwrite=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _compare(self, module, args_factory) -> dict:
        raw = module.run(args_factory(str(LOG_SAMPLE)))
        converted = module.run(args_factory(str(self.run_h5)))
        self.assertEqual(raw.exit_code, converted.exit_code)
        return {"raw": _normalize(raw.payload), "converted": _normalize(converted.payload)}

    def test_stats_matches_the_converted_run(self) -> None:
        payloads = self._compare(stats_module, _stats_args)
        self.assertEqual(payloads["raw"], payloads["converted"])

    def test_query_matches_the_converted_run(self) -> None:
        payloads = self._compare(query_module, _query_args)
        self.assertEqual(payloads["raw"], payloads["converted"])

    def test_analyse_matches_the_converted_run(self) -> None:
        payloads = self._compare(analyse_module, _analyse_args)
        raw_shock = payloads["raw"]["result"]["shock"]
        converted_shock = payloads["converted"]["result"]["shock"]
        self.assertEqual(raw_shock["time_s"], converted_shock["time_s"])
        self.assertEqual(raw_shock["zone_index"], converted_shock["zone_index"])
        self.assertEqual(raw_shock["breakout_time_s"], converted_shock["breakout_time_s"])
        self.assertEqual(payloads["raw"], payloads["converted"])

    def test_bpf_header_lookup(self) -> None:
        from helios_diag.source import raw_header

        view = raw_header(BPF_SAMPLE)
        self.assertEqual(view.format, "bpf")
        self.assertEqual(view.n_zones, 50)
        self.assertEqual(view.n_snapshots, 901)
        self.assertGreater(len(view.list_fields()), 50)

    def test_classic_netcdf_exodus_is_named_not_crashed(self) -> None:
        from helios_diag.source import UnsupportedInputError, detect_input, raw_header

        with tempfile.TemporaryDirectory() as tmp:
            classic = Path(tmp) / "classic.exo"
            classic.write_bytes(b"CDF\x01\x00\x00\x03\x85")
            self.assertEqual(detect_input(classic), "netcdf3")
            with self.assertRaises(UnsupportedInputError) as caught:
                raw_header(classic)
        self.assertIn("classic NetCDF-3", str(caught.exception))

    def test_inspect_reports_the_raw_format(self) -> None:
        from helios_diag import inspect as inspect_module

        result = inspect_module.run(types.SimpleNamespace(path=str(LOG_SAMPLE)))
        self.assertEqual(result.payload["file"]["format"], "log")
        self.assertGreater(len(result.payload["fields"]), 20)
        self.assertTrue(result.payload["physics"]["laser"])
        self.assertIsNone(result.payload["visar"])


if __name__ == "__main__":
    unittest.main()