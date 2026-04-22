from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser, HeliosRun, write_hdf5


ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT / "outputs" / "hdf5"


def _truncate_after_partial_final_block(source: Path) -> str:
    text = source.read_text(encoding="utf-8", errors="ignore")
    cycle_headers = list(re.finditer(r"^\s*Cycle\s+Time\s+\(s\)", text, re.MULTILINE))
    if not cycle_headers:
        raise AssertionError(f"No snapshot blocks found in {source}.")
    last_start = cycle_headers[-1].start()
    tail = text[last_start:]
    cut_markers = ("Ion energy", "ENERGY SUMMARY:", "Radiation Cooling Rates and Boundary Fluxes")
    cut_offset = -1
    for marker in cut_markers:
        candidate = tail.find(marker)
        if candidate > 0:
            cut_offset = candidate
            break
    if cut_offset < 0:
        cut_offset = min(len(tail), 2500)
    return text[: last_start + cut_offset].rstrip() + "\n"


class RunStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = HeliosParser()

    def test_footer_status_distinguishes_completed_and_unknown_runs(self) -> None:
        completed = self.parser.parse(ROOT / "Cu1e17.log")
        self.assertIsNotNone(completed.run_status)
        assert completed.run_status is not None
        self.assertEqual(completed.run_status.state, "completed")

        cylindrical_completed = self.parser.parse(ROOT / "Cu1e17_cyl.log")
        self.assertIsNotNone(cylindrical_completed.run_status)
        assert cylindrical_completed.run_status is not None
        self.assertEqual(cylindrical_completed.run_status.state, "completed")

        unknown = self.parser.parse(ROOT / "5Fe+4.9TW+light.log")
        self.assertIsNotNone(unknown.run_status)
        assert unknown.run_status is not None
        self.assertEqual(unknown.run_status.state, "unknown")

    def test_partial_final_block_is_dropped_and_statused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            truncated_log = tmpdir_path / "5Fe_truncated.log"
            truncated_log.write_text(
                _truncate_after_partial_final_block(ROOT / "5Fe+4.9TW+light.log"),
                encoding="utf-8",
            )

            simulation = self.parser.parse(truncated_log)
            self.assertIsNotNone(simulation.run_status)
            assert simulation.run_status is not None
            self.assertEqual(simulation.run_status.state, "truncated")
            self.assertTrue(simulation.run_status.dropped_partial_final_block)
            self.assertEqual(simulation.metadata["n_snapshots"], 7)
            self.assertEqual(self.parser.count_snapshots(truncated_log), 7)
            self.assertAlmostEqual(float(simulation.time["time"][-1]), float(simulation.run_status.last_valid_snapshot_time_s))
            self.assertIsNotNone(simulation.run_status.damaged_final_block_reason)
            self.assertIn("final indexed snapshot block was structurally incomplete", " ".join(simulation.run_status.notes))

            hdf5_path = tmpdir_path / "5Fe_truncated.h5"
            write_hdf5(truncated_log, hdf5_path, overwrite=True, parser=self.parser)
            with HeliosRun(hdf5_path) as run:
                status = run.get_run_status()
                self.assertEqual(status["state"], "truncated")
                self.assertTrue(bool(status["dropped_partial_final_block"]))
                self.assertEqual(run.n_snapshots, 7)
                self.assertAlmostEqual(float(run.get_time()[-1]), float(status["last_valid_snapshot_time_s"]))

    def test_legacy_hdf5_defaults_to_unknown_status_without_silent_completion_claim(self) -> None:
        with HeliosRun(HDF5_ROOT / "Cu_0166_stabilized.h5") as run:
            status = run.get_run_status()
        self.assertEqual(status["state"], "unknown")
        self.assertEqual(status["source"], "legacy_hdf5")
        self.assertFalse(bool(status["dropped_partial_final_block"]))


if __name__ == "__main__":
    unittest.main()
