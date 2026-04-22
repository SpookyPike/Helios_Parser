from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _test_bootstrap  # noqa: F401

from helios_parser import HeliosParser, HeliosRun, write_hdf5


ROOT = Path(__file__).resolve().parents[1]


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


class StreamingConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = HeliosParser()

    def test_streaming_iterator_drops_partial_final_block_without_full_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            truncated_log = Path(tmpdir) / "5Fe_truncated.log"
            truncated_log.write_text(
                _truncate_after_partial_final_block(ROOT / "5Fe+4.9TW+light.log"),
                encoding="utf-8",
            )
            with self.parser.open_document(truncated_log) as document:
                header = document.inspect()
                iterator = document.iter_snapshots_streaming(header=header)
                snapshots = list(iterator)

            self.assertEqual(len(snapshots), 7)
            self.assertIsNotNone(iterator.run_status)
            assert iterator.run_status is not None
            self.assertEqual(iterator.run_status.state, "truncated")
            self.assertTrue(iterator.run_status.dropped_partial_final_block)
            self.assertAlmostEqual(float(snapshots[-1].time), float(iterator.run_status.last_valid_snapshot_time_s))

    def test_streaming_write_matches_full_parse_output_for_valid_run(self) -> None:
        source = ROOT / "5Fe+4.9TW+light.log"
        simulation = self.parser.parse(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "5Fe_streaming.h5"
            progress_events: list[tuple[str, int, int]] = []
            write_hdf5(
                source,
                output,
                overwrite=True,
                parser=self.parser,
                progress_callback=lambda event: progress_events.append((event.stage, event.current, event.total)),
            )

            with HeliosRun(output) as run:
                np.testing.assert_allclose(run.get_time(), simulation.time["time"])
                np.testing.assert_allclose(run.get_field("density"), simulation.fields["density"], equal_nan=True)
                np.testing.assert_allclose(
                    run.get_diagnostic("energy_summary/current/ions"),
                    simulation.diagnostics["energy_summary"]["current"]["ions"],
                    equal_nan=True,
                )
                status = run.get_run_status()
                self.assertEqual(status["state"], simulation.run_status.state if simulation.run_status is not None else "unknown")
                self.assertEqual(run.n_snapshots, simulation.time["time"].size)

            self.assertTrue(any(stage == "snapshots" and current == 1 for stage, current, _ in progress_events))
            first_snapshot_event = next(index for index, event in enumerate(progress_events) if event[0] == "snapshots")
            finalize_event = next(index for index, event in enumerate(progress_events) if event[0] == "finalize")
            self.assertLess(first_snapshot_event, finalize_event)


if __name__ == "__main__":
    unittest.main()
