from __future__ import annotations

import ctypes
import os
import tempfile
import unittest
from ctypes import wintypes
from pathlib import Path

import _test_bootstrap  # noqa: F401
from helios_parser import write_hdf5


ROOT = Path(__file__).resolve().parents[1]
BPF_SAMPLE = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"


class ConversionAtomicityTests(unittest.TestCase):
    def test_failure_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.h5"
            output.write_bytes(b"previous result")
            truncated = Path(tmpdir) / "broken.bpf"
            truncated.write_bytes(b"not a BPF")
            with self.assertRaises(Exception):
                write_hdf5(truncated, output, overwrite=True)
            self.assertEqual(output.read_bytes(), b"previous result")
            self.assertEqual(list(Path(tmpdir).glob(".result.h5.*.tmp")), [])

    def test_prepublication_callback_failure_preserves_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.h5"
            output.write_bytes(b"previous result")
            def callback(event):
                if event.stage == "snapshots":
                    raise RuntimeError("consumer stopped")
            with self.assertRaisesRegex(RuntimeError, "consumer stopped"):
                write_hdf5(BPF_SAMPLE, output, overwrite=True, progress_callback=callback)
            self.assertEqual(output.read_bytes(), b"previous result")
            self.assertEqual(list(Path(tmpdir).glob(".result.h5.*.tmp")), [])

    def test_done_callback_failure_is_contained_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.h5"
            def callback(event):
                if event.stage == "done":
                    raise RuntimeError("ui closed")
            self.assertEqual(write_hdf5(BPF_SAMPLE, output, progress_callback=callback), output)
            self.assertTrue(output.is_file())

    def test_no_clobber_publish_detects_destination_created_during_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.h5"
            def callback(event):
                if event.stage == "finalize":
                    output.write_bytes(b"racing writer")
            with self.assertRaises(FileExistsError):
                write_hdf5(BPF_SAMPLE, output, progress_callback=callback)
            self.assertEqual(output.read_bytes(), b"racing writer")

    def test_output_cannot_alias_selected_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alias = Path(tmpdir) / "input.bpf"
            try:
                os.link(BPF_SAMPLE, alias)
            except OSError as exc:
                self.skipTest(f"hard links unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "aliases a selected input"):
                write_hdf5(BPF_SAMPLE, alias, overwrite=True)
            self.assertEqual(alias.read_bytes(), BPF_SAMPLE.read_bytes())

    @unittest.skipUnless(os.name == "posix", "POSIX file mode semantics")
    def test_overwrite_preserves_existing_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.h5"
            output.write_bytes(b"previous")
            output.chmod(0o640)
            write_hdf5(BPF_SAMPLE, output, overwrite=True)
            self.assertEqual(output.stat().st_mode & 0o777, 0o640)

    @unittest.skipUnless(os.name == "nt", "Windows file sharing semantics")
    def test_overwrite_preserves_destination_locked_against_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "locked-result.h5"
            original = b"previous result"
            output.write_bytes(original)

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            kernel32.CreateFileW.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.CreateFileW(
                str(output),
                0x80000000,  # GENERIC_READ
                0,  # deny read/write/delete sharing while replacement is attempted
                None,
                3,  # OPEN_EXISTING
                0x80,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            if handle is None or handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                with self.assertRaises(PermissionError):
                    write_hdf5(BPF_SAMPLE, output, overwrite=True)
            finally:
                self.assertTrue(kernel32.CloseHandle(handle))

            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(Path(tmpdir).glob(".locked-result.h5.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
