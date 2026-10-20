from __future__ import annotations

import shutil
import re
import struct
import tempfile
import unittest
from decimal import Decimal
from unittest import mock
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun, write_hdf5
from helios_parser.bpf import BpfFile
from helios_parser.parser import HeliosParser
from helios.services.derived.common import load_run_data


ROOT = Path(__file__).resolve().parents[1]
BPF_SAMPLE = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
LOG_BPF_SAMPLE = ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log"
LOG_BPF_COMPANION = LOG_BPF_SAMPLE.with_suffix(".bpf")


def _assert_scale_close(testcase: unittest.TestCase, actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float = 0.0) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    testcase.assertEqual(actual_array.shape, expected_array.shape)
    diff = np.abs(actual_array - expected_array)
    scale = max(1.0, float(np.nanmax(np.abs(actual_array))), float(np.nanmax(np.abs(expected_array))))
    testcase.assertLessEqual(float(np.nanmax(diff)), atol + rtol * scale)


class BpfPipelineTests(unittest.TestCase):
    def test_bpf_layout_and_integrity_checks(self) -> None:
        with BpfFile(BPF_SAMPLE) as bpf:
            layout = bpf.layout
            self.assertEqual(layout.n_zones, 500)
            self.assertEqual(layout.n_nodes, 501)
            self.assertEqual(layout.n_freq_bins, 200)
            self.assertEqual(layout.n_snapshots, 8)
            self.assertTrue(layout.recognized_zero_trailer)
            snapshot = bpf.extract_snapshot(0)
            self.assertEqual(snapshot.fields["mass_density_g_cm3"].shape, (500,))
            self.assertEqual(snapshot.fields["radiation_net_flux_rmin_j_s_cm2_eV"].shape, (200,))
            self.assertEqual(snapshot.fields["bpf_record_03"].shape, (50,))
            self.assertEqual(snapshot.fields["ionization_fractions_by_zone_charge"].shape, (500, 27))
            self.assertEqual(snapshot.fields["zone_outer_velocity_cm_s"].shape, (500,))
            self.assertEqual(snapshot.fields["radiation_energy_density_j_cm3"].shape, (500,))
            np.testing.assert_allclose(snapshot.fields["zone_outer_velocity_cm_s"], snapshot.fields["interface_velocity_cm_s"][1:])
            np.testing.assert_allclose(snapshot.fields["radiation_energy_density_j_cm3"], snapshot.fields["bpf_record_14"][1:-1])
            np.testing.assert_allclose(snapshot.fields["radiation_pressure_j_cm3"], snapshot.fields["radiation_energy_density_j_cm3"] / 3.0)
            np.testing.assert_allclose(snapshot.fields["ionization_fractions_by_zone_charge"].sum(axis=1), 1.0, rtol=1.0e-6, atol=1.0e-8)
            np.testing.assert_allclose(
                np.tensordot(snapshot.fields["ionization_fractions_by_zone_charge"], snapshot.fields["charge_state"], axes=([1], [0])),
                snapshot.fields["mean_charge"],
                rtol=1.0e-6,
                atol=1.0e-8,
            )
            del snapshot





    def test_valid_one_frame_bpf_with_documented_trailer_is_complete(self) -> None:
        with BpfFile(BPF_SAMPLE) as source:
            first_trailer = 2 + source.layout.block_records
            first_trailer_start = source._payload_offsets[first_trailer] - 4
            last_trailer_start = source._payload_offsets[-1] - 4
            original = BPF_SAMPLE.read_bytes()
            raw = original[:first_trailer_start] + original[last_trailer_start:]
        with tempfile.TemporaryDirectory() as tmpdir:
            one_frame = Path(tmpdir) / "one_frame.bpf"
            one_frame.write_bytes(raw)
            with BpfFile(one_frame) as parsed:
                self.assertEqual(parsed.layout.n_snapshots, 1)
                self.assertTrue(parsed.layout.recognized_zero_trailer)
            output = Path(tmpdir) / "one_frame.h5"
            write_hdf5(one_frame, output)
            with HeliosRun(output) as run:
                self.assertEqual(run.n_snapshots, 1)
                self.assertEqual(run.get_run_status()["state"], "complete")

    def test_truncated_first_bpf_snapshot_is_not_recovered(self) -> None:
        with BpfFile(BPF_SAMPLE) as source:
            record_index = BpfFile.snapshot_start(0, source.layout.block_records) + 5
            cut_at = source._payload_offsets[record_index] - 4
            raw = BPF_SAMPLE.read_bytes()[:cut_at]
        with tempfile.TemporaryDirectory() as tmpdir:
            truncated = Path(tmpdir) / "truncated_first.bpf"
            truncated.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "complete BPF snapshot block"):
                BpfFile(truncated)

    def test_corrupt_partial_record_after_valid_prefix_is_fatal(self) -> None:
        with BpfFile(BPF_SAMPLE) as source:
            second_snapshot = BpfFile.snapshot_start(1, source.layout.block_records)
            tail_start = source._payload_offsets[second_snapshot + 1] - 4
            raw = BPF_SAMPLE.read_bytes()[:tail_start] + struct.pack("<i", 1234567) + b"garbage"
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted = Path(tmpdir) / "corrupt_partial_tail.bpf"
            corrupted.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "corrupt partial BPF data"):
                BpfFile(corrupted)
            output = Path(tmpdir) / "existing.h5"
            output.write_bytes(b"sentinel")
            with self.assertRaisesRegex(ValueError, "corrupt partial BPF data"):
                write_hdf5(corrupted, output, overwrite=True)
            self.assertEqual(b"sentinel", output.read_bytes())

    def test_corruption_in_complete_middle_record_is_fatal(self) -> None:
        with BpfFile(BPF_SAMPLE) as source:
            record_index = BpfFile.snapshot_start(3, source.layout.block_records) + 5
            payload_end = source._payload_offsets[record_index] + source._payload_lengths[record_index]
            raw = bytearray(BPF_SAMPLE.read_bytes())
        closing = struct.unpack_from("<i", raw, payload_end)[0]
        struct.pack_into("<i", raw, payload_end, closing + 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted = Path(tmpdir) / "corrupted_middle.bpf"
            corrupted.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "marker mismatch"):
                BpfFile(corrupted)

    def test_negative_bpf_density_is_rejected(self) -> None:
        with BpfFile(BPF_SAMPLE) as source:
            record_index = BpfFile.snapshot_start(0, source.layout.block_records) + 8
            payload_start = source._payload_offsets[record_index]
            raw = bytearray(BPF_SAMPLE.read_bytes())
        struct.pack_into("<d", raw, payload_start, -1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted = Path(tmpdir) / "negative_density.bpf"
            corrupted.write_bytes(raw)
            with BpfFile(corrupted) as bpf:
                with self.assertRaisesRegex(ValueError, "negative mass density"):
                    next(bpf.iter_snapshots())

    def test_partial_final_bpf_snapshot_is_dropped_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            truncated = Path(tmpdir) / "truncated.bpf"
            with BpfFile(BPF_SAMPLE) as source:
                record_index = BpfFile.snapshot_start(7, source.layout.block_records) + 5
                record_start = source._payload_offsets[record_index] - 4
                payload_length = source._payload_lengths[record_index]
                cut_at = record_start + 4 + max(1, payload_length // 2)
                raw = BPF_SAMPLE.read_bytes()[:cut_at]
            truncated.write_bytes(raw)
            with BpfFile(truncated) as bpf:
                self.assertEqual(bpf.layout.n_snapshots, 7)
                self.assertTrue(bpf.layout.incomplete_final_snapshot)
                self.assertGreater(bpf.layout.partial_final_record_bytes, 0)
            output = Path(tmpdir) / "truncated.h5"
            complete_output = Path(tmpdir) / "complete.h5"
            write_hdf5(truncated, output)
            write_hdf5(BPF_SAMPLE, complete_output)
            with HeliosRun(output) as run, HeliosRun(complete_output) as complete_run:
                status = run.get_run_status()
                self.assertEqual(status["state"], "truncated")
                self.assertEqual(status["valid_snapshot_count"], 7)
                self.assertTrue(status["dropped_partial_final_block"])
                self.assertEqual(run.n_snapshots, 7)
                for field_name in run.list_fields():
                    np.testing.assert_array_equal(run.get_field(field_name), complete_run.get_field(field_name)[:7])

    def test_nonzero_bpf_trailer_is_not_accepted_as_documented_trailer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted = Path(tmpdir) / "corrupted_trailer.bpf"
            payload = bytearray(BPF_SAMPLE.read_bytes())
            trailer_start = len(payload) - (4 + 240 + 4)
            payload[trailer_start + 4] = 1
            corrupted.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "unrecognized trailing BPF records"):
                BpfFile(corrupted)

    def test_bpf_hdf5_schema_field_discovery_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "bpf_sample.h5"
            write_hdf5(BPF_SAMPLE, output, overwrite=True, compression="lzf")
            with HeliosRun(output) as run:
                self.assertEqual(run.n_snapshots, 8)
                self.assertEqual(run.n_zones, 500)
                fields = set(run.list_fields())
                self.assertIn("mass_density_g_cm3", fields)
                self.assertIn("density", fields)
                self.assertIn("ionization_fractions_by_zone_charge", fields)
                self.assertIn("radiation_net_flux_rmin_j_s_cm2_eV", fields)
                self.assertIn("bpf_record_03", fields)
                self.assertIn("bpf_record_29", fields)
                self.assertIn("planck_mean_opacity_absorption_cm2_g", fields)
                self.assertIn("planck_mean_opacity_emission_cm2_g", fields)
                self.assertIn("rosseland_mean_opacity_cm2_g", fields)
                self.assertIn("laser_source_j_g", fields)
                self.assertIn("radiation_net_heating_j_g_s", fields)
                self.assertIn("radiation_energy_density_j_cm3", fields)
                self.assertIn("radiation_energy_j_g", fields)
                self.assertIn("plasma_pressure_j_cm3", fields)
                self.assertIn("total_pressure_j_cm3", fields)
                self.assertIn("laser_source_bpf_sampled_j_g", fields)
                plasma = run.get_field("pressure_j_cm3")
                explicit_plasma = run.get_field("plasma_pressure_j_cm3")
                total = run.get_field("total_pressure_j_cm3")
                np.testing.assert_array_equal(explicit_plasma, plasma)
                np.testing.assert_array_equal(run.get_field("pressure"), total)
                np.testing.assert_array_equal(total, plasma + run.get_field("pressure_radiation"))
                self.assertEqual(run.get_field_metadata("pressure").alias_of, "total_pressure_j_cm3")
                self.assertEqual(run.get_field_metadata("pressure").quantity, "total_pressure")

                density = run.get_field("density")
                canonical_density = run.get_field("mass_density_g_cm3")
                np.testing.assert_allclose(density, canonical_density)
                self.assertEqual(run.get_field_metadata("density").alias_of, "mass_density_g_cm3")
                self.assertEqual(run.get_field_metadata("velocity").alias_of, "zone_outer_velocity_cm_s")
                self.assertEqual(run.get_field_metadata("laser_source").alias_of, "laser_source_j_g")

                ion_meta = run.get_field_metadata("ionization_fractions_by_zone_charge")
                self.assertEqual(ion_meta.source, "bpf")
                self.assertEqual(ion_meta.dimensions, ("time", "zone", "charge_state"))
                self.assertEqual(ion_meta.status, "validated")
                self.assertIn("ionization_fraction", run.plotting_modes_for_field("ionization_fractions_by_zone_charge"))

                flux_meta = run.get_field_metadata("radiation_net_flux_rmin_j_s_cm2_eV")
                self.assertEqual(flux_meta.unit, "J/s/cm2/eV")
                self.assertEqual(flux_meta.dimensions, ("time", "frequency"))
                self.assertIn("spectral_evolution", run.plotting_modes_for_field("radiation_net_flux_rmin_j_s_cm2_eV"))

                unknown_meta = run.get_field_metadata("bpf_record_03")
                self.assertEqual(unknown_meta.status, "unknown_bpf_record")
                self.assertEqual(unknown_meta.unit, "")
                self.assertEqual(unknown_meta.dimensions, ("time", "bpf_record_value"))

                radiation_meta = run.get_field_metadata("radiation_energy_density_j_cm3")
                self.assertEqual(radiation_meta.status, "validated")
                self.assertEqual(radiation_meta.unit, "J/cm3")
                self.assertEqual(run.get_field_metadata("pressure_radiation").alias_of, "radiation_pressure_j_cm3")
                self.assertEqual(run.get_field_metadata("radiation_energy").alias_of, "radiation_energy_j_g")

                record14_meta = run.get_field_metadata("bpf_record_14")
                self.assertEqual(record14_meta.status, "validated")
                self.assertEqual(record14_meta.unit, "J/cm3")

                raw_laser_meta = run.get_field_metadata("bpf_record_29")
                self.assertEqual(raw_laser_meta.status, "unknown_bpf_record")
                self.assertEqual(raw_laser_meta.label, "BPF record 29")

                opacity_meta = run.get_field_metadata("rosseland_mean_opacity_cm2_g")
                self.assertEqual(opacity_meta.status, "mapped")
                self.assertEqual(opacity_meta.unit, "cm2/g")
                self.assertEqual(opacity_meta.dimensions, ("time", "zone"))

                ion = run.get_field("ionization_fractions_by_zone_charge")
                np.testing.assert_allclose(ion.sum(axis=2), 1.0, rtol=1.0e-6, atol=1.0e-8)
            derived = load_run_data(output)
            self.assertIsNone(derived.zone_atomic_weight)
            self.assertEqual(8, derived.time_s.size)

    def test_bpf_primary_log_input_cross_validates_overlapping_log_fields(self) -> None:
        log_simulation = HeliosParser().parse(LOG_BPF_SAMPLE)
        overlapping_fields = (
            "radius",
            "zone_width",
            "density",
            "velocity",
            "temperature_radiation",
            "temperature_i",
            "temperature_e",
            "pressure_i",
            "pressure_e",
            "pressure_radiation",
            "electron_density",
            "mean_charge",
            "artificial_viscosity",
            "ion_energy",
            "radiation_energy",
            "radiation_heating",
            "radiation_cooling",
            "radiation_net_heating",
            "laser_deposition",
            "laser_source",
            "pressure",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cu_from_log.h5"
            with mock.patch.object(HeliosParser, "parse", side_effect=AssertionError("BPF-primary conversion must stream LOG snapshots")):
                write_hdf5(LOG_BPF_SAMPLE, output, overwrite=True, compression="lzf")
            with HeliosRun(output) as run:
                self.assertEqual(run.get_metadata()["source_precedence"], "bpf_primary_log_metadata_exo_optional")
                for name in overlapping_fields:
                    with self.subTest(field=name):
                        if name == "pressure":
                            np.testing.assert_allclose(
                                run.get_field("total_pressure_j_cm3"),
                                log_simulation.fields[name],
                                rtol=1.0e-3,
                                atol=1.0,
                            )
                        else:
                            _assert_scale_close(self, run.get_field(name), log_simulation.fields[name], rtol=1.0e-3, atol=1.0e-10)
                np.testing.assert_allclose(
                    run.get_field("pressure_j_cm3"),
                    log_simulation.fields["pressure_i"] + log_simulation.fields["pressure_e"],
                    rtol=1.0e-3,
                    atol=1.0,
                )

    def test_input_routing_accepts_log_bpf_and_missing_companions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            from_log = tmpdir_path / "from_log.h5"
            from_bpf = tmpdir_path / "from_bpf.h5"
            write_hdf5(LOG_BPF_SAMPLE, from_log, overwrite=True, compression="lzf")
            write_hdf5(LOG_BPF_COMPANION, from_bpf, overwrite=True, compression="lzf")
            with HeliosRun(from_log) as log_run, HeliosRun(from_bpf) as bpf_run:
                self.assertEqual(log_run.get_metadata()["source_precedence"], "bpf_primary_log_metadata_exo_optional")
                self.assertEqual(bpf_run.get_metadata()["source_precedence"], "bpf_primary_log_metadata_exo_optional")
                self.assertIn("bpf", log_run.get_metadata()["source_files"])
                self.assertIn("log", bpf_run.get_metadata()["source_files"])
                np.testing.assert_allclose(log_run.get_field("density"), bpf_run.get_field("density"))

            log_only_source = tmpdir_path / LOG_BPF_SAMPLE.name
            shutil.copyfile(LOG_BPF_SAMPLE, log_only_source)
            log_only = tmpdir_path / "log_only.h5"
            write_hdf5(log_only_source, log_only, overwrite=True, compression="lzf")
            with HeliosRun(log_only) as run:
                self.assertEqual(run.get_metadata()["source_precedence"], "log_only")
                self.assertIn("density", run.list_fields())
                self.assertFalse(run.has_field("ionization_fractions_by_zone_charge"))

            bpf_only_source = tmpdir_path / BPF_SAMPLE.name
            shutil.copyfile(BPF_SAMPLE, bpf_only_source)
            bpf_only = tmpdir_path / "bpf_only.h5"
            write_hdf5(bpf_only_source, bpf_only, overwrite=True, compression="lzf")
            with HeliosRun(bpf_only) as run:
                self.assertEqual(run.get_metadata()["source_precedence"], "bpf_primary")
                self.assertIn("ionization_fractions_by_zone_charge", run.list_fields())
                self.assertIn("density", run.list_fields())
                self.assertEqual(run.get_field_metadata("laser_source_j_g").source, "derived")

    def test_aligned_log_companion_supplies_cumulative_laser_source(self) -> None:
        log_simulation = HeliosParser().parse(LOG_BPF_SAMPLE)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            log_copy = tmpdir_path / LOG_BPF_SAMPLE.name
            bpf_copy = tmpdir_path / LOG_BPF_COMPANION.name
            shutil.copyfile(LOG_BPF_SAMPLE, log_copy)
            shutil.copyfile(LOG_BPF_COMPANION, bpf_copy)
            output = tmpdir_path / "cu_with_companion.h5"
            write_hdf5(log_copy, output, overwrite=True, compression="lzf")
            with HeliosRun(output) as run:
                self.assertEqual(run.get_metadata()["source_precedence"], "bpf_primary_log_metadata_exo_optional")
                self.assertEqual(run.get_field_metadata("laser_source_j_g").source, "log")
                np.testing.assert_allclose(run.get_field("laser_source"), log_simulation.fields["laser_source"])



    def test_malformed_optional_log_falls_back_to_standalone_bpf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bpf_copy = tmp / BPF_SAMPLE.name
            shutil.copyfile(BPF_SAMPLE, bpf_copy)
            bpf_copy.with_suffix(".log").write_text("broken optional companion\n", encoding="utf-8")
            output = tmp / "standalone.h5"
            write_hdf5(bpf_copy, output)
            with HeliosRun(output) as run:
                metadata = run.get_metadata()
                self.assertEqual(metadata["source_precedence"], "bpf_primary")
                self.assertNotIn("log", metadata["source_files"])
                self.assertEqual(run.get_field_metadata("laser_source_j_g").source, "derived")

    def test_interior_time_misalignment_discards_companion_metadata_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_copy = tmp / LOG_BPF_SAMPLE.name
            bpf_copy = tmp / LOG_BPF_COMPANION.name
            shutil.copyfile(LOG_BPF_COMPANION, bpf_copy)
            text = LOG_BPF_SAMPLE.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            header_lines = [index for index, line in enumerate(lines) if re.match(r"^\s*Cycle\s+Time\s+\(s\)", line)]
            time_row = header_lines[1] + 1
            tokens = lines[time_row].split()
            original = Decimal(tokens[1].replace("D", "E").replace("d", "E"))
            quantum = Decimal(1).scaleb(original.as_tuple().exponent)
            tokens[1] = str(original + quantum)
            lines[time_row] = "  ".join(tokens)
            log_copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
            output = tmp / "misaligned.h5"
            write_hdf5(log_copy, output)
            with HeliosRun(output) as run:
                metadata = run.get_metadata()
                self.assertEqual(metadata["source_precedence"], "bpf_primary")
                self.assertNotIn("log", metadata["source_files"])
                self.assertEqual(run.get_field_metadata("laser_source_j_g").source, "derived")
                self.assertIn("laser_source_bpf_sampled_j_g", run.list_fields())

    def test_legacy_sparse_file_still_opens_without_field_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_sparse.h5"
            with h5py.File(path, "w") as handle:
                grid = handle.create_group("grid")
                grid.create_dataset("zone_id", data=np.asarray([1, 2], dtype=np.int32))
                grid.create_dataset("x", data=np.asarray([0.5, 1.5], dtype=np.float64))
                grid.create_dataset("zone_width", data=np.asarray([1.0, 1.0], dtype=np.float64))
                grid.create_dataset("zone_region_id", data=np.asarray([1, 1], dtype=np.int32))
                grid.create_dataset("zone_material_index", data=np.asarray([1, 1], dtype=np.int32))
                time = handle.create_group("time")
                time.create_dataset("time", data=np.asarray([0.0, 1.0], dtype=np.float64))
                fields = handle.create_group("fields")
                fields.create_dataset("density", data=np.ones((2, 2), dtype=np.float64))
                regions = handle.create_group("regions")
                regions.create_dataset("region_index", data=np.asarray([1], dtype=np.int32))
                regions.create_dataset("min_zone_index", data=np.asarray([1], dtype=np.int32))
                regions.create_dataset("max_zone_index", data=np.asarray([2], dtype=np.int32))
                materials = handle.create_group("materials")
                materials.create_dataset("index", data=np.asarray([1], dtype=np.int32))
                handle.create_group("diagnostics")
                handle.create_group("metadata")
            with HeliosRun(path) as run:
                self.assertEqual(run.list_fields(), ["density"])
                metadata = run.get_field_metadata("density")
                self.assertEqual(metadata.dimensions, ("time", "zone"))
                self.assertEqual(metadata.status, "legacy")
                self.assertTrue(run.has_field("density"))
                self.assertFalse(run.has_field("temperature_e"))
                with self.assertRaises(KeyError):
                    run.get_field("temperature_e")


if __name__ == "__main__":
    unittest.main()
