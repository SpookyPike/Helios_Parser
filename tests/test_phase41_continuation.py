from __future__ import annotations

from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, get_app
from helios.platform.archive_utils import ARCHIVE_TAR_GZ, ARCHIVE_ZIP, archive_type_for_path, extract_archive, inspect_archive
from helios.platform.registry import _inspect_hdf5, _inspect_log, build_dataset_registry
from helios.runtime import RunContext
from helios.services.derived import DerivedAnalysisParameters, build_cold_attenuation_request, load_run_data
from helios.services.derived.selection import build_analysis_geometry
from helios_viewer.icon import application_icon_path, canonical_icon_png_path, ensure_packaging_icon, load_application_icon


class _FakeHdf5Run:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def summary(self):
        return {
            "n_zones": 12,
            "n_snapshots": 7,
            "geometry": "PLANAR",
            "available_fields": ("density",),
        }

    def get_metadata(self):
        return {}

    def get_regions(self):
        raise KeyError("regions missing")


class Phase41ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_archive_helpers_support_windows_safe_zip_and_tar_gz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload"
            payload.mkdir()
            (payload / "alpha.txt").write_text("alpha", encoding="utf-8")

            zip_path = root / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as bundle:
                bundle.write(payload / "alpha.txt", arcname="bundle/alpha.txt")

            tar_path = root / "bundle.tar.gz"
            with tarfile.open(tar_path, "w:gz") as bundle:
                bundle.add(payload / "alpha.txt", arcname="bundle/alpha.txt")

            self.assertEqual(archive_type_for_path(zip_path), ARCHIVE_ZIP)
            self.assertEqual(archive_type_for_path(tar_path), ARCHIVE_TAR_GZ)

            zip_info = inspect_archive(zip_path)
            tar_info = inspect_archive(tar_path)
            self.assertEqual(zip_info.archive_type, ARCHIVE_ZIP)
            self.assertEqual(tar_info.archive_type, ARCHIVE_TAR_GZ)
            self.assertIn("bundle", zip_info.top_level_entries)
            self.assertIn("bundle", tar_info.top_level_entries)

            zip_out = extract_archive(zip_path, root / "zip_out")
            tar_out = extract_archive(tar_path, root / "tar_out")
            self.assertTrue((zip_out / "bundle" / "alpha.txt").exists())
            self.assertTrue((tar_out / "bundle" / "alpha.txt").exists())

    def test_registry_discovers_archives_and_assets_in_temp_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "three_icons.png").write_bytes(b"fake-png")
            (root / "NRL_Formulary_2023.pdf").write_bytes(b"%PDF-1.4\n")

            zip_path = root / "helios_xcom_integration.zip"
            with zipfile.ZipFile(zip_path, "w") as bundle:
                bundle.writestr("helios_xcom_integration/readme.txt", "ok")

            tar_path = root / "XCOM.tar.gz"
            with tarfile.open(tar_path, "w:gz") as bundle:
                source = root / "xcom.txt"
                source.write_text("xcom", encoding="utf-8")
                bundle.add(source, arcname="XCOM/readme.txt")

            registry = build_dataset_registry(root, include_outputs=False)
            records = {record.filename: record for record in registry.records}
            self.assertEqual(records["helios_xcom_integration.zip"].artifact_type, "archive")
            self.assertEqual(records["XCOM.tar.gz"].artifact_type, "archive")
            self.assertEqual(records["three_icons.png"].artifact_type, "image_asset")
            self.assertEqual(records["NRL_Formulary_2023.pdf"].artifact_type, "pdf_reference")
            self.assertIn("xcom_bundle", records["helios_xcom_integration.zip"].notes)

    def test_registry_log_and_hdf5_inspection_degrade_gracefully_for_legacy_layouts(self) -> None:
        fake_header = type(
            "Header",
            (),
            {
                "n_zones": 42,
                "n_snapshots": 9,
                "geometry": "PLANAR",
                "laser_source": None,
                "photon_energy_grid": (1.0, 2.0, 3.0),
            },
        )()
        with mock.patch("helios.platform.registry.inspect", return_value=fake_header):
            record = _inspect_log(Path("legacy.log"))
        self.assertEqual(record.artifact_type, "helios_log")
        self.assertIn("header_without_spatial_regions", record.notes)
        self.assertIn("photon_grid_present", record.notes)

        with mock.patch("helios.platform.registry.HeliosRun", return_value=_FakeHdf5Run()):
            record = _inspect_hdf5(Path("legacy.h5"))
        self.assertEqual(record.artifact_type, "hdf5")
        self.assertTrue(record.directly_usable)
        self.assertIn("missing_regions_group", record.notes)

    def test_icon_loading_uses_real_asset_and_missing_asset_fallback_is_safe(self) -> None:
        self.assertIsNotNone(canonical_icon_png_path())
        actual_icon_path = application_icon_path()
        self.assertIsNotNone(actual_icon_path)
        assert actual_icon_path is not None
        self.assertTrue(actual_icon_path.exists())
        self.assertEqual(actual_icon_path.suffix.lower(), ".png")
        packaging_icon = ensure_packaging_icon()
        self.assertIsNotNone(packaging_icon)
        assert packaging_icon is not None
        self.assertTrue(packaging_icon.exists())
        self.assertEqual(packaging_icon.suffix.lower(), ".ico")
        self.assertFalse(load_application_icon().isNull())

        missing = Path.cwd() / "__missing_icon__.png"
        with mock.patch("helios_viewer.icon.icon_candidate_paths", return_value=(missing,)):
            self.assertIsNone(application_icon_path())
            self.assertTrue(load_application_icon().isNull())

    def test_optional_xcom_request_seam_builds_zone_payload_for_real_run(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        region_ids = tuple(int(value) for value in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32)))
        material_ids = tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))))
        context = RunContext(
            path=dataset.path,
            summary=dict(dataset.summary),
            metadata=dict(dataset.metadata),
            fields=(),
            diagnostics=(),
            time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
            static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
            zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
            zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
            has_dynamic_radius=dataset.radius_cm is not None,
            snapshot_index=0,
            map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
            slice_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
            selected_region_ids=region_ids,
            selected_material_ids=material_ids,
        )
        parameters = DerivedAnalysisParameters()
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        request = build_cold_attenuation_request(
            dataset,
            context,
            snapshot_index=0,
            parameters=parameters,
            geometry=geometry,
            photon_energies_kev=(8.0, 10.0, 12.0),
        )
        self.assertEqual(request.snapshot_index, 0)
        self.assertEqual(request.observation_side, "front")
        self.assertEqual(request.photon_energies_kev, (8.0, 10.0, 12.0))
        self.assertEqual(len(request.zones), 500)
        self.assertGreater(request.zones[0].density_g_cm3, 0.0)
        self.assertGreater(request.zones[0].path_length_cm, 0.0)


if __name__ == "__main__":
    unittest.main()
