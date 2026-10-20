from __future__ import annotations

import hashlib
import json
import unittest
from importlib import resources
from pathlib import Path

import _test_bootstrap  # noqa: F401

from helios.services.derived.plasmon_electron_policy import _baseline_table_bundle
from helios.services.derived.plasmon_reference_data import (
    USER_DRIVEN_AL_DISPERSION_REFERENCE,
)
from helios.services.derived.xcom_hook import fallback_table_available


class PackagedScientificResourceTests(unittest.TestCase):
    def test_runtime_json_tables_are_packaged_and_match_repository_sources(
        self,
    ) -> None:
        resource_root = resources.files("helios.services.derived").joinpath("resources")
        repo_root = Path(__file__).resolve().parents[1]
        pairs = (
            (
                resource_root.joinpath("hed_helios_cold_electronic_baseline_core.json"),
                repo_root / "hed_helios_cold_electronic_baseline_core.json",
            ),
            (
                resource_root.joinpath("xcom_fallback_1keV_12keV_extended.json"),
                repo_root / "x-com_fallback" / "xcom_fallback_1keV_12keV_extended.json",
            ),
        )
        for packaged, repository in pairs:
            with self.subTest(resource=packaged.name):
                self.assertTrue(packaged.is_file())
                self.assertTrue(repository.is_file())
                self.assertEqual(
                    hashlib.sha256(packaged.read_bytes()).digest(),
                    hashlib.sha256(repository.read_bytes()).digest(),
                )
                self.assertIsInstance(
                    json.loads(packaged.read_text(encoding="utf-8")), dict
                )

    def test_scientific_loaders_use_packaged_resources(self) -> None:
        self.assertTrue(fallback_table_available())
        self.assertIn(
            "materials",
            json.loads(
                resources.files("helios.services.derived")
                .joinpath("resources", "xcom_fallback_1keV_12keV_extended.json")
                .read_text(encoding="utf-8")
            ),
        )
        baseline, source = _baseline_table_bundle()
        self.assertTrue(baseline)
        self.assertTrue(
            source.endswith("hed_helios_cold_electronic_baseline_core.json")
        )
        self.assertTrue(USER_DRIVEN_AL_DISPERSION_REFERENCE)

    def test_packaging_configs_include_scientific_resources(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        spec = (
            repo_root / "packaging/pyinstaller/helios_parser_viewer.spec"
        ).read_text(encoding="utf-8")
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"resources/*.json"', pyproject)
        self.assertIn('"reference_data/plasmon/*.json"', pyproject)
        self.assertGreaterEqual(spec.count('"helios/services/derived/resources"'), 2)
        self.assertGreaterEqual(
            spec.count('"helios/services/derived/reference_data/plasmon"'), 2
        )


if __name__ == "__main__":
    unittest.main()
