from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock

import _test_bootstrap  # noqa: F401
from helios.services.derived.xcom_hook import (
    ColdAttenuationRequest, ColdAttenuationZone, PrecomputedColdTableBackend,
    lookup_fallback_mu_rho,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "x-com_fallback/xcom_fallback_1keV_12keV_extended.json"


class XcomTablePrecisionTests(unittest.TestCase):
    def test_independent_copper_nist_anchors(self):
        # NIST X-Ray Mass Attenuation Coefficients, Copper (z29), cm^2/g.
        for energy, coefficient in ((1., 10570.), (8., 52.55), (10., 215.9)):
            actual = lookup_fallback_mu_rho("Cu", energy).mu_rho_cm2_g
            with self.subTest(energy=energy):
                self.assertAlmostEqual(coefficient, actual, delta=coefficient * .001)

    def test_copper_and_iron_edge_sides_match_native_reference(self):
        references = (("Cu", 8.95, 38.63), ("Cu", 8.9788, 38.29),
                      ("Cu", 8.9790, 278.3), ("Fe", 7.1119, 53.20),
                      ("Fe", 7.1121, 407.7), ("Fe", 7.2, 395.7))
        for material, energy, coefficient in references:
            with self.subTest(material=material, energy=energy):
                actual = lookup_fallback_mu_rho(material, energy).mu_rho_cm2_g
                self.assertAlmostEqual(coefficient, actual, delta=coefficient * .0005)

    def test_edges_keep_both_limits_and_do_not_interpolate_across_jump(self):
        table = json.loads(TABLE.read_text())
        for key, material in table["materials"].items():
            rows = material["rows"]
            for i, row in enumerate(rows):
                if row.get("edge_side") != "above":
                    continue
                below = rows[i-1]
                with self.subTest(material=key, edge=row["energy_eV"]):
                    self.assertEqual("below", below["edge_side"])
                    self.assertEqual(row["energy_eV"], below["energy_eV"])
                    self.assertGreater(row["mu_rho_total_cm2_g"], below["mu_rho_total_cm2_g"])
                    energy = row["energy_eV"] / 1000
                    exact = lookup_fallback_mu_rho(key, energy)
                    self.assertAlmostEqual(below["mu_rho_total_cm2_g"], exact.mu_rho_cm2_g)

    def test_provenance_validation_and_component_sum(self):
        table = json.loads(TABLE.read_text())
        metadata = table["metadata"]
        self.assertEqual(hashlib.sha256((ROOT / "XCOM.tar.gz").read_bytes()).hexdigest(), metadata["source_archive_sha256"])
        self.assertEqual(2, metadata["schema_version"])
        for key, material in table["materials"].items():
            self.assertLessEqual(material["validation"]["max_holdout_relative_error"], 2e-4)
            self.assertGreaterEqual(material["validation"]["holdout_probes"], 200)
            xs = [r["energy_eV"] for r in material["rows"]]
            self.assertEqual(sorted(xs), xs)
            self.assertEqual((1000., 12000.), (xs[0], xs[-1]))
            for row in material["rows"]:
                total = sum(row[c] for c in ("coherent_cm2_g", "incoherent_cm2_g", "photoelectric_cm2_g", "pair_nuclear_cm2_g", "pair_electron_cm2_g"))
                with self.subTest(material=key, energy=row["energy_eV"]):
                    self.assertTrue(math.isclose(total, row["mu_rho_total_cm2_g"], rel_tol=1e-12))

    def test_default_lookup_rejects_unvalidated_domain(self):
        for energy in (.9, 12.01, 0., -1., float("nan"), float("inf")):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                lookup_fallback_mu_rho("Cu", energy)

    def test_repeated_calculations_hash_the_immutable_table_once(self):
        request = ColdAttenuationRequest(0, "front", 1., (8.,),
            (ColdAttenuationZone(1, 1, 1, "Cu", 8.96, 1e-4),))
        backend = PrecomputedColdTableBackend()
        with mock.patch("helios.services.derived.xcom_hook._fallback_table_fingerprint", return_value="table-hash") as digest:
            for _ in range(10):
                backend.compute_transmission(request)
        self.assertEqual(1, digest.call_count)

    def test_standalone_helper_agrees_at_edges_and_rejects_extrapolation(self):
        spec = importlib.util.spec_from_file_location("standalone_xcom_table", ROOT / "x-com_fallback/xcom_table_fallback_extended.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for energy in (8.95, 8.9788, 8.979, 9.):
            self.assertAlmostEqual(helper.lookup_mu_rho("Cu", energy*1000), lookup_fallback_mu_rho("Cu", energy).mu_rho_cm2_g)
        with self.assertRaises(ValueError):
            helper.lookup_mu_rho("Cu", 900.)

    @unittest.skipUnless(shutil.which("gfortran"), "native XCOM validation needs gfortran")
    def test_committed_table_against_independent_native_holdouts(self):
        result = subprocess.run([sys.executable, str(ROOT / "x-com_fallback/build_xcom_fallback_from_xcom.py"), "--verify-only"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
