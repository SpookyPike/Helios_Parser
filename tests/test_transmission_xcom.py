from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.selection import build_analysis_geometry
from helios.services.derived.transmission import (
    _classify_zone_regimes,
    apply_transmission_model,
    evaluate_transmission,
    refine_transmission_with_xcom,
)
from helios.services.derived.xcom_hook import (
    ColdAttenuationBackendStatus,
    ColdAttenuationRequest,
    ColdAttenuationResult,
    ColdAttenuationZone,
    HeliosXcomBackend,
    PersistentColdAttenuationCache,
    build_cold_attenuation_request,
    canonical_fallback_table_key,
    load_precomputed_cold_backend,
    describe_optional_cold_backend,
    lookup_fallback_mu_rho,
    load_optional_cold_backend,
    material_display_labels_by_id,
    normalize_material_key,
    resolve_material_identities,
)


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int = 0) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        selected_region_ids=tuple(int(value) for value in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


class _FakeBackend:
    def __init__(self, *, transmission: tuple[float, ...] = (0.73,), optical_depth: tuple[float, ...] = (0.314,)) -> None:
        self.backend_fingerprint = "fake-backend-fingerprint"
        self.calls = 0
        self._transmission = tuple(float(value) for value in transmission)
        self._optical_depth = tuple(float(value) for value in optical_depth)

    def compute_transmission(self, request) -> ColdAttenuationResult:
        self.calls += 1
        return ColdAttenuationResult(
            energies_kev=np.asarray(request.photon_energies_kev, dtype=np.float64),
            transmission=np.asarray(self._transmission, dtype=np.float64),
            metadata={
                "backend_name": "XCOM",
                "backend_fingerprint": self.backend_fingerprint,
                "attenuation_mode": "total_with_coherent",
                "optical_depth": list(self._optical_depth),
                "material_budgets": [
                    {
                        "label": str(request.zones[0].material_label or "unknown"),
                        "areal_density_g_cm2": float(request.zones[0].density_g_cm3) * float(request.zones[0].path_length_cm),
                        "optical_depth": list(self._optical_depth),
                    }
                ],
            },
        )


class _FakeTableBackend:
    def __init__(self, *, transmission: tuple[float, ...] = (0.61,), optical_depth: tuple[float, ...] = (0.494,)) -> None:
        self.backend_fingerprint = "fake-table-fingerprint"
        self.backend_name = "XCOM table"
        self.calls = 0
        self._transmission = tuple(float(value) for value in transmission)
        self._optical_depth = tuple(float(value) for value in optical_depth)

    def compute_transmission(self, request) -> ColdAttenuationResult:
        self.calls += 1
        return ColdAttenuationResult(
            energies_kev=np.asarray(request.photon_energies_kev, dtype=np.float64),
            transmission=np.asarray(self._transmission, dtype=np.float64),
            metadata={
                "backend_name": self.backend_name,
                "backend_fingerprint": self.backend_fingerprint,
                "attenuation_mode": "total_with_coherent",
                "optical_depth": list(self._optical_depth),
                "material_budgets": [
                    {
                        "label": str(request.zones[0].material_label or "unknown"),
                        "areal_density_g_cm2": float(request.zones[0].density_g_cm3) * float(request.zones[0].path_length_cm),
                        "optical_depth": list(self._optical_depth),
                    }
                ],
                "source": "precomputed_xcom_table",
                "interpolation_mode": "interpolated",
                "interpolation_note": "Precomputed XCOM table interpolated in log-log space across energy.",
            },
        )


class TransmissionXcomTests(unittest.TestCase):
    def _fallback_table_json(self) -> dict[str, object]:
        path = Path(__file__).resolve().parents[1] / "x-com_fallback" / "xcom_fallback_1keV_12keV_extended.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_transmission_inputs(self):
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=0)
        parameters = DerivedAnalysisParameters(transmission_photon_energy_kev=8.0)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        return path, dataset, context, parameters, geometry

    def _active_zone_material_labels(self, dataset, context, parameters, geometry) -> tuple[np.ndarray, dict[int, str | None]]:
        request = build_cold_attenuation_request(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            photon_energies_kev=(8.0,),
        )
        active_zone_indices = np.asarray([int(zone.zone_index) - 1 for zone in request.zones], dtype=np.int32)
        labels = {int(zone.zone_index) - 1: zone.material_label for zone in request.zones}
        return active_zone_indices, labels

    def test_thomson_baseline_survives_without_backend(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        unavailable = ColdAttenuationBackendStatus(
            available=False,
            status="unavailable",
            message="Optional XCOM backend is not installed.",
            backend_name="XCOM",
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable):
            result = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        self.assertEqual(result.model_type, "thomson")
        self.assertEqual(result.source, "baseline")
        self.assertIsNotNone(result.cold_refinement)
        assert result.cold_refinement is not None
        self.assertEqual(result.cold_refinement.backend_status, "unavailable")
        self.assertGreater(result.thomson_transmission, 0.0)

    def test_build_cold_request_resolves_materials_from_eos_metadata(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        request = build_cold_attenuation_request(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            photon_energies_kev=(8.0,),
        )
        self.assertGreater(len(request.zones), 0)
        first_zone = request.zones[0]
        self.assertEqual("Fe", first_zone.material_label)
        self.assertEqual("resolved_from_eos", first_zone.material_resolution_status)
        self.assertEqual("Fe", first_zone.material_display_label)

    def test_material_resolution_prefers_eos_and_formats_viewer_labels(self) -> None:
        materials = {
            "index": np.asarray([1, 2, 3], dtype=np.int32),
            "eos_model": np.asarray(["EOSOPA", "EOSOPA", ""], dtype=object),
            "eos_file_path": np.asarray(["C:/tables/Cu.prp", "C:/tables/C2H4O.prp", ""], dtype=object),
            "opacity_model": np.asarray(["EOSOPA", "", "OpacityOnly"], dtype=object),
            "opacity_file_path": np.asarray(["", "", "C:/tables/SiO2.prp"], dtype=object),
        }
        resolutions = resolve_material_identities(materials)
        labels = material_display_labels_by_id(materials)
        self.assertEqual("resolved_from_eos", resolutions[1].status)
        self.assertEqual("Cu", resolutions[1].backend_label)
        self.assertEqual("resolved_from_eos", resolutions[2].status)
        self.assertEqual("C2H4O", resolutions[2].backend_label)
        self.assertEqual("epoxy (from EOS)", labels[2])
        self.assertEqual("resolved_from_opacity", resolutions[3].status)
        self.assertEqual("silica (from opacity)", labels[3])

    def test_material_resolution_reports_guessed_and_unresolved_cases(self) -> None:
        materials = {
            "index": np.asarray([1, 2], dtype=np.int32),
            "eos_model": np.asarray(["EOSOPA", ""], dtype=object),
            "eos_file_path": np.asarray(["C:/tables/glue_block.prp", ""], dtype=object),
            "opacity_model": np.asarray(["", ""], dtype=object),
            "opacity_file_path": np.asarray(["", ""], dtype=object),
        }
        resolutions = resolve_material_identities(materials)
        self.assertEqual("guessed", resolutions[1].status)
        self.assertEqual("C2H4O", resolutions[1].backend_label)
        self.assertEqual("unresolved", resolutions[2].status)

    def test_backend_probe_reports_source_only_without_wrapper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcom_probe_") as temp_dir:
            source_path = Path(temp_dir) / "XCOM.tar.gz"
            source_path.write_bytes(b"source")
            missing_wrapper = Path(temp_dir) / "missing_wrapper.zip"
            with (
                mock.patch("helios.services.derived.xcom_hook._wrapper_archive_path", return_value=missing_wrapper),
                mock.patch("helios.services.derived.xcom_hook._source_archive_path", return_value=source_path),
            ):
                status = describe_optional_cold_backend(force=True)
        self.assertEqual("source_only", status.status)
        self.assertFalse(status.wrapper_present)
        self.assertFalse(status.available)

    def test_backend_probe_reports_compute_failure_honestly(self) -> None:
        class _FakeModule:
            @staticmethod
            def default_client():
                return SimpleNamespace(backend=SimpleNamespace(backend_fingerprint="probe-fingerprint", backend_name="XCOM"))

            @staticmethod
            def compute_multizone_transmission(*args, **kwargs):
                raise RuntimeError("probe compute failed")

        with tempfile.TemporaryDirectory(prefix="xcom_probe_") as temp_dir:
            wrapper_path = Path(temp_dir) / "helios_xcom_integration.zip"
            wrapper_path.write_bytes(b"wrapper")
            missing_source = Path(temp_dir) / "missing_source.tar.gz"
            with (
                mock.patch("helios.services.derived.xcom_hook._wrapper_archive_path", return_value=wrapper_path),
                mock.patch("helios.services.derived.xcom_hook._source_archive_path", return_value=missing_source),
                mock.patch("helios.services.derived.xcom_hook._load_helios_xcom_module", return_value=_FakeModule()),
            ):
                status = describe_optional_cold_backend(probe_compute=True, force=True)
        self.assertEqual("compute_failed", status.status)
        self.assertTrue(status.wrapper_present)
        self.assertTrue(status.import_ok)
        self.assertTrue(status.client_ok)
        self.assertFalse(status.compute_ok)
        self.assertFalse(status.available)

    def test_backend_probe_reports_active_when_smoke_compute_succeeds(self) -> None:
        class _FakeModule:
            @staticmethod
            def default_client():
                return SimpleNamespace(backend=SimpleNamespace(backend_fingerprint="probe-fingerprint", backend_name="XCOM"))

            @staticmethod
            def compute_multizone_transmission(*args, **kwargs):
                del args, kwargs
                return SimpleNamespace(transmission=np.asarray([0.997], dtype=np.float64))

        with tempfile.TemporaryDirectory(prefix="xcom_probe_") as temp_dir:
            wrapper_path = Path(temp_dir) / "helios_xcom_integration.zip"
            wrapper_path.write_bytes(b"wrapper")
            missing_source = Path(temp_dir) / "missing_source.tar.gz"
            with (
                mock.patch("helios.services.derived.xcom_hook._wrapper_archive_path", return_value=wrapper_path),
                mock.patch("helios.services.derived.xcom_hook._source_archive_path", return_value=missing_source),
                mock.patch("helios.services.derived.xcom_hook._load_helios_xcom_module", return_value=_FakeModule()),
            ):
                status = describe_optional_cold_backend(probe_compute=True, force=True)
        self.assertEqual("active", status.status)
        self.assertTrue(status.wrapper_present)
        self.assertTrue(status.import_ok)
        self.assertTrue(status.client_ok)
        self.assertTrue(status.compute_ok)
        self.assertTrue(status.available)
        self.assertEqual("probe-fingerprint", status.backend_fingerprint)

    def test_fallback_table_alias_mapping_and_exact_lookup(self) -> None:
        self.assertEqual("c2h4o", normalize_material_key("epoxy"))
        self.assertEqual("c2h4o", normalize_material_key("epoxy_c2h4o"))
        self.assertEqual("ch", normalize_material_key("CH"))
        self.assertEqual("sio2", normalize_material_key("glass"))
        self.assertEqual("sio2", normalize_material_key("SiO2"))
        self.assertEqual("c22h10n2o5", normalize_material_key("kapton"))
        self.assertEqual("cu", canonical_fallback_table_key("Cu"))
        self.assertEqual("c2h4o", canonical_fallback_table_key("epoxy"))
        self.assertEqual("c2h4o", canonical_fallback_table_key("epoxy_c2h4o"))
        self.assertEqual("ch", canonical_fallback_table_key("CH"))
        self.assertEqual("sio2", canonical_fallback_table_key("glass"))
        self.assertEqual("sio2", canonical_fallback_table_key("SiO2"))
        self.assertEqual("c22h10n2o5", canonical_fallback_table_key("kapton"))
        self.assertEqual("cu", canonical_fallback_table_key("Cu"))
        self.assertEqual("cu", canonical_fallback_table_key("copper"))
        lookup = lookup_fallback_mu_rho("copper", 1.0)
        self.assertEqual("cu", lookup.material_key)
        self.assertEqual("cu", lookup.table_key)
        self.assertEqual("exact", lookup.interpolation_kind)
        table = self._fallback_table_json()
        expected = float(table["materials"]["cu"]["rows"][0]["mu_rho_total_cm2_g"])
        self.assertAlmostEqual(expected, lookup.mu_rho_cm2_g, places=12)

    def test_fallback_table_lookup_succeeds_for_canonical_alias_family(self) -> None:
        cases = (
            ("epoxy", "c2h4o", "epoxy_c2h4o"),
            ("epoxy_c2h4o", "c2h4o", "epoxy_c2h4o"),
            ("CH", "ch", "ch"),
            ("glass", "sio2", "sio2"),
            ("SiO2", "sio2", "sio2"),
            ("kapton", "c22h10n2o5", "kapton_c22h10n2o5"),
        )
        for label, canonical_key, table_key in cases:
            lookup = lookup_fallback_mu_rho(label, 8.0)
            self.assertEqual(canonical_key, lookup.material_key)
            self.assertEqual(table_key, lookup.table_key)
            self.assertGreater(lookup.mu_rho_cm2_g, 0.0)

    def test_canonical_keys_resolve_to_unique_valid_fallback_entries(self) -> None:
        canonical_keys = ("al", "cu", "fe", "si", "ti", "au", "be", "c", "ch", "c2h4o", "sio2", "c22h10n2o5")
        seen_table_keys: dict[str, str] = {}
        for canonical_key in canonical_keys:
            lookup = lookup_fallback_mu_rho(canonical_key, 8.0)
            self.assertEqual(canonical_key, lookup.material_key)
            self.assertTrue(lookup.table_key)
            previous = seen_table_keys.get(canonical_key)
            if previous is not None:
                self.assertEqual(previous, lookup.table_key)
            seen_table_keys[canonical_key] = lookup.table_key

    def test_precomputed_table_backend_uses_mu_rho_times_areal_density_for_tau(self) -> None:
        backend = load_precomputed_cold_backend()
        self.assertIsNotNone(backend)
        request = ColdAttenuationRequest(
            snapshot_index=0,
            observation_side="front",
            line_of_sight_cosine=1.0,
            photon_energies_kev=(8.0,),
            zones=(
                ColdAttenuationZone(
                    zone_index=1,
                    region_id=1,
                    material_id=1,
                    material_label="Cu",
                    material_display_label="Cu",
                    material_resolution_status="resolved_from_eos",
                    material_canonical_key="cu",
                    density_g_cm3=2.2,
                    path_length_cm=25.0e-4,
                ),
            ),
        )
        result = backend.compute_transmission(request)
        lookup = lookup_fallback_mu_rho("Cu", 8.0)
        expected_tau = float(lookup.mu_rho_cm2_g) * 2.2 * 25.0e-4
        actual_tau = float(result.metadata["optical_depth"][0])
        self.assertAlmostEqual(expected_tau, actual_tau, places=12)
        self.assertAlmostEqual(math.exp(-expected_tau), float(result.transmission[0]), places=12)

    def test_fallback_table_loglog_interpolation_and_transmission_from_areal_density(self) -> None:
        table = self._fallback_table_json()
        cu_rows = table["materials"]["cu"]["rows"]
        row0 = next(row for row in cu_rows if int(row["energy_eV"]) == 2500)
        row1 = next(row for row in cu_rows if int(row["energy_eV"]) == 3000)
        lookup = lookup_fallback_mu_rho("cu", 2.8)
        self.assertEqual("interpolated", lookup.interpolation_kind)
        expected_mu = math.exp(
            math.log(float(row0["mu_rho_total_cm2_g"]))
            + (
                (math.log(2800.0) - math.log(float(row0["energy_eV"])))
                / (math.log(float(row1["energy_eV"])) - math.log(float(row0["energy_eV"])))
            )
            * (math.log(float(row1["mu_rho_total_cm2_g"])) - math.log(float(row0["mu_rho_total_cm2_g"])))
        )
        self.assertAlmostEqual(expected_mu, lookup.mu_rho_cm2_g, places=10)
        areal_density = 2.2 * 25.0e-4
        transmission = math.exp(-lookup.mu_rho_cm2_g * areal_density)
        self.assertGreaterEqual(transmission, 0.0)
        self.assertLessEqual(transmission, 1.0)

    def test_apply_transmission_model_free_free_plus_thomson_populates_region_taus(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
        )
        model_parameters = DerivedAnalysisParameters(
            transmission_mode="free_free_thomson",
            transmission_photon_energy_kev=8.0,
        )
        updated = apply_transmission_model(
            baseline,
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=model_parameters,
            geometry=geometry,
        )
        self.assertEqual(updated.selected_mode, "free_free_thomson")
        self.assertIsNotNone(updated.selected_tau)
        self.assertIsNotNone(updated.selected_transmission)
        self.assertEqual(updated.region_budgets[0].dominant_regime, "free_free_thomson")
        self.assertTrue(all(budget.total_tau >= budget.thomson_tau for budget in updated.region_budgets))
        self.assertIsNotNone(updated.partition)
        assert updated.partition is not None
        self.assertTrue(any("Free-free" in note for note in updated.partition.notes))

    def test_apply_transmission_model_auto_builds_partition_and_uses_xcom_coefficients(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
        )
        auto_parameters = DerivedAnalysisParameters(
            transmission_mode="auto_hybrid",
            transmission_photon_energy_kev=8.0,
        )

        def _fake_classify(_dataset, *, snapshot_index, active_zone_indices, material_labels):
            regimes = np.full(active_zone_indices.shape, "thomson", dtype=object)
            if active_zone_indices.size:
                regimes[0] = "xcom"
            if active_zone_indices.size > 1:
                regimes[1] = "free_free_thomson"
            return regimes, [], ("Synthetic partition for test.",)

        with mock.patch(
            "helios.services.derived.transmission._classify_zone_regimes",
            side_effect=_fake_classify,
        ), mock.patch(
            "helios.services.derived.transmission._xcom_material_tau_coefficients",
            return_value=(
                {"fe": 0.5},
                None,
                (),
                "precomputed_xcom_table",
                "refined",
            ),
        ):
            updated = apply_transmission_model(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=auto_parameters,
                geometry=geometry,
        )
        self.assertEqual(updated.selected_mode, "auto_hybrid")
        self.assertEqual(updated.source, "precomputed_xcom_table")
        self.assertIsNotNone(updated.partition)
        assert updated.partition is not None
        self.assertTrue(any(summary.regime == "xcom" for summary in updated.partition.regime_summaries))
        self.assertTrue(any(summary.regime == "free_free_thomson" for summary in updated.partition.regime_summaries))
        self.assertTrue(any(budget.xcom_tau > 0.0 for budget in updated.region_budgets))
        self.assertTrue(any("Selected mode: Auto hybrid." in note for note in updated.partition.notes))

    def test_apply_transmission_model_auto_falls_back_per_zone_when_xcom_coefficients_are_missing(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
        )
        auto_parameters = DerivedAnalysisParameters(
            transmission_mode="auto_hybrid",
            transmission_photon_energy_kev=8.0,
        )

        def _all_xcom(_dataset, *, snapshot_index, active_zone_indices, material_labels):
            del snapshot_index, material_labels
            return np.full(active_zone_indices.shape, "xcom", dtype=object), [], ("Synthetic all-XCOM partition.",)

        with mock.patch(
            "helios.services.derived.transmission._classify_zone_regimes",
            side_effect=_all_xcom,
        ), mock.patch(
            "helios.services.derived.transmission._xcom_material_tau_coefficients",
            return_value=({}, None, (), "baseline", "unavailable"),
        ):
            updated = apply_transmission_model(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=auto_parameters,
                geometry=geometry,
            )
        self.assertEqual(updated.selected_mode, "auto_hybrid")
        self.assertEqual(updated.model_type, "thomson")
        self.assertAlmostEqual(float(updated.selected_tau or 0.0), float(baseline.thomson_tau), places=12)
        self.assertTrue(any(summary.regime == "thomson_fallback" for summary in updated.partition.regime_summaries))
        self.assertTrue(any("fell back to Thomson" in warning.message for warning in updated.warnings))

    def test_apply_transmission_model_builds_selected_mode_time_traces_and_replaces_thomson_only_warning(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=True,
        )
        model_parameters = DerivedAnalysisParameters(
            transmission_mode="free_free_thomson",
            transmission_photon_energy_kev=8.0,
        )
        updated = apply_transmission_model(
            baseline,
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=model_parameters,
            geometry=geometry,
        )
        self.assertEqual(("selected_tau", "selected_transmission"), tuple(bundle.key for bundle in updated.time_plots))
        self.assertTrue(any("Free-free + Thomson" in bundle.title for bundle in updated.time_plots))
        self.assertFalse(
            any(
                warning.message.startswith("Transmission is a Thomson-only quick-look estimate;")
                for warning in updated.warnings
            )
        )
        self.assertTrue(any("weak-coupling" in note for note in updated.partition.notes))

    def test_cold_metal_with_nonzero_mean_charge_stays_xcom_candidate(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        active_zone_indices, material_labels = self._active_zone_material_labels(dataset, context, parameters, geometry)
        baseline_regimes, _, _ = _classify_zone_regimes(
            dataset,
            snapshot_index=context.snapshot_index,
            active_zone_indices=active_zone_indices,
            material_labels=material_labels,
        )
        modified_mean_charge = np.asarray(dataset.mean_charge, dtype=np.float64).copy()
        modified_mean_charge[int(context.snapshot_index), active_zone_indices] = 2.5
        modified_dataset = replace(dataset, mean_charge=modified_mean_charge)
        modified_regimes, _, _ = _classify_zone_regimes(
            modified_dataset,
            snapshot_index=context.snapshot_index,
            active_zone_indices=active_zone_indices,
            material_labels=material_labels,
        )
        np.testing.assert_array_equal(baseline_regimes, modified_regimes)
        self.assertTrue(np.any(np.asarray(modified_regimes, dtype=object) == "xcom"))

    def test_hot_or_expanded_zone_is_not_classified_as_xcom(self) -> None:
        path, dataset, _context0, parameters, _geometry0 = self._load_transmission_inputs()
        context = _context_from_dataset(path, dataset, snapshot_index=1)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        active_zone_indices, material_labels = self._active_zone_material_labels(dataset, context, parameters, geometry)
        self.assertGreater(active_zone_indices.size, 0)
        hot_zone = int(active_zone_indices[0])
        modified_temperature = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        modified_density = np.asarray(dataset.density_g_cm3, dtype=np.float64).copy()
        modified_temperature[1, hot_zone] = 60.0
        modified_density[1, hot_zone] = max(modified_density[0, hot_zone] * 0.2, 1.0e-6)
        modified_dataset = replace(
            dataset,
            temperature_e_ev=modified_temperature,
            density_g_cm3=modified_density,
        )
        regimes, _, _ = _classify_zone_regimes(
            modified_dataset,
            snapshot_index=1,
            active_zone_indices=active_zone_indices,
            material_labels=material_labels,
        )
        hot_offset = int(np.where(active_zone_indices == hot_zone)[0][0])
        self.assertNotEqual("xcom", str(regimes[hot_offset]))

    def test_auto_region_budgets_report_regime_fractions_and_tau_dominance(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
        )

        def _mixed_partition(_dataset, *, snapshot_index, active_zone_indices, material_labels):
            del snapshot_index, material_labels
            regimes = np.full(active_zone_indices.shape, "thomson_fallback", dtype=object)
            if active_zone_indices.size:
                regimes[min(2, active_zone_indices.size - 1)] = "xcom"
            return regimes, [], ("Synthetic mixed partition.",)

        with mock.patch(
            "helios.services.derived.transmission._classify_zone_regimes",
            side_effect=_mixed_partition,
        ), mock.patch(
            "helios.services.derived.transmission._xcom_material_tau_coefficients",
            return_value=({"fe": 500.0}, None, (), "cache", "refined"),
        ):
            updated = apply_transmission_model(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=DerivedAnalysisParameters(
                    transmission_mode="auto_hybrid",
                    transmission_photon_energy_kev=8.0,
                ),
                geometry=geometry,
                include_time_plots=False,
            )
        budget = updated.region_budgets[0]
        self.assertIsNotNone(budget.xcom_path_fraction)
        self.assertIsNotNone(budget.thomson_fallback_path_fraction)
        self.assertIsNotNone(budget.xcom_tau_fraction)
        self.assertIsNotNone(budget.thomson_fallback_tau_fraction)
        self.assertGreater(float(budget.thomson_fallback_path_fraction or 0.0), float(budget.xcom_path_fraction or 0.0))
        self.assertGreater(float(budget.xcom_tau_fraction or 0.0), float(budget.thomson_fallback_tau_fraction or 0.0))
        self.assertEqual("xcom", budget.dominant_regime)
        self.assertTrue(any("Mixed region:" in note for note in budget.notes))

    def test_time_traces_change_with_selected_mode(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=True,
        )
        with mock.patch(
            "helios.services.derived.transmission._free_free_absorption_m_inv",
            return_value=(1.0e6, 4.2, None),
        ):
            updated = apply_transmission_model(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=DerivedAnalysisParameters(
                    transmission_mode="free_free_thomson",
                    transmission_photon_energy_kev=8.0,
                ),
                geometry=geometry,
                include_time_plots=True,
            )
        self.assertEqual(("selected_tau", "selected_transmission"), tuple(bundle.key for bundle in baseline.time_plots))
        self.assertEqual(("selected_tau", "selected_transmission"), tuple(bundle.key for bundle in updated.time_plots))
        self.assertFalse(
            np.allclose(
                np.asarray(baseline.time_plots[1].y_series[0], dtype=np.float64),
                np.asarray(updated.time_plots[1].y_series[0], dtype=np.float64),
                equal_nan=True,
            )
        )

    def test_backend_probe_compute_health_is_session_cached(self) -> None:
        status = ColdAttenuationBackendStatus(
            available=True,
            status="active",
            message="cached",
            backend_name="XCOM",
            backend_fingerprint="probe-fingerprint",
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=True,
        )
        with mock.patch("helios.services.derived.xcom_hook._probe_optional_cold_backend", return_value=status) as probe:
            first = describe_optional_cold_backend(probe_compute=True, force=True)
            second = describe_optional_cold_backend(probe_compute=True)
        self.assertEqual("active", first.status)
        self.assertEqual("active", second.status)
        self.assertEqual(1, probe.call_count)

    def test_cu0166_reference_curve_matches_xcom_backend_within_tolerance_when_backend_is_active(self) -> None:
        reference_points = (
            (1000.0, 3.14248e-17),
            (2800.0, 1.20652e-02),
            (4600.0, 4.53259e-03),
            (6400.0, 6.47505e-02),
            (8200.0, 2.1526e-01),
            (10000.0, 3.6210e-01),
        )
        status = describe_optional_cold_backend(probe_compute=True, force=True)
        if not status.available:
            self.skipTest(f"XCOM backend is not compute-capable in this environment: {status.status}")
        backend = load_optional_cold_backend(require_compute_ok=True)
        if backend is None:
            self.skipTest("XCOM backend could not be loaded even though probe reported compute_ok.")
        request = ColdAttenuationRequest(
            snapshot_index=0,
            observation_side="front",
            line_of_sight_cosine=1.0,
            photon_energies_kev=tuple(float(energy_ev) * 1.0e-3 for energy_ev, _value in reference_points),
            zones=(
                ColdAttenuationZone(
                    zone_index=0,
                    region_id=1,
                    material_id=1,
                    material_label="Cu",
                    density_g_cm3=2.2,
                    path_length_cm=25.0e-4,
                    material_display_label="Cu",
                    material_resolution_status="resolved_from_eos",
                ),
            ),
        )
        result = backend.compute_transmission(request)
        self.assertEqual(len(reference_points), len(result.transmission))
        for (_energy_ev, reference_value), actual_value in zip(reference_points, np.asarray(result.transmission, dtype=np.float64)):
            tolerance = max(abs(reference_value) * 0.15, 1.0e-20)
            self.assertAlmostEqual(float(actual_value), float(reference_value), delta=tolerance)

    def test_refine_transmission_with_xcom_uses_live_backend_and_marks_result(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        backend = _FakeBackend()
        cache_path = Path(tempfile.mkdtemp(prefix="xcom_cache_test_")) / "cache.json"
        cache = PersistentColdAttenuationCache(cache_path)
        available = ColdAttenuationBackendStatus(
            available=True,
            status="available",
            message="XCOM wrapper detected.",
            backend_name="XCOM",
            backend_fingerprint=backend.backend_fingerprint,
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=backend),
            mock.patch("helios.services.derived.transmission.persistent_cold_attenuation_cache", return_value=cache),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("recommended", 1.0, 2.5, 0.8, "The active subset is mostly cold / weakly ionized."),
            ),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual(refined.model_type, "xcom")
        self.assertEqual(refined.source, "live_xcom_backend")
        self.assertIsNotNone(refined.cold_refinement)
        assert refined.cold_refinement is not None
        self.assertEqual(refined.cold_refinement.source, "live_xcom_backend")
        self.assertEqual(refined.cold_refinement.transmission, (0.73,))
        self.assertGreaterEqual(backend.calls, 1)
        self.assertTrue(cache.path.exists())

    def test_refine_transmission_with_xcom_uses_persistent_cache_before_backend(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        cache_path = Path(tempfile.mkdtemp(prefix="xcom_cache_test_")) / "cache.json"
        cache = PersistentColdAttenuationCache(cache_path)
        available = ColdAttenuationBackendStatus(
            available=True,
            status="available",
            message="XCOM wrapper detected.",
            backend_name="XCOM",
            backend_fingerprint="fake-backend-fingerprint",
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        backend_live = _FakeBackend()
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=backend_live),
            mock.patch("helios.services.derived.transmission.load_precomputed_cold_backend", return_value=None),
            mock.patch("helios.services.derived.transmission.persistent_cold_attenuation_cache", return_value=cache),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("recommended", 1.0, 2.5, 0.8, "The active subset is mostly cold / weakly ionized."),
            ),
        ):
            first = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual(first.source, "live_xcom_backend")
        self.assertGreaterEqual(backend_live.calls, 1)

        backend_cached = _FakeBackend()
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=backend_cached),
            mock.patch("helios.services.derived.transmission.load_precomputed_cold_backend", return_value=None),
            mock.patch("helios.services.derived.transmission.persistent_cold_attenuation_cache", return_value=cache),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("recommended", 1.0, 2.5, 0.8, "The active subset is mostly cold / weakly ionized."),
            ),
        ):
            second = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual(second.source, "live_xcom_backend")
        self.assertEqual(second.model_type, "xcom")
        self.assertEqual(backend_cached.calls, 0)
        assert second.cold_refinement is not None
        self.assertEqual("refined_cached", second.cold_refinement.backend_status)

    def test_refine_transmission_with_xcom_uses_precomputed_table_when_live_backend_is_unavailable(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        table_backend = _FakeTableBackend()
        cache_path = Path(tempfile.mkdtemp(prefix="xcom_cache_test_")) / "cache.sqlite3"
        cache = PersistentColdAttenuationCache(cache_path)
        unavailable = ColdAttenuationBackendStatus(
            available=False,
            status="compute_failed",
            message="XCOM smoke compute failed.",
            backend_name="XCOM",
            backend_fingerprint="live-failed",
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=False,
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=None),
            mock.patch("helios.services.derived.transmission.load_precomputed_cold_backend", return_value=table_backend),
            mock.patch("helios.services.derived.transmission.persistent_cold_attenuation_cache", return_value=cache),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("recommended", 1.0, 2.5, 0.8, "The active subset is mostly cold / weakly ionized."),
            ),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual("xcom", refined.model_type)
        self.assertEqual("precomputed_xcom_table", refined.source)
        assert refined.cold_refinement is not None
        self.assertEqual("precomputed_xcom_table", refined.cold_refinement.source)
        self.assertIn("precomputed XCOM table", refined.cold_refinement.message)
        self.assertGreaterEqual(table_backend.calls, 1)

    def test_refine_transmission_with_xcom_prefers_live_backend_over_table(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        live_backend = _FakeBackend()
        table_backend = _FakeTableBackend()
        cache_path = Path(tempfile.mkdtemp(prefix="xcom_cache_test_")) / "cache.sqlite3"
        cache = PersistentColdAttenuationCache(cache_path)
        available = ColdAttenuationBackendStatus(
            available=True,
            status="active",
            message="XCOM wrapper imported and smoke compute succeeded.",
            backend_name="XCOM",
            backend_fingerprint=live_backend.backend_fingerprint,
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=True,
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=live_backend),
            mock.patch("helios.services.derived.transmission.load_precomputed_cold_backend", return_value=table_backend),
            mock.patch("helios.services.derived.transmission.persistent_cold_attenuation_cache", return_value=cache),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("recommended", 1.0, 2.5, 0.8, "The active subset is mostly cold / weakly ionized."),
            ),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual("live_xcom_backend", refined.source)
        self.assertGreaterEqual(live_backend.calls, 1)
        self.assertEqual(0, table_backend.calls)

    def test_cu_cold_case_uses_precomputed_table_as_valid_xcom_not_thomson_fallback(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=0)
        parameters = DerivedAnalysisParameters(transmission_photon_energy_kev=8.0)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        unavailable = ColdAttenuationBackendStatus(
            available=False,
            status="compute_failed",
            message="XCOM smoke compute failed.",
            backend_name="XCOM",
            backend_fingerprint="live-failed",
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=False,
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=None),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual("xcom", refined.selected_mode)
        self.assertEqual("xcom", refined.model_type)
        self.assertEqual("precomputed_xcom_table", refined.source)
        self.assertGreater(float(refined.selected_tau or 0.0), 0.0)
        self.assertIsNotNone(refined.cold_refinement)
        assert refined.cold_refinement is not None
        self.assertEqual("precomputed_xcom_table", refined.cold_refinement.source)
        self.assertTrue(any("Cu [resolved_from_eos]" == item for item in refined.cold_refinement.resolved_materials))
        self.assertTrue(any(budget.xcom_tau > 0.0 for budget in refined.region_budgets))
        self.assertTrue(any((budget.xcom_path_fraction or 0.0) > 0.0 for budget in refined.region_budgets))
        self.assertFalse(all((budget.thomson_fallback_path_fraction or 0.0) >= 0.999 for budget in refined.region_budgets))

    def test_composite_cold_case_uses_precomputed_table_for_multiple_materials(self) -> None:
        path = Path(__file__).resolve().parents[1] / "50Al+10E+25CH+3.5TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=0)
        parameters = DerivedAnalysisParameters(transmission_photon_energy_kev=8.0)
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        unavailable = ColdAttenuationBackendStatus(
            available=False,
            status="compute_failed",
            message="XCOM smoke compute failed.",
            backend_name="XCOM",
            backend_fingerprint="live-failed",
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=False,
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=unavailable),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=None),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual("xcom", refined.model_type)
        self.assertEqual("precomputed_xcom_table", refined.source)
        assert refined.cold_refinement is not None
        resolved = set(refined.cold_refinement.resolved_materials)
        self.assertIn("Al [resolved_from_eos]", resolved)
        self.assertIn("CH [resolved_from_eos]", resolved)
        self.assertIn("epoxy (from EOS) [resolved_from_eos]", resolved)
        self.assertTrue(all((budget.xcom_tau > 0.0) for budget in refined.region_budgets))
        self.assertTrue(all((budget.thomson_fallback_path_fraction or 0.0) < 0.999 for budget in refined.region_budgets))

    def test_late_auto_hybrid_snapshot_can_be_mixed_but_tau_dominant_xcom(self) -> None:
        path = Path(__file__).resolve().parents[1] / "50Al+10E+25CH+3.5TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=750)
        baseline = evaluate_transmission(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=DerivedAnalysisParameters(),
            geometry=build_analysis_geometry(
                dataset,
                context,
                observation_side="front",
                line_of_sight_angle_deg=0.0,
                line_of_sight_impact_parameter_cm=0.0,
                profile_coordinate_mode="viewer",
            ),
            include_time_plots=False,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side="front",
            line_of_sight_angle_deg=0.0,
            line_of_sight_impact_parameter_cm=0.0,
            profile_coordinate_mode="viewer",
        )
        refined = apply_transmission_model(
            baseline,
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            parameters=DerivedAnalysisParameters(
                transmission_mode="auto_hybrid",
                transmission_photon_energy_kev=8.0,
            ),
            geometry=geometry,
            include_time_plots=False,
        )
        region_three = next(budget for budget in refined.region_budgets if budget.region_id == 3)
        self.assertGreater(float(region_three.free_free_thomson_path_fraction or 0.0), 0.9)
        self.assertGreater(float(region_three.xcom_tau_fraction or 0.0), 0.9)
        self.assertEqual("xcom", region_three.dominant_regime)
        self.assertTrue(any("Mixed region:" in note for note in region_three.notes))

    def test_persistent_cold_cache_uses_bounded_sqlite_lru_eviction(self) -> None:
        cache_path = Path(tempfile.mkdtemp(prefix="xcom_cache_test_")) / "cache.sqlite3"
        cache = PersistentColdAttenuationCache(cache_path, max_size_bytes=1_000_000)

        def _put(index: int) -> None:
            cache.put(
                f"key-{index}",
                request_payload={"index": index, "blob": "r" * 256},
                result_payload={"transmission": [1.0 - index * 0.01], "blob": "s" * 256},
            )

        _put(0)
        first_stats = cache.stats()
        entry_size = int(first_stats["total_size_bytes"])
        self.assertGreater(entry_size, 0)

        cache.max_size_bytes = int(entry_size * 2 + 16)
        _put(1)
        time.sleep(0.02)
        self.assertIsNotNone(cache.get("key-0"))
        time.sleep(0.02)
        _put(2)

        self.assertIsNotNone(cache.get("key-0"))
        self.assertIsNone(cache.get("key-1"))
        self.assertIsNotNone(cache.get("key-2"))

        reloaded = PersistentColdAttenuationCache(cache_path, max_size_bytes=cache.max_size_bytes)
        self.assertIsNotNone(reloaded.get("key-0"))
        self.assertIsNone(reloaded.get("key-1"))
        self.assertIsNotNone(reloaded.get("key-2"))
        stats = reloaded.stats()
        self.assertLessEqual(int(stats["total_size_bytes"]), int(stats["max_size_bytes"]))
        self.assertEqual(2, len(reloaded.debug_entries()))

    def test_refine_transmission_with_xcom_falls_back_when_out_of_domain(self) -> None:
        _path, dataset, context, parameters, geometry = self._load_transmission_inputs()
        available = ColdAttenuationBackendStatus(
            available=True,
            status="available",
            message="XCOM wrapper detected.",
            backend_name="XCOM",
            backend_fingerprint="fake-backend-fingerprint",
        )
        with mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available):
            baseline = evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
            )
        backend = _FakeBackend()
        with (
            mock.patch("helios.services.derived.transmission.describe_optional_cold_backend", return_value=available),
            mock.patch("helios.services.derived.transmission.load_optional_cold_backend", return_value=backend),
            mock.patch(
                "helios.services.derived.transmission._assess_xcom_applicability",
                return_value=("not_recommended", 0.1, 45.0, 8.0, "The active subset is mostly ionized; Thomson remains the safer default."),
            ),
        ):
            refined = refine_transmission_with_xcom(
                baseline,
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
            )
        self.assertEqual(refined.model_type, "thomson")
        self.assertEqual(refined.source, "baseline")
        self.assertEqual(refined.thomson_transmission, baseline.thomson_transmission)
        self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
