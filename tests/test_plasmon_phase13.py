from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from scripts.benchmark_plasmon_article_cases import (
    ARTICLE_DRIVEN_BENCHMARK_POLICIES,
    _assess_article_al_policy_state,
    _compute_density_averaged_point_result,
    _policy_state_row,
    _uniform_policy_state_summary,
)
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon_config import (
    PLASMON_MODEL_AUTO_BEST,
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_OBSERVABLE_MODE_XRTS,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
)
from helios.services.derived.plasmon_electron_policy import (
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
    PLASMON_ELECTRON_POLICY_RAW,
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
    _baseline_table_source,
    material_policy_registry,
    normalize_policy,
    resolve_effective_electron_fields,
)
from helios.services.derived.plasmon_driven_response import (
    PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
    PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    DrivenElectronResponseState,
    apply_driven_response_model,
    response_model_fields_equal,
)
from helios.services.derived.plasmon_stls import solve_static_stls_state
from helios.services.derived.plasmon_validation import compute_plasmon, q_to_angle_deg, uniform_al_dataset
from helios.services.derived.selection import AnalysisStateCache


def _low_z_uniform_al_dataset(*, rho_g_cm3: float = 4.0, te_ev: float = 0.6, zbar_raw: float = 0.2) -> tuple[DerivedRunData, object]:
    dataset, context = uniform_al_dataset(rho_g_cm3, te_ev)
    raw_ne = np.full_like(dataset.electron_density_cm3, float(rho_g_cm3) / 26.9815 * 6.02214076e23 * float(zbar_raw))
    dataset = replace(
        dataset,
        electron_density_cm3=raw_ne,
        mean_charge=np.full_like(dataset.mean_charge, float(zbar_raw)),
    )
    return dataset, context


def _mixed_al_ch_dataset() -> tuple[DerivedRunData, object]:
    dataset, context = uniform_al_dataset(2.7, 0.3, zones=6)
    zone_material = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    materials = {
        "index": np.asarray([1, 2], dtype=np.int32),
        "eos_file_path": np.asarray(["Al.prp", "CH.prp"], dtype=object),
        "opacity_file_path": np.asarray(["Al.prp", "CH.prp"], dtype=object),
        "eos_model": np.asarray(["EOSOPA", "EOSOPA"], dtype=object),
        "opacity_model": np.asarray(["EOSOPA", "EOSOPA"], dtype=object),
    }
    dataset = replace(
        dataset,
        materials=materials,
        zone_material_index=zone_material,
        zone_atomic_weight=np.asarray([26.9815, 26.9815, 26.9815, 13.0, 13.0, 13.0], dtype=np.float64),
    )
    return dataset, context


class PlasmonPhase13Tests(unittest.TestCase):
    def test_noop_driven_response_model_keeps_cold_baseline(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        ion_density = np.asarray(baseline_payload.electron_density_cm3, dtype=np.float64) / np.asarray(baseline_payload.mean_charge, dtype=np.float64)
        response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="test",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
        )
        self.assertTrue(np.allclose(response.effective_mean_charge, 3.0))
        self.assertTrue(np.allclose(response.increment_mean_charge, 0.0))
        self.assertEqual(response.model, PLASMON_DRIVEN_RESPONSE_MODEL_NONE)

    def test_material_registry_contains_expected_benchmark_entries(self) -> None:
        registry = material_policy_registry()
        self.assertTrue(math.isclose(registry["Al"].benchmark_valence_per_nucleus, 3.0, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertEqual(registry["Al"].baseline_entry_path, "elements.Al")
        self.assertTrue(math.isclose(registry["CH"].benchmark_valence_per_nucleus, 0.0, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertEqual(registry["CH"].baseline_entry_path, "compound_and_polymer_baselines.CH_generic")
        self.assertTrue(math.isclose(registry["C2H4O"].benchmark_valence_per_nucleus, 0.0, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertEqual(registry["C2H4O"].baseline_entry_path, "compound_and_polymer_baselines.Epoxy_generic")
        self.assertTrue(_baseline_table_source().endswith("hed_helios_cold_electronic_baseline_core.json"))

    def test_normalize_policy_falls_back_to_raw(self) -> None:
        self.assertEqual(normalize_policy("does_not_exist"), PLASMON_ELECTRON_POLICY_RAW)

    def test_benchmark_valence_aware_recomputes_effective_al_electrons(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset()
        payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE)
        expected_ne = 4.0 / 26.9815 * 6.02214076e23 * 3.0
        self.assertEqual(payload.policy, PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE)
        self.assertTrue(np.allclose(payload.mean_charge, 3.0))
        self.assertTrue(np.allclose(payload.electron_density_cm3, expected_ne))
        self.assertIn("Al@elements.Al", payload.summary)
        self.assertIn("cold_baseline_plus_increment", payload.summary)
        self.assertIn("elements.Al", payload.baseline_entries[0])

    def test_plasmon_service_uses_valence_policy_in_effective_state(self) -> None:
        dataset, context = _low_z_uniform_al_dataset()
        angle = q_to_angle_deg(1.28, 8.307)
        raw = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=2401,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_RAW,
        )
        aware = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=2401,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
        )
        locked = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=2401,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
        )
        self.assertTrue(math.isclose(raw.mean_charge, 0.2, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(aware.mean_charge, 3.0, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(locked.mean_charge, 3.0, rel_tol=0.0, abs_tol=1.0e-12))
        self.assertGreater(aware.electron_density_cm3, raw.electron_density_cm3 * 10.0)
        self.assertGreater(aware.peak_energy_ev, raw.peak_energy_ev)
        self.assertEqual(aware.electron_policy, PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE)
        self.assertEqual(locked.electron_policy, PLASMON_ELECTRON_POLICY_VALENCE_LOCKED)

    def test_article_al_benchmark_only_remaps_al_and_reports_raw_kept_non_al(self) -> None:
        dataset, _ = _mixed_al_ch_dataset()
        raw_zbar = np.asarray(dataset.mean_charge, dtype=np.float64).copy()
        payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        self.assertEqual(payload.policy, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        self.assertTrue(np.allclose(payload.mean_charge[:, :3], 3.0))
        self.assertTrue(np.allclose(payload.mean_charge[:, 3:], raw_zbar[:, 3:]))
        self.assertIn("Al@elements.Al", payload.summary)
        self.assertIn("CH", payload.raw_kept_materials)

    def test_plasmon_result_reports_requested_policy_and_benchmark_preset(self) -> None:
        dataset, context = _low_z_uniform_al_dataset()
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_integration_mode='effective_state',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
            plasmon_benchmark_preset='al_ambient_article',
        )
        self.assertEqual(result.requested_electron_policy, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        self.assertEqual(result.electron_policy, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        self.assertEqual(result.benchmark_preset, 'al_ambient_article')
        self.assertEqual(result.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_NONE)
        self.assertIn('article', result.electron_density_source.lower())

    def test_active_subset_material_reporting_ignores_filtered_non_al_layers(self) -> None:
        dataset, context = _mixed_al_ch_dataset()
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_integration_mode='effective_state',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
            derived_material_ids=(1,),
        )
        self.assertIn("Al@elements.Al", result.material_policy_summary)
        self.assertEqual(result.raw_kept_materials, ())
        self.assertNotIn("CH", result.material_policy_summary)

    def test_article_al_driven_increment_keeps_cold_al_at_baseline_and_raises_driven_state(self) -> None:
        cold_dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=2.7, te_ev=0.025, zbar_raw=0.2)
        driven_dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        cold_payload = resolve_effective_electron_fields(cold_dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT)
        driven_payload = resolve_effective_electron_fields(driven_dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT)
        self.assertTrue(np.allclose(cold_payload.mean_charge, 3.0))
        self.assertTrue(np.allclose(cold_payload.increment_mean_charge, 0.0))
        self.assertGreater(float(np.nanmean(np.asarray(driven_payload.increment_mean_charge, dtype=np.float64))), 0.0)
        self.assertGreater(float(np.nanmean(np.asarray(driven_payload.mean_charge, dtype=np.float64))), 3.0)
        self.assertEqual(driven_payload.increment_mode, "benchmark_driven_increment")
        self.assertEqual(driven_payload.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)
        self.assertIn("baseline", driven_payload.summary.lower())

    def test_plasmon_result_reports_driven_response_model_for_article_increment(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
        )
        self.assertEqual(result.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)
        self.assertIn("baseline", result.driven_response_summary.lower())

    def test_quantum_hydrodynamic_backend_executes_and_reports_provenance(self) -> None:
        dataset, context = uniform_al_dataset(4.125, 0.6)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model=PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_collision_model="benchmark_dense",
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
        )
        self.assertEqual(result.model_name, PLASMON_MODEL_QUANTUM_HYDRODYNAMIC)
        self.assertEqual(result.response_backend, "quantum_hydrodynamic")
        self.assertIn("beta_eff", result.backend_summary)
        self.assertTrue(np.asarray(result.spectrum_energy_ev, dtype=np.float64).size > 0)
        self.assertTrue(math.isfinite(float(result.peak_energy_ev)))

    def test_auto_best_does_not_silently_switch_to_quantum_hydrodynamic(self) -> None:
        dataset, context = uniform_al_dataset(4.125, 0.6)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model=PLASMON_MODEL_AUTO_BEST,
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
        )
        self.assertNotEqual(result.model_name, PLASMON_MODEL_QUANTUM_HYDRODYNAMIC)

    def test_driven_response_scalar_control_matches_article_increment_policy_fields(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        policy_payload = resolve_effective_electron_fields(
            dataset,
            PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        ion_density = np.asarray(baseline_payload.electron_density_cm3, dtype=np.float64) / np.asarray(baseline_payload.mean_charge, dtype=np.float64)
        response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="test",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        policy_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(policy_payload.baseline_mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="policy",
            ),
            policy_payload.driven_response_model,
        )
        self.assertTrue(response_model_fields_equal(response, policy_response))
        self.assertTrue(np.allclose(policy_payload.mean_charge, response.effective_mean_charge))
        self.assertTrue(np.allclose(policy_payload.electron_density_cm3, response.effective_electron_density_cm3))

    def test_experimental_driven_response_keeps_scalar_control_fields_but_adds_weighting(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        ion_density = np.asarray(baseline_payload.electron_density_cm3, dtype=np.float64) / np.asarray(baseline_payload.mean_charge, dtype=np.float64)
        scalar_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="scalar",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        experimental_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="experimental",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
        )
        self.assertTrue(response_model_fields_equal(scalar_response, experimental_response))
        self.assertEqual(str(experimental_response.response_modifiers.get("ensemble_weight_mode", "")), "effective_electron_column")
        weight = np.asarray(experimental_response.response_modifiers.get("ensemble_weight_multiplier"), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(weight)))
        self.assertTrue(np.all(weight > 0.0))

    def test_experimental_driven_response_policy_summary_reports_weighting(self) -> None:
        summary = _uniform_policy_state_summary(
            rho_g_cm3=4.2,
            te_ev=0.6,
            policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
        )
        self.assertEqual(str(summary["driven_response_model"]), PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED)
        self.assertEqual(str(summary["driven_response_weight_mode"]), "effective_electron_column")
        self.assertGreater(float(summary["driven_response_weight_mean"]), 1.0e20)
        self.assertIn("ensemble-weight", str(summary["driven_response_summary"]))

    def test_collision_shape_driven_response_keeps_scalar_fields_but_adds_shape_modifier(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        ion_density = np.asarray(baseline_payload.electron_density_cm3, dtype=np.float64) / np.asarray(baseline_payload.mean_charge, dtype=np.float64)
        scalar_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="scalar",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        shape_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="shape",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
        )
        self.assertTrue(response_model_fields_equal(scalar_response, shape_response))
        self.assertEqual(str(shape_response.response_modifiers.get("shape_modifier_mode", "")), "collision_hbar_nu_dense_gaussian")
        shape_fwhm = np.asarray(shape_response.response_modifiers.get("shape_modifier_fwhm_ev"), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(shape_fwhm)))
        self.assertTrue(np.all(shape_fwhm >= 0.0))
        self.assertIn("shape-modifier", str(shape_response.summary))

    def test_collision_shape_policy_summary_reports_shape_metadata(self) -> None:
        summary = _uniform_policy_state_summary(
            rho_g_cm3=4.2,
            te_ev=0.6,
            policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
        )
        self.assertEqual(str(summary["driven_response_model"]), PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE)
        self.assertEqual(str(summary["driven_response_shape_mode"]), "collision_hbar_nu_dense_gaussian")
        self.assertGreaterEqual(float(summary["driven_response_shape_mean_ev"]), 0.0)
        self.assertIn("shape-modifier", str(summary["driven_response_summary"]))

    def test_collision_shape_override_does_not_change_ambient_article_baseline(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=2.7, te_ev=0.025, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        shape_payload = resolve_effective_electron_fields(
            dataset,
            PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
        )
        self.assertTrue(np.allclose(shape_payload.mean_charge, baseline_payload.mean_charge))
        self.assertTrue(np.allclose(shape_payload.electron_density_cm3, baseline_payload.electron_density_cm3))
        self.assertEqual(shape_payload.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_NONE)
        self.assertEqual(str(shape_payload.driven_response_shape_mode), "")

    def test_collision_shape_density_averaged_point_result_executes(self) -> None:
        row = _compute_density_averaged_point_result(
            model="rpa_static_lfc",
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
            q_value=1.28,
            densities_g_cm3=(3.75, 4.0, 4.25, 4.5),
            te_ev=0.6,
            instrument_fwhm_ev=0.2,
            benchmark_preset="al_driven_article",
        )
        self.assertEqual(str(row["driven_response_model"]), PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE)
        self.assertEqual(str(row["driven_response_shape_mode"]), "collision_hbar_nu_dense_gaussian")
        self.assertGreaterEqual(float(row["driven_response_shape_mean_ev"]), 0.0)
        self.assertTrue(np.asarray(row["energy_ev"], dtype=np.float64).size > 0)
        self.assertTrue(np.asarray(row["spectrum"], dtype=np.float64).size > 0)

    def test_response_function_ensemble_driven_response_keeps_scalar_fields_but_adds_ensemble_mode(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        ion_density = np.asarray(baseline_payload.electron_density_cm3, dtype=np.float64) / np.asarray(baseline_payload.mean_charge, dtype=np.float64)
        scalar_response = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="scalar",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        response_ensemble = apply_driven_response_model(
            DrivenElectronResponseState(
                material_formula="Al",
                density_g_cm3=np.asarray(dataset.density_g_cm3, dtype=np.float64),
                electron_temperature_ev=np.asarray(dataset.temperature_e_ev, dtype=np.float64),
                ion_temperature_ev=np.asarray(dataset.temperature_i_ev, dtype=np.float64),
                ion_density_cm3=ion_density,
                raw_electron_density_cm3=np.asarray(dataset.electron_density_cm3, dtype=np.float64),
                raw_mean_charge=np.asarray(dataset.mean_charge, dtype=np.float64),
                baseline_mean_charge=np.asarray(baseline_payload.mean_charge, dtype=np.float64),
                baseline_entry="elements.Al",
                baseline_table_source=_baseline_table_source(),
                state_origin="ensemble",
            ),
            PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
        )
        self.assertTrue(response_model_fields_equal(scalar_response, response_ensemble))
        self.assertEqual(str(response_ensemble.response_modifiers.get("ensemble_response_mode", "")), "epsilon_average_before_loss")
        self.assertIn("ensemble-response", str(response_ensemble.summary))

    def test_response_function_ensemble_policy_summary_reports_ensemble_metadata(self) -> None:
        summary = _uniform_policy_state_summary(
            rho_g_cm3=4.2,
            te_ev=0.6,
            policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
        )
        self.assertEqual(str(summary["driven_response_model"]), PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE)
        self.assertEqual(str(summary["driven_response_ensemble_mode"]), "epsilon_average_before_loss")
        self.assertIn("ensemble-response", str(summary["driven_response_summary"]))

    def test_plasmon_result_reports_response_function_ensemble_provenance(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa_static_lfc",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            plasmon_driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
        )
        self.assertEqual(result.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE)
        self.assertEqual(result.driven_response_ensemble_mode, "epsilon_average_before_loss")

    def test_response_function_ensemble_override_does_not_change_ambient_article_baseline(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=2.7, te_ev=0.025, zbar_raw=0.2)
        baseline_payload = resolve_effective_electron_fields(dataset, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        ensemble_payload = resolve_effective_electron_fields(
            dataset,
            PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
        )
        self.assertTrue(np.allclose(ensemble_payload.mean_charge, baseline_payload.mean_charge))
        self.assertTrue(np.allclose(ensemble_payload.electron_density_cm3, baseline_payload.electron_density_cm3))
        self.assertEqual(ensemble_payload.driven_response_model, PLASMON_DRIVEN_RESPONSE_MODEL_NONE)
        self.assertEqual(str(ensemble_payload.driven_response_ensemble_mode), "")

    def test_response_function_ensemble_single_state_matches_scalar_control_density_average_result(self) -> None:
        control = _compute_density_averaged_point_result(
            model="rpa_static_lfc",
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
            q_value=1.28,
            densities_g_cm3=(4.0,),
            te_ev=0.6,
            instrument_fwhm_ev=0.2,
            benchmark_preset="al_driven_article",
        )
        ensemble = _compute_density_averaged_point_result(
            model="rpa_static_lfc",
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
            q_value=1.28,
            densities_g_cm3=(4.0,),
            te_ev=0.6,
            instrument_fwhm_ev=0.2,
            benchmark_preset="al_driven_article",
        )
        self.assertEqual(str(ensemble["driven_response_ensemble_mode"]), "epsilon_average_before_loss")
        self.assertTrue(
            np.allclose(
                np.asarray(control["energy_ev"], dtype=np.float64),
                np.asarray(ensemble["energy_ev"], dtype=np.float64),
                atol=1.0e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(control["spectrum"], dtype=np.float64),
                np.asarray(ensemble["spectrum"], dtype=np.float64),
                atol=1.0e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(math.isclose(float(control["peak_energy_ev"]), float(ensemble["peak_energy_ev"]), rel_tol=0.0, abs_tol=1.0e-12))

    def test_static_stls_solver_converges_for_driven_al_control_state(self) -> None:
        dataset, _ = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        payload = resolve_effective_electron_fields(
            dataset,
            PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        ne_cm3 = float(np.nanmean(np.asarray(payload.electron_density_cm3, dtype=np.float64)))
        solution = solve_static_stls_state(
            ne_cm3=ne_cm3,
            te_ev=0.6,
            imag_shift_ev=1.0e-9,
            benchmark=True,
        )
        self.assertTrue(bool(solution["converged"]))
        self.assertGreater(int(solution["iterations"]), 0)
        self.assertLessEqual(float(solution["relative_residual"]), 5.0e-4)
        self.assertEqual(str(solution["closure_name"]), "static_stls_isotropic_angle_integral")
        self.assertGreater(int(solution["q_grid_count"]), 0)
        self.assertGreater(int(solution["energy_grid_count"]), 0)

    def test_plasmon_result_reports_stls_backend_provenance(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model=PLASMON_MODEL_FINITE_T_STLS,
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            plasmon_driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        self.assertEqual(result.response_backend, "finite_t_stls")
        self.assertTrue(result.stls_converged)
        self.assertGreater(result.stls_iteration_count, 0)
        self.assertTrue(math.isfinite(float(result.stls_convergence_residual)))
        self.assertTrue(math.isfinite(float(result.stls_convergence_relative_residual)))
        self.assertEqual(result.stls_closure_name, "static_stls_isotropic_angle_integral")
        self.assertEqual(result.benchmark_status, "valid")
        self.assertIn("self-consistent static G(q)", result.backend_summary)

    def test_article_driven_density_average_policy_set_excludes_raw_helios(self) -> None:
        self.assertNotIn(PLASMON_ELECTRON_POLICY_RAW, ARTICLE_DRIVEN_BENCHMARK_POLICIES)
        self.assertIn(PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK, ARTICLE_DRIVEN_BENCHMARK_POLICIES)
        self.assertIn(PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT, ARTICLE_DRIVEN_BENCHMARK_POLICIES)
        self.assertIn(PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE, ARTICLE_DRIVEN_BENCHMARK_POLICIES)
        self.assertIn(PLASMON_ELECTRON_POLICY_VALENCE_LOCKED, ARTICLE_DRIVEN_BENCHMARK_POLICIES)

    def test_benchmark_dense_mermin_is_not_trivially_invalid_for_dense_al(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model='lindhard_mermin',
            plasmon_execution_mode='benchmark',
            plasmon_integration_mode='effective_state',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
            plasmon_collision_model='benchmark_dense',
            plasmon_lfc_model='esa_static',
        )
        self.assertEqual(result.collision_source, 'benchmark_dense')
        self.assertIn('v_eff/a_i', result.collision_summary)
        self.assertEqual(result.benchmark_status, 'valid')
        self.assertTrue(result.model_executed_fully)

    def test_article_benchmark_flags_raw_helios_condensed_al_input_as_invalid_policy(self) -> None:
        selection = {
            "rho_weighted_g_cm3": 2.7,
            "ne_weighted_cm3": 7.764e9,
            "effective_valence_from_ne": 1.289e-13,
            "raw_ne_weighted_cm3": 7.764e9,
            "raw_effective_valence_from_ne": 1.289e-13,
        }
        assessment = _assess_article_al_policy_state("ambient_al_t0", PLASMON_ELECTRON_POLICY_RAW, selection)
        self.assertEqual(assessment["input_policy_status"], "invalid_input_policy")
        self.assertIn("not physically credible", str(assessment["input_policy_reason"]))

    def test_best_hydro_slab_also_flags_raw_helios_as_invalid_policy(self) -> None:
        selection = {
            "rho_weighted_g_cm3": 4.15,
            "ne_weighted_cm3": 1.7071e22,
            "effective_valence_from_ne": 0.1845,
            "raw_ne_weighted_cm3": 1.7071e22,
            "raw_effective_valence_from_ne": 0.1845,
        }
        assessment = _assess_article_al_policy_state("driven_al_best_hydro_slab", PLASMON_ELECTRON_POLICY_RAW, selection)
        self.assertEqual(assessment["input_policy_status"], "invalid_input_policy")
        self.assertIn("not physically credible", str(assessment["input_policy_reason"]))

    def test_article_benchmark_policy_row_carries_effective_valence_and_status(self) -> None:
        selection = {
            "snapshot_index": 0,
            "time_ns": 0.0,
            "zone_index_lower": 1,
            "zone_index_upper": 1000,
            "zone_count": 1000,
            "rho_weighted_g_cm3": 2.7,
            "te_weighted_ev": 0.025,
            "raw_ne_weighted_cm3": 7.764e9,
            "ne_weighted_cm3": 7.764e9,
            "effective_ne_weighted_cm3": 7.764e9,
            "raw_zbar_weighted": 1.289e-13,
            "zbar_weighted": 1.289e-13,
            "effective_zbar_weighted": 1.289e-13,
            "ion_density_weighted_cm3": 6.02214076e22,
            "effective_valence_from_ne": 1.289e-13,
            "raw_effective_valence_from_ne": 1.289e-13,
        }
        row = _policy_state_row("ambient_al_t0", PLASMON_ELECTRON_POLICY_RAW, selection)
        self.assertEqual(row["input_policy_status"], "invalid_input_policy")
        self.assertTrue(math.isclose(float(row["raw_effective_valence_from_ne"]), 1.289e-13, rel_tol=0.0, abs_tol=1.0e-20))

    def test_los_cluster_cache_does_not_mix_raw_and_article_policies(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        shared_cache = AnalysisStateCache()
        raw = compute_plasmon(
            dataset,
            context,
            analysis_cache=shared_cache,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_RAW,
        )
        article = compute_plasmon(
            dataset,
            context,
            analysis_cache=shared_cache,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
        )
        fresh_article = compute_plasmon(
            dataset,
            context,
            analysis_cache=AnalysisStateCache(),
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
        )
        self.assertEqual(raw.electron_policy, PLASMON_ELECTRON_POLICY_RAW)
        self.assertEqual(article.electron_policy, PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
        self.assertEqual(article.benchmark_status, "valid")
        self.assertEqual(fresh_article.benchmark_status, "valid")
        self.assertTrue(math.isclose(float(article.peak_energy_ev), float(fresh_article.peak_energy_ev), rel_tol=0.0, abs_tol=1.0e-9))

    def test_xrts_observable_mode_returns_component_arrays_and_provenance(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_observable_mode=PLASMON_OBSERVABLE_MODE_XRTS,
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=3.5,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            plasmon_driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        self.assertEqual(result.observable_mode, PLASMON_OBSERVABLE_MODE_XRTS)
        self.assertEqual(result.observable_decomposition_mode, "minimal_chihara_like_al")
        self.assertEqual(result.observable_peak_extraction_mode, "positive_branch_excluding_elastic_core")
        self.assertEqual(result.spectrum_energy_ev.shape, result.spectrum_intensity.shape)
        self.assertEqual(result.spectrum_free_component.shape, result.spectrum_energy_ev.shape)
        self.assertEqual(result.spectrum_bound_component.shape, result.spectrum_energy_ev.shape)
        self.assertEqual(result.spectrum_elastic_component.shape, result.spectrum_energy_ev.shape)
        self.assertGreater(float(np.nanmax(np.asarray(result.spectrum_elastic_component, dtype=np.float64))), 0.0)
        self.assertIn("Chihara-like", result.observable_summary)
        self.assertGreater(float(result.observable_elastic_fraction), 0.0)

    def test_density_averaged_xrts_observable_returns_component_bookkeeping(self) -> None:
        row = _compute_density_averaged_point_result(
            model=PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
            q_value=1.28,
            densities_g_cm3=(3.75, 4.0, 4.25, 4.5),
            te_ev=0.6,
            instrument_fwhm_ev=3.5,
            benchmark_preset="al_driven_article",
            observable_mode=PLASMON_OBSERVABLE_MODE_XRTS,
        )
        self.assertEqual(str(row["observable_mode"]), PLASMON_OBSERVABLE_MODE_XRTS)
        self.assertEqual(np.asarray(row["energy_ev"], dtype=np.float64).shape, np.asarray(row["spectrum"], dtype=np.float64).shape)
        self.assertEqual(np.asarray(row["spectrum_free_component"], dtype=np.float64).shape, np.asarray(row["energy_ev"], dtype=np.float64).shape)
        self.assertEqual(np.asarray(row["spectrum_bound_component"], dtype=np.float64).shape, np.asarray(row["energy_ev"], dtype=np.float64).shape)
        self.assertEqual(np.asarray(row["spectrum_elastic_component"], dtype=np.float64).shape, np.asarray(row["energy_ev"], dtype=np.float64).shape)
        self.assertTrue(math.isfinite(float(row["observable_free_fraction"])))
        self.assertTrue(math.isfinite(float(row["observable_elastic_fraction"])))

    def test_article_native_observable_reports_material_specific_provenance(self) -> None:
        dataset, context = _low_z_uniform_al_dataset(rho_g_cm3=4.2, te_ev=0.6, zbar_raw=0.2)
        angle = q_to_angle_deg(1.28, 8.307)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model=PLASMON_MODEL_FINITE_T_STLS,
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_observable_mode=PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=3.5,
            plasmon_electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            plasmon_driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        )
        self.assertEqual(result.observable_mode, PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE)
        self.assertEqual(result.observable_decomposition_mode, "article_native_al_chihara_like")
        self.assertEqual(result.observable_comparison_mode, "inelastic_branch_after_explicit_elastic_subtraction")
        self.assertEqual(result.observable_subtraction_mode, "explicit_elastic_component_removed_before_peak_fit")
        self.assertEqual(result.observable_normalization_mode, "peak")
        self.assertTrue(math.isfinite(float(result.observable_elastic_form_factor_total)))
        self.assertTrue(math.isfinite(float(result.observable_elastic_form_factor_core)))
        self.assertTrue(math.isfinite(float(result.observable_elastic_screening_form_factor)))
        self.assertEqual(result.observable_bound_core_mode, "shell_thresholded_zero_below_al_l_shell")
        self.assertIn("L(8@72.6eV)=inactive", result.observable_bound_shell_summary)
        self.assertGreater(float(result.peak_energy_ev), 1.0)

    def test_density_averaged_minimal_xrts_peak_extraction_no_longer_collapses_for_stls(self) -> None:
        row = _compute_density_averaged_point_result(
            model=PLASMON_MODEL_FINITE_T_STLS,
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
            q_value=1.57,
            densities_g_cm3=(3.75, 4.0, 4.25, 4.5),
            te_ev=0.6,
            instrument_fwhm_ev=3.5,
            benchmark_preset="al_driven_article",
            observable_mode=PLASMON_OBSERVABLE_MODE_XRTS,
        )
        self.assertGreater(float(row["peak_energy_ev"]), 10.0)
        self.assertIn(str(row["observable_peak_fit_status"]), {"accepted_local_quadratic", "fallback_from_unstable_local_quadratic", "edge_discrete_fallback"})
        self.assertEqual(np.asarray(row["spectrum_free_component"], dtype=np.float64).shape, np.asarray(row["energy_ev"], dtype=np.float64).shape)

    def test_density_averaged_article_native_stls_peak_stays_on_inelastic_branch(self) -> None:
        row = _compute_density_averaged_point_result(
            model=PLASMON_MODEL_FINITE_T_STLS,
            electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
            q_value=2.27,
            densities_g_cm3=(3.75, 4.0, 4.25, 4.5),
            te_ev=0.6,
            instrument_fwhm_ev=3.5,
            benchmark_preset="al_driven_article",
            observable_mode=PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
        )
        self.assertGreater(float(row["peak_energy_ev"]), 10.0)
        self.assertEqual(str(row["observable_peak_extraction_mode"]), "inelastic_branch_after_elastic_subtraction")
        self.assertEqual(str(row["observable_comparison_mode"]), "inelastic_branch_after_explicit_elastic_subtraction")
        self.assertEqual(str(row["observable_subtraction_mode"]), "explicit_elastic_component_removed_before_peak_fit")
        self.assertTrue(math.isfinite(float(row["observable_elastic_form_factor_total"])))
        self.assertIn("L(8@72.6eV)=inactive", str(row["observable_bound_shell_summary"]))
        self.assertEqual(np.asarray(row["spectrum_elastic_component"], dtype=np.float64).shape, np.asarray(row["energy_ev"], dtype=np.float64).shape)


if __name__ == "__main__":
    unittest.main()
