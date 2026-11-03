from __future__ import annotations

import unittest
from dataclasses import replace

import _test_bootstrap  # noqa: F401

from helios.services.derived.analysis import (
    DerivedAnalysisParameters,
    _base_module_key_matches,
    compute_analysis_result,
)
from helios.services.derived.plasmon_validation import uniform_al_dataset


class DerivedParameterKeyTest(unittest.TestCase):
    def test_core_key_ignores_module_parameters(self) -> None:
        base = DerivedAnalysisParameters()
        for changed in (
            replace(base, plasmon_photon_energy_kev=7.5),
            replace(base, plasmon_model="rpa"),
            replace(base, plasmon_scan_points=64),
            replace(base, xrd_photon_energy_kev=9.0),
            replace(base, spectroscopy_line_wavelength_nm=532.0),
            replace(base, preheat_target_region_id=2),
        ):
            self.assertEqual(base.core_key(), changed.core_key())

    def test_core_key_tracks_geometry_and_selection(self) -> None:
        base = DerivedAnalysisParameters()
        for changed in (
            replace(base, observation_side="rear"),
            replace(base, line_of_sight_angle_deg=30.0),
            replace(base, line_of_sight_impact_parameter_cm=0.02),
            replace(base, profile_coordinate_mode="zone"),
            replace(base, reuse_viewer_subset=not base.reuse_viewer_subset),
            replace(base, derived_region_ids=(1, 2)),
            replace(base, derived_material_ids=(3,)),
            replace(base, exclude_entry_region=True),
            replace(base, exclude_low_density=True),
            replace(base, min_density_g_cm3=1.0),
            replace(base, exclude_opposite_velocity=True),
            replace(base, zone_index_lower=2),
            replace(base, zone_index_upper=9),
            replace(base, weighting_mode="mass"),
        ):
            self.assertNotEqual(base.core_key(), changed.core_key(), changed)

    def test_module_key_tracks_module_parameters_only(self) -> None:
        base = DerivedAnalysisParameters()
        self.assertNotEqual(base.module_key(), replace(base, plasmon_model="rpa").module_key())
        self.assertNotEqual(base.module_key(), replace(base, xrd_photon_energy_kev=9.0).module_key())
        self.assertNotEqual(base.module_key(), replace(base, preheat_target_region_id=2).module_key())
        self.assertEqual(base.module_key(), replace(base, observation_side="rear").module_key())

    def test_full_key_is_core_plus_module(self) -> None:
        base = DerivedAnalysisParameters()
        self.assertEqual(base.key(), (*base.core_key(), *base.module_key()))
        # Any parameter change must move the full key.
        self.assertNotEqual(base.key(), replace(base, observation_side="rear").key())
        self.assertNotEqual(base.key(), replace(base, plasmon_model="rpa").key())


class ModuleReuseGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, self.context = uniform_al_dataset(2.7, 50.0)
        self.base = DerivedAnalysisParameters()

    def _compute(self, parameters: DerivedAnalysisParameters, *, base_result=None):
        return compute_analysis_result(
            self.dataset,
            self.context,
            parameters=parameters,
            context_key=("guard", parameters.key()),
            requested_modules=(),
            include_wavefront=False,
            base_result=base_result,
        )

    def test_module_payload_is_not_reused_across_a_module_parameter_change(self) -> None:
        first = self._compute(self.base)
        changed = replace(self.base, plasmon_photon_energy_kev=7.5)
        self.assertFalse(_base_module_key_matches(first, changed))
        second = self._compute(changed, base_result=first)
        self.assertEqual(second.module_key, changed.module_key())
        self.assertIsNot(second.plasmon, first.plasmon)

    def test_module_payload_is_reused_when_the_module_key_matches(self) -> None:
        first = self._compute(self.base)
        self.assertTrue(_base_module_key_matches(first, self.base))
        second = self._compute(self.base, base_result=first)
        self.assertIs(second.plasmon, first.plasmon)

    def test_geometry_gating_is_owned_by_the_shock_predicate(self) -> None:
        # The module key deliberately ignores geometry: geometry reuse is decided
        # by _can_reuse_shock and _can_reuse_wave_tracking, which compare the
        # geometry and selection payloads. Assert both halves of that split so a
        # future change cannot silently move the responsibility.
        first = self._compute(self.base)
        changed = replace(self.base, observation_side="rear")
        self.assertTrue(_base_module_key_matches(first, changed))
        self.assertNotEqual(self.base.core_key(), changed.core_key())


if __name__ == "__main__":
    unittest.main()