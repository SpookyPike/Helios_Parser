from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import _resolved_plasmon_collision_rate_s, evaluate_plasmon_regime
from helios.services.derived.plasmon_spectrum import energy_axis_ev, epsilon_mermin, epsilon_rpa
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry


def _synthetic_dataset(*, te_ev: float = 120.0, ne_cm3: float = 8.0e20) -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 3
    n_zones = 6
    time_s = np.asarray([0.0, 1.0e-9, 2.0e-9], dtype=np.float64)
    static_x = np.linspace(1.0e-4, 6.0e-4, n_zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, 6.5e-4, n_zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, n_zones), 1.0e-4, dtype=np.float64)
    density = np.ones((n_snapshots, n_zones), dtype=np.float64)
    velocity = np.zeros_like(density)
    temperature_e = np.full_like(density, te_ev)
    temperature_i = np.full_like(density, 80.0)
    electron_density = np.full_like(density, ne_cm3)
    mean_charge = np.full_like(density, 6.0)
    zone_region_id = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    zone_material = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    regions = {
        "region_index": np.asarray([1, 2], dtype=np.int32),
        "min_zone_index": np.asarray([1, 4], dtype=np.int32),
        "max_zone_index": np.asarray([3, 6], dtype=np.int32),
        "atomic_weight": np.asarray([27.0, 63.5], dtype=np.float64),
        "initial_mass_density": np.asarray([1.0, 1.0], dtype=np.float64),
        "initial_temperature": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    dataset = DerivedRunData(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={"geometry": "PLANAR", "coordinate_model": {"coordinate_name": "x"}},
        regions=regions,
        materials={"index": np.asarray([1, 2], dtype=np.int32)},
        time_s=time_s,
        static_x_cm=static_x,
        static_x_edge_cm=static_x_edges,
        zone_width_cm=zone_width,
        density_g_cm3=density,
        velocity_cm_s=velocity,
        temperature_e_ev=temperature_e,
        temperature_i_ev=temperature_i,
        temperature_radiation_ev=None,
        electron_density_cm3=electron_density,
        mean_charge=mean_charge,
        radius_cm=None,
        radius_edge_cm=None,
        zone_region_id=zone_region_id,
        zone_material_index=zone_material,
        zone_atomic_weight=np.asarray([27.0, 27.0, 27.0, 63.5, 63.5, 63.5], dtype=np.float64),
        zone_initial_density_g_cm3=np.full(n_zones, 1.0, dtype=np.float64),
        zone_initial_temperature_ev=np.full(n_zones, 1.0, dtype=np.float64),
        laser_entry=None,
    )
    context = RunContext(
        path=Path("synthetic_plasmon.h5"),
        summary={"n_zones": n_zones, "n_snapshots": n_snapshots},
        metadata={},
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=time_s.copy(),
        static_x_values=static_x.copy(),
        zone_region_id=zone_region_id.copy(),
        zone_material_index=zone_material.copy(),
        has_dynamic_radius=False,
        snapshot_index=1,
        map_coordinate="static_x",
        slice_coordinate="zone",
        selected_region_ids=(1, 2),
        selected_material_ids=(1, 2),
    )
    return dataset, context


class PlasmonPhase3Tests(unittest.TestCase):
    def _compute(self, *, collision_rate_s: float = 0.0, collision_model: str = "manual_constant", if_fwhm_ev: float = 0.0) -> object:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="mermin",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_instrument_fwhm_ev=if_fwhm_ev,
            plasmon_collision_model=collision_model,
            plasmon_manual_collision_rate_s=collision_rate_s,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        return evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
        )

    def test_nrl_collision_resolver_uses_piecewise_fallback_when_quicklook_rate_is_nan(self) -> None:
        parameters = DerivedAnalysisParameters(plasmon_collision_model="nrl_constant", plasmon_collision_scale=1.0)
        resolved = _resolved_plasmon_collision_rate_s(
            {"collision_rate_s": float("nan"), "te_ev": 100.0, "ne_cm3": 1.0e20, "zbar": 4.0},
            parameters,
        )
        self.assertTrue(np.isfinite(resolved))
        self.assertGreater(resolved, 0.0)

    def test_epsilon_mermin_reduces_to_rpa_when_nu_goes_to_zero(self) -> None:
        energy = energy_axis_ev(20.0, 401)
        _, eps_rpa = epsilon_rpa(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, imag_shift_ev=0.05)
        _, eps_mermin = epsilon_mermin(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, collision_rate_s=0.0, imag_shift_ev=0.05)
        self.assertTrue(np.allclose(eps_rpa, eps_mermin, equal_nan=True, rtol=1.0e-11, atol=1.0e-11))

    def test_mermin_result_builds_finite_spectrum_and_resolved_collision_rate(self) -> None:
        result = self._compute(collision_rate_s=1.5e15)
        self.assertEqual(result.model_name, "mermin")
        self.assertEqual(result.requested_model_name, "mermin")
        self.assertGreater(result.spectrum_energy_ev.size, 100)
        self.assertEqual(result.spectrum_energy_ev.size, result.spectrum_intensity.size)
        finite_fraction = np.count_nonzero(np.isfinite(result.spectrum_intensity)) / float(result.spectrum_intensity.size)
        self.assertGreater(finite_fraction, 0.98)
        self.assertTrue(np.nanmax(result.spectrum_intensity) > 0.0)
        self.assertTrue(np.isfinite(result.electron_collision_rate_s))
        self.assertAlmostEqual(float(result.electron_collision_rate_s), 1.5e15, delta=1.0e8)
        self.assertGreaterEqual(float(result.peak_energy_ev), 0.0)

    def test_increasing_constant_nu_broadens_peak(self) -> None:
        weak = self._compute(collision_rate_s=2.0e14)
        strong = self._compute(collision_rate_s=2.0e15)
        self.assertTrue(np.nanmax(weak.spectrum_intensity) > 0.0)
        self.assertTrue(np.nanmax(strong.spectrum_intensity) > 0.0)
        self.assertGreater(float(strong.peak_fwhm_ev), float(weak.peak_fwhm_ev))

    def test_mermin_rpa_cache_bucket_reuses_identical_request(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="mermin",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=8.0e14,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        cache = AnalysisStateCache()
        first = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=cache,
        )
        second = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=cache,
        )
        stats = cache.stats()
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertTrue(np.array_equal(first.spectrum_energy_ev, second.spectrum_energy_ev))
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
