from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon import evaluate_plasmon_regime, electron_debye_length_cm
from helios.services.derived.plasmon_units import electron_debye_length_cm as electron_debye_length_cm_units
from helios.services.derived.selection import build_analysis_geometry


def _synthetic_dataset() -> tuple[DerivedRunData, RunContext]:
    n_snapshots = 3
    n_zones = 6
    time_s = np.asarray([0.0, 1.0e-9, 2.0e-9], dtype=np.float64)
    static_x = np.linspace(1.0e-4, 6.0e-4, n_zones, dtype=np.float64)
    static_x_edges = np.linspace(5.0e-5, 6.5e-4, n_zones + 1, dtype=np.float64)
    zone_width = np.full((n_snapshots, n_zones), 1.0e-4, dtype=np.float64)
    density = np.ones((n_snapshots, n_zones), dtype=np.float64)
    velocity = np.zeros_like(density)
    temperature_e = np.full_like(density, 120.0)
    temperature_i = np.full_like(density, 80.0)
    electron_density = np.full_like(density, 8.0e20)
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


class PlasmonPhase1Tests(unittest.TestCase):
    def test_new_plasmon_parameters_participate_in_key(self) -> None:
        baseline = DerivedAnalysisParameters()
        updated = DerivedAnalysisParameters(plasmon_model="mermin", plasmon_energy_window_ev=120.0)
        self.assertNotEqual(baseline.key(), updated.key())
        self.assertEqual(updated.plasmon_model, "mermin")
        self.assertEqual(updated.plasmon_energy_points, 1201)

    def test_quicklook_formulas_reexport_through_plasmon_module(self) -> None:
        self.assertAlmostEqual(electron_debye_length_cm(100.0, 1.0e21), electron_debye_length_cm_units(100.0, 1.0e21), places=12)

    def test_phase_seams_preserve_requested_settings_even_as_models_arrive(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model="mermin",
            plasmon_integration_mode="los_integrated",
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=1.2e14,
            plasmon_instrument_fwhm_ev=3.5,
            plasmon_energy_window_ev=120.0,
            plasmon_energy_points=1501,
            plasmon_lfc_model="esa_static",
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        result = evaluate_plasmon_regime(
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
        self.assertEqual(result.requested_model_name, "mermin")
        self.assertEqual(result.collision_model, "manual_constant")
        self.assertAlmostEqual(result.manual_collision_rate_s, 1.2e14)
        self.assertAlmostEqual(result.instrument_fwhm_ev, 3.5)
        self.assertEqual(result.spectrum_points, 1501)
        messages = [warning.message.lower() for warning in result.warnings]
        self.assertTrue(any("los-integrated" in message for message in messages))
        self.assertTrue(any("static-lfc" in message or "non-lfc" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
