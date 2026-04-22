from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_lfc import esa_domain_contains, esa_static_local_field_correction
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from test_plasmon_phase2 import _synthetic_dataset


class PlasmonPhase4Tests(unittest.TestCase):
    def _compute_synthetic(self, model: str = "rpa_static_lfc", **kwargs):
        dataset, context = _synthetic_dataset(te_ev=kwargs.pop("te_ev", 120.0), ne_cm3=kwargs.pop("ne_cm3", 8.0e20))
        parameters = DerivedAnalysisParameters(
            plasmon_model=model,
            plasmon_photon_energy_kev=kwargs.pop("photon_energy_kev", 8.0),
            plasmon_scattering_angle_deg=kwargs.pop("scattering_angle_deg", 35.0),
            plasmon_energy_window_ev=kwargs.pop("energy_window_ev", 40.0),
            plasmon_energy_points=kwargs.pop("energy_points", 801),
            plasmon_instrument_fwhm_ev=kwargs.pop("instrument_fwhm_ev", 1.5),
            plasmon_lfc_model=kwargs.pop("lfc_model", "esa_static"),
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
            analysis_cache=kwargs.pop("analysis_cache", None),
        )

    def test_esa_backend_returns_bounded_static_lfc(self) -> None:
        values = esa_static_local_field_correction(np.asarray([0.0, 1.0, 3.0, 10.0]), 2.0, 1.0)
        self.assertAlmostEqual(float(values[0]), 0.0, places=12)
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertGreater(float(values[-1]), float(values[1]))

    def test_rpa_static_lfc_builds_finite_spectrum_and_differs_from_rpa(self) -> None:
        base = self._compute_synthetic(model="rpa")
        corr = self._compute_synthetic(model="rpa_static_lfc")
        self.assertEqual(corr.model_name, "rpa_static_lfc")
        self.assertTrue(np.all(np.isfinite(corr.spectrum_intensity)))
        self.assertGreater(corr.static_lfc_value, 0.0)
        self.assertGreater(corr.q_over_qf, 0.0)
        delta = np.nanmax(np.abs(corr.spectrum_intensity - base.spectrum_intensity))
        self.assertGreater(delta, 1.0e-5)

    def test_domain_warning_triggers_outside_esa_range(self) -> None:
        result = self._compute_synthetic(model="rpa_static_lfc", te_ev=300.0, ne_cm3=2.0e19)
        self.assertFalse(esa_domain_contains(result.wigner_seitz_rs, result.theta_degeneracy))
        messages = [warning.message.lower() for warning in result.warnings]
        self.assertTrue(any("validated only" in message for message in messages))

    def test_cache_bucket_reuses_rpa_static_lfc_payload(self) -> None:
        cache = AnalysisStateCache()
        first = self._compute_synthetic(model="rpa_static_lfc", analysis_cache=cache)
        second = self._compute_synthetic(model="rpa_static_lfc", analysis_cache=cache)
        stats = cache.stats()
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))

    def test_reference_example_static_lfc_is_finite_and_compares_to_rpa(self) -> None:
        path = _test_bootstrap.example_data_path("5Fe+4.9TW+light_stabilized.h5")
        dataset = load_run_data(path)
        context = RunContext(
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
            snapshot_index=min(5, max(0, len(dataset.time_s) - 1)),
            map_coordinate="static_x",
            slice_coordinate="zone",
            selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
            selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
        )
        def compute(model: str):
            params = DerivedAnalysisParameters(
                plasmon_model=model,
                plasmon_photon_energy_kev=8.0,
                plasmon_scattering_angle_deg=25.0,
                plasmon_energy_window_ev=60.0,
                plasmon_energy_points=1001,
                plasmon_lfc_model="esa_static",
            )
            geometry = build_analysis_geometry(
                dataset, context,
                observation_side=params.observation_side,
                line_of_sight_angle_deg=params.line_of_sight_angle_deg,
                line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
                profile_coordinate_mode=params.profile_coordinate_mode,
            )
            return evaluate_plasmon_regime(
                dataset, context, snapshot_index=context.snapshot_index,
                photon_energy_kev=params.plasmon_photon_energy_kev,
                scattering_angle_deg=params.plasmon_scattering_angle_deg,
                adiabatic_index=params.plasmon_adiabatic_index,
                parameters=params, geometry=geometry, include_time_plots=False,
            )
        rpa = compute("rpa")
        lfc = compute("rpa_static_lfc")
        self.assertTrue(np.all(np.isfinite(lfc.spectrum_intensity)))
        self.assertGreater(lfc.static_lfc_value, 0.0)
        self.assertGreater(np.nanmax(np.abs(lfc.spectrum_intensity - rpa.spectrum_intensity)), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
