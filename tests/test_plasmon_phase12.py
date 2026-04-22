from __future__ import annotations

from pathlib import Path
import os
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon_validation import (
    compute_plasmon,
    make_run_context,
    q_to_angle_deg,
    shocked_al_slab_summary,
    uniform_al_dataset,
)

HYDRO_CANDIDATES = (
    Path(os.environ.get("HELIOS_DRIVEN_AL_H5", "")),
    Path("/mnt/data/50Al+10E+25CH+3.5TW_stabilized.h5"),
    Path("50Al+10E+25CH+3.5TW_stabilized.h5"),
)


def _hydro_path() -> Path:
    for candidate in HYDRO_CANDIDATES:
        if str(candidate) and candidate.is_file():
            return candidate
    raise unittest.SkipTest("Driven Al hydro file is not available for phase12 validation.")


class PlasmonPhase12Tests(unittest.TestCase):
    def test_shocked_al_slab_summary_is_in_article_like_range(self) -> None:
        for target_time_ns in (6.3, 6.4, 6.5):
            dataset = load_run_data(_hydro_path())
            time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
            snapshot_index = int(np.argmin(np.abs(time_ns - target_time_ns)))
            summary = shocked_al_slab_summary(dataset, snapshot_index=snapshot_index, density_floor_g_cm3=3.75, material_id=1)

            self.assertGreaterEqual(int(summary["zone_count"]), 350)
            self.assertLessEqual(int(summary["zone_count"]), 500)
            self.assertGreaterEqual(float(summary["rho_weighted_g_cm3"]), 4.0)
            self.assertLessEqual(float(summary["rho_weighted_g_cm3"]), 4.3)
            self.assertGreaterEqual(float(summary["te_weighted_ev"]), 0.45)
            self.assertLessEqual(float(summary["te_weighted_ev"]), 0.55)
            self.assertLess(float(summary["zbar_weighted"]), 0.5)
            self.assertGreaterEqual(int(summary["zone_index_lower"]), 500)
            self.assertLessEqual(int(summary["zone_index_upper"]), 1000)


    def test_native_hydro_path_stays_far_below_literature_scale_but_valence_locked_state_recovers_it(self) -> None:
        hydro_path = _hydro_path()
        dataset = load_run_data(hydro_path)
        time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
        snapshot_index = int(np.argmin(np.abs(time_ns - 6.4)))
        summary = shocked_al_slab_summary(dataset, snapshot_index=snapshot_index, density_floor_g_cm3=3.75, material_id=1)
        context = make_run_context(dataset, hydro_path, snapshot_index=snapshot_index)
        angle = q_to_angle_deg(1.57, 8.307)

        native = compute_plasmon(
            dataset,
            context,
            plasmon_model="rpa_static_lfc",
            plasmon_execution_mode="quicklook",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=601,
            plasmon_instrument_fwhm_ev=0.20,
            plasmon_lfc_model="esa_static",
            derived_material_ids=(1,),
            zone_index_lower=int(summary["zone_index_lower"]),
            zone_index_upper=int(summary["zone_index_upper"]),
        )

        uniform_dataset, uniform_context = uniform_al_dataset(float(summary["rho_weighted_g_cm3"]), float(summary["te_weighted_ev"]))
        valence_locked = compute_plasmon(
            uniform_dataset,
            uniform_context,
            plasmon_model="lindhard",
            plasmon_execution_mode="quicklook",
            plasmon_integration_mode="effective_state",
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=angle,
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=601,
            plasmon_instrument_fwhm_ev=0.20,
            plasmon_lfc_model="esa_static",
        )

        self.assertLess(float(native.peak_energy_ev), 15.0)
        self.assertGreaterEqual(float(valence_locked.peak_energy_ev), 30.0)
        self.assertLessEqual(float(valence_locked.peak_energy_ev), 36.0)
        self.assertGreater(float(valence_locked.peak_energy_ev - native.peak_energy_ev), 20.0)
