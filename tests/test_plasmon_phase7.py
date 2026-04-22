from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import tempfile
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.services.derived.models import PlasmonResult
from helios.services.derived.plasmon_export import plasmon_export_columns, plasmon_export_is_ready, write_plasmon_spectrum_csv


class PlasmonPhase7Tests(unittest.TestCase):
    def _result(self) -> PlasmonResult:
        energy = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
        return PlasmonResult(
            snapshot_index=1,
            weighting_mode='electron_column',
            geometry_summary='front side',
            photon_energy_kev=7.5,
            scattering_angle_deg=20.0,
            adiabatic_index=1.0,
            electron_density_cm3=1.0e20,
            electron_temperature_ev=100.0,
            ion_temperature_ev=80.0,
            mean_charge=4.0,
            ion_mass_mu=27.0,
            debye_length_cm=1.0e-7,
            plasma_frequency_rad_s=1.0e16,
            plasma_frequency_ev=10.0,
            electron_collision_rate_s=1.0e14,
            coulomb_logarithm=2.0,
            ion_sound_speed_cm_s=1.0e6,
            probe_wavelength_angstrom=1.0,
            scattering_wavevector_cm_inv=1.0e7,
            spectrum_energy_ev=energy,
            spectrum_intensity=np.linspace(0.0, 1.0, energy.size, dtype=np.float64),
            dielectric_real=np.linspace(1.0, 2.0, energy.size, dtype=np.float64),
            dielectric_imag=np.linspace(0.0, 0.5, energy.size, dtype=np.float64),
            loss_function=np.linspace(0.1, 0.3, energy.size, dtype=np.float64),
            model_name='mermin_static_lfc',
            requested_model_name='mermin_static_lfc',
        )

    def test_export_columns_include_aligned_optional_arrays(self) -> None:
        result = self._result()
        columns = plasmon_export_columns(result)
        self.assertEqual(list(columns), ['energy_transfer_ev', 'observed_intensity', 'dielectric_real', 'dielectric_imag', 'loss_function'])
        self.assertTrue(plasmon_export_is_ready(result))

    def test_export_columns_omit_misaligned_optional_arrays(self) -> None:
        result = self._result()
        result = replace(result, dielectric_real=np.asarray([1.0, 2.0], dtype=np.float64))
        columns = plasmon_export_columns(result)
        self.assertEqual(list(columns), ['energy_transfer_ev', 'observed_intensity', 'dielectric_imag', 'loss_function'])

    def test_write_plasmon_spectrum_csv_writes_expected_header_and_rows(self) -> None:
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plasmon.csv'
            write_plasmon_spectrum_csv(path, result)
            with path.open('r', encoding='utf-8', newline='') as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ['energy_transfer_ev', 'observed_intensity', 'dielectric_real', 'dielectric_imag', 'loss_function'])
        self.assertEqual(len(rows), result.spectrum_energy_ev.size + 1)


if __name__ == '__main__':
    unittest.main()
