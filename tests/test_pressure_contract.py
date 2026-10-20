from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
import numpy as np
from helios_parser.quantities import plasma_pressure, total_pressure
from helios_viewer.units import DisplayUnitChoices, convert_field_values


class PressureContractTests(unittest.TestCase):
    def test_plasma_and_total_pressure_are_distinct(self) -> None:
        ion = np.asarray([1.0])
        electron = np.asarray([2.0])
        radiation = np.asarray([4.0])
        np.testing.assert_array_equal(plasma_pressure(ion, electron), [3.0])
        np.testing.assert_array_equal(total_pressure(ion, electron, radiation), [7.0])

    def test_explicit_pressure_fields_use_pressure_display_units(self) -> None:
        units = DisplayUnitChoices(
            time_unit="ns", length_unit="um", pressure_unit="GPa", density_unit="g/cm3",
            temperature_unit="eV", velocity_unit="cm/s", specific_energy_unit="J/g",
            rate_unit="J/g/s", heat_capacity_unit="J/g/eV", number_density_unit="1/cm3",
        )
        for field_name in ("pressure_j_cm3", "plasma_pressure_j_cm3", "total_pressure_j_cm3"):
            values, label = convert_field_values(field_name, np.asarray([1.0]), "J/cm3", units)
            self.assertEqual(label, "GPa")
            np.testing.assert_allclose(values, convert_field_values("pressure", np.asarray([1.0]), "J/cm3", units)[0])


if __name__ == "__main__":
    unittest.main()
