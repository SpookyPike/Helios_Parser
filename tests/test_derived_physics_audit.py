"""Analytic checks on derived quantities that the audit of 2026-11-04 fixed.

Each test pins a number against a closed form or a measured convention, so the
next edit that silently changes the physics fails here instead of in a figure:

- the finite-T Lindhard susceptibility against the Thomas-Fermi static limit and
  the Debye limit, which is what fixes the k quadrature order for a degenerate gas
- the Doppler width coefficient against ``2 sqrt(ln 2) sqrt(2 k T / m) / c``, which
  is what makes it a FWHM rather than a sigma
- the laser-power unit, which the log header mislabels as TW while printing watts
- the per-group flux unit, which the vendor names "per eV" for a bin content
- the geometry placeholder in the mass unit, which HELIOS never substitutes
- the residual tolerance used to compare radiation heating against cooling

Run with ``.venv/bin/python -m unittest tests.test_derived_physics_audit``.
"""

from __future__ import annotations

import math
import unittest

import _test_bootstrap  # noqa: F401
import numpy as np

from helios.services.constants import C_LIGHT_CM_S
from helios.services.derived.common import _finite_allclose_residual
from helios.services.derived.plasmon_lindhard import (
    finite_t_lindhard_susceptibility,
    lindhard_k_order,
    lindhard_polarization,
)
from helios.services.derived.plasmon_units import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    HBAR_J_S,
    electron_debye_length_m,
    electron_fermi_energy_ev,
    electron_fermi_wavevector_m_inv,
    electron_plasma_energy_ev,
)
from helios.services.derived.spectroscopy import doppler_width_fraction
from helios_parser.exodus import FLUX_FIELDS
from helios_parser.hdf5 import _input_parameter_unit, _resolve_geometry_unit

_BOLTZMANN_ERG_PER_K = 1.380649e-16
_EV_ERG = 1.602176634e-12
_PROTON_G = 1.67262192369e-24


def _thomas_fermi_static_limit(ne_cm3: float, q_m_inv: float) -> float:
    """``chi(q->0, omega=0) = k_TF^2/q^2`` with ``k_TF^2 = 3 omega_p^2 / v_F^2``."""

    kf = electron_fermi_wavevector_m_inv(ne_cm3)
    v_f = HBAR_J_S * kf / ELECTRON_MASS_KG
    omega_p = electron_plasma_energy_ev(ne_cm3) * ELECTRON_CHARGE_C / HBAR_J_S
    return 3.0 * omega_p**2 / v_f**2 / q_m_inv**2


class LindhardQuadratureTests(unittest.TestCase):
    """The k quadrature must resolve the Fermi surface a degenerate gas puts in range."""

    def test_static_limit_is_accurate_at_moderate_degeneracy(self) -> None:
        ne_cm3 = 1.0e22
        theta = 0.05 / electron_fermi_energy_ev(ne_cm3)
        self.assertLess(theta, 0.05)
        q = 1.0e-2 * electron_fermi_wavevector_m_inv(ne_cm3)
        chi = finite_t_lindhard_susceptibility(
            np.array([0.0]), k_m_inv=q, te_ev=0.05, ne_cm3=ne_cm3, imag_shift_ev=1.0e-5
        )[0].real
        ratio = chi / _thomas_fermi_static_limit(ne_cm3, q)
        self.assertAlmostEqual(ratio, 1.0, delta=0.02, msg=f"chi/limit = {ratio:.4f}")

    def test_the_fixed_order_was_measurably_wrong(self) -> None:
        """Guard the guard: a fixed 64-point rule is 1.8x high at the same state."""

        ne_cm3 = 1.0e22
        q = 1.0e-2 * electron_fermi_wavevector_m_inv(ne_cm3)
        e, eps0 = ELECTRON_CHARGE_C, 8.8541878128e-12
        polarization = lindhard_polarization(
            np.array([0.0]), k_m_inv=q, te_ev=0.05, ne_cm3=ne_cm3, imag_shift_ev=1.0e-5, k_order=64, mu_order=40
        )
        chi = (-(e * e / (eps0 * q * q)) * polarization)[0].real
        ratio = chi / _thomas_fermi_static_limit(ne_cm3, q)
        self.assertGreater(ratio, 1.5, f"the 64-point rule should be visibly wrong, got {ratio:.4f}")
        self.assertGreater(lindhard_k_order(ne_cm3, 0.05, q), 64)

    def test_debye_limit_for_a_classical_plasma(self) -> None:
        """Non-degenerate limit: chi(q->0, 0) -> k_D^2/q^2 with k_D = 1/lambda_De."""

        ne_cm3 = 1.0e20
        te_ev = 1000.0
        k_th = math.sqrt(2.0 * ELECTRON_MASS_KG * te_ev * ELECTRON_CHARGE_C) / HBAR_J_S
        k_debye_sq = 1.0 / electron_debye_length_m(te_ev, ne_cm3) ** 2
        for fraction in (1.0e-3, 1.0e-1):
            with self.subTest(fraction=fraction):
                q = fraction * k_th
                chi = finite_t_lindhard_susceptibility(
                    np.array([0.0]), k_m_inv=q, te_ev=te_ev, ne_cm3=ne_cm3, imag_shift_ev=1.0e-3
                )[0].real
                self.assertAlmostEqual(chi / (k_debye_sq / q**2), 1.0, delta=0.01)

    def test_plasma_limit_at_high_frequency(self) -> None:
        """chi -> -omega_p^2/omega^2 for omega >> omega_p, the f-sum normalisation."""

        ne_cm3 = 1.0e20
        omega_p_ev = electron_plasma_energy_ev(ne_cm3)
        q = 0.1 * electron_fermi_wavevector_m_inv(ne_cm3)
        for multiple in (100.0, 1000.0):
            with self.subTest(multiple=multiple):
                energy_ev = multiple * omega_p_ev
                chi = finite_t_lindhard_susceptibility(
                    np.array([energy_ev]), k_m_inv=q, te_ev=100.0, ne_cm3=ne_cm3, imag_shift_ev=1.0e-4
                )[0].real
                self.assertAlmostEqual(chi / (-(omega_p_ev**2) / energy_ev**2), 1.0, delta=0.01)


class DopplerWidthTests(unittest.TestCase):
    """NRL Eq. (25) is the FWHM, so the coefficient must match the FWHM closed form."""

    def test_coefficient_is_the_full_width_at_half_maximum(self) -> None:
        temperature_ev = 1.0
        ion_mass_mu = 1.0
        # 2 sqrt(ln 2) * sqrt(2 k T / m) / c, all cgs, with T in eV.
        thermal_speed_cm_s = math.sqrt(
            2.0 * _BOLTZMANN_ERG_PER_K * (temperature_ev * _EV_ERG / _BOLTZMANN_ERG_PER_K) / (_PROTON_G * ion_mass_mu)
        )
        fwhm_fraction = 2.0 * math.sqrt(math.log(2.0)) * thermal_speed_cm_s / C_LIGHT_CM_S
        # NRL prints the coefficient as 7.7e-5, i.e. rounded to two digits.
        self.assertAlmostEqual(fwhm_fraction / 7.7e-5, 1.0, delta=5.0e-3)
        self.assertAlmostEqual(doppler_width_fraction(temperature_ev, ion_mass_mu) / fwhm_fraction, 1.0, delta=5.0e-3)
        # And it is not the 1/e half-width or the Gaussian sigma, which differ by
        # 1.665x and 2.355x respectively.
        self.assertLess(doppler_width_fraction(temperature_ev, ion_mass_mu) / (fwhm_fraction / 2.355), 2.4)

    def test_scaling_is_square_root_temperature(self) -> None:
        base = doppler_width_fraction(100.0, 27.0)
        quadrupled = doppler_width_fraction(400.0, 27.0)
        self.assertAlmostEqual(quadrupled / base, 2.0, delta=1.0e-12)


class ConverterUnitTests(unittest.TestCase):
    """Units the converter publishes, each pinned by a measurement on real output."""

    def test_laser_power_is_declared_in_watts(self) -> None:
        """The log header says TW but prints watts: `25Cu+9.33TW.log` reads 9.33e12."""

        self.assertEqual(_input_parameter_unit(("laser_source", "power_table", "power")), "W")
        self.assertEqual(_input_parameter_unit(("laser_source", "power_table", "time")), "s")

    def test_flux_bin_contents_are_declared_per_group(self) -> None:
        """A group sum, not a dE integral, reproduces TotalFluxRmin/Rmax."""

        for vendor_name, (dataset, unit) in FLUX_FIELDS.items():
            if dataset in {"incident_rmax", "incident_rmin", "net_interface"}:
                with self.subTest(dataset=dataset):
                    self.assertEqual(unit, "J/(cm2 s)/group", vendor_name)

    def test_the_exodus_path_resolves_the_mass_unit_too(self) -> None:
        """Both writers have to agree: a planar zone mass is rho*dx per unit area,
        so a bare "g" overstates it by the cross-section."""
        import tempfile
        from pathlib import Path

        import h5py

        from helios_parser import write_hdf5

        root = Path(__file__).resolve().parents[1]
        fixture = root / "tests" / "fixtures" / "radiation" / "cu_freqrad_tiny_t0p02ns.exo"
        if not fixture.is_file():
            self.skipTest("the exodus fixture is not in this tree")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exo.h5"
            write_hdf5(fixture, out, overwrite=True)
            with h5py.File(out, "r") as handle:
                self.assertEqual(handle["grid/zone_mass"].attrs["unit"], "g/cm2")
                mass = handle["grid/zone_mass"][()]
                density = handle["fields/density"][0]
                width = handle["grid/zone_width"][()]
                np.testing.assert_allclose(mass / (density * width), 1.0, rtol=1e-9)

    def test_the_exodus_geometry_id_keeps_cylindrical_and_spherical_apart(self) -> None:
        """Both report the axis name 'r', but a mass per unit length and a mass are
        different quantities, so the exponent cannot come from the axis name."""
        import shutil
        import tempfile
        from pathlib import Path

        import h5py

        from helios_parser import write_hdf5

        root = Path(__file__).resolve().parents[1]
        fixture = root / "tests" / "fixtures" / "radiation" / "cu_freqrad_tiny_t0p02ns.exo"
        if not fixture.is_file():
            self.skipTest("the exodus fixture is not in this tree")
        attribute = "Geometry ID (0 = planar, 1 = cylindrical, 2 = spherical)"
        expected = {0: "g/cm2", 1: "g/cm", 2: "g"}
        with tempfile.TemporaryDirectory() as tmpdir:
            for geometry_id, unit in expected.items():
                source = Path(tmpdir) / f"g{geometry_id}.exo"
                shutil.copy2(fixture, source)
                # copy2 preserves the source mode, and the fixture is read-only in
                # the materialised test tree.
                source.chmod(0o600)
                with h5py.File(source, "r+") as handle:
                    handle.attrs[attribute] = geometry_id
                out = Path(tmpdir) / f"g{geometry_id}.h5"
                write_hdf5(source, out, overwrite=True)
                with h5py.File(out, "r") as handle:
                    self.assertEqual(handle["grid/zone_mass"].attrs["unit"], unit, f"id={geometry_id}")

    def test_geometry_placeholder_resolves_per_geometry(self) -> None:
        self.assertEqual(_resolve_geometry_unit("g/cm**X", "PLANAR"), "g/cm2")
        self.assertEqual(_resolve_geometry_unit("g/cm**X", "CYLINDRICAL"), "g/cm")
        self.assertEqual(_resolve_geometry_unit("g/cm**X", "SPHERICAL"), "g")
        # An unknown geometry keeps the placeholder: a guess would be worse.
        self.assertEqual(_resolve_geometry_unit("g/cm**X", None), "g/cm**X")
        self.assertEqual(_resolve_geometry_unit("g/cm3", "PLANAR"), "g/cm3")


class CancellingResidualTests(unittest.TestCase):
    """A cancelling difference is compared against the terms, not against itself."""

    def test_radiation_heating_cooling_cancellation_is_not_a_false_alarm(self) -> None:
        heating = np.array([1.218e10, 1.218e10])
        cooling = np.array([1.218e10, 1.218e10])
        net = np.array([10.83, 10.83])
        scale = np.maximum(np.abs(heating), np.abs(cooling))
        matches, worst = _finite_allclose_residual(net - (heating - cooling), scale, rtol=2.0e-3, atol=1.0e-9)
        self.assertTrue(matches)
        self.assertLess(worst, 1.0)

    def test_a_real_discrepancy_still_fails(self) -> None:
        heating = np.array([1.0e10])
        cooling = np.array([0.0])
        net = np.array([1.0e10])
        matches, _ = _finite_allclose_residual(net - (heating - cooling), np.abs(heating), rtol=2.0e-3, atol=1.0e-9)
        self.assertTrue(matches)
        net = np.array([5.0e9])
        matches, _ = _finite_allclose_residual(net - (heating - cooling), np.abs(heating), rtol=2.0e-3, atol=1.0e-9)
        self.assertFalse(matches)


if __name__ == "__main__":
    unittest.main()