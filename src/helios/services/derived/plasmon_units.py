"""Quick-look plasmon/XRTS constants and unit helpers.

This module deliberately keeps the current NRL-style quick-look formulas in one
place so later advanced dielectric/spectrum kernels can import the same unit
conversions without duplicating constants.
"""

from __future__ import annotations

import math

from helios.services.constants.nrl import HBAR_EV_S
from helios.services.units.conversions import photon_energy_kev_to_wavelength_cm


def electron_debye_length_cm(te_ev: float, ne_cm3: float) -> float:
    """NRL 2023 p. 29, Fundamental Plasma Parameters, Debye length."""

    if te_ev <= 0.0 or ne_cm3 <= 0.0:
        return float("nan")
    return 7.43e2 * math.sqrt(float(te_ev) / float(ne_cm3))


def electron_plasma_frequency_rad_s(ne_cm3: float) -> float:
    """NRL 2023 p. 29, electron plasma frequency, output in rad/s."""

    if ne_cm3 <= 0.0:
        return float("nan")
    return 5.64e4 * math.sqrt(float(ne_cm3))


def electron_plasma_energy_ev(ne_cm3: float) -> float:
    """Return hbar*omega_pe in eV for a density expressed in cm^-3."""

    omega_pe = electron_plasma_frequency_rad_s(ne_cm3)
    return float(HBAR_EV_S * omega_pe) if math.isfinite(omega_pe) else float("nan")


def coulomb_logarithm_ei(te_ev: float, ne_cm3: float, z_eff: float) -> float:
    """NRL 2023 p. 29 / p. 37 Coulomb-log quick look."""

    if te_ev <= 0.0 or ne_cm3 <= 0.0:
        return float("nan")
    lambda_d = electron_debye_length_cm(te_ev, ne_cm3)
    classical_min_cm = 1.44e-7 * max(float(z_eff), 1.0e-6) / float(te_ev)
    quantum_min_cm = 2.76e-8 / math.sqrt(float(te_ev))
    r_min = max(classical_min_cm, quantum_min_cm)
    if not math.isfinite(lambda_d) or lambda_d <= r_min or r_min <= 0.0:
        return float("nan")
    return math.log(lambda_d / r_min)




def coulomb_logarithm_ei_nrl_piecewise(te_ev: float, ne_cm3: float, z_eff: float) -> float:
    """Return the NRL-style piecewise e-i Coulomb log estimate.

    The quick-look Debye-over-rmin estimate is convenient and explicit, but it
    can return NaN in dense or partially degenerate states where the ordering of
    the characteristic lengths becomes ambiguous. For the constant-ν collision
    branch we therefore keep a direct fallback to the standard piecewise NRL
    electron-ion Coulomb-log formulas in cgs/eV units.
    """

    if te_ev <= 0.0 or ne_cm3 <= 0.0:
        return float("nan")
    z_value = max(float(z_eff), 1.0)
    try:
        if te_ev < 10.0 * z_value * z_value:
            value = 23.0 - math.log(math.sqrt(float(ne_cm3)) * z_value * float(te_ev) ** (-1.5))
        else:
            value = 24.0 - math.log(math.sqrt(float(ne_cm3)) * float(te_ev) ** (-1.0))
    except (ValueError, OverflowError):
        return float("nan")
    return value if math.isfinite(value) and value > 0.0 else float("nan")

def electron_collision_rate_s(ne_cm3: float, te_ev: float, coulomb_log: float) -> float:
    """NRL 2023 p. 29, electron collision rate, output in s^-1."""

    if ne_cm3 <= 0.0 or te_ev <= 0.0 or not math.isfinite(coulomb_log) or coulomb_log <= 0.0:
        return float("nan")
    return 2.91e-6 * float(ne_cm3) * float(coulomb_log) * float(te_ev) ** (-1.5)


def ion_sound_speed_cm_s(te_ev: float, mean_charge: float, ion_mass_mu: float, gamma: float) -> float:
    """NRL 2023 p. 30, ion sound speed, output in cm/s."""

    if te_ev <= 0.0 or mean_charge <= 0.0 or ion_mass_mu <= 0.0 or gamma <= 0.0:
        return float("nan")
    return 9.79e5 * math.sqrt(float(gamma) * float(mean_charge) * float(te_ev) / float(ion_mass_mu))


def plasmon_probe_wavelength_cm(photon_energy_kev: float) -> float:
    """Return the probe wavelength in cm for a photon energy given in keV."""

    return photon_energy_kev_to_wavelength_cm(photon_energy_kev)


def plasmon_probe_wavelength_angstrom(photon_energy_kev: float) -> float:
    """Return the probe wavelength in angstrom for a photon energy given in keV."""

    return plasmon_probe_wavelength_cm(photon_energy_kev) / 1.0e-8


def scattering_wavevector_cm_inv(photon_energy_kev: float, scattering_angle_deg: float) -> float:
    """Return the XRTS scattering wavevector in cm^-1."""

    wavelength_cm = plasmon_probe_wavelength_cm(photon_energy_kev)
    if not math.isfinite(wavelength_cm) or wavelength_cm <= 0.0:
        return float("nan")
    return 4.0 * math.pi * math.sin(math.radians(float(scattering_angle_deg)) / 2.0) / wavelength_cm


ELECTRON_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
EPSILON_0_F_M = 8.8541878128e-12
HBAR_J_S = 1.054571817e-34
BOHR_RADIUS_M = 5.29177210903e-11


def electron_density_m3_from_cm3(ne_cm3: float) -> float:
    return float(ne_cm3) * 1.0e6


def electron_debye_length_m(te_ev: float, ne_cm3: float) -> float:
    value_cm = electron_debye_length_cm(te_ev, ne_cm3)
    return value_cm * 1.0e-2 if math.isfinite(value_cm) else float("nan")


def scattering_wavevector_m_inv(photon_energy_kev: float, scattering_angle_deg: float) -> float:
    value_cm_inv = scattering_wavevector_cm_inv(photon_energy_kev, scattering_angle_deg)
    return value_cm_inv * 1.0e2 if math.isfinite(value_cm_inv) else float("nan")


def electron_thermal_speed_m_s(te_ev: float) -> float:
    if te_ev <= 0.0:
        return float("nan")
    return math.sqrt(float(te_ev) * ELECTRON_CHARGE_C / ELECTRON_MASS_KG)


def electron_fermi_wavevector_m_inv(ne_cm3: float) -> float:
    ne_m3 = electron_density_m3_from_cm3(ne_cm3)
    if ne_m3 <= 0.0:
        return float("nan")
    return (3.0 * math.pi * math.pi * ne_m3) ** (1.0 / 3.0)


def electron_fermi_energy_ev(ne_cm3: float) -> float:
    kf = electron_fermi_wavevector_m_inv(ne_cm3)
    if not math.isfinite(kf) or kf <= 0.0:
        return float("nan")
    energy_j = (HBAR_J_S * HBAR_J_S * kf * kf) / (2.0 * ELECTRON_MASS_KG)
    return energy_j / ELECTRON_CHARGE_C


def electron_theta_degeneracy(te_ev: float, ne_cm3: float) -> float:
    ef = electron_fermi_energy_ev(ne_cm3)
    if not math.isfinite(ef) or ef <= 0.0:
        return float("nan")
    return float(te_ev) / ef


def electron_wigner_seitz_rs(ne_cm3: float) -> float:
    ne_m3 = electron_density_m3_from_cm3(ne_cm3)
    if ne_m3 <= 0.0:
        return float("nan")
    radius_m = (3.0 / (4.0 * math.pi * ne_m3)) ** (1.0 / 3.0)
    return radius_m / BOHR_RADIUS_M


def electron_k_over_kf(k_m_inv: float, ne_cm3: float) -> float:
    kf = electron_fermi_wavevector_m_inv(ne_cm3)
    if not math.isfinite(kf) or kf <= 0.0 or not math.isfinite(k_m_inv) or k_m_inv < 0.0:
        return float("nan")
    return float(k_m_inv) / kf
