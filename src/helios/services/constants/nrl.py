"""NRL Formulary constants and formula metadata.

This module centralizes the physical constants and short reference metadata used
by the Phase 4 Derived / Analysis services. The primary source is:

    NRL Plasma Formulary 2023, repository file ``NRL_Formulary_2023.pdf``

Important unit discipline from the formulary:
- most plasma quick-look formulas are in Gaussian CGS
- temperatures in the plasma sections are typically expressed in eV
- ion mass is frequently expressed as ``mu = m_i / m_p``

The service layer uses these constants explicitly instead of scattering them
through individual widgets or derived calculators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormulaReference:
    """Compact documentation bundle for a quick-look formula."""

    section: str
    page: int
    equation: str | None
    input_units: str
    output_units: str
    assumptions: str


# NRL Plasma Formulary 2023, Physical Constants (CGS), p. 17.
BOLTZMANN_CGS = 1.38065e-16  # erg / K
ELEMENTARY_CHARGE_STATC = 4.80320e-10  # statcoulomb
ELECTRON_MASS_G = 9.10938e-28  # g
PROTON_MASS_G = 1.67262e-24  # g
HBAR_CGS = 1.05457e-27  # erg s
C_LIGHT_CM_S = 2.99792e10  # cm / s
CLASSICAL_ELECTRON_RADIUS_CM = 2.81794e-13  # cm
THOMSON_CROSS_SECTION_CM2 = 6.65246e-25  # cm^2
PLANCK_EV_S = 4.135667696e-15  # eV s
HBAR_EV_S = 6.582119569e-16  # eV s
WAVELENGTH_PER_EV_CM = 1.23984e-4  # cm associated with 1 eV, NRL p. 17

# Exact useful conversions.
EV_TO_J = 1.602176634e-19
EV_TO_ERG = EV_TO_J * 1.0e7
EV_TO_K = 11604.518121550082
CM_PER_ANGSTROM = 1.0e-8
CM_PER_NM = 1.0e-7
WAVELENGTH_ANGSTROM_KEV = 12.398419843320025  # lambda [A] = const / E_keV


DEBYE_LENGTH_REF = FormulaReference(
    section="Fundamental Plasma Parameters",
    page=29,
    equation=None,
    input_units="electron temperature in eV, electron density in cm^-3",
    output_units="Debye length in cm",
    assumptions="Gaussian CGS quick-look formula with temperature expressed in eV.",
)

PLASMA_FREQUENCY_REF = FormulaReference(
    section="Fundamental Plasma Parameters",
    page=29,
    equation=None,
    input_units="electron density in cm^-3",
    output_units="angular plasma frequency in rad/s",
    assumptions="Gaussian CGS electron plasma frequency quick-look formula.",
)

COLLISION_RATE_REF = FormulaReference(
    section="Fundamental Plasma Parameters",
    page=29,
    equation=None,
    input_units="electron density in cm^-3, electron temperature in eV, Coulomb logarithm dimensionless",
    output_units="electron collision rate in s^-1",
    assumptions="Maxwellian electron-ion quick-look rate with Gaussian CGS units and eV temperature.",
)

ION_SOUND_REF = FormulaReference(
    section="Fundamental Plasma Parameters",
    page=30,
    equation=None,
    input_units="electron temperature in eV, charge state Z, ion mass mu = m_i / m_p, adiabatic index gamma",
    output_units="ion sound speed in cm/s",
    assumptions="Single-fluid ion-acoustic quick-look estimate with T_e >> T_i style scaling.",
)

DOPPLER_WIDTH_REF = FormulaReference(
    section="Radiation",
    page=58,
    equation="(25)",
    input_units="emitter temperature in eV, ion mass mu = M / m_p, line wavelength in any consistent wavelength unit",
    output_units="fractional Doppler width Delta(lambda)/lambda (dimensionless)",
    assumptions="Thermal Doppler width with temperatures expressed in eV.",
)

OPTICAL_DEPTH_REF = FormulaReference(
    section="Radiation",
    page=58,
    equation="(26)",
    input_units="oscillator strength f, wavelength in cm, mass ratio mu, absorber temperature in eV, number density in cm^-3, path length in cm",
    output_units="line-center optical depth (dimensionless)",
    assumptions="Doppler-broadened line optical depth quick-look estimate; optically thin means tau < 1.",
)
