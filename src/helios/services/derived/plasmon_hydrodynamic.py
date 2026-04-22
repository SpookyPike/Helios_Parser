"""Quantum-hydrodynamic dielectric backend for plasmon benchmarking.

This backend is intentionally distinct from the existing response families:

- the classical Maxwellian finite-temperature susceptibility;
- the finite-temperature Lindhard family.

It models the longitudinal electron response with a damped quantum-fluid
dielectric that keeps collective pressure and quantum-recoil terms explicit:

    epsilon(q, omega) =
        1 - omega_p^2 / (omega * (omega + i nu) - beta_eff^2 q^2 - omega_B^2)

with

    beta_eff^2 = 3 v_th^2 + (3/5) v_F^2
    omega_B = hbar q^2 / (2 m_e)

The intent is not to claim article-native physics, but to add one structurally
different backend that changes the response object itself instead of layering
another surrogate correction on top of the current families.
"""

from __future__ import annotations

import math

import numpy as np

from helios.services.constants.nrl import HBAR_EV_S
from helios.services.derived.plasmon_units import (
    ELECTRON_MASS_KG,
    HBAR_J_S,
    electron_fermi_wavevector_m_inv,
    electron_plasma_frequency_rad_s,
    electron_thermal_speed_m_s,
)


_DEF_NUMERICAL_IMAG_SHIFT_EV = 1.0e-9


def hydrodynamic_backend_name() -> str:
    return "quantum_hydrodynamic"


def epsilon_quantum_hydrodynamic(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    collision_rate_s: float,
    imag_shift_ev: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Return ``(chi, epsilon, metadata)`` for a damped QHD dielectric."""

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    nan_array = np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128)
    metadata: dict[str, float | str] = {
        "backend_name": hydrodynamic_backend_name(),
        "backend_summary": "",
        "beta_eff_m_s": float("nan"),
        "thermal_speed_m_s": float("nan"),
        "fermi_speed_m_s": float("nan"),
        "bohm_recoil_energy_ev": float("nan"),
    }
    q_m_inv = float(k_m_inv)
    if (
        not math.isfinite(q_m_inv)
        or q_m_inv <= 0.0
        or not math.isfinite(float(te_ev))
        or float(te_ev) <= 0.0
        or not math.isfinite(float(ne_cm3))
        or float(ne_cm3) <= 0.0
    ):
        metadata["backend_summary"] = "QHD backend rejected a non-finite or non-positive state."
        return nan_array.copy(), nan_array.copy(), metadata
    nu_s = float(collision_rate_s)
    if not math.isfinite(nu_s) or nu_s < 0.0:
        metadata["backend_summary"] = "QHD backend rejected a non-finite collision rate."
        return nan_array.copy(), nan_array.copy(), metadata

    omega = (energy + 1j * max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)) / HBAR_EV_S
    omega_pe = float(electron_plasma_frequency_rad_s(float(ne_cm3)))
    v_th = float(electron_thermal_speed_m_s(float(te_ev)))
    k_fermi = float(electron_fermi_wavevector_m_inv(float(ne_cm3)))
    v_fermi = (HBAR_J_S * k_fermi / ELECTRON_MASS_KG) if math.isfinite(k_fermi) and k_fermi > 0.0 else float("nan")
    thermal_term_sq = 3.0 * max(v_th, 0.0) ** 2
    fermi_term_sq = (3.0 / 5.0) * max(v_fermi if math.isfinite(v_fermi) else 0.0, 0.0) ** 2
    beta_eff_sq = thermal_term_sq + fermi_term_sq
    beta_eff = math.sqrt(beta_eff_sq) if math.isfinite(beta_eff_sq) and beta_eff_sq >= 0.0 else float("nan")
    omega_bohm = (HBAR_J_S * (q_m_inv**2)) / (2.0 * ELECTRON_MASS_KG)
    omega_bohm_sq = omega_bohm**2

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = omega * (omega + 1j * nu_s) - beta_eff_sq * (q_m_inv**2) - omega_bohm_sq
        epsilon = 1.0 - (omega_pe**2) / denominator
    chi = np.asarray(epsilon - 1.0, dtype=np.complex128)
    epsilon = np.asarray(epsilon, dtype=np.complex128)
    bad = (~np.isfinite(np.real(chi))) | (~np.isfinite(np.imag(chi))) | (~np.isfinite(np.real(epsilon))) | (~np.isfinite(np.imag(epsilon)))
    if np.any(bad):
        chi = np.asarray(chi, dtype=np.complex128)
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        chi[bad] = np.nan + 0.0j
        epsilon[bad] = np.nan + 0.0j

    recoil_energy_ev = float(HBAR_EV_S * omega_bohm) if math.isfinite(omega_bohm) else float("nan")
    metadata.update(
        {
            "beta_eff_m_s": float(beta_eff),
            "thermal_speed_m_s": float(v_th),
            "fermi_speed_m_s": float(v_fermi),
            "bohm_recoil_energy_ev": recoil_energy_ev,
            "backend_summary": (
                "QHD dielectric with beta_eff^2 = 3 v_th^2 + 3/5 v_F^2, "
                f"Bohm recoil {recoil_energy_ev:.4g} eV, nu = {nu_s:.4g} 1/s."
            ),
        }
    )
    return chi, epsilon, metadata
