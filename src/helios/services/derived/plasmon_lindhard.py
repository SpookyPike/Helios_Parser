"""Finite-temperature Lindhard response helpers.

This module provides a numerically evaluated finite-T Lindhard / Mermin-ready
collisionless dielectric baseline for the homogeneous electron gas. The
implementation intentionally favors robustness and explicit domain handling over
maximal asymptotic optimization so it can serve as a stronger-response backend
for benchmark and auto-selected plasmon models.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
try:
    from scipy.optimize import brentq as _scipy_brentq
except ModuleNotFoundError:  # pragma: no cover - optional numerical helper
    _scipy_brentq = None

from helios.services.derived.plasmon_units import (
    BOHR_RADIUS_M,
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    EPSILON_0_F_M,
    HBAR_J_S,
    electron_density_m3_from_cm3,
    electron_fermi_energy_ev,
    electron_fermi_wavevector_m_inv,
)

_PI2 = math.pi * math.pi
_MIN_IMAG_SHIFT_EV = 1.0e-9


def _bracketed_root_solve(
    func,
    lower: float,
    upper: float,
    *,
    maxiter: int = 200,
    xtol: float = 1.0e-8,
    rtol: float = 1.0e-10,
) -> float:
    if _scipy_brentq is not None:
        return float(_scipy_brentq(func, lower, upper, maxiter=maxiter, xtol=xtol, rtol=rtol))
    a = float(lower)
    b = float(upper)
    fa = float(func(a))
    fb = float(func(b))
    if not math.isfinite(fa) or not math.isfinite(fb) or fa == 0.0 and fb == 0.0 or fa * fb > 0.0:
        raise ValueError("Root is not bracketed for bracketed_root_solve().")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(max(int(maxiter), 1)):
        midpoint = 0.5 * (a + b)
        fm = float(func(midpoint))
        if not math.isfinite(fm):
            raise ValueError("Non-finite function value during bracketed_root_solve().")
        interval = abs(b - a)
        tolerance = max(float(xtol), float(rtol) * max(abs(a), abs(b), abs(midpoint), 1.0))
        if fm == 0.0 or interval <= tolerance:
            return float(midpoint)
        if fa * fm < 0.0:
            b = midpoint
            fb = fm
        else:
            a = midpoint
            fa = fm
    return float(0.5 * (a + b))


@lru_cache(maxsize=16)
def _leggauss_cached(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _clipped_exp_argument(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), -700.0, 700.0)


def _energy_from_k_j(k_m_inv: np.ndarray) -> np.ndarray:
    k = np.asarray(k_m_inv, dtype=np.float64)
    return (HBAR_J_S * HBAR_J_S * k * k) / (2.0 * ELECTRON_MASS_KG)


def _fermi_occupation(energy_j: np.ndarray, mu_j: float, te_ev: float) -> np.ndarray:
    thermal_j = float(te_ev) * ELECTRON_CHARGE_C
    if not math.isfinite(thermal_j) or thermal_j <= 0.0:
        return np.zeros_like(np.asarray(energy_j, dtype=np.float64))
    arg = _clipped_exp_argument((np.asarray(energy_j, dtype=np.float64) - float(mu_j)) / thermal_j)
    return 1.0 / (1.0 + np.exp(arg))


def _k_scales(ne_cm3: float, te_ev: float, q_m_inv: float = 0.0) -> tuple[float, float, float]:
    kf = float(electron_fermi_wavevector_m_inv(ne_cm3))
    thermal_j = float(te_ev) * ELECTRON_CHARGE_C
    kth = math.sqrt(max(2.0 * ELECTRON_MASS_KG * thermal_j, 0.0)) / HBAR_J_S if thermal_j > 0.0 else 0.0
    k_char = max(kf, kth, abs(float(q_m_inv)), 1.0e7)
    k_max = max(6.0 * max(kf, 1.0e7) + 2.0 * abs(float(q_m_inv)), 10.0 * max(kth, 1.0e7) + 2.0 * abs(float(q_m_inv)), 8.0 * k_char)
    return kf, kth, k_max


def _density_from_mu_cm3(mu_ev: float, ne_cm3: float, te_ev: float, *, order: int = 128) -> float:
    _, _, k_max = _k_scales(ne_cm3, te_ev, 0.0)
    nodes, weights = _leggauss_cached(order)
    k = 0.5 * (nodes + 1.0) * k_max
    wk = 0.5 * k_max * weights
    occ = _fermi_occupation(_energy_from_k_j(k), float(mu_ev) * ELECTRON_CHARGE_C, te_ev)
    density_m3 = float(np.sum((k * k) * occ * wk, dtype=np.float64) / _PI2)
    return density_m3 * 1.0e-6


@lru_cache(maxsize=256)
def _chemical_potential_ev_cached(ne_cm3_rounded: float, te_ev_rounded: float) -> float:
    ne_cm3 = float(ne_cm3_rounded)
    te_ev = float(te_ev_rounded)
    if not math.isfinite(ne_cm3) or ne_cm3 <= 0.0 or not math.isfinite(te_ev) or te_ev <= 0.0:
        return float("nan")
    ef_ev = float(electron_fermi_energy_ev(ne_cm3))
    theta = te_ev / ef_ev if math.isfinite(ef_ev) and ef_ev > 0.0 else float("nan")
    if math.isfinite(theta) and theta < 0.03:
        return float(ef_ev * (1.0 - (math.pi * math.pi / 12.0) * theta * theta))
    ne_target = ne_cm3
    lambda_th_m = math.sqrt(2.0 * math.pi * HBAR_J_S * HBAR_J_S / (ELECTRON_MASS_KG * te_ev * ELECTRON_CHARGE_C))
    classical_mu = te_ev * math.log(max(electron_density_m3_from_cm3(ne_cm3) * (lambda_th_m ** 3) / 2.0, 1.0e-300))
    lower = min(classical_mu - 40.0 * te_ev, -8.0 * max(ef_ev, te_ev), -400.0 * te_ev)
    upper = max(ef_ev + 40.0 * te_ev, classical_mu + 80.0 * te_ev, 20.0 * te_ev)

    def residual(mu_ev: float) -> float:
        return _density_from_mu_cm3(mu_ev, ne_cm3, te_ev) - ne_target

    f_low = residual(lower)
    f_high = residual(upper)
    expand = 0
    while (not math.isfinite(f_low) or not math.isfinite(f_high) or f_low > 0.0 or f_high < 0.0) and expand < 12:
        lower -= max(te_ev, ef_ev, 1.0) * 4.0
        upper += max(te_ev, ef_ev, 1.0) * 4.0
        f_low = residual(lower)
        f_high = residual(upper)
        expand += 1
    if not math.isfinite(f_low) or not math.isfinite(f_high) or f_low > 0.0 or f_high < 0.0:
        return float(classical_mu if math.isfinite(classical_mu) else ef_ev)
    return _bracketed_root_solve(residual, lower, upper, maxiter=200, xtol=1.0e-8, rtol=1.0e-10)


def chemical_potential_ev(ne_cm3: float, te_ev: float) -> float:
    return _chemical_potential_ev_cached(round(float(ne_cm3), 3), round(float(te_ev), 6))


def lindhard_polarization(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    k_order: int = 72,
    mu_order: int = 48,
    batch_size: int = 256,
) -> np.ndarray:
    """Return the finite-T Lindhard polarization Π(q,ω) in SI units.

    The return value has units of m^-3 J^-1. The dielectric correction is then
    obtained through ``chi = -v(q) Π`` with ``v(q)=e^2/(eps0 q^2)``.
    """

    energy_ev = np.asarray(delta_energy_ev, dtype=np.float64)
    q = float(k_m_inv)
    if energy_ev.size == 0:
        return np.asarray([], dtype=np.complex128)
    if not math.isfinite(q) or q <= 0.0 or not math.isfinite(te_ev) or te_ev <= 0.0 or not math.isfinite(ne_cm3) or ne_cm3 <= 0.0:
        return np.full(energy_ev.shape, np.nan + 0.0j, dtype=np.complex128)
    mu_ev = chemical_potential_ev(ne_cm3, te_ev)
    if not math.isfinite(mu_ev):
        return np.full(energy_ev.shape, np.nan + 0.0j, dtype=np.complex128)
    _, _, k_max = _k_scales(ne_cm3, te_ev, q)
    knodes, kweights = _leggauss_cached(int(k_order))
    munodes, muweights = _leggauss_cached(int(mu_order))
    k = 0.5 * (knodes + 1.0) * k_max
    wk = 0.5 * k_max * kweights
    mu_cos = munodes
    wmu = muweights

    k_grid = k[:, None]
    mu_grid = mu_cos[None, :]
    kq = np.sqrt(np.clip(k_grid * k_grid + q * q + 2.0 * k_grid * q * mu_grid, a_min=0.0, a_max=None))
    eps_k = _energy_from_k_j(k_grid)
    eps_kq = _energy_from_k_j(kq)
    f_k = _fermi_occupation(eps_k, mu_ev * ELECTRON_CHARGE_C, te_ev)
    f_kq = _fermi_occupation(eps_kq, mu_ev * ELECTRON_CHARGE_C, te_ev)
    delta_eps_j = eps_k - eps_kq
    phase_weight = (k_grid * k_grid) * wk[:, None] * wmu[None, :] / (2.0 * _PI2)
    numerator = phase_weight * (f_k - f_kq)
    flat_num = numerator.reshape(-1)
    flat_delta = delta_eps_j.reshape(-1)
    omega_j = energy_ev * ELECTRON_CHARGE_C
    imag_j = max(float(imag_shift_ev), _MIN_IMAG_SHIFT_EV) * ELECTRON_CHARGE_C
    result = np.empty(energy_ev.shape, dtype=np.complex128)
    for start in range(0, energy_ev.size, max(int(batch_size), 1)):
        stop = min(energy_ev.size, start + max(int(batch_size), 1))
        denom = flat_delta[:, None] + omega_j[None, start:stop] + 1j * imag_j
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            block = np.sum(flat_num[:, None] / denom, axis=0, dtype=np.complex128)
        result[start:stop] = block
    bad = ~np.isfinite(np.real(result)) | ~np.isfinite(np.imag(result))
    if np.any(bad):
        result = np.asarray(result, dtype=np.complex128)
        result[bad] = np.nan + 0.0j
    return result


def _finite_t_lindhard_susceptibility_uncached(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    benchmark: bool = False,
) -> np.ndarray:
    """Return the dielectric correction ``chi = epsilon - 1``.

    Benchmark mode uses a somewhat denser quadrature than quicklook mode.
    """

    q = float(k_m_inv)
    if not math.isfinite(q) or q <= 0.0:
        return np.full(np.asarray(delta_energy_ev, dtype=np.float64).shape, np.nan + 0.0j, dtype=np.complex128)
    pi_qw = lindhard_polarization(
        delta_energy_ev,
        k_m_inv=q,
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=float(imag_shift_ev),
        k_order=(96 if benchmark else 64),
        mu_order=(64 if benchmark else 40),
        batch_size=(384 if benchmark else 256),
    )
    v_q = (ELECTRON_CHARGE_C * ELECTRON_CHARGE_C) / (EPSILON_0_F_M * q * q)
    return -v_q * pi_qw



def _energy_axis_cache_key(delta_energy_ev: np.ndarray) -> tuple[object, ...]:
    energy = np.asarray(delta_energy_ev, dtype=np.float64).ravel()
    if energy.size == 0:
        return ("empty",)
    if energy.size == 1:
        return ("single", round(float(energy[0]), 9))
    diffs = np.diff(energy)
    step = float(diffs[0])
    if np.allclose(diffs, step, rtol=0.0, atol=max(1.0e-12, abs(step) * 1.0e-12)):
        return ("lin", int(energy.size), round(float(energy[0]), 9), round(float(energy[-1]), 9))
    rounded = tuple(round(float(value), 9) for value in energy.tolist())
    return ("grid", rounded)


def _energy_axis_from_cache_key(key: tuple[object, ...]) -> np.ndarray:
    tag = str(key[0])
    if tag == "empty":
        return np.asarray([], dtype=np.float64)
    if tag == "single":
        return np.asarray([float(key[1])], dtype=np.float64)
    if tag == "lin":
        _, size, first, last = key
        return np.linspace(float(first), float(last), int(size), dtype=np.float64)
    if tag == "grid":
        return np.asarray(key[1], dtype=np.float64)
    raise ValueError(f"Unsupported Lindhard energy-axis cache key: {key!r}")


@lru_cache(maxsize=256)
def _finite_t_lindhard_susceptibility_cached(
    energy_key: tuple[object, ...],
    k_m_inv_rounded: float,
    te_ev_rounded: float,
    ne_cm3_rounded: float,
    imag_shift_ev_rounded: float,
    benchmark: bool,
) -> np.ndarray:
    energy = _energy_axis_from_cache_key(energy_key)
    return _finite_t_lindhard_susceptibility_uncached(
        energy,
        k_m_inv=float(k_m_inv_rounded),
        te_ev=float(te_ev_rounded),
        ne_cm3=float(ne_cm3_rounded),
        imag_shift_ev=float(imag_shift_ev_rounded),
        benchmark=bool(benchmark),
    )


def finite_t_lindhard_susceptibility(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    benchmark: bool = False,
) -> np.ndarray:
    """Cached finite-T Lindhard susceptibility.

    Repeated LOS-cluster sweeps hit the same `(q, Te, ne, imag_shift, energy-grid)` tuples
    across Lindhard, Lindhard+LFC, and Mermin-style closures. Caching this level keeps the
    full hydro-driven Lindhard family practical for validation runs without changing the
    underlying numerical quadrature.
    """

    return np.asarray(
        _finite_t_lindhard_susceptibility_cached(
            _energy_axis_cache_key(delta_energy_ev),
            round(float(k_m_inv), 6),
            round(float(te_ev), 6),
            round(float(ne_cm3), 3),
            round(float(imag_shift_ev), 9),
            bool(benchmark),
        ),
        dtype=np.complex128,
    ).copy()


def finite_t_lindhard_cache_info() -> dict[str, int]:
    info = _finite_t_lindhard_susceptibility_cached.cache_info()
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": int(info.maxsize or 0),
        "currsize": int(info.currsize),
    }


def clear_finite_t_lindhard_cache() -> None:
    _finite_t_lindhard_susceptibility_cached.cache_clear()


def finite_t_lindhard_backend_name() -> str:
    return "numerical_finite_t_lindhard"
