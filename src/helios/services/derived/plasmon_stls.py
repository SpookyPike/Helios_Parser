"""Finite-temperature self-consistent static STLS backend.

This module implements the minimal honest Phase-1 STLS core requested for the
article-facing plasmon benchmark:

- no collisions;
- no dynamic local-field factor ``G(q, omega)``;
- no qSTLS / VS compressibility enforcement;
- an explicit self-consistent loop
  ``G(q) -> chi(q, omega) -> S(q) -> G(q)``.

The implementation is intentionally separate from the existing
``static_lfc`` surrogate. Here ``G(q)`` is not prescribed. It is iterated
numerically from an STLS closure using the finite-temperature Lindhard
polarization as the ideal-gas baseline.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from helios.services.derived.plasmon_lindhard import (
    finite_t_lindhard_backend_name,
    lindhard_polarization,
)
from helios.services.derived.plasmon_units import (
    ELECTRON_CHARGE_C,
    EPSILON_0_F_M,
    electron_density_m3_from_cm3,
    electron_fermi_energy_ev,
    electron_fermi_wavevector_m_inv,
    electron_plasma_energy_ev,
)

_DEF_NUMERICAL_IMAG_SHIFT_EV = 1.0e-9
_STLS_CLOSURE_NAME = "static_stls_isotropic_angle_integral"
_STLS_KERNEL_SUMMARY = (
    "Finite-T STLS with self-consistent static G(q): "
    "Pi0 from numerical finite-T Lindhard, S(q) from FDT on the interacting "
    "density response, and G(q) updated from the 3D isotropic STLS closure."
)


def _trapezoid_compat(y: np.ndarray, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


@lru_cache(maxsize=8)
def _leggauss_cached(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def stls_backend_name() -> str:
    return "finite_t_stls"


def _q_grid_m_inv(ne_cm3: float, *, benchmark: bool) -> np.ndarray:
    kf = float(electron_fermi_wavevector_m_inv(float(ne_cm3)))
    if not math.isfinite(kf) or kf <= 0.0:
        return np.asarray([], dtype=np.float64)
    q_max = max(6.0 * kf, 6.0e10)
    q_min = max(0.05 * kf, 5.0e8)
    count = 30 if benchmark else 22
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)
    grid = q_min + (q_max - q_min) * (u**1.35)
    return np.asarray(grid, dtype=np.float64)


def _positive_energy_grid_ev(ne_cm3: float, te_ev: float, *, benchmark: bool) -> np.ndarray:
    ef_ev = float(electron_fermi_energy_ev(float(ne_cm3)))
    plasma_ev = float(electron_plasma_energy_ev(float(ne_cm3)))
    e_max = max(80.0, 5.0 * plasma_ev if math.isfinite(plasma_ev) else 0.0, 6.0 * ef_ev if math.isfinite(ef_ev) else 0.0, 20.0 * float(te_ev))
    count = 321 if benchmark else 241
    return np.linspace(1.0e-3, e_max, count, dtype=np.float64)


def _coth_half_beta_energy(energy_ev: np.ndarray, te_ev: float) -> np.ndarray:
    energy = np.asarray(energy_ev, dtype=np.float64)
    thermal = max(float(te_ev), 1.0e-8)
    arg = np.clip(energy / (2.0 * thermal), -350.0, 350.0)
    small = np.abs(arg) < 1.0e-6
    result = np.empty_like(arg, dtype=np.float64)
    if np.any(small):
        local = arg[small]
        # Stable series for coth(x) near x = 0.
        result[small] = 1.0 / np.where(local != 0.0, local, 1.0e-12) + local / 3.0
    if np.any(~small):
        result[~small] = 1.0 / np.tanh(arg[~small])
    return result


def _density_response_from_local_field(
    pi0_qw: np.ndarray,
    *,
    q_m_inv: float,
    local_field: float,
) -> tuple[np.ndarray, np.ndarray]:
    v_q = (ELECTRON_CHARGE_C * ELECTRON_CHARGE_C) / (EPSILON_0_F_M * float(q_m_inv) * float(q_m_inv))
    epsilon = 1.0 - v_q * (1.0 - float(local_field)) * np.asarray(pi0_qw, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        density_response = np.asarray(pi0_qw, dtype=np.complex128) / epsilon
    bad = (
        ~np.isfinite(np.real(density_response))
        | ~np.isfinite(np.imag(density_response))
        | ~np.isfinite(np.real(epsilon))
        | ~np.isfinite(np.imag(epsilon))
    )
    if np.any(bad):
        density_response = np.asarray(density_response, dtype=np.complex128)
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        density_response[bad] = np.nan + 0.0j
        epsilon[bad] = np.nan + 0.0j
    return density_response, epsilon


def _static_structure_factor_from_response(
    q_grid_m_inv: np.ndarray,
    energy_ev: np.ndarray,
    pi0_grid_qw: np.ndarray,
    local_field_grid: np.ndarray,
    *,
    ne_m3: float,
    te_ev: float,
) -> np.ndarray:
    q_grid = np.asarray(q_grid_m_inv, dtype=np.float64)
    pi0_grid = np.asarray(pi0_grid_qw, dtype=np.complex128)
    g_grid = np.asarray(local_field_grid, dtype=np.float64)
    thermal_kernel = _coth_half_beta_energy(energy_ev, te_ev)
    s_grid = np.full(q_grid.shape, np.nan, dtype=np.float64)
    for index, q_value in enumerate(q_grid.tolist()):
        density_response, _epsilon = _density_response_from_local_field(
            pi0_grid[index],
            q_m_inv=float(q_value),
            local_field=float(g_grid[index]),
        )
        integrand = thermal_kernel * np.imag(density_response)
        value = -ELECTRON_CHARGE_C / (math.pi * float(ne_m3)) * float(_trapezoid_compat(np.asarray(integrand, dtype=np.float64), energy_ev))
        s_grid[index] = value
    if np.any(~np.isfinite(s_grid)):
        return np.full(q_grid.shape, np.nan, dtype=np.float64)
    strongly_negative = np.nanmin(s_grid) < -5.0e-2
    if strongly_negative:
        return np.full(q_grid.shape, np.nan, dtype=np.float64)
    return np.clip(s_grid, 0.0, None)


def _interpolate_structure_factor(q_values: np.ndarray, s_grid: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(q_values, dtype=np.float64)
    s = np.asarray(s_grid, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    flat = np.interp(target.reshape(-1), q, s, left=float(s[0]), right=1.0)
    return np.asarray(flat.reshape(target.shape), dtype=np.float64)


def _update_local_field_from_structure_factor(
    q_grid_m_inv: np.ndarray,
    s_grid: np.ndarray,
    *,
    ne_m3: float,
    angle_order: int,
) -> np.ndarray:
    q_grid = np.asarray(q_grid_m_inv, dtype=np.float64)
    s = np.asarray(s_grid, dtype=np.float64)
    mu_nodes, mu_weights = _leggauss_cached(int(angle_order))
    updated = np.zeros_like(q_grid, dtype=np.float64)
    prefactor = -1.0 / (4.0 * math.pi * math.pi * float(ne_m3))
    for iq, q_value in enumerate(q_grid.tolist()):
        k_values = q_grid
        p = np.sqrt(
            np.clip(
                q_value * q_value + k_values[:, None] * k_values[:, None] - 2.0 * q_value * k_values[:, None] * mu_nodes[None, :],
                a_min=0.0,
                a_max=None,
            )
        )
        s_interp = _interpolate_structure_factor(q_grid, s, p)
        angular = np.sum(mu_weights[None, :] * mu_nodes[None, :] * (s_interp - 1.0), axis=1, dtype=np.float64)
        # Coulomb STLS closure:
        #   G(q) = -(1/n) ∫ d^3k/(2π)^3 (q·k / k^2) [S(|q-k|) - 1]
        # The `v(k)/v(q) = q^2/k^2` factor is what structurally separates the
        # closure from the earlier static-LFC surrogate path.
        integrand = float(q_value) * k_values * angular
        updated[iq] = prefactor * float(_trapezoid_compat(integrand, k_values))
    return np.asarray(updated, dtype=np.float64)


def _stls_iteration_parameters(benchmark: bool) -> dict[str, float | int]:
    return {
        "max_iter": (56 if benchmark else 40),
        "tol": (1.0e-3 if benchmark else 2.0e-3),
        "relative_tol": 5.0e-4,
        "mixing": 0.20,
        "angle_order": (18 if benchmark else 14),
        "k_order": (56 if benchmark else 40),
        "mu_order": (36 if benchmark else 28),
        "batch_size": (384 if benchmark else 256),
    }


@lru_cache(maxsize=128)
def _solve_static_stls_cached(
    ne_cm3_rounded: float,
    te_ev_rounded: float,
    imag_shift_ev_rounded: float,
    benchmark: bool,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], bool, int, float, float, float, float, int, int]:
    ne_cm3 = float(ne_cm3_rounded)
    te_ev = float(te_ev_rounded)
    imag_shift_ev = max(float(imag_shift_ev_rounded), _DEF_NUMERICAL_IMAG_SHIFT_EV)
    if not math.isfinite(ne_cm3) or ne_cm3 <= 0.0 or not math.isfinite(te_ev) or te_ev <= 0.0:
        return (), (), (), False, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0

    q_grid = _q_grid_m_inv(ne_cm3, benchmark=bool(benchmark))
    energy_grid = _positive_energy_grid_ev(ne_cm3, te_ev, benchmark=bool(benchmark))
    params = _stls_iteration_parameters(bool(benchmark))
    if q_grid.size == 0 or energy_grid.size == 0:
        return (), (), (), False, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0

    pi0_rows: list[np.ndarray] = []
    for q_value in q_grid.tolist():
        pi0_rows.append(
            lindhard_polarization(
                energy_grid,
                k_m_inv=float(q_value),
                te_ev=float(te_ev),
                ne_cm3=float(ne_cm3),
                imag_shift_ev=float(imag_shift_ev),
                k_order=int(params["k_order"]),
                mu_order=int(params["mu_order"]),
                batch_size=int(params["batch_size"]),
            )
        )
    pi0_grid = np.asarray(pi0_rows, dtype=np.complex128)
    if pi0_grid.shape != (q_grid.size, energy_grid.size) or np.any(~np.isfinite(np.real(pi0_grid)) | ~np.isfinite(np.imag(pi0_grid))):
        return tuple(float(value) for value in q_grid.tolist()), (), (), False, 0, float("nan"), float("nan"), float("nan"), float("nan"), int(q_grid.size), int(energy_grid.size)

    ne_m3 = float(electron_density_m3_from_cm3(ne_cm3))
    g_grid = np.zeros(q_grid.shape, dtype=np.float64)
    s_grid = np.ones(q_grid.shape, dtype=np.float64)
    mixing = float(params["mixing"])
    converged = False
    residual = float("inf")
    relative_residual = float("inf")
    for iteration in range(1, int(params["max_iter"]) + 1):
        s_trial = _static_structure_factor_from_response(
            q_grid,
            energy_grid,
            pi0_grid,
            g_grid,
            ne_m3=ne_m3,
            te_ev=te_ev,
        )
        if np.any(~np.isfinite(s_trial)):
            return (
                tuple(float(value) for value in q_grid.tolist()),
                tuple(float(value) for value in g_grid.tolist()),
                tuple(),
                False,
                int(iteration),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                int(q_grid.size),
                int(energy_grid.size),
            )
        g_trial = _update_local_field_from_structure_factor(
            q_grid,
            s_trial,
            ne_m3=ne_m3,
            angle_order=int(params["angle_order"]),
        )
        if np.any(~np.isfinite(g_trial)):
            return (
                tuple(float(value) for value in q_grid.tolist()),
                tuple(float(value) for value in g_grid.tolist()),
                tuple(float(value) for value in s_trial.tolist()),
                False,
                int(iteration),
                float("nan"),
                float(np.nanmin(s_trial)),
                float(np.nanmax(s_trial)),
                float("nan"),
                int(q_grid.size),
                int(energy_grid.size),
            )
        g_next = (1.0 - mixing) * g_grid + mixing * g_trial
        residual = float(np.nanmax(np.abs(g_next - g_grid))) if g_next.size else float("nan")
        scale = float(max(1.0, float(np.nanmax(np.abs(g_next))))) if g_next.size else 1.0
        relative_residual = (float(residual / scale) if math.isfinite(residual) and math.isfinite(scale) and scale > 0.0 else float("nan"))
        g_grid = np.asarray(g_next, dtype=np.float64)
        s_grid = np.asarray(s_trial, dtype=np.float64)
        if (
            (math.isfinite(residual) and residual <= float(params["tol"]))
            or (math.isfinite(relative_residual) and relative_residual <= float(params["relative_tol"]))
        ):
            converged = True
            return (
                tuple(float(value) for value in q_grid.tolist()),
                tuple(float(value) for value in g_grid.tolist()),
                tuple(float(value) for value in s_grid.tolist()),
                True,
                int(iteration),
                float(residual),
                float(relative_residual),
                float(np.nanmin(s_grid)),
                float(np.nanmax(s_grid)),
                int(q_grid.size),
                int(energy_grid.size),
            )
    return (
        tuple(float(value) for value in q_grid.tolist()),
        tuple(float(value) for value in g_grid.tolist()),
        tuple(float(value) for value in s_grid.tolist()),
        False,
        int(params["max_iter"]),
        float(residual),
        float(relative_residual),
        float(np.nanmin(s_grid)) if s_grid.size else float("nan"),
        float(np.nanmax(s_grid)) if s_grid.size else float("nan"),
        int(q_grid.size),
        int(energy_grid.size),
    )


def solve_static_stls_state(
    *,
    ne_cm3: float,
    te_ev: float,
    imag_shift_ev: float,
    benchmark: bool,
) -> dict[str, object]:
    q_grid, g_grid, s_grid, converged, iterations, residual, relative_residual, s_min, s_max, q_count, energy_count = _solve_static_stls_cached(
        round(float(ne_cm3), 3),
        round(float(te_ev), 6),
        round(float(imag_shift_ev), 9),
        bool(benchmark),
    )
    return {
        "q_grid_m_inv": np.asarray(q_grid, dtype=np.float64),
        "local_field_grid": np.asarray(g_grid, dtype=np.float64),
        "structure_factor_grid": np.asarray(s_grid, dtype=np.float64),
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual": float(residual),
        "relative_residual": float(relative_residual),
        "closure_name": _STLS_CLOSURE_NAME,
        "structure_factor_min": float(s_min),
        "structure_factor_max": float(s_max),
        "q_grid_count": int(q_count),
        "energy_grid_count": int(energy_count),
        "kernel_backend": finite_t_lindhard_backend_name(),
    }


def stls_cache_info() -> dict[str, int]:
    info = _solve_static_stls_cached.cache_info()
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": int(info.maxsize or 0),
        "currsize": int(info.currsize),
    }


def clear_stls_cache() -> None:
    _solve_static_stls_cached.cache_clear()


def epsilon_finite_t_stls(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    benchmark: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool | int]]:
    """Return ``(chi, epsilon, metadata)`` for the static finite-T STLS backend."""

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    nan_array = np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128)
    metadata: dict[str, float | str | bool | int] = {
        "backend_name": stls_backend_name(),
        "backend_summary": "",
        "closure_name": _STLS_CLOSURE_NAME,
        "converged": False,
        "iterations": 0,
        "residual": float("nan"),
        "relative_residual": float("nan"),
        "local_field_value": float("nan"),
        "q_over_qf": float("nan"),
        "structure_factor_min": float("nan"),
        "structure_factor_max": float("nan"),
        "q_grid_count": 0,
        "energy_grid_count": 0,
    }
    q_value = float(k_m_inv)
    if (
        energy.size == 0
        or not math.isfinite(q_value)
        or q_value <= 0.0
        or not math.isfinite(float(te_ev))
        or float(te_ev) <= 0.0
        or not math.isfinite(float(ne_cm3))
        or float(ne_cm3) <= 0.0
    ):
        metadata["backend_summary"] = "Finite-T STLS rejected a non-finite or non-positive state."
        return nan_array.copy(), nan_array.copy(), metadata

    solution = solve_static_stls_state(
        ne_cm3=float(ne_cm3),
        te_ev=float(te_ev),
        imag_shift_ev=max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV),
        benchmark=bool(benchmark),
    )
    q_grid = np.asarray(solution["q_grid_m_inv"], dtype=np.float64)
    g_grid = np.asarray(solution["local_field_grid"], dtype=np.float64)
    s_grid = np.asarray(solution["structure_factor_grid"], dtype=np.float64)
    metadata.update(
        {
            "converged": bool(solution["converged"]),
            "iterations": int(solution["iterations"]),
            "residual": float(solution["residual"]),
            "relative_residual": float(solution["relative_residual"]),
            "structure_factor_min": float(solution["structure_factor_min"]),
            "structure_factor_max": float(solution["structure_factor_max"]),
            "q_grid_count": int(solution["q_grid_count"]),
            "energy_grid_count": int(solution["energy_grid_count"]),
        }
    )
    if not bool(solution["converged"]) or q_grid.size == 0 or g_grid.size != q_grid.size or s_grid.size != q_grid.size:
        metadata["backend_summary"] = (
                f"{_STLS_KERNEL_SUMMARY} No spectral result was produced because the self-consistent loop "
            f"did not converge (iterations={int(solution['iterations'])}, residual={float(solution['residual']):.3e}, "
            f"relative={float(solution['relative_residual']):.3e})."
        )
        return nan_array.copy(), nan_array.copy(), metadata
    if q_value > float(q_grid[-1]):
        metadata["backend_summary"] = (
            f"{_STLS_KERNEL_SUMMARY} Requested q={q_value:.4e} 1/m lies beyond the converged STLS support grid "
            f"(q_max={float(q_grid[-1]):.4e} 1/m)."
        )
        return nan_array.copy(), nan_array.copy(), metadata

    kf = float(electron_fermi_wavevector_m_inv(float(ne_cm3)))
    g_value = float(np.interp(q_value, q_grid, g_grid, left=float(g_grid[0]), right=float(g_grid[-1])))
    s_value = float(np.interp(q_value, q_grid, s_grid, left=float(s_grid[0]), right=float(s_grid[-1])))
    pi0 = lindhard_polarization(
        energy,
        k_m_inv=float(q_value),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV),
        k_order=(96 if benchmark else 64),
        mu_order=(64 if benchmark else 40),
        batch_size=(384 if benchmark else 256),
    )
    density_response, epsilon = _density_response_from_local_field(
        pi0,
        q_m_inv=float(q_value),
        local_field=float(g_value),
    )
    chi = np.asarray(density_response, dtype=np.complex128)
    metadata.update(
        {
            "local_field_value": float(g_value),
            "structure_factor_value": float(s_value),
            "q_over_qf": (float(q_value / kf) if math.isfinite(kf) and kf > 0.0 else float("nan")),
            "backend_summary": (
                f"{_STLS_KERNEL_SUMMARY} Closure={_STLS_CLOSURE_NAME}; "
                f"iterations={int(solution['iterations'])}, residual={float(solution['residual']):.3e}, "
                f"relative={float(solution['relative_residual']):.3e}, "
                f"G(q)={g_value:.4f}, S(q)={s_value:.4f}, q/kF="
                f"{(float(q_value / kf) if math.isfinite(kf) and kf > 0.0 else float('nan')):.4f}, "
                f"ideal-kernel={finite_t_lindhard_backend_name()}."
            ),
        }
    )
    return chi, np.asarray(epsilon, dtype=np.complex128), metadata
