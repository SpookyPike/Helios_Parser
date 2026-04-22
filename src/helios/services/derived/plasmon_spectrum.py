"""Spectral helpers for the plasmon/XRTS module.

Phase 2 introduced a first one-state spectral baseline using a classical
finite-temperature susceptibility for a Maxwellian electron gas. Phase 3 keeps
that baseline but adds a constant-ν Mermin collision branch on top of the same
susceptibility so the GUI/result plumbing can evolve toward a fuller
finite-temperature Lindhard/Mermin stack without another architectural pass.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

try:
    from scipy.special import wofz as _scipy_wofz
except ModuleNotFoundError:  # pragma: no cover - optional numerical fallback
    _scipy_wofz = None

from helios.services.constants.nrl import HBAR_EV_S
from helios.services.derived.plasmon_lfc import esa_static_local_field_correction
from helios.services.derived.plasmon_lindhard import finite_t_lindhard_backend_name, finite_t_lindhard_susceptibility
from helios.services.derived.plasmon_units import electron_debye_length_m, electron_k_over_kf, electron_thermal_speed_m_s


_DEF_NUMERICAL_IMAG_SHIFT_EV = 1.0e-9


def _trapezoid_compat(y: np.ndarray, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for NumPy versions without ``np.trapezoid``."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _cache_float_token(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return float(f"{value:.12g}")


def _energy_axis_cache_key(delta_energy_ev: np.ndarray) -> tuple[object, ...]:
    energy = np.asarray(delta_energy_ev, dtype=np.float64).reshape(-1)
    if energy.size == 0:
        return ("empty", 0)
    if energy.size == 1:
        scalar = _cache_float_token(float(energy[0]))
        return ("single", scalar)
    step = float(energy[1] - energy[0])
    if np.allclose(
        energy,
        float(energy[0]) + step * np.arange(energy.size, dtype=np.float64),
        rtol=0.0,
        atol=max(1.0e-12, abs(step) * 1.0e-12),
    ):
        return ("linspace", int(energy.size), _cache_float_token(float(energy[0])), _cache_float_token(float(energy[-1])))
    if energy.size <= 32:
        return ("tuple", tuple(_cache_float_token(float(value)) for value in energy.tolist()))
    return ("linspace", int(energy.size), _cache_float_token(float(energy[0])), _cache_float_token(float(energy[-1])))


def _energy_axis_from_cache_key(axis_key: tuple[object, ...]) -> np.ndarray:
    if not axis_key:
        return np.zeros(0, dtype=np.float64)
    kind = str(axis_key[0])
    if kind == "empty":
        return np.zeros(0, dtype=np.float64)
    if kind == "single":
        return np.asarray([float(axis_key[1])], dtype=np.float64)
    if kind == "linspace":
        return np.linspace(float(axis_key[2]), float(axis_key[3]), int(axis_key[1]), dtype=np.float64)
    if kind == "tuple":
        return np.asarray(tuple(float(value) for value in axis_key[1]), dtype=np.float64)
    raise ValueError(f"Unsupported plasmon energy-axis cache key: {axis_key!r}")


def energy_axis_ev(window_ev: float, points: int) -> np.ndarray:
    """Return a symmetric energy-transfer axis in eV."""

    half_window = max(float(window_ev), 1.0)
    count = max(int(points), 101)
    if count % 2 == 0:
        count += 1
    return np.linspace(-half_window, half_window, count, dtype=np.float64)


@lru_cache(maxsize=8)
def _dispersion_quadrature_cache(samples: int = 1537, span: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    count = int(samples)
    x = np.linspace(-float(span), float(span), count, dtype=np.float64)
    if count <= 1:
        weights = np.ones_like(x, dtype=np.float64)
    else:
        dx = float(x[1] - x[0])
        weights = np.full_like(x, dx, dtype=np.float64)
        weights[0] = 0.5 * dx
        weights[-1] = 0.5 * dx
    weighted_kernel = np.exp(-(x**2), dtype=np.float64) * weights / math.sqrt(math.pi)
    return x, weighted_kernel


def _plasma_dispersion_function_quadrature(
    zeta: np.ndarray,
    *,
    samples: int = 1537,
    span: float = 8.0,
) -> np.ndarray:
    zeta_array = np.asarray(zeta, dtype=np.complex128)
    x, weighted_kernel = _dispersion_quadrature_cache(samples=samples, span=span)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        integrand = weighted_kernel[:, None] / (x[:, None] - zeta_array[None, :])
    return np.sum(integrand, axis=0, dtype=np.complex128)


def plasma_dispersion_backend_name() -> str:
    return "scipy_wofz" if _scipy_wofz is not None else "quadrature_fallback"


def plasma_dispersion_function(
    zeta: np.ndarray,
    *,
    backend: str = "auto",
    samples: int = 1537,
    span: float = 8.0,
) -> np.ndarray:
    """Return the Fried-Conte plasma dispersion function ``Z(zeta)``.

    Primary path: the exact Faddeeva backend via ``scipy.special.wofz`` using
    the standard identity ``Z(ζ) = i*sqrt(pi)*w(ζ)``. The older fixed-grid real
    axis quadrature is retained only as an optional dependency-free fallback.
    """

    zeta_array = np.asarray(zeta, dtype=np.complex128)
    selected = str(backend or "auto").lower()
    if selected in {"auto", "faddeeva", "scipy", "scipy_wofz", "wofz"} and _scipy_wofz is not None:
        return 1j * math.sqrt(math.pi) * _scipy_wofz(zeta_array)
    if selected in {"quadrature", "fallback", "quadrature_fallback"} or _scipy_wofz is None:
        return _plasma_dispersion_function_quadrature(zeta_array, samples=samples, span=span)
    raise ValueError(f"Unknown plasma-dispersion backend: {backend!r}")


@lru_cache(maxsize=512)
def _finite_t_susceptibility_cached(
    axis_key: tuple[object, ...],
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
) -> np.ndarray:
    energy = _energy_axis_from_cache_key(axis_key)
    if not math.isfinite(float(k_m_inv)) or float(k_m_inv) <= 0.0 or te_ev <= 0.0 or ne_cm3 <= 0.0:
        result = np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128)
        result.setflags(write=False)
        return result
    lambda_d_m = electron_debye_length_m(float(te_ev), float(ne_cm3))
    v_th_m_s = electron_thermal_speed_m_s(float(te_ev))
    if not math.isfinite(lambda_d_m) or lambda_d_m <= 0.0 or not math.isfinite(v_th_m_s) or v_th_m_s <= 0.0:
        result = np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128)
        result.setflags(write=False)
        return result
    omega = (energy + 1j * max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)) / HBAR_EV_S
    zeta = omega / (math.sqrt(2.0) * float(k_m_inv) * v_th_m_s)
    z_value = plasma_dispersion_function(np.asarray(zeta, dtype=np.complex128))
    prefactor = 1.0 / ((float(k_m_inv) * lambda_d_m) ** 2)
    result = np.asarray(prefactor * (1.0 + zeta * z_value), dtype=np.complex128)
    result.setflags(write=False)
    return result


def classical_response_cache_info() -> dict[str, object]:
    info = _finite_t_susceptibility_cached.cache_info()
    return {
        "finite_t_susceptibility_hits": int(info.hits),
        "finite_t_susceptibility_misses": int(info.misses),
        "finite_t_susceptibility_currsize": int(info.currsize),
        "finite_t_susceptibility_maxsize": int(info.maxsize or 0),
    }


def clear_classical_response_cache() -> None:
    _finite_t_susceptibility_cached.cache_clear()


def _apply_static_lfc_response(chi_rpa: np.ndarray, g_value: float) -> tuple[np.ndarray, np.ndarray]:
    chi_array = np.asarray(chi_rpa, dtype=np.complex128)
    g_scalar = float(g_value)
    if not math.isfinite(g_scalar):
        empty = np.full(chi_array.shape, np.nan + 0.0j, dtype=np.complex128)
        return empty, empty
    if abs(g_scalar) <= 1.0e-15:
        epsilon = 1.0 + chi_array
        return chi_array, np.asarray(epsilon, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = 1.0 + g_scalar * chi_array
        chi_eff = chi_array / denominator
        epsilon = 1.0 + chi_eff
    bad = (~np.isfinite(np.real(chi_eff))) | (~np.isfinite(np.imag(chi_eff))) | (~np.isfinite(np.real(epsilon))) | (~np.isfinite(np.imag(epsilon)))
    if np.any(bad):
        chi_eff = np.asarray(chi_eff, dtype=np.complex128)
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        chi_eff[bad] = np.nan + 0.0j
        epsilon[bad] = np.nan + 0.0j
    return chi_eff, epsilon



def finite_t_susceptibility(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
) -> np.ndarray:
    """Return the classical finite-temperature electron susceptibility.

    This is the Maxwellian/Vlasov finite-temperature susceptibility

    ``chi = (k^2 lambda_D^2)^-1 * (1 + zeta Z(zeta))``

    with ``zeta = (omega + i eta) / (sqrt(2) k v_th)``.

    It is intentionally retained as the baseline kernel until a later phase
    replaces it with a finite-temperature Lindhard susceptibility. The current
    Mermin branch therefore inherits a classical warm-plasma baseline and emits
    warnings at the service layer when the selected state is noticeably
    degenerate.
    """

    axis_key = _energy_axis_cache_key(np.asarray(delta_energy_ev, dtype=np.float64))
    cached = _finite_t_susceptibility_cached(
        axis_key,
        k_m_inv=_cache_float_token(float(k_m_inv)),
        te_ev=_cache_float_token(float(te_ev)),
        ne_cm3=_cache_float_token(float(ne_cm3)),
        imag_shift_ev=_cache_float_token(float(imag_shift_ev)),
    )
    return np.array(cached, dtype=np.complex128, copy=True)



def epsilon_rpa(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chi, epsilon)`` for the classical warm RPA baseline."""

    chi = finite_t_susceptibility(
        delta_energy_ev,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=float(imag_shift_ev),
    )
    epsilon = 1.0 + chi
    return chi, epsilon





def epsilon_rpa_static_lfc(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    rs: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(chi_eff, epsilon, G(k), q/k_F)`` for the static-LFC RPA branch."""

    chi_rpa = finite_t_susceptibility(
        delta_energy_ev,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=float(imag_shift_ev),
    )
    q_over_qf = electron_k_over_kf(float(k_m_inv), float(ne_cm3))
    g_value = float(esa_static_local_field_correction(q_over_qf, float(rs), float(theta)))
    chi_eff, epsilon = _apply_static_lfc_response(chi_rpa, g_value)
    return chi_eff, epsilon, g_value, float(q_over_qf)

def epsilon_mermin(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    collision_rate_s: float,
    imag_shift_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chi, epsilon)`` for a constant-ν Mermin collision branch.

    The branch uses the classical warm-plasma dielectric function as the
    collisionless baseline and applies the number-conserving Mermin closure
    with a constant real collision rate ``ν``. In the limit ``ν -> 0`` it
    reduces to the current warm RPA baseline.
    """

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    nu_s = float(collision_rate_s)
    if not math.isfinite(nu_s) or nu_s < 0.0:
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
        )
    numerical_imag_shift_ev = max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)
    if nu_s <= 0.0:
        return epsilon_rpa(
            energy,
            k_m_inv=float(k_m_inv),
            te_ev=float(te_ev),
            ne_cm3=float(ne_cm3),
            imag_shift_ev=numerical_imag_shift_ev,
        )

    nu_ev = HBAR_EV_S * nu_s
    chi_shift, eps_shift = epsilon_rpa(
        energy,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev + nu_ev,
    )
    del chi_shift
    _, eps_static_arr = epsilon_rpa(
        np.asarray([0.0], dtype=np.float64),
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev,
    )
    eps_static = complex(eps_static_arr[0])
    if not math.isfinite(float(np.real(eps_static))) or not math.isfinite(float(np.imag(eps_static))):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
        )

    omega = energy / HBAR_EV_S
    ratio = np.zeros(energy.shape, dtype=np.complex128)
    nonzero = np.abs(omega) > 1.0e-30
    ratio[nonzero] = 1j * nu_s / omega[nonzero]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = (1.0 + ratio) * (eps_shift - 1.0)
        denominator = 1.0 + ratio * (eps_shift - 1.0) / (eps_static - 1.0)
        epsilon = 1.0 + numerator / denominator

    zero_mask = ~nonzero
    if np.any(zero_mask):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[zero_mask] = eps_static

    bad = ~np.isfinite(np.real(epsilon)) | ~np.isfinite(np.imag(epsilon))
    if np.any(bad):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[bad] = np.nan + 0.0j
    chi = epsilon - 1.0
    return chi, epsilon


def epsilon_mermin_static_lfc(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    collision_rate_s: float,
    imag_shift_ev: float,
    rs: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(chi, epsilon, G(k), q/k_F)`` for a constant-ν Mermin + static-LFC branch.

    Phase 6 keeps the same compact ESA-style static-LFC surrogate introduced in
    the collisionless branch and applies the constant-ν Mermin closure to that
    correlated collisionless dielectric function. In the limits ``ν -> 0`` or
    ``G -> 0`` the result should reduce to ``RPA + static LFC`` or plain
    ``Mermin`` respectively.
    """

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    q_over_qf = electron_k_over_kf(float(k_m_inv), float(ne_cm3))
    g_value = float(esa_static_local_field_correction(q_over_qf, float(rs), float(theta)))
    nu_s = float(collision_rate_s)
    if not math.isfinite(g_value):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float("nan"),
            float(q_over_qf),
        )
    if not math.isfinite(nu_s) or nu_s < 0.0:
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float(g_value),
            float(q_over_qf),
        )
    numerical_imag_shift_ev = max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)
    if nu_s <= 0.0:
        chi, epsilon, _, _ = epsilon_rpa_static_lfc(
            energy,
            k_m_inv=float(k_m_inv),
            te_ev=float(te_ev),
            ne_cm3=float(ne_cm3),
            imag_shift_ev=numerical_imag_shift_ev,
            rs=float(rs),
            theta=float(theta),
        )
        return chi, epsilon, float(g_value), float(q_over_qf)

    nu_ev = HBAR_EV_S * nu_s
    chi_shift, eps_shift, _, _ = epsilon_rpa_static_lfc(
        energy,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev + nu_ev,
        rs=float(rs),
        theta=float(theta),
    )
    del chi_shift
    _, eps_static_arr, _, _ = epsilon_rpa_static_lfc(
        np.asarray([0.0], dtype=np.float64),
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev,
        rs=float(rs),
        theta=float(theta),
    )
    eps_static = complex(eps_static_arr[0])
    if not math.isfinite(float(np.real(eps_static))) or not math.isfinite(float(np.imag(eps_static))):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float(g_value),
            float(q_over_qf),
        )

    omega = energy / HBAR_EV_S
    ratio = np.zeros(energy.shape, dtype=np.complex128)
    nonzero = np.abs(omega) > 1.0e-30
    ratio[nonzero] = 1j * nu_s / omega[nonzero]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = (1.0 + ratio) * (eps_shift - 1.0)
        denominator = 1.0 + ratio * (eps_shift - 1.0) / (eps_static - 1.0)
        epsilon = 1.0 + numerator / denominator

    zero_mask = ~nonzero
    if np.any(zero_mask):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[zero_mask] = eps_static

    bad = ~np.isfinite(np.real(epsilon)) | ~np.isfinite(np.imag(epsilon))
    if np.any(bad):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[bad] = np.nan + 0.0j
    chi = epsilon - 1.0
    return chi, epsilon, float(g_value), float(q_over_qf)



def epsilon_lindhard(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    benchmark: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chi, epsilon)`` for the finite-temperature Lindhard baseline."""

    chi = finite_t_lindhard_susceptibility(
        delta_energy_ev,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=float(imag_shift_ev),
        benchmark=bool(benchmark),
    )
    epsilon = 1.0 + chi
    return chi, epsilon


def epsilon_lindhard_static_lfc(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    imag_shift_ev: float,
    rs: float,
    theta: float,
    benchmark: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(chi_eff, epsilon, G(k), q/k_F)`` for the finite-T Lindhard + static-LFC branch."""

    chi_rpa = finite_t_lindhard_susceptibility(
        delta_energy_ev,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=float(imag_shift_ev),
        benchmark=bool(benchmark),
    )
    q_over_qf = electron_k_over_kf(float(k_m_inv), float(ne_cm3))
    g_value = float(esa_static_local_field_correction(q_over_qf, float(rs), float(theta)))
    chi_eff, epsilon = _apply_static_lfc_response(chi_rpa, g_value)
    return chi_eff, epsilon, g_value, float(q_over_qf)


def epsilon_lindhard_mermin(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    collision_rate_s: float,
    imag_shift_ev: float,
    benchmark: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chi, epsilon)`` for the finite-T Lindhard baseline with a Mermin closure."""

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    nu_s = float(collision_rate_s)
    if not math.isfinite(nu_s) or nu_s < 0.0:
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
        )
    numerical_imag_shift_ev = max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)
    if nu_s <= 0.0:
        return epsilon_lindhard(
            energy,
            k_m_inv=float(k_m_inv),
            te_ev=float(te_ev),
            ne_cm3=float(ne_cm3),
            imag_shift_ev=numerical_imag_shift_ev,
            benchmark=bool(benchmark),
        )

    nu_ev = HBAR_EV_S * nu_s
    _, eps_shift = epsilon_lindhard(
        energy,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev + nu_ev,
        benchmark=bool(benchmark),
    )
    _, eps_static_arr = epsilon_lindhard(
        np.asarray([0.0], dtype=np.float64),
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev,
        benchmark=bool(benchmark),
    )
    eps_static = complex(eps_static_arr[0])
    if not math.isfinite(float(np.real(eps_static))) or not math.isfinite(float(np.imag(eps_static))):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
        )

    omega = energy / HBAR_EV_S
    ratio = np.zeros(energy.shape, dtype=np.complex128)
    nonzero = np.abs(omega) > 1.0e-30
    ratio[nonzero] = 1j * nu_s / omega[nonzero]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = (1.0 + ratio) * (eps_shift - 1.0)
        denominator = 1.0 + ratio * (eps_shift - 1.0) / (eps_static - 1.0)
        epsilon = 1.0 + numerator / denominator

    zero_mask = ~nonzero
    if np.any(zero_mask):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[zero_mask] = eps_static

    bad = ~np.isfinite(np.real(epsilon)) | ~np.isfinite(np.imag(epsilon))
    if np.any(bad):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[bad] = np.nan + 0.0j
    chi = epsilon - 1.0
    return chi, epsilon


def epsilon_lindhard_mermin_static_lfc(
    delta_energy_ev: np.ndarray,
    *,
    k_m_inv: float,
    te_ev: float,
    ne_cm3: float,
    collision_rate_s: float,
    imag_shift_ev: float,
    rs: float,
    theta: float,
    benchmark: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(chi, epsilon, G(k), q/k_F)`` for finite-T Lindhard + Mermin + static LFC."""

    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    q_over_qf = electron_k_over_kf(float(k_m_inv), float(ne_cm3))
    g_value = float(esa_static_local_field_correction(q_over_qf, float(rs), float(theta)))
    nu_s = float(collision_rate_s)
    if not math.isfinite(g_value):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float("nan"),
            float(q_over_qf),
        )
    if not math.isfinite(nu_s) or nu_s < 0.0:
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float(g_value),
            float(q_over_qf),
        )
    numerical_imag_shift_ev = max(float(imag_shift_ev), _DEF_NUMERICAL_IMAG_SHIFT_EV)
    if nu_s <= 0.0:
        chi, epsilon, _, _ = epsilon_lindhard_static_lfc(
            energy,
            k_m_inv=float(k_m_inv),
            te_ev=float(te_ev),
            ne_cm3=float(ne_cm3),
            imag_shift_ev=numerical_imag_shift_ev,
            rs=float(rs),
            theta=float(theta),
            benchmark=bool(benchmark),
        )
        return chi, epsilon, float(g_value), float(q_over_qf)

    nu_ev = HBAR_EV_S * nu_s
    _, eps_shift, _, _ = epsilon_lindhard_static_lfc(
        energy,
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev + nu_ev,
        rs=float(rs),
        theta=float(theta),
        benchmark=bool(benchmark),
    )
    _, eps_static_arr, _, _ = epsilon_lindhard_static_lfc(
        np.asarray([0.0], dtype=np.float64),
        k_m_inv=float(k_m_inv),
        te_ev=float(te_ev),
        ne_cm3=float(ne_cm3),
        imag_shift_ev=numerical_imag_shift_ev,
        rs=float(rs),
        theta=float(theta),
        benchmark=bool(benchmark),
    )
    eps_static = complex(eps_static_arr[0])
    if not math.isfinite(float(np.real(eps_static))) or not math.isfinite(float(np.imag(eps_static))):
        return (
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            np.full(energy.shape, np.nan + 0.0j, dtype=np.complex128),
            float(g_value),
            float(q_over_qf),
        )

    omega = energy / HBAR_EV_S
    ratio = np.zeros(energy.shape, dtype=np.complex128)
    nonzero = np.abs(omega) > 1.0e-30
    ratio[nonzero] = 1j * nu_s / omega[nonzero]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = (1.0 + ratio) * (eps_shift - 1.0)
        denominator = 1.0 + ratio * (eps_shift - 1.0) / (eps_static - 1.0)
        epsilon = 1.0 + numerator / denominator

    zero_mask = ~nonzero
    if np.any(zero_mask):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[zero_mask] = eps_static

    bad = ~np.isfinite(np.real(epsilon)) | ~np.isfinite(np.imag(epsilon))
    if np.any(bad):
        epsilon = np.asarray(epsilon, dtype=np.complex128)
        epsilon[bad] = np.nan + 0.0j
    chi = epsilon - 1.0
    return chi, epsilon, float(g_value), float(q_over_qf)


def loss_function_from_epsilon(epsilon: np.ndarray) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = -np.imag(1.0 / eps)
    loss[~np.isfinite(loss)] = np.nan
    return loss.astype(np.float64, copy=False)



def dsf_from_loss(loss: np.ndarray, delta_energy_ev: np.ndarray, te_ev: float) -> np.ndarray:
    """Build a simple free-electron DSF proxy from the loss function."""

    loss_array = np.asarray(loss, dtype=np.float64)
    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    if te_ev <= 0.0:
        result = np.where(np.isfinite(loss_array), loss_array, np.nan)
        result[result < 0.0] = 0.0
        return result
    scaled = -energy / float(te_ev)
    scaled = np.clip(scaled, -700.0, 700.0)
    denom = -np.expm1(scaled)
    small = np.abs(denom) < 1.0e-9
    if np.any(small):
        denom = denom.copy()
        denom[small] = energy[small] / float(te_ev)
    with np.errstate(divide="ignore", invalid="ignore"):
        spectrum = loss_array / denom
    spectrum = np.asarray(np.real(spectrum), dtype=np.float64)
    bad = ~np.isfinite(spectrum)
    if np.any(bad):
        idx = np.flatnonzero(bad)
        for center in idx:
            left = center - 1
            right = center + 1
            while left >= 0 and not math.isfinite(float(spectrum[left])):
                left -= 1
            while right < spectrum.size and not math.isfinite(float(spectrum[right])):
                right += 1
            if left >= 0 and right < spectrum.size and math.isfinite(float(spectrum[left])) and math.isfinite(float(spectrum[right])):
                spectrum[center] = 0.5 * (float(spectrum[left]) + float(spectrum[right]))
            else:
                spectrum[center] = 0.0
    negative = np.isfinite(spectrum) & (spectrum < 0.0)
    spectrum[negative] = 0.0
    return spectrum



def gaussian_convolve(delta_energy_ev: np.ndarray, spectrum: np.ndarray, fwhm_ev: float) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    if not math.isfinite(float(fwhm_ev)) or float(fwhm_ev) <= 0.0 or energy.size < 3:
        return values.copy()
    spacing = float(np.nanmedian(np.diff(energy)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        return values.copy()
    sigma = float(fwhm_ev) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    half_span = max(3, int(math.ceil(4.0 * sigma / spacing)))
    offsets = np.arange(-half_span, half_span + 1, dtype=np.float64) * spacing
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(np.nan_to_num(values, nan=0.0), (half_span, half_span), mode="edge")
    convolved = np.convolve(padded, kernel, mode="valid")
    return np.asarray(convolved, dtype=np.float64)



def normalize_spectrum(delta_energy_ev: np.ndarray, spectrum: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    if mode == "none":
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    if mode == "area":
        area = float(np.trapezoid(np.clip(values[finite], a_min=0.0, a_max=None), np.asarray(delta_energy_ev, dtype=np.float64)[finite]))
        if math.isfinite(area) and area > 0.0:
            return values / area
        return values
    peak = float(np.nanmax(values[finite]))
    if math.isfinite(peak) and peak > 0.0:
        return values / peak
    return values



def _quadratic_peak_fit(energy: np.ndarray, values: np.ndarray, peak_index: int, *, half_window: int) -> tuple[float, float]:
    """Return a locally fitted peak position and amplitude.

    The fit is deliberately light-weight: a quadratic polynomial is fit over a
    small local window around the discrete maximum. This preserves interactive
    speed while removing the worst grid-locking from peak extraction.
    """

    count = int(energy.size)
    if count < 3 or peak_index <= 0 or peak_index >= count - 1:
        return float(energy[peak_index]), float(values[peak_index])
    left = max(0, int(peak_index) - max(int(half_window), 1))
    right = min(count, int(peak_index) + max(int(half_window), 1) + 1)
    x = np.asarray(energy[left:right], dtype=np.float64)
    y = np.asarray(values[left:right], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return float(energy[peak_index]), float(values[peak_index])
    x = x[valid]
    y = y[valid]
    x0 = float(energy[peak_index])
    shifted = x - x0
    weights = np.sqrt(np.clip(y, a_min=1.0e-30, a_max=None))
    try:
        coeffs = np.polyfit(shifted, y, 2, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return float(energy[peak_index]), float(values[peak_index])
    a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    if (not math.isfinite(a)) or (not math.isfinite(b)) or (not math.isfinite(c)) or a >= 0.0 or abs(a) <= 1.0e-30:
        return float(energy[peak_index]), float(values[peak_index])
    x_vertex = -b / (2.0 * a)
    x_min = float(np.min(shifted))
    x_max = float(np.max(shifted))
    if (not math.isfinite(x_vertex)) or x_vertex < x_min or x_vertex > x_max:
        return float(energy[peak_index]), float(values[peak_index])
    y_vertex = a * x_vertex * x_vertex + b * x_vertex + c
    if not math.isfinite(y_vertex):
        return float(energy[peak_index]), float(values[peak_index])
    return float(x0 + x_vertex), float(max(y_vertex, float(values[peak_index])))


def _interp_half_max_crossing(energy: np.ndarray, values: np.ndarray, idx0: int, idx1: int, half_max: float) -> float:
    x0 = float(energy[idx0])
    x1 = float(energy[idx1])
    y0 = float(values[idx0])
    y1 = float(values[idx1])
    if not math.isfinite(y0) or not math.isfinite(y1) or y1 == y0:
        return x0
    fraction = (half_max - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def estimate_peak_metrics(
    delta_energy_ev: np.ndarray,
    spectrum: np.ndarray,
    *,
    method: str = "quadratic",
    local_half_window_points: int = 2,
) -> tuple[float, float]:
    energy = np.asarray(delta_energy_ev, dtype=np.float64)
    values = np.asarray(spectrum, dtype=np.float64)
    finite = np.isfinite(energy) & np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return float("nan"), float("nan")
    positive = finite & (energy >= 0.0)
    active = positive if np.any(positive) else finite
    idx_candidates = np.flatnonzero(active)
    if idx_candidates.size == 0:
        return float("nan"), float("nan")
    local = values[idx_candidates]
    peak_rel = int(np.nanargmax(local))
    peak_index = int(idx_candidates[peak_rel])
    discrete_peak_value = float(values[peak_index])
    if not math.isfinite(discrete_peak_value) or discrete_peak_value <= 0.0:
        return float("nan"), float("nan")
    fit_mode = str(method or "quadratic").lower()
    if fit_mode in {"quadratic", "local_quadratic", "publication"}:
        half_window = max(int(local_half_window_points), 1 if fit_mode == "quadratic" else 2)
        peak_energy, peak_value = _quadratic_peak_fit(energy, values, peak_index, half_window=half_window)
    else:
        peak_energy = float(energy[peak_index])
        peak_value = discrete_peak_value
    if not math.isfinite(peak_energy) or not math.isfinite(peak_value) or peak_value <= 0.0:
        return float("nan"), float("nan")

    half_max = 0.5 * peak_value
    left = peak_index
    while left > 0 and math.isfinite(float(values[left])) and float(values[left]) >= half_max:
        left -= 1
    right = peak_index
    last_index = values.size - 1
    while right < last_index and math.isfinite(float(values[right])) and float(values[right]) >= half_max:
        right += 1
    if left == peak_index or right == peak_index or right <= left:
        return float(peak_energy), float("nan")
    left_cross = _interp_half_max_crossing(energy, values, left, min(left + 1, peak_index), half_max)
    right_cross = _interp_half_max_crossing(energy, values, max(right - 1, peak_index), right, half_max)
    width = float(right_cross - left_cross) if math.isfinite(right_cross - left_cross) else float("nan")
    return float(peak_energy), width if width > 0.0 else float("nan")
