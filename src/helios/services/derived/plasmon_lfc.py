"""Static-LFC helpers for the plasmon/XRTS module.

Phase 4 adds a lightweight static-LFC backend keyed as ``esa_static``.
The current implementation is intentionally compact and dependency-free: it
preserves the essential ESA-style asymptotes ``G(0)=0`` and
``G(q→inf)=1-g(0)`` and uses a smooth activation around
``q∼3 q_F``. This is sufficient to expose the correlation seam,
cache plumbing, domain warnings, and UI/result paths without introducing a
heavy external model dependency.

Important: this backend is an ESA-style surrogate, not the full 2021 analytic
ESA parametrization. The service layer emits an explicit warning so the user is
not misled about the fidelity level.
"""

from __future__ import annotations

import math
import numpy as np

ESA_RS_MIN = 0.7
ESA_RS_MAX = 20.0
ESA_THETA_MIN = 0.0
ESA_THETA_MAX = 4.0


def esa_domain_contains(rs: float, theta: float) -> bool:
    return math.isfinite(rs) and math.isfinite(theta) and ESA_RS_MIN <= rs <= ESA_RS_MAX and ESA_THETA_MIN <= theta <= ESA_THETA_MAX


def esa_domain_message(rs: float, theta: float) -> str | None:
    if not math.isfinite(rs) or not math.isfinite(theta):
        return "ESA static-LFC requires finite r_s and Theta descriptors."
    if rs < ESA_RS_MIN or rs > ESA_RS_MAX or theta < ESA_THETA_MIN or theta > ESA_THETA_MAX:
        return (
            f"ESA static-LFC surrogate is validated only for approximately {ESA_RS_MIN:g} <= r_s <= {ESA_RS_MAX:g} "
            f"and {ESA_THETA_MIN:g} <= Theta <= {ESA_THETA_MAX:g}; current state has r_s={rs:.3g}, Theta={theta:.3g}."
        )
    if rs > 15.0:
        return (
            f"ESA static-LFC surrogate is near its strong-coupling edge (r_s={rs:.3g}); correlation corrections may become less reliable as r_s approaches {ESA_RS_MAX:g}."
        )
    return None


def esa_on_top_pair_distribution_surrogate(rs: float, theta: float) -> float:
    """Return a compact surrogate for the on-top PDF ``g(0)``.

    The functional form is empirical and chosen only to provide a smooth,
    bounded high-q limit for the ESA-style backend. It is *not* a reproduction
    of the published ESA/NN fit coefficients.
    """

    if not math.isfinite(rs) or rs <= 0.0 or not math.isfinite(theta) or theta < 0.0:
        return float('nan')
    ground_state = 0.78 * math.exp(-0.62 * rs)
    temperature_softening = 1.0 / (1.0 + 0.16 * theta + 0.035 * theta * theta)
    value = ground_state * temperature_softening
    return min(max(value, 0.0), 0.98)


def esa_activation(q_over_qf: np.ndarray | float) -> np.ndarray:
    q = np.asarray(q_over_qf, dtype=np.float64)
    x = np.clip(q / 3.0, 0.0, None)
    return 1.0 - np.exp(-(x**4))


def esa_static_local_field_correction(q_over_qf: np.ndarray | float, rs: float, theta: float) -> np.ndarray:
    """Return an ESA-style static local-field correction ``G(q)``.

    This compact surrogate matches the correct qualitative limits used by the
    effective static approximation:
      - ``G(0) = 0``
      - ``G(q→∞) = 1 - g(0)``

    and it transitions between them around ``q ≈ 3 q_F``.
    """

    q = np.asarray(q_over_qf, dtype=np.float64)
    q_abs = np.clip(np.abs(q), 0.0, None)
    g0 = esa_on_top_pair_distribution_surrogate(float(rs), float(theta))
    if not math.isfinite(g0):
        return np.full(q_abs.shape, np.nan, dtype=np.float64)
    high_limit = max(0.0, min(1.0, 1.0 - g0))
    width = 1.0 + 0.12 * float(rs) + 0.20 * float(theta)
    low_q = high_limit * (q_abs * q_abs) / (q_abs * q_abs + width * width)
    act = esa_activation(q_abs)
    values = act * high_limit + (1.0 - act) * low_q
    return np.clip(values, 0.0, 1.0)
