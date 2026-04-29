"""Central feature flags for production and development GUI surfaces."""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUE_VALUES


def dev_mode_enabled() -> bool:
    """Return whether development-only UI surfaces may be shown."""

    return _env_flag("HELIOS_DEV_MODE")


def experimental_features_enabled() -> bool:
    """Return whether experimental physics pipelines may be shown in the GUI."""

    return dev_mode_enabled() or _env_flag("HELIOS_ENABLE_EXPERIMENTAL")


def production_feature_visible(feature_name: str) -> bool:
    """Return whether a named feature is visible in the production GUI."""

    normalized = str(feature_name or "").strip().lower().replace("-", "_")
    experimental = {
        "plasmon",
        "xrts",
        "transmission",
    }
    return normalized not in experimental or experimental_features_enabled()
