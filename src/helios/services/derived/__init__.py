"""Derived scientific-analysis services for HELIOS Analyzer."""

from .analysis import (
    DerivedAnalysisParameters,
    analysis_result_time_plot_modules,
    compute_analysis_result,
    normalize_time_plot_modules,
    registered_module_contracts,
    refresh_analysis_result_for_snapshot,
)
from .common import load_run_data
from .plasmon import evaluate_plasmon_regime
from .shock_tracking import track_shock_front
from .spectroscopy import evaluate_spectroscopy
from .transmission import evaluate_transmission
from .xcom_hook import build_cold_attenuation_request
from .xrd import estimate_xrd

__all__ = [
    "DerivedAnalysisParameters",
    "compute_analysis_result",
    "refresh_analysis_result_for_snapshot",
    "analysis_result_time_plot_modules",
    "normalize_time_plot_modules",
    "registered_module_contracts",
    "estimate_xrd",
    "evaluate_plasmon_regime",
    "evaluate_spectroscopy",
    "evaluate_transmission",
    "build_cold_attenuation_request",
    "load_run_data",
    "track_shock_front",
]
