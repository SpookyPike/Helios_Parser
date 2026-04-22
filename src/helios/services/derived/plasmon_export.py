"""CSV export helpers for plasmon/XRTS spectra."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from helios.services.derived.models import PlasmonResult


def plasmon_export_columns(plasmon: PlasmonResult) -> dict[str, np.ndarray]:
    """Return aligned plasmon spectrum arrays ready for CSV export.

    Required columns are always ``energy_transfer_ev`` and ``observed_intensity``.
    Optional dielectric/loss arrays are exported only when they have the same
    length as the energy axis.
    """

    energy = np.asarray(plasmon.spectrum_energy_ev, dtype=np.float64)
    intensity = np.asarray(plasmon.spectrum_intensity, dtype=np.float64)
    if energy.size == 0 or intensity.size == 0 or energy.shape != intensity.shape:
        return {}
    columns: dict[str, np.ndarray] = {
        'energy_transfer_ev': energy,
        'observed_intensity': intensity,
    }
    optional = {
        'free_component': np.asarray(plasmon.spectrum_free_component, dtype=np.float64),
        'bound_component': np.asarray(plasmon.spectrum_bound_component, dtype=np.float64),
        'elastic_component': np.asarray(plasmon.spectrum_elastic_component, dtype=np.float64),
        'dielectric_real': np.asarray(plasmon.dielectric_real, dtype=np.float64),
        'dielectric_imag': np.asarray(plasmon.dielectric_imag, dtype=np.float64),
        'loss_function': np.asarray(plasmon.loss_function, dtype=np.float64),
    }
    for name, values in optional.items():
        if values.shape == energy.shape:
            columns[name] = values
    return columns


def plasmon_export_is_ready(plasmon: PlasmonResult | None) -> bool:
    if plasmon is None:
        return False
    return bool(plasmon_export_columns(plasmon))


def write_plasmon_spectrum_csv(path: str | Path, plasmon: PlasmonResult) -> Path:
    columns = plasmon_export_columns(plasmon)
    if not columns:
        raise ValueError('No aligned plasmon spectrum arrays are available for export.')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns.keys())
    row_count = len(columns[names[0]])
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for index in range(row_count):
            writer.writerow([f'{float(columns[name][index]):.17g}' for name in names])
    return destination
