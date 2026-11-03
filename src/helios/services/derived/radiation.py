"""Radiation-spectral helpers: group integration and the boundary spectra.

WHY THIS EXISTS.  A converted run carries 250 photon groups per spectrum, and the
two questions a campaign asks most often are "what is the total loss" and "show
me the spectrum at this instant".  Both are three lines of numpy, but the group
integration convention is not obvious and getting it wrong is silent, so it lives
here once with the measurement that fixes it.

THE CONVENTION.  `sum(FdIntgPowerLossRmin * dE)` reproduces
`FreqIntgRadEnLossRmin` to 1e-16 relative on all three reference fixtures, every
dump, so HELIOS treats the per-group arrays as bin contents and weights them by
the bin width.  Trapezoid weights are off by up to 1.5 % on the Rmin boundary,
where the group grid is coarse next to the 10 keV split.  `integrate_over_groups`
uses the rectangle rule and is checked against the stored integrated scalars.

THE TWO AXES.  The spectra here are on the `flux/` group, i.e. the `time_whole`
axis.  The node-centred spectral field is on `radiation/`, whose axis is
`time_freq`; `radiation/hydro_index` maps one onto the other.  Nothing in this
module resamples between them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# Boundary side -> the flux dataset that carries its spectrum, and which row of
# it when the dataset is per-boundary.  `net` is the informative one in the
# reference fixtures: the incident arrays are identically zero there.
SIDE_DATASETS = {"rmax": "incident_rmax", "rmin": "incident_rmin"}
NET_ROWS = {"rmin": 0, "rmax": -1}


def integrate_over_groups(values: np.ndarray, group_edges: np.ndarray) -> np.ndarray:
    """Integrate over the group axis with the bin width as the weight.

    `values` has one more dimension entry on its last axis than `group_edges` has
    entries: N group values against N+1 edges.  A per-dump grid pairs with a
    per-dump array, so both may be 2-D.
    """
    array = np.asarray(values, dtype=np.float64)
    edges = np.asarray(group_edges, dtype=np.float64)
    if edges.shape[-1] != array.shape[-1] + 1:
        raise ValueError(
            f"group_edges has {edges.shape[-1]} entries for {array.shape[-1]} group values; "
            "expected one more edge than values."
        )
    return (array * np.diff(edges, axis=-1)).sum(axis=-1)


def _resolve_dataset(side: str, quantity: str) -> tuple[str, int | None]:
    if side not in SIDE_DATASETS:
        raise ValueError(f"side must be one of {sorted(SIDE_DATASETS)}, got {side!r}.")
    if quantity == "incident":
        return SIDE_DATASETS[side], None
    if quantity == "net":
        return "net_interface", NET_ROWS[side]
    raise ValueError(f"quantity must be 'incident' or 'net', got {quantity!r}.")


def _read_flux(
    path: str | Path,
    side: str,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset, row = _resolve_dataset(side, quantity)
    source = Path(path)
    with h5py.File(source, "r") as handle:
        if "flux" not in handle:
            raise ValueError(
                f"{source} carries no radiation output: there is no flux group. "
                "Convert a run whose deck set `Write freq-dependent radiation output = 1`."
            )
        time_s = np.asarray(handle["flux/time"][()], dtype=np.float64)
        group_edges = np.asarray(handle["flux/group_edges"][()], dtype=np.float64)
        values = np.asarray(handle[f"flux/{dataset}"][()], dtype=np.float64)
    if row is not None:
        values = values[:, row, :]
    return time_s, group_edges, values


def spectrum_map(
    path: str | Path,
    side: str = "rmax",
    *,
    quantity: str = "incident",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The `(n_time, n_group)` array a spectrum-versus-time contour needs.

    Returns the time axis in seconds, the photon energy edges in eV with shape
    `(n_time, n_group + 1)`, and the spectra.  `quantity="net"` reads the net
    flux at that boundary, which is the one that carries energy in the reference
    fixtures.
    """
    return _read_flux(path, side, quantity)


def spectrum_at(
    path: str | Path,
    time_s: float,
    side: str = "rmax",
    *,
    quantity: str = "incident",
) -> tuple[float, np.ndarray, np.ndarray]:
    """The spectrum nearest `time_s`, with the sample time it actually came from.

    The sample time is returned so a caller cannot mistake it for the requested
    one: the spectral dumps are a skip count over hydro dumps, not a fixed grid.
    """
    times, group_edges, values = _read_flux(path, side, quantity)
    if times.size == 0:
        raise ValueError(f"{path} has an empty flux time axis, so there is no spectrum to sample.")
    index = int(np.abs(times - float(time_s)).argmin())
    return float(times[index]), group_edges[index], values[index]