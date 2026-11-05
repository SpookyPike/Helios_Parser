"""Radiation-spectral helpers: group integration and the boundary spectra.

WHY THIS EXISTS.  A converted run carries 250 photon groups per spectrum, and the
two questions a campaign asks most often are "what is the total loss" and "show
me the spectrum at this instant".  Both are three lines of numpy, but the group
integration convention is not obvious and getting it wrong is silent, so it lives
here once with the measurement that fixes it.

THE CONVENTION.  Two per-group families in the same file use opposite
normalizations, and the vendor's dataset names call both "per eV":

- `FdIntgPowerLoss*` is a density.  `sum(FdIntgPowerLossRmin * dE)` reproduces
  `FreqIntgRadEnLossRmin` to 1e-16 relative on all three fixtures, every dump;
  trapezoid weights are off by 1.5 % on the Rmin boundary, and a plain sum is off
  by 1e15.  `integrate_over_groups` uses the rectangle rule for this family.
- `net_interface` is a bin content.  `sum(net_interface[:, 0, :])` reproduces
  `-TotalFluxRmin` exactly, while weighting by the group width is off by a factor
  of 13 to 2.8e3 depending on the dump.  Sum over groups, do not integrate.

The writer records which is which in a `normalization` dataset attribute, and the
`unit` attribute follows that measurement (`J/(cm2 s)/group` for a bin content)
rather than the vendor's "per eV" name.

THE TWO AXES.  The spectra here are on the `flux/` group, i.e. the `time_whole`
axis.  The node-centred spectral field is on `radiation/`, whose axis is
`time_freq`; `radiation/hydro_index` maps one onto the other.  Nothing in this
module resamples between them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from helios.services.derived.common import cached_run_container, run_cache_key
from helios.services.derived.models import DerivedPlotBundle, DerivedWarning, RadiationResult

# An HDF5 container starts with this signature; a raw HELIOS deck does not.
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

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

def evaluate_radiation(
    dataset: object,
    *,
    snapshot_index: int,
) -> RadiationResult:
    """Build the plot bundles the Radiation tab renders for one snapshot.

    Reads only the `flux/` block and the integrated scalars, never the
    `(dump, zone, group)` spectral field: the tab shows boundary spectra, and a
    26 MB container should not be re-read every time the snapshot slider moves.

    A run whose deck left the radiation output off gets `available=False` and a
    reason rather than an exception, because "this run has no spectra" is a
    normal state, not a failure.
    """
    path = Path(getattr(dataset, "path"))
    snapshot_index = int(snapshot_index)
    requested_time_s = float("nan")
    time_s = getattr(dataset, "time_s", None)
    if time_s is not None and np.asarray(time_s).size > snapshot_index >= 0:
        requested_time_s = float(np.asarray(time_s)[snapshot_index])

    def _read_container() -> dict[str, object]:
        # A raw deck is not an HDF5 container, so it carries no radiation group.
        # Report that as the normal "output was off" state instead of an OSError
        # about a file signature, which is what a converted run reports too.
        try:
            with path.open("rb") as probe:
                if probe.read(8) != HDF5_MAGIC:
                    return {"missing": True}
        except OSError:
            return {"missing": True}
        with h5py.File(path, "r") as handle:
            if "flux" not in handle or "radiation" not in handle:
                return {"missing": True}
            return {
                "missing": False,
                "flux_time": np.asarray(handle["flux/time"][()], dtype=np.float64),
                "group_edges": np.asarray(handle["flux/group_edges"][()], dtype=np.float64),
                "net_interface": np.asarray(handle["flux/net_interface"][()], dtype=np.float64),
                "integrated": {
                    name: np.asarray(handle[f"radiation/integrated/{name}"][()], dtype=np.float64)
                    for name in ("loss_rmax", "loss_rmin", "total_flux_rmax", "total_flux_rmin")
                    if f"radiation/integrated/{name}" in handle
                },
                "metadata": {
                    name: int(handle[f"metadata/{name}"][()])
                    for name in ("n_groups", "n_freq_snapshots", "n_boundaries", "n_regions")
                    if f"metadata/{name}" in handle
                },
            }

    try:
        # The container does not depend on the snapshot, so it is cached per run
        # revision. Without this, every slider move re-opened the file and read
        # the whole flux block to change one index.
        container = cached_run_container(run_cache_key(dataset), "radiation_container", _read_container)
        assert isinstance(container, dict)
        if container.get("missing"):
            return _unavailable_radiation(
                snapshot_index,
                requested_time_s,
                "This run carries no radiation output: the deck left "
                "`Write freq-dependent radiation output` off.",
            )
        flux_time = np.asarray(container["flux_time"], dtype=np.float64)
        group_edges = np.asarray(container["group_edges"], dtype=np.float64)
        net_interface = np.asarray(container["net_interface"], dtype=np.float64)
        integrated = container["integrated"]
        metadata = container["metadata"]
        assert isinstance(integrated, dict) and isinstance(metadata, dict)
    except OSError as error:
        return _unavailable_radiation(
            snapshot_index,
            requested_time_s,
            f"The radiation block could not be read: {error}.",
        )

    if flux_time.size == 0 or group_edges.size == 0:
        return _unavailable_radiation(snapshot_index, requested_time_s, "The flux group is empty.")

    sample_index = int(np.abs(flux_time - requested_time_s).argmin()) if np.isfinite(requested_time_s) else flux_time.size - 1
    sample_time_s = float(flux_time[sample_index])
    n_groups = int(group_edges.shape[-1] - 1)
    rmin_spectrum = net_interface[:, 0, :]
    rmax_spectrum = net_interface[:, -1, :]

    profile_plots = tuple(
        DerivedPlotBundle(
            key=f"spectrum_{row}",
            title=f"Boundary spectra at {float(flux_time[row]):.4g} s",
            x_label="photon energy [eV]",
            # `net_interface` is a per-group bin content: the group sum, not a
            # dE-weighted integral, reproduces TotalFluxRmin/Rmax.  Labelling it
            # per-eV would invite an integration that overstates the total by
            # the group width.
            y_label="net flux per group [J/(cm2 s)]",
            x_values=np.sqrt(group_edges[row, :-1] * group_edges[row, 1:]),
            y_series=(rmin_spectrum[row], rmax_spectrum[row]),
            curve_names=("Rmin", "Rmax"),
            value_scale_mode="symlog",
        )
        for row in range(flux_time.size)
    )

    # The time traces read the stored integrated scalars rather than integrating
    # the spectra here: `net_interface` is a bin content, and the file already
    # carries the vendor's own totals.
    time_bundles: list[DerivedPlotBundle] = []
    if "total_flux_rmin" in integrated and "total_flux_rmax" in integrated:
        time_bundles.append(
            DerivedPlotBundle(
                key="total_flux",
                title="Boundary flux",
                x_label="time [s]",
                y_label="total flux [J/(cm2 s)]",
                x_values=flux_time,
                y_series=(integrated["total_flux_rmin"], integrated["total_flux_rmax"]),
                curve_names=("Rmin", "Rmax"),
                value_scale_mode="symlog",
            )
        )
    if "loss_rmin" in integrated and "loss_rmax" in integrated:
        time_bundles.append(
            DerivedPlotBundle(
                key="loss",
                title="Frequency-integrated radiation loss per boundary",
                x_label="time [s]",
                y_label="loss",
                x_values=flux_time,
                y_series=(integrated["loss_rmin"], integrated["loss_rmax"]),
                curve_names=("Rmin", "Rmax"),
                value_scale_mode="symlog",
            )
        )
    time_plots = tuple(time_bundles)

    summary = (
        f"radiation output on | {n_groups} groups | {flux_time.size} flux dumps | "
        f"{metadata.get('n_freq_snapshots', 0)} spectral dumps | "
        f"{metadata.get('n_boundaries', 2)} boundaries | {metadata.get('n_regions', 1)} regions | "
        f"sampled {sample_time_s:.4g} s"
    )
    warnings: tuple[DerivedWarning, ...] = ()
    if not np.any(net_interface):
        warnings = (
            DerivedWarning(
                "radiation",
                "The net interface flux is zero in every dump of this run.",
                severity="caution",
            ),
        )
    return RadiationResult(
        snapshot_index=snapshot_index,
        available=True,
        summary=summary,
        n_groups=n_groups,
        n_flux_dumps=int(flux_time.size),
        n_freq_snapshots=int(metadata.get("n_freq_snapshots", 0)),
        n_boundaries=int(net_interface.shape[1]),
        n_regions=int(metadata.get("n_regions", 1)),
        sample_time_s=sample_time_s,
        requested_time_s=requested_time_s,
        total_flux_rmin=float(integrated.get("total_flux_rmin", np.array([np.nan]))[sample_index]),
        total_flux_rmax=float(integrated.get("total_flux_rmax", np.array([np.nan]))[sample_index]),
        time_plots=time_plots,
        profile_plots=profile_plots,
        warnings=warnings,
    )


def _unavailable_radiation(snapshot_index: int, requested_time_s: float, reason: str) -> RadiationResult:
    return RadiationResult(
        snapshot_index=snapshot_index,
        available=False,
        summary=reason,
        n_groups=0,
        n_flux_dumps=0,
        n_freq_snapshots=0,
        n_boundaries=0,
        n_regions=0,
        sample_time_s=float("nan"),
        requested_time_s=requested_time_s,
        total_flux_rmin=float("nan"),
        total_flux_rmax=float("nan"),
        time_plots=(),
        profile_plots=(),
        warnings=(DerivedWarning("radiation", reason, severity="info"),),
    )
