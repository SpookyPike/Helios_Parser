from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

from inspect_bpf import extract_common_snapshot, infer_layout, read_fortran_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "new_data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "bpf_extra_fields"


@dataclass(frozen=True, slots=True)
class IonizationSeries:
    run_name: str
    source_path: Path
    times_s: np.ndarray
    zone_centers_cm: np.ndarray
    ionization_fractions: np.ndarray
    mean_charge: np.ndarray
    dominant_charge: np.ndarray
    charge_states: np.ndarray
    photon_group_boundaries_eV: np.ndarray
    radiation_flux_rmin_j_s_cm2_eV: np.ndarray
    radiation_flux_rmax_j_s_cm2_eV: np.ndarray
    radiation_loss_rmin_j_cm2_eV: np.ndarray
    radiation_loss_rmax_j_cm2_eV: np.ndarray


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "run"


def _axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Axis values must be one-dimensional.")
    if values.size == 0:
        raise ValueError("Axis values must not be empty.")
    if values.size == 1:
        delta = max(abs(float(values[0])) * 0.05, 1.0)
        return np.asarray([values[0] - delta, values[0] + delta], dtype=np.float64)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - (edges[1] - values[0])
    edges[-1] = values[-1] + (values[-1] - edges[-2])
    return edges


def _discover_bpf_files(paths: list[str]) -> list[Path]:
    candidates = [Path(value) for value in paths] if paths else [DEFAULT_DATA_ROOT]
    discovered: list[Path] = []
    for path in candidates:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.bpf")))
        elif path.is_file() and path.suffix.lower() == ".bpf":
            discovered.append(path)
        else:
            raise FileNotFoundError(f"No BPF file or directory found at {path}.")
    return discovered


def load_ionization_series(path: Path) -> IonizationSeries:
    records = read_fortran_records(path)
    layout = infer_layout(path, records)
    times = np.empty(layout.n_snapshots, dtype=np.float64)
    zone_centers = np.empty((layout.n_snapshots, layout.n_zones), dtype=np.float64)
    fractions: np.ndarray | None = None
    flux_rmin = np.empty((layout.n_snapshots, layout.n_freq_bins), dtype=np.float64)
    flux_rmax = np.empty((layout.n_snapshots, layout.n_freq_bins), dtype=np.float64)
    loss_rmin = np.empty((layout.n_snapshots, layout.n_freq_bins), dtype=np.float64)
    loss_rmax = np.empty((layout.n_snapshots, layout.n_freq_bins), dtype=np.float64)
    photon_boundaries: np.ndarray | None = None

    for snapshot_index in range(layout.n_snapshots):
        fields = extract_common_snapshot(records, layout, snapshot_index)
        node_position = np.asarray(fields["node_position_cm"], dtype=np.float64)
        ionization = np.asarray(fields["ionization_fractions"], dtype=np.float64)
        current_boundaries = np.asarray(
            fields["frequency_group_boundaries_eV"], dtype=np.float64
        )
        if photon_boundaries is None:
            photon_boundaries = current_boundaries
        elif not np.allclose(photon_boundaries, current_boundaries):
            raise ValueError(
                f"{path} has changing photon group boundaries at snapshot {snapshot_index}."
            )
        if fractions is None:
            fractions = np.empty(
                (layout.n_snapshots, layout.n_zones, ionization.shape[1]),
                dtype=np.float64,
            )
        elif ionization.shape != fractions.shape[1:]:
            raise ValueError(
                f"Inconsistent ionization shape at snapshot {snapshot_index}: "
                f"{ionization.shape} != {fractions.shape[1:]}"
            )
        times[snapshot_index] = float(fields["time_s"])
        zone_centers[snapshot_index, :] = 0.5 * (
            node_position[:-1] + node_position[1:]
        )
        fractions[snapshot_index, :, :] = ionization
        flux_rmin[snapshot_index, :] = np.asarray(
            fields["radiation_net_flux_rmin_j_s_cm2_eV"], dtype=np.float64
        )
        flux_rmax[snapshot_index, :] = np.asarray(
            fields["radiation_net_flux_rmax_j_s_cm2_eV"], dtype=np.float64
        )
        loss_rmin[snapshot_index, :] = np.asarray(
            fields["radiation_loss_rmin_j_cm2_eV"], dtype=np.float64
        )
        loss_rmax[snapshot_index, :] = np.asarray(
            fields["radiation_loss_rmax_j_cm2_eV"], dtype=np.float64
        )

    if fractions is None or photon_boundaries is None:
        raise ValueError(f"{path} has no ionization fraction data.")
    if not np.all(np.isfinite(fractions)):
        raise ValueError(f"{path} contains non-finite ionization fractions.")
    for name, values in {
        "radiation_flux_rmin": flux_rmin,
        "radiation_flux_rmax": flux_rmax,
        "radiation_loss_rmin": loss_rmin,
        "radiation_loss_rmax": loss_rmax,
    }.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite {name} values.")
    if not np.all(np.diff(photon_boundaries) > 0.0):
        raise ValueError(f"{path} photon group boundaries are not monotonic.")
    row_sums = fractions.sum(axis=2)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"{path} ionization fractions do not sum to one: "
            f"min={row_sums.min():.6g}, max={row_sums.max():.6g}."
        )

    charge_states = np.arange(fractions.shape[2], dtype=np.float64)
    mean_charge = np.tensordot(fractions, charge_states, axes=([2], [0]))
    dominant_charge = np.argmax(fractions, axis=2).astype(np.float64)
    return IonizationSeries(
        run_name=path.stem,
        source_path=path,
        times_s=times,
        zone_centers_cm=zone_centers,
        ionization_fractions=fractions,
        mean_charge=mean_charge,
        dominant_charge=dominant_charge,
        charge_states=charge_states,
        photon_group_boundaries_eV=photon_boundaries,
        radiation_flux_rmin_j_s_cm2_eV=flux_rmin,
        radiation_flux_rmax_j_s_cm2_eV=flux_rmax,
        radiation_loss_rmin_j_cm2_eV=loss_rmin,
        radiation_loss_rmax_j_cm2_eV=loss_rmax,
    )


def save_series_npz(series: IonizationSeries, output_dir: Path) -> Path:
    output_path = output_dir / f"{_slugify(series.run_name)}_bpf_extra_fields.npz"
    np.savez_compressed(
        output_path,
        source_path=str(series.source_path),
        times_s=series.times_s,
        zone_centers_cm=series.zone_centers_cm,
        ionization_fractions=series.ionization_fractions,
        mean_charge=series.mean_charge,
        dominant_charge=series.dominant_charge,
        charge_states=series.charge_states,
        photon_group_boundaries_eV=series.photon_group_boundaries_eV,
        radiation_flux_rmin_j_s_cm2_eV=series.radiation_flux_rmin_j_s_cm2_eV,
        radiation_flux_rmax_j_s_cm2_eV=series.radiation_flux_rmax_j_s_cm2_eV,
        radiation_loss_rmin_j_cm2_eV=series.radiation_loss_rmin_j_cm2_eV,
        radiation_loss_rmax_j_cm2_eV=series.radiation_loss_rmax_j_cm2_eV,
    )
    return output_path


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_final_charge_zone(series: IonizationSeries, output_dir: Path) -> Path:
    final_index = series.times_s.size - 1
    x_edges = _axis_edges(series.zone_centers_cm[final_index])
    y_edges = np.arange(series.charge_states.size + 1, dtype=np.float64) - 0.5
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        series.ionization_fractions[final_index].T,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"{series.run_name}: final charge-state fractions")
    ax.set_xlabel("Zone center position (cm)")
    ax.set_ylabel("Charge state q")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Ion fraction f_q (dimensionless)")
    return _save_figure(
        fig,
        output_dir / f"{_slugify(series.run_name)}_final_charge_state_contour.png",
    )


def plot_selected_zone_time_charge(series: IonizationSeries, output_dir: Path) -> Path:
    zone_index = int(np.nanargmax(np.nanmax(series.mean_charge, axis=0)))
    time_edges_ns = _axis_edges(series.times_s * 1.0e9)
    y_edges = np.arange(series.charge_states.size + 1, dtype=np.float64) - 0.5
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax.pcolormesh(
        time_edges_ns,
        y_edges,
        series.ionization_fractions[:, zone_index, :].T,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(
        f"{series.run_name}: charge-state history, zone {zone_index + 1}"
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Charge state q")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Ion fraction f_q (dimensionless)")
    return _save_figure(
        fig,
        output_dir / f"{_slugify(series.run_name)}_selected_zone_time_charge_contour.png",
    )


def plot_dominant_charge_time_zone(series: IonizationSeries, output_dir: Path) -> Path:
    zone_edges = np.arange(series.dominant_charge.shape[1] + 1, dtype=np.float64) + 0.5
    time_edges_ns = _axis_edges(series.times_s * 1.0e9)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax.pcolormesh(
        zone_edges,
        time_edges_ns,
        series.dominant_charge,
        shading="auto",
        cmap="plasma",
    )
    ax.set_title(f"{series.run_name}: dominant charge state over time")
    ax.set_xlabel("Zone index")
    ax.set_ylabel("Time (ns)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Dominant charge state q")
    return _save_figure(
        fig,
        output_dir / f"{_slugify(series.run_name)}_dominant_charge_time_zone_contour.png",
    )


def _signed_flux_norm(values: np.ndarray) -> SymLogNorm:
    max_abs = float(np.nanmax(np.abs(values)))
    linear_threshold = max(max_abs * 1.0e-6, 1.0e-12)
    return SymLogNorm(linthresh=linear_threshold, vmin=-max_abs, vmax=max_abs, base=10)


def _positive_norm(values: np.ndarray) -> LogNorm | None:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return None
    return LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))


def plot_radiation_flux_time_energy(
    series: IonizationSeries,
    output_dir: Path,
    *,
    boundary: str,
    values: np.ndarray,
) -> Path:
    time_edges_ns = _axis_edges(series.times_s * 1.0e9)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    mesh = ax.pcolormesh(
        time_edges_ns,
        series.photon_group_boundaries_eV,
        values.T,
        shading="auto",
        cmap="coolwarm",
        norm=_signed_flux_norm(values),
    )
    ax.set_title(f"{series.run_name}: spectral net radiation flux at {boundary}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Photon energy (eV)")
    ax.set_yscale("log")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Net flux (+R direction) (J/s/cm2/eV)")
    return _save_figure(
        fig,
        output_dir
        / f"{_slugify(series.run_name)}_radiation_flux_{boundary.lower()}_time_energy_contour.png",
    )


def plot_radiation_loss_time_energy(
    series: IonizationSeries,
    output_dir: Path,
    *,
    boundary: str,
    values: np.ndarray,
) -> Path:
    time_edges_ns = _axis_edges(series.times_s * 1.0e9)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    norm = _positive_norm(values)
    mesh = ax.pcolormesh(
        time_edges_ns,
        series.photon_group_boundaries_eV,
        values.T,
        shading="auto",
        cmap="inferno",
        norm=norm,
    )
    ax.set_title(f"{series.run_name}: cumulative spectral radiation loss at {boundary}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Photon energy (eV)")
    ax.set_yscale("log")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Cumulative loss (J/cm2/eV)")
    return _save_figure(
        fig,
        output_dir
        / f"{_slugify(series.run_name)}_radiation_loss_{boundary.lower()}_time_energy_contour.png",
    )


def plot_series(series: IonizationSeries, output_dir: Path) -> list[Path]:
    return [
        plot_final_charge_zone(series, output_dir),
        plot_selected_zone_time_charge(series, output_dir),
        plot_dominant_charge_time_zone(series, output_dir),
        plot_radiation_flux_time_energy(
            series,
            output_dir,
            boundary="Rmin",
            values=series.radiation_flux_rmin_j_s_cm2_eV,
        ),
        plot_radiation_flux_time_energy(
            series,
            output_dir,
            boundary="Rmax",
            values=series.radiation_flux_rmax_j_s_cm2_eV,
        ),
        plot_radiation_loss_time_energy(
            series,
            output_dir,
            boundary="Rmin",
            values=series.radiation_loss_rmin_j_cm2_eV,
        ),
        plot_radiation_loss_time_energy(
            series,
            output_dir,
            boundary="Rmax",
            values=series.radiation_loss_rmax_j_cm2_eV,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot BPF-only charge-state ionization fraction data as contour figures."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="BPF files or directories to scan. Defaults to repository new_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated .npz files and figures.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in _discover_bpf_files(args.paths):
        series = load_ionization_series(path)
        npz_path = save_series_npz(series, output_dir)
        figure_paths = plot_series(series, output_dir)
        print(f"{path}")
        print(
            "  extracted ionization fractions: "
            f"snapshots={series.times_s.size}, zones={series.zone_centers_cm.shape[1]}, "
            f"charge_states={series.charge_states.size}"
        )
        print(
            "  extracted spectral radiation data: "
            f"photon_bins={series.photon_group_boundaries_eV.size - 1}, "
            "flux_units=J/s/cm2/eV, loss_units=J/cm2/eV"
        )
        print(f"  saved data: {npz_path}")
        for figure_path in figure_paths:
            print(f"  saved figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
