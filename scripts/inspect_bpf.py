from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "new_data"


@dataclass(frozen=True, slots=True)
class FortranRecord:
    index: int
    offset: int
    payload: bytes

    @property
    def length(self) -> int:
        return len(self.payload)


@dataclass(frozen=True, slots=True)
class BpfLayout:
    path: Path
    run_date: str
    run_clock: str
    record_count: int
    n_nodes: int
    n_zones: int
    n_freq_bins: int
    block_records: int
    n_snapshots: int
    trailing_records: int
    trailer_lengths: tuple[int, ...]
    first_time_s: float
    last_time_s: float
    first_cycle: int
    last_cycle: int


@dataclass(frozen=True, slots=True)
class ExoCheck:
    name: str
    passed: bool
    max_abs_error: float


@dataclass(frozen=True, slots=True)
class BpfRecordSpec:
    rel: int
    key: str
    dtype: str
    shape: str
    status: str
    unit: str = ""
    description: str = ""


BPF_FIXED_RECORD_SPECS: tuple[BpfRecordSpec, ...] = (
    BpfRecordSpec(0, "snapshot_header", "f64", "30", "known", "", "Snapshot metadata header."),
    BpfRecordSpec(1, "global_radiation_energy_summary", "f64", "70", "partial", "", "Global scalar diagnostics; selected entries match radiation flux/loss and energy-balance log values."),
    BpfRecordSpec(2, "global_source_summary", "f64", "n_freq_bins", "partial", "", "Global/source vector; first entries match laser source/exchange scalars."),
    BpfRecordSpec(3, "bpf_record_03", "f64", "n_zones", "unknown"),
    BpfRecordSpec(4, "node_position_cm", "f64", "n_nodes", "known", "cm", "Dynamic node coordinate."),
    BpfRecordSpec(5, "zone_mass", "f64", "n_zones", "known", "g/cm**X", "Zone mass in HELIOS geometry convention."),
    BpfRecordSpec(6, "ion_temperature_eV", "f64", "n_zones", "known", "eV"),
    BpfRecordSpec(7, "radiation_temperature_eV", "f64", "n_zones", "known", "eV"),
    BpfRecordSpec(8, "mass_density_g_cm3", "f64", "n_zones", "known", "g/cm3"),
    BpfRecordSpec(9, "mean_charge", "f64", "n_zones", "known", "", "Charge-state weighted mean charge."),
    BpfRecordSpec(10, "ion_pressure_j_cm3", "f64", "n_zones", "known", "J/cm3"),
    BpfRecordSpec(11, "radiation_cooling_j_g_s", "f64", "n_zones", "known", "J/g/s"),
    BpfRecordSpec(12, "radiation_heating_j_g_s", "f64", "n_zones", "known", "J/g/s"),
    BpfRecordSpec(13, "ion_energy_j_g", "f64", "n_zones", "known", "J/g"),
    BpfRecordSpec(14, "bpf_record_14", "f64", "n_zones+2", "known", "J/cm3"),
    BpfRecordSpec(15, "bpf_record_15", "f64", "n_nodes", "partial"),
    BpfRecordSpec(16, "interface_velocity_cm_s", "f64", "n_nodes", "known", "cm/s", "Interface velocity; zone average equals EXO/log fluid velocity."),
    BpfRecordSpec(17, "laser_deposition_j_g_s", "f64", "n_zones", "known", "J/g/s"),
    BpfRecordSpec(18, "bpf_record_18", "f64", "n_nodes", "partial"),
    BpfRecordSpec(19, "bpf_record_19", "f64", "n_nodes", "partial"),
    BpfRecordSpec(20, "artificial_viscosity_j_cm3", "f64", "n_zones", "known", "J/cm3"),
    BpfRecordSpec(21, "bpf_record_21", "f64", "n_nodes", "partial"),
    BpfRecordSpec(22, "bpf_record_22", "f64", "n_nodes", "partial"),
    BpfRecordSpec(23, "bpf_record_23", "f64", "n_nodes", "partial"),
    BpfRecordSpec(24, "bpf_record_24", "f64", "n_nodes", "partial"),
    BpfRecordSpec(25, "bpf_record_25", "f64", "n_nodes", "partial"),
    BpfRecordSpec(26, "bpf_record_26", "f64", "n_nodes", "partial"),
    BpfRecordSpec(27, "bpf_record_27", "f64", "n_nodes", "partial"),
    BpfRecordSpec(28, "bpf_record_28", "f64", "n_nodes", "partial"),
    BpfRecordSpec(29, "bpf_record_29", "f64", "n_zones", "unknown"),
    BpfRecordSpec(30, "planck_mean_opacity_absorption_cm2_g", "f64", "n_zones", "mapped", "cm2/g"),
    BpfRecordSpec(31, "planck_mean_opacity_emission_cm2_g", "f64", "n_zones", "mapped", "cm2/g"),
    BpfRecordSpec(32, "rosseland_mean_opacity_cm2_g", "f64", "n_zones", "mapped", "cm2/g"),
    BpfRecordSpec(33, "ion_density_cm3", "f64", "n_zones", "known", "1/cm3"),
    BpfRecordSpec(34, "boundary_net_flux_pair_j_s_cm2", "f64", "2", "known", "J/s/cm2", "Signed Rmin/Rmax boundary flux pair."),
    BpfRecordSpec(35, "bpf_record_35", "i32", "1", "mapped", "count"),
    BpfRecordSpec(36, "bpf_record_36", "f64", "2", "partial"),
    BpfRecordSpec(37, "region_net_cooling_rate_j_s_cm2", "f64", "1", "known", "J/s/cm2"),
    BpfRecordSpec(38, "region_ion_flux_boundary_j_s_cm2", "f64", "1", "known", "J/s/cm2"),
    BpfRecordSpec(39, "region_electron_flux_boundary_j_s_cm2", "f64", "1", "known", "J/s/cm2"),
    BpfRecordSpec(40, "bpf_record_40", "f64", "1", "partial"),
    BpfRecordSpec(41, "electron_temperature_eV", "f64", "n_zones", "known", "eV"),
    BpfRecordSpec(42, "electron_pressure_j_cm3", "f64", "n_zones", "known", "J/cm3"),
    BpfRecordSpec(43, "electron_density_cm3", "f64", "n_zones", "known", "1/cm3"),
    BpfRecordSpec(44, "bpf_record_44", "f64", "n_zones", "unknown"),
    BpfRecordSpec(45, "frequency_group_boundaries_eV", "f64", "n_freq_bins+1", "known", "eV"),
    BpfRecordSpec(46, "radiation_source_escaping_rmax_j_s_cm2_eV", "f64", "n_freq_bins", "known", "J/s/cm2/eV"),
    BpfRecordSpec(47, "radiation_source_escaping_rmin_j_s_cm2_eV", "f64", "n_freq_bins", "known", "J/s/cm2/eV"),
    BpfRecordSpec(48, "radiation_loss_rmax_j_cm2_eV", "f64", "n_freq_bins", "known", "J/cm2/eV"),
    BpfRecordSpec(49, "radiation_loss_rmin_j_cm2_eV", "f64", "n_freq_bins", "known", "J/cm2/eV"),
    BpfRecordSpec(50, "radiation_net_flux_rmin_j_s_cm2_eV", "f64", "n_freq_bins", "known", "J/s/cm2/eV"),
    BpfRecordSpec(51, "radiation_net_flux_rmax_j_s_cm2_eV", "f64", "n_freq_bins", "known", "J/s/cm2/eV"),
    BpfRecordSpec(52, "external_rad_source_rmin", "f64", "n_freq_bins", "known"),
    BpfRecordSpec(53, "external_rad_source_rmax", "f64", "n_freq_bins", "known"),
    BpfRecordSpec(54, "bpf_record_54", "i32", "1", "partial"),
    BpfRecordSpec(55, "bpf_record_55", "f64", "1", "partial"),
    BpfRecordSpec(56, "bpf_record_56", "i32", "1", "partial"),
    BpfRecordSpec(57, "region_index_by_zone", "i32", "n_zones", "known"),
    BpfRecordSpec(58, "bpf_record_58", "i32", "1", "partial"),
    BpfRecordSpec(59, "bpf_record_59", "i32", "1", "partial"),
    BpfRecordSpec(60, "bpf_record_60", "i32", "1", "partial"),
    BpfRecordSpec(61, "bpf_record_61", "i32", "1", "partial"),
    BpfRecordSpec(62, "ionization_fraction_width", "i32", "1", "known", "", "Z+1 charge-state count."),
    BpfRecordSpec(63, "bpf_record_63", "i32", "1", "partial"),
    BpfRecordSpec(64, "atomic_number_z", "i32", "1", "known"),
    BpfRecordSpec(65, "bpf_record_65", "i32", "1", "partial"),
    BpfRecordSpec(66, "bpf_record_66", "f64", "1", "partial"),
    BpfRecordSpec(67, "bpf_record_67", "i32", "1", "partial"),
    BpfRecordSpec(68, "ionization_fraction_width_repeat", "i32", "1", "known"),
)


def iter_fortran_records(path: Path) -> Iterable[FortranRecord]:
    data = path.read_bytes()
    offset = 0
    index = 0
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"Short Fortran record marker at byte {offset}.")
        length = struct.unpack_from("<i", data, offset)[0]
        payload_start = offset + 4
        payload_end = payload_start + length
        if length < 0 or payload_end + 4 > len(data):
            raise ValueError(
                f"Invalid record length {length} at record {index}, byte {offset}."
            )
        closing_length = struct.unpack_from("<i", data, payload_end)[0]
        if closing_length != length:
            raise ValueError(
                "Fortran record marker mismatch at "
                f"record {index}: {length} != {closing_length}."
            )
        yield FortranRecord(index=index, offset=offset, payload=data[payload_start:payload_end])
        offset = payload_end + 4
        index += 1


def read_fortran_records(path: Path) -> tuple[FortranRecord, ...]:
    return tuple(iter_fortran_records(path))


def _ascii_record(record: FortranRecord) -> str:
    return record.payload.decode("ascii", errors="replace").strip("\x00 ")


def _f64(record: FortranRecord) -> np.ndarray:
    if record.length % 8 != 0:
        raise ValueError(f"Record {record.index} is not float64-aligned.")
    return np.frombuffer(record.payload, dtype="<f8")


def _i32(record: FortranRecord) -> np.ndarray:
    if record.length % 4 != 0:
        raise ValueError(f"Record {record.index} is not int32-aligned.")
    return np.frombuffer(record.payload, dtype="<i4")


def infer_layout(path: Path, records: tuple[FortranRecord, ...]) -> BpfLayout:
    if len(records) < 3:
        raise ValueError(f"{path} has too few records to be a HELIOS BPF file.")
    header = _f64(records[2])
    if header.size < 7:
        raise ValueError(f"{path} first header record is too short.")
    n_nodes = int(round(float(header[4])))
    n_zones = n_nodes - 1
    n_freq_bins = int(round(float(header[6])))
    if n_nodes <= 1 or n_freq_bins <= 0:
        raise ValueError(
            f"{path} has implausible layout values: nodes={n_nodes}, "
            f"freq_bins={n_freq_bins}."
        )

    block_records = 69 + 2 * n_zones
    start = 2
    n_snapshots = 0
    while start + block_records <= len(records):
        n_snapshots += 1
        start += block_records
    if n_snapshots == 0:
        raise ValueError(f"{path} does not contain a complete snapshot block.")

    first_header = _f64(records[2])
    last_header = _f64(records[snapshot_start(n_snapshots - 1, block_records)])
    return BpfLayout(
        path=path,
        run_date=_ascii_record(records[0]),
        run_clock=_ascii_record(records[1]),
        record_count=len(records),
        n_nodes=n_nodes,
        n_zones=n_zones,
        n_freq_bins=n_freq_bins,
        block_records=block_records,
        n_snapshots=n_snapshots,
        trailing_records=len(records) - start,
        trailer_lengths=tuple(record.length for record in records[start:]),
        first_time_s=float(first_header[1]),
        last_time_s=float(last_header[1]),
        first_cycle=int(round(float(first_header[2]))),
        last_cycle=int(round(float(last_header[2]))),
    )


def snapshot_start(snapshot_index: int, block_records: int) -> int:
    return 2 + snapshot_index * block_records


def extract_common_snapshot(
    records: tuple[FortranRecord, ...],
    layout: BpfLayout,
    snapshot_index: int,
) -> dict[str, np.ndarray | float | int]:
    if snapshot_index < 0:
        snapshot_index += layout.n_snapshots
    if snapshot_index < 0 or snapshot_index >= layout.n_snapshots:
        raise IndexError(
            f"Snapshot index {snapshot_index} outside 0..{layout.n_snapshots - 1}."
        )

    start = snapshot_start(snapshot_index, layout.block_records)
    header = _f64(records[start])
    interface_velocity = _f64(records[start + 16])
    if interface_velocity.size != layout.n_nodes:
        raise ValueError(
            "Interface velocity record does not match node count: "
            f"{interface_velocity.size} != {layout.n_nodes}."
        )

    region_index = _i32(records[start + 57])
    ion_flags: list[int] = []
    ion_fractions: list[np.ndarray] = []
    ion_fraction_start = start + 69
    for zone_index in range(layout.n_zones):
        flag_record = records[ion_fraction_start + 2 * zone_index]
        fraction_record = records[ion_fraction_start + 2 * zone_index + 1]
        flag_values = _i32(flag_record)
        ion_flags.append(int(flag_values[0]) if flag_values.size else 0)
        ion_fractions.append(_f64(fraction_record))

    fraction_widths = {value.size for value in ion_fractions}
    if len(fraction_widths) != 1:
        raise ValueError(f"Ionization fraction widths are inconsistent: {fraction_widths}.")

    return {
        "snapshot_index": snapshot_index,
        "time_s": float(header[1]),
        "cycle": int(round(float(header[2]))),
        "header": header.copy(),
        "node_position_cm": _f64(records[start + 4]).copy(),
        "zone_mass": _f64(records[start + 5]).copy(),
        "ion_temperature_eV": _f64(records[start + 6]).copy(),
        "mass_density_g_cm3": _f64(records[start + 8]).copy(),
        "interface_velocity_cm_s": interface_velocity.copy(),
        "fluid_velocity_cm_s": 0.5 * (interface_velocity[:-1] + interface_velocity[1:]),
        "ion_density_cm3_inferred": _f64(records[start + 33]).copy(),
        "electron_temperature_eV": _f64(records[start + 41]).copy(),
        "frequency_group_boundaries_eV": _f64(records[start + 45]).copy(),
        "radiation_source_escaping_rmax_j_s_cm2_eV": _f64(records[start + 46]).copy(),
        "radiation_source_escaping_rmin_j_s_cm2_eV": _f64(records[start + 47]).copy(),
        "radiation_loss_rmax_j_cm2_eV": _f64(records[start + 48]).copy(),
        "radiation_loss_rmin_j_cm2_eV": _f64(records[start + 49]).copy(),
        "radiation_net_flux_rmin_j_s_cm2_eV": _f64(records[start + 50]).copy(),
        "radiation_net_flux_rmax_j_s_cm2_eV": _f64(records[start + 51]).copy(),
        "external_rad_source_rmin": _f64(records[start + 52]).copy(),
        "external_rad_source_rmax": _f64(records[start + 53]).copy(),
        "region_index_by_zone": region_index.copy(),
        "ion_fraction_flags": np.asarray(ion_flags, dtype=np.int32),
        "ionization_fractions": np.vstack(ion_fractions),
    }


def extract_all_snapshot_records(
    records: tuple[FortranRecord, ...],
    layout: BpfLayout,
    snapshot_index: int,
) -> dict[str, np.ndarray]:
    """Return every known-position record in one BPF snapshot.

    Unknown records are intentionally retained under stable ``bpf_record_XX``
    keys so a future parser can preserve data losslessly while names and units
    are still being validated.
    """

    if snapshot_index < 0:
        snapshot_index += layout.n_snapshots
    if snapshot_index < 0 or snapshot_index >= layout.n_snapshots:
        raise IndexError(
            f"Snapshot index {snapshot_index} outside 0..{layout.n_snapshots - 1}."
        )

    start = snapshot_start(snapshot_index, layout.block_records)
    values: dict[str, np.ndarray] = {}
    for spec in BPF_FIXED_RECORD_SPECS:
        record = records[start + spec.rel]
        if spec.dtype == "f64":
            values[spec.key] = _f64(record).copy()
        elif spec.dtype == "i32":
            values[spec.key] = _i32(record).copy()
        else:
            values[spec.key] = np.frombuffer(record.payload, dtype=np.uint8).copy()

    ion_flags: list[np.ndarray] = []
    ion_fractions: list[np.ndarray] = []
    ion_fraction_start = start + 69
    for zone_index in range(layout.n_zones):
        ion_flags.append(_i32(records[ion_fraction_start + 2 * zone_index]).copy())
        ion_fractions.append(_f64(records[ion_fraction_start + 2 * zone_index + 1]).copy())
    values["ionization_fraction_flags_by_zone"] = np.vstack(ion_flags)
    values["ionization_fractions_by_zone_charge"] = np.vstack(ion_fractions)
    return values


def _check_close(name: str, actual: np.ndarray, expected: np.ndarray) -> ExoCheck:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return ExoCheck(name, False, float("inf"))
    error = np.abs(actual - expected)
    max_error = float(np.nanmax(error)) if error.size else 0.0
    return ExoCheck(name, bool(np.allclose(actual, expected)), max_error)


def validate_against_exo(
    records: tuple[FortranRecord, ...],
    layout: BpfLayout,
    exo_path: Path,
) -> tuple[ExoCheck, ...]:
    try:
        from scipy.io import netcdf_file
    except Exception as exc:  # pragma: no cover - optional local dependency
        return (ExoCheck(f"EXO validation unavailable: {exc}", False, float("inf")),)

    checks: list[ExoCheck] = []
    with netcdf_file(str(exo_path), "r", mmap=False) as exo:
        time_whole = np.asarray(exo.variables["time_whole"].data, dtype=np.float64)
        sample_indices = sorted({0, layout.n_snapshots // 2, layout.n_snapshots - 1})
        for index in sample_indices:
            fields = extract_common_snapshot(records, layout, index)
            checks.append(
                _check_close(
                    f"time_whole[{index}]",
                    np.asarray([fields["time_s"]], dtype=np.float64),
                    np.asarray([time_whole[index]], dtype=np.float64),
                )
            )
            checks.append(
                _check_close(
                    f"node_position_cm[{index}]",
                    fields["node_position_cm"],
                    exo.variables["vals_nod_var"].data[index, 0, :],
                )
            )
            checks.append(
                _check_close(
                    f"mass_density_g_cm3[{index}]",
                    fields["mass_density_g_cm3"],
                    exo.variables["vals_elem_var2eb1"].data[index, :],
                )
            )
            checks.append(
                _check_close(
                    f"electron_temperature_eV[{index}]",
                    fields["electron_temperature_eV"],
                    exo.variables["vals_elem_var3eb1"].data[index, :],
                )
            )
            checks.append(
                _check_close(
                    f"ion_temperature_eV[{index}]",
                    fields["ion_temperature_eV"],
                    exo.variables["vals_elem_var4eb1"].data[index, :],
                )
            )
            checks.append(
                _check_close(
                    f"fluid_velocity_cm_s[{index}]",
                    fields["fluid_velocity_cm_s"],
                    exo.variables["vals_elem_var5eb1"].data[index, :],
                )
            )
            checks.append(
                _check_close(
                    f"external_rad_source_rmin[{index}]",
                    fields["external_rad_source_rmin"],
                    exo.variables["vals_fd_var1"].data[index, :],
                )
            )
            checks.append(
                _check_close(
                    f"external_rad_source_rmax[{index}]",
                    fields["external_rad_source_rmax"],
                    exo.variables["vals_fd_var2"].data[index, :],
                )
            )
        first_fields = extract_common_snapshot(records, layout, 0)
        checks.append(
            _check_close(
                "frequency_group_boundaries_eV",
                first_fields["frequency_group_boundaries_eV"],
                exo.variables["FREQ_GROUP_BOUNDARIES"].data[:],
            )
        )
    return tuple(checks)


def discover_bpf_paths(inputs: list[str]) -> list[Path]:
    paths = [Path(value) for value in inputs] if inputs else [DEFAULT_DATA_ROOT]
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.bpf")))
        elif path.is_file() and path.suffix.lower() == ".bpf":
            discovered.append(path)
        else:
            raise FileNotFoundError(f"No BPF file or directory found at {path}.")
    return discovered


def print_layout(layout: BpfLayout) -> None:
    print(f"BPF: {layout.path}")
    print(f"  written: {layout.run_date} {layout.run_clock}")
    print(
        "  layout: "
        f"records={layout.record_count}, snapshots={layout.n_snapshots}, "
        f"block_records={layout.block_records}, trailing_records={layout.trailing_records}"
    )
    print(
        "  mesh/frequency: "
        f"zones={layout.n_zones}, nodes={layout.n_nodes}, "
        f"frequency_bins={layout.n_freq_bins}"
    )
    print(
        "  time/cycle: "
        f"{layout.first_time_s:.12g}s cycle {layout.first_cycle} -> "
        f"{layout.last_time_s:.12g}s cycle {layout.last_cycle}"
    )
    if layout.trailer_lengths:
        print(f"  trailer record lengths: {layout.trailer_lengths}")


def print_field_specs() -> None:
    print("Fixed per-snapshot BPF record map:")
    for spec in BPF_FIXED_RECORD_SPECS:
        unit = f", unit={spec.unit}" if spec.unit else ""
        description = f" - {spec.description}" if spec.description else ""
        print(
            f"  rel={spec.rel:02d} key={spec.key} dtype={spec.dtype} "
            f"shape={spec.shape} status={spec.status}{unit}{description}"
        )
    print("  rel=69+2*z key=ionization_fraction_flags_by_zone dtype=i32 shape=(n_zones, 1)")
    print(
        "  rel=70+2*z key=ionization_fractions_by_zone_charge "
        "dtype=f64 shape=(n_zones, Z+1)"
    )


def save_npz(path: Path, fields: dict[str, np.ndarray | float | int]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, value in fields.items():
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect HELIOS .bpf binary plot files and extract validated fields."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="BPF files or directories to scan. Defaults to the repository new_data folder.",
    )
    parser.add_argument(
        "--validate-exo",
        action="store_true",
        help="Validate common BPF fields against a same-stem .exo file when present.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="Print the fixed BPF snapshot record map.",
    )
    parser.add_argument(
        "--extract-snapshot",
        type=int,
        help="Extract validated fields for one snapshot index. Negative indices are allowed.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="Write extracted snapshot fields to a compressed .npz file.",
    )
    args = parser.parse_args(argv)

    if args.list_fields:
        print_field_specs()
        if not args.paths:
            return 0

    paths = discover_bpf_paths(args.paths)
    if args.output_npz is not None and (args.extract_snapshot is None or len(paths) != 1):
        parser.error("--output-npz requires --extract-snapshot and exactly one BPF file.")

    for path in paths:
        records = read_fortran_records(path)
        layout = infer_layout(path, records)
        print_layout(layout)

        if args.extract_snapshot is not None:
            fields = extract_common_snapshot(records, layout, args.extract_snapshot)
            print(
                "  extracted snapshot: "
                f"index={fields['snapshot_index']}, time={fields['time_s']:.12g}s, "
                f"cycle={fields['cycle']}"
            )
            print(
                "  extracted fields: "
                + ", ".join(
                    key
                    for key in fields
                    if key not in {"snapshot_index", "time_s", "cycle"}
                )
            )
            if args.output_npz is not None:
                save_npz(args.output_npz, fields)
                print(f"  wrote: {args.output_npz}")

        if args.validate_exo:
            exo_path = path.with_suffix(".exo")
            if not exo_path.exists():
                print(f"  EXO validation: skipped, missing {exo_path}")
            else:
                checks = validate_against_exo(records, layout, exo_path)
                passed = sum(1 for check in checks if check.passed)
                print(f"  EXO validation: {passed}/{len(checks)} checks passed")
                for check in checks:
                    status = "PASS" if check.passed else "FAIL"
                    print(
                        f"    {status} {check.name} "
                        f"max_abs_error={check.max_abs_error:.6g}"
                    )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
