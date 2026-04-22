"""Semantic HELIOS log parser built on the indexed document architecture.

The active production path is:

    buffer -> structural index -> semantic header/snapshot parse

This module also retains explicit compatibility handling for older and newer
observed HELIOS table layouts. Those compatibility branches are intentional and
remain part of the supported parser surface for the current file set.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from helios.instrumentation import timed_block
from .coordinates import CoordinateValidationIssue, build_coordinate_model, coordinate_name_for_geometry
from .model import HeliosHeader, HeliosPreview, Simulation, Snapshot
from .tokenizer import RegexTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_TOKENIZER = RegexTokenizer()
NUMBER_TOKEN_RE = DEFAULT_TOKENIZER.number_pattern
CYCLE_HEADER_RE = DEFAULT_TOKENIZER.cycle_header_pattern
BLOCK_DELIMITER = "--------------------------------------------------------------------"
# Compatibility table layouts for the observed HELIOS variants. These are kept
# intentionally so older logs and newer extra-column logs stay on one parser.
KNOWN_TABLE_LAYOUTS: dict[tuple[str, int], tuple[list[str], list[str]]] = {
    ("primary", 14): (
        [
            "Radius",
            "Zone width",
            "Mass dens",
            "Velocity",
            "Rad temp",
            "Ion temp",
            "Elec temp",
            "Ion press",
            "Elec press",
            "Rad press",
            "Compression",
            "Elec Dens",
            "Mean Chg",
            "Art. Visc.",
        ],
        [
            "cm",
            "cm",
            "g/cm3",
            "cm/s",
            "eV",
            "eV",
            "eV",
            "J/cm3",
            "J/cm3",
            "J/cm3",
            "rho/rho0",
            "1/cm3",
            "",
            "J/cm3",
        ],
    ),
    ("secondary", 11): (
        [
            "Ion energy",
            "Ele energy",
            "IonHeatCap",
            "EleHeatCap",
            "Rad energy",
            "Kin energy",
            "RadHeating",
            "RadCooling",
            "RadNetHeat",
            "LaserSrc",
            "LaserDep",
        ],
        ["J/g", "J/g", "J/g/eV", "J/g/eV", "J/g", "J/g", "J/g/s", "J/g/s", "J/g/s", "J/g", "J/g/s"],
    ),
    ("secondary", 12): (
        [
            "Ion energy",
            "Ele energy",
            "IonHeatCap",
            "EleHeatCap",
            "Rad energy",
            "Kin energy",
            "RadHeating",
            "RadCooling",
            "RadSink",
            "RadNetHeat",
            "LaserSrc",
            "LaserDep",
        ],
        ["J/g", "J/g", "J/g/eV", "J/g/eV", "J/g", "J/g", "J/g/s", "J/g/s", "J/g/s", "J/g/s", "J/g", "J/g/s"],
    ),
}


def _normalize_label(label: str) -> str:
    return DEFAULT_TOKENIZER.normalize_label(label)


def _normalize_unit(unit: str) -> str:
    return DEFAULT_TOKENIZER.normalize_unit(unit)


def _normalize_identifier(label: str) -> str:
    return DEFAULT_TOKENIZER.normalize_identifier(label)


def _canonical_diagnostic_name(label: str) -> str:
    return DEFAULT_TOKENIZER.canonical_diagnostic_name(label)


def _extract_number_tokens(line: str) -> list[str]:
    return DEFAULT_TOKENIZER.extract_number_tokens(line)


def _canonical_field_name(label: str) -> str:
    return DEFAULT_TOKENIZER.canonical_field_name(label)


def _is_int_token(token: str) -> bool:
    if not token:
        return False
    if token[0] in "+-":
        return token[1:].isdigit()
    return token.isdigit()


def _is_zero_int_token(token: str) -> bool:
    return _is_int_token(token) and token.lstrip("+-") == "0"


def _extract_first(lines: Iterable[str], pattern: str, cast=str, default=None):
    regex = re.compile(pattern)
    for line in lines:
        match = regex.search(line)
        if match:
            return cast(match.group(1))
    return default


def _find_line_index(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if needle in line:
            return index
    return -1


def _slice_between(lines: list[str], start_needle: str, stop_needles: tuple[str, ...]) -> list[str]:
    start = _find_line_index(lines, start_needle)
    if start < 0:
        return []
    collected: list[str] = []
    for line in lines[start + 1 :]:
        if any(needle in line for needle in stop_needles):
            break
        collected.append(line.rstrip("\n"))
    return collected


def _coerce_value(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(NUMBER_TOKEN_RE.pattern, stripped):
        return float(_extract_number_tokens(stripped)[0])
    return stripped


def _parse_simple_table(
    lines: list[str],
    section_label: str,
    row_regex: re.Pattern[str],
    column_names: tuple[str, ...],
) -> list[dict[str, object]]:
    start = _find_line_index(lines, section_label)
    if start < 0:
        return []
    rows: list[dict[str, object]] = []
    for line in lines[start + 1 :]:
        if rows and not line.strip():
            break
        match = row_regex.match(line)
        if not match:
            continue
        row: dict[str, object] = {}
        for key, value in zip(column_names, match.groups()):
            if key.endswith("_index") or key == "index":
                row[key] = int(value)
            elif key.endswith("_path") or key.endswith("_model"):
                row[key] = value.strip()
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _parse_float_block(lines: list[str], section_label: str, stop_label: str) -> np.ndarray:
    start = _find_line_index(lines, section_label)
    if start < 0:
        return np.array([], dtype=np.float64)
    values: list[float] = []
    for line in lines[start + 1 :]:
        if stop_label in line:
            break
        if not line.strip():
            continue
        values.extend(float(token) for token in _extract_number_tokens(line))
    return np.asarray(values, dtype=np.float64)


def _parse_field_table_rows(
    data_lines: list[str],
    header_line: str,
    units_line: str,
    n_zones: int,
    logger: logging.Logger,
    *,
    geometry: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str], dict[str, np.ndarray | float | None]]:
    def extract_zone_zero_boundary() -> float | None:
        for raw_line in data_lines:
            stripped = raw_line.lstrip()
            if not stripped:
                continue
            first_space = stripped.find(" ")
            first_token = stripped if first_space < 0 else stripped[:first_space]
            if not _is_zero_int_token(first_token):
                continue
            tokens = DEFAULT_TOKENIZER.split_table_row_tokens(stripped)
            if len(tokens) >= 2:
                try:
                    return float(tokens[1])
                except ValueError:
                    pass
            extracted = _extract_number_tokens(stripped)
            if len(extracted) >= 2 and _is_zero_int_token(extracted[0]):
                try:
                    return float(extracted[1])
                except ValueError:
                    return None
        return None

    def block_requires_normalization(block_text: str) -> bool:
        return (
            DEFAULT_TOKENIZER.malformed_exponent_pattern.search(block_text) is not None
            or "D" in block_text
            or "d" in block_text
        )

    layout_key = "secondary" if "Ion energy" in header_line else "primary"
    if layout_key == "primary":
        known_layout = KNOWN_TABLE_LAYOUTS.get(("primary", 14))
    else:
        known_layout = KNOWN_TABLE_LAYOUTS.get(("secondary", 12 if "RadSink" in header_line else 11))
    if known_layout is not None:
        raw_labels, raw_units = known_layout
        column_count = len(raw_labels)
    else:
        sample_line = ""
        sample_tokens: list[str] = []
        best_token_count = -1
        for line in data_lines:
            tokens = line.split()
            if not tokens or not _is_int_token(tokens[0]):
                continue
            if len(tokens) > best_token_count:
                best_token_count = len(tokens)
                sample_line = line
                sample_tokens = tokens
        if not sample_line:
            raise ValueError("Failed to infer table layout from data rows.")

        column_count = len(sample_tokens) - 1
        token_spans = list(NUMBER_TOKEN_RE.finditer(sample_line))
        value_spans = token_spans[1:]
        padded_header = header_line.ljust(len(sample_line))
        padded_units = units_line.ljust(len(sample_line))
        raw_labels, raw_units = [], []
        for index, start in enumerate(match.start() for match in value_spans):
            end = value_spans[index + 1].start() if index + 1 < column_count else len(sample_line)
            raw_labels.append(_normalize_label(padded_header[start:end]))
            raw_units.append(_normalize_unit(padded_units[start:end]))

    canonical_labels = [_canonical_field_name(label) for label in raw_labels]
    units = {name: unit for name, unit in zip(canonical_labels, raw_units)}
    raw_field_map = {name: label for name, label in zip(canonical_labels, raw_labels)}
    normalize_number_token = DEFAULT_TOKENIZER.normalize_number_token
    normalize_numeric_text = DEFAULT_TOKENIZER.normalize_numeric_text
    extract_number_tokens = _extract_number_tokens
    expected_row_width = column_count + 1
    extras: dict[str, np.ndarray | float | None | int | tuple[dict[str, str], ...]] = {
        "parsed_row_count": 0,
        "expected_row_count": int(n_zones),
        "padded_row_count": 0,
        "truncated_row_count": 0,
        "coordinate_issues": (),
    }

    filtered_lines: list[str] = []
    for raw_line in data_lines:
        stripped = raw_line.lstrip()
        if not stripped:
            continue
        first_space = stripped.find(" ")
        first_token = stripped if first_space < 0 else stripped[:first_space]
        if not _is_int_token(first_token) or _is_zero_int_token(first_token):
            continue
        filtered_lines.append(stripped)

    if filtered_lines:
        block_text = "\n".join(filtered_lines)
        parsed_block = np.empty(0, dtype=np.float64)
        if not block_requires_normalization(block_text):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                parsed_block = np.fromstring(block_text, sep=" ", dtype=np.float64)
        if parsed_block.size != len(filtered_lines) * expected_row_width:
            parsed_block = np.fromstring(normalize_numeric_text(block_text), sep=" ", dtype=np.float64)
        if parsed_block.size == len(filtered_lines) * expected_row_width:
            rows = parsed_block.reshape((-1, expected_row_width))
            zone_indices = rows[:, 0].astype(np.int32)
            valid = (zone_indices >= 1) & (zone_indices <= n_zones)
            matrix = np.full((n_zones, column_count), np.nan, dtype=np.float64)
            matrix[zone_indices[valid] - 1, :] = rows[valid, 1:]
            parsed_rows = int(np.count_nonzero(valid))
            if parsed_rows != n_zones:
                logger.warning("Parsed %s zone rows for a table, expected %s.", parsed_rows, n_zones)
            arrays = {name: matrix[:, index] for index, name in enumerate(canonical_labels)}
            if layout_key == "primary" and "radius" in arrays and "zone_width" in arrays:
                boundary_edge = extract_zone_zero_boundary()
                coordinate_issues: list[CoordinateValidationIssue] = []
                coordinate_edge, coordinate_center = build_coordinate_model(
                    arrays["radius"],
                    arrays["zone_width"],
                    boundary_edge=boundary_edge,
                    geometry=geometry,
                    issues=coordinate_issues,
                )
                extras = {
                    "coordinate_edge": coordinate_edge,
                    "coordinate_center": coordinate_center,
                    "boundary_edge": boundary_edge,
                    "parsed_row_count": int(parsed_rows),
                    "expected_row_count": int(n_zones),
                    "padded_row_count": 0,
                    "truncated_row_count": 0,
                    "coordinate_issues": tuple(
                        {"code": issue.code, "message": issue.message, "severity": issue.severity}
                        for issue in coordinate_issues
                    ),
                }
                arrays["radius"] = coordinate_center
                raw_field_map["radius"] = "Radius (zone-center alias derived from HELIOS edge coordinates)"
            else:
                extras["parsed_row_count"] = int(parsed_rows)
            return arrays, units, raw_field_map, extras

    matrix = np.full((n_zones, column_count), np.nan, dtype=np.float64)

    parsed_rows = 0
    padded_row_count = 0
    truncated_row_count = 0
    for raw_line in data_lines:
        tokens = DEFAULT_TOKENIZER.split_table_row_tokens(raw_line)
        if not tokens or not _is_int_token(tokens[0]):
            continue
        zone_index = int(tokens[0])
        if zone_index < 1 or zone_index > n_zones:
            continue
        values = tokens[1:]
        if len(values) != column_count:
            tokens = extract_number_tokens(raw_line)
            if not tokens or not _is_int_token(tokens[0]):
                continue
            values = tokens[1:]
        if len(values) < column_count:
            logger.warning("Row for zone %s has %s values, expected %s. Padding with NaN.", zone_index, len(values), column_count)
            values = values + ["nan"] * (column_count - len(values))
            padded_row_count += 1
        elif len(values) > column_count:
            logger.warning("Row for zone %s has %s values, expected %s. Truncating extras.", zone_index, len(values), column_count)
            values = values[:column_count]
            truncated_row_count += 1
        row = matrix[zone_index - 1]
        for column_index, token in enumerate(values):
            row[column_index] = float(normalize_number_token(token))
        parsed_rows += 1

    if parsed_rows != n_zones:
        logger.warning("Parsed %s zone rows for a table, expected %s.", parsed_rows, n_zones)
    arrays = {name: matrix[:, index] for index, name in enumerate(canonical_labels)}
    if layout_key == "primary" and "radius" in arrays and "zone_width" in arrays:
        boundary_edge = extract_zone_zero_boundary()
        coordinate_issues: list[CoordinateValidationIssue] = []
        coordinate_edge, coordinate_center = build_coordinate_model(
            arrays["radius"],
            arrays["zone_width"],
            boundary_edge=boundary_edge,
            geometry=geometry,
            issues=coordinate_issues,
        )
        extras = {
            "coordinate_edge": coordinate_edge,
            "coordinate_center": coordinate_center,
            "boundary_edge": boundary_edge,
            "parsed_row_count": int(parsed_rows),
            "expected_row_count": int(n_zones),
            "padded_row_count": int(padded_row_count),
            "truncated_row_count": int(truncated_row_count),
            "coordinate_issues": tuple(
                {"code": issue.code, "message": issue.message, "severity": issue.severity}
                for issue in coordinate_issues
            ),
        }
        arrays["radius"] = coordinate_center
        raw_field_map["radius"] = "Radius (zone-center alias derived from HELIOS edge coordinates)"
    else:
        extras["parsed_row_count"] = int(parsed_rows)
        extras["expected_row_count"] = int(n_zones)
        extras["padded_row_count"] = int(padded_row_count)
        extras["truncated_row_count"] = int(truncated_row_count)
    return arrays, units, raw_field_map, extras


def _parse_key_value_lines(lines: Iterable[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("Index"):
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
        elif "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            continue
        values[_normalize_identifier(key)] = _coerce_value(value)
    return values


def _merge_material_tables(
    material_rows: list[dict[str, object]],
    opacity_rows: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    # The parsed EOS/opacity tables are intentionally preserved verbatim because
    # downstream tooling uses these fields as the practical material-identity
    # contract when no explicit chemical-formula field exists in the source log.
    eos_by_index = {int(row["index"]): row for row in material_rows}
    opacity_by_index = {int(row["index"]): row for row in opacity_rows}
    indices = sorted(set(eos_by_index) | set(opacity_by_index))
    materials = {
        "index": np.asarray(indices, dtype=np.int32),
        "eos_model": np.asarray([str(eos_by_index.get(i, {}).get("eos_model", "")) for i in indices], dtype=object),
        "eos_file_path": np.asarray([str(eos_by_index.get(i, {}).get("file_path", "")) for i in indices], dtype=object),
        "opacity_model": np.asarray([str(opacity_by_index.get(i, {}).get("opacity_model", "")) for i in indices], dtype=object),
        "opacity_file_path": np.asarray([str(opacity_by_index.get(i, {}).get("file_path", "")) for i in indices], dtype=object),
    }
    return materials, {name: "" for name in materials}


def _rows_to_region_arrays(
    spatial_regions: list[dict[str, object]],
    region_masses: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    region_by_index = {int(row["region_index"]): row for row in spatial_regions}
    mass_by_index = {int(row["region_index"]): float(row["mass"]) for row in region_masses}
    indices = sorted(region_by_index)
    regions = {
        "region_index": np.asarray(indices, dtype=np.int32),
        "min_zone_index": np.asarray([int(region_by_index[i]["min_index"]) for i in indices], dtype=np.int32),
        "max_zone_index": np.asarray([int(region_by_index[i]["max_index"]) for i in indices], dtype=np.int32),
        "material_index": np.asarray([int(region_by_index[i]["material_index"]) for i in indices], dtype=np.int32),
        "material_table_index": np.asarray([abs(int(region_by_index[i]["material_index"])) for i in indices], dtype=np.int32),
        "atomic_weight": np.asarray([float(region_by_index[i]["atomic_weight"]) for i in indices], dtype=np.float64),
        "initial_temperature": np.asarray([float(region_by_index[i]["temperature"]) for i in indices], dtype=np.float64),
        "initial_mass_density": np.asarray([float(region_by_index[i]["mass_density"]) for i in indices], dtype=np.float64),
        "initial_ion_density": np.asarray([float(region_by_index[i]["ion_density"]) for i in indices], dtype=np.float64),
        "initial_velocity": np.asarray([float(region_by_index[i]["velocity"]) for i in indices], dtype=np.float64),
        "mass": np.asarray([mass_by_index.get(i, np.nan) for i in indices], dtype=np.float64),
    }
    units = {
        "region_index": "",
        "min_zone_index": "",
        "max_zone_index": "",
        "material_index": "",
        "material_table_index": "",
        "atomic_weight": "amu",
        "initial_temperature": "eV",
        "initial_mass_density": "g/cm**3",
        "initial_ion_density": "ions/cm**3",
        "initial_velocity": "cm/s",
        "mass": "g/cm**X",
    }
    return regions, units


def _parse_frequency_gridding(lines: list[str]) -> dict[str, np.ndarray]:
    rows = _parse_simple_table(
        lines,
        "Frequency gridding parameters:",
        re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s*$"),
        ("section", "number_of_groups", "min_photon_energy", "max_photon_energy"),
    )
    if not rows:
        return {}
    return {
        "section": np.asarray([int(row["section"]) for row in rows], dtype=np.int32),
        "number_of_groups": np.asarray([int(row["number_of_groups"]) for row in rows], dtype=np.int32),
        "min_photon_energy": np.asarray([float(row["min_photon_energy"]) for row in rows], dtype=np.float64),
        "max_photon_energy": np.asarray([float(row["max_photon_energy"]) for row in rows], dtype=np.float64),
    }


def _parse_time_step_controls(lines: list[str]) -> dict[str, float]:
    block = _slice_between(lines, "Time step controls (dX/X):", ("Photon energy grid:",))
    controls: dict[str, float] = {}
    for line in block:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = _normalize_identifier(key)
        if key:
            controls[key] = float(_extract_number_tokens(value)[0])
    return controls


def _parse_hydro_input(lines: list[str]) -> dict[str, Any]:
    quiet_start_disabled = any("Quiet start is disabled." in line for line in lines)
    return {
        "plasma_model": _extract_first(lines, r"Plasma model:\s*(.+)", lambda value: value.strip()),
        "boundary_temperature_rmin": _extract_first(lines, r"Boundary temperature at Rmin \(eV\):\s*([0-9.E+-]+)", float),
        "boundary_temperature_rmax": _extract_first(lines, r"Boundary temperature at Rmax \(eV\):\s*([0-9.E+-]+)", float),
        "quiet_start_enabled": not quiet_start_disabled,
        "quiet_start_temperature": np.nan if quiet_start_disabled else _extract_first(
            lines,
            r"Quiet start temperature \(eV\):\s*([0-9.E+-]+)",
            float,
        ),
    }


def _parse_boundary_source_block(block_lines: list[str]) -> dict[str, Any]:
    stripped = [line.strip() for line in block_lines if line.strip()]
    if not stripped:
        return {}
    if any("No radiation source." in line for line in stripped):
        return {"has_source": False}
    parsed = _parse_key_value_lines(stripped)
    parsed["has_source"] = True
    return parsed


def _parse_radiation_source_input(lines: list[str]) -> dict[str, Any]:
    return {
        "rmin": _parse_boundary_source_block(
            _slice_between(lines, "Parameters at Rmin:", ("Parameters at Rmax:", "LASER SOURCE PARAMETERS:", "TIME CONTROL PARAMETERS:"))
        ),
        "rmax": _parse_boundary_source_block(
            _slice_between(lines, "Parameters at Rmax:", ("LASER SOURCE PARAMETERS:", "TIME CONTROL PARAMETERS:"))
        ),
    }


def _parse_laser_input(lines: list[str], laser_power: list[dict[str, object]]) -> dict[str, Any]:
    data = {
        "wavelength": _extract_first(lines, r"Laser wavelength \(microns\)\s*=\s*([0-9.E+-]+)", float),
        "incident_angle_cosine": _extract_first(lines, r"Cosine of incident angle\s*=\s*([0-9.E+-]+)", float),
        "propagation_direction": _extract_first(lines, r"Laser propagates toward\s+(\S+)", str),
        "origin_zone_index": _extract_first(lines, r"Originates in zone index\s*=\s*(\d+)", int),
    }
    data["power_table"] = {
        "index": np.asarray([int(row["index"]) for row in laser_power], dtype=np.int32),
        "time": np.asarray([float(row["time"]) for row in laser_power], dtype=np.float64),
        "power": np.asarray([float(row["power"]) for row in laser_power], dtype=np.float64),
    } if laser_power else {}
    return data


def _parse_time_control_input(lines: list[str]) -> dict[str, Any]:
    return {
        "max_cycle_count": _extract_first(lines, r"Max\. cycle count\s*=\s*(\d+)", int),
        "max_simulation_time": _extract_first(lines, r"Max\. simulation time \(sec\)\s*=\s*([0-9.E+-]+)", float),
        "initial_time_step": _extract_first(lines, r"Initial time step \(sec\)\s*=\s*([0-9.E+-]+)", float),
        "min_time_step": _extract_first(lines, r"Min\. time step \(sec\)\s*=\s*([0-9.E+-]+)", float),
        "max_time_step": _extract_first(lines, r"Max\. time step \(sec\)\s*=\s*([0-9.E+-]+)", float),
        "time_step_controls": _parse_time_step_controls(lines),
    }


def _parse_radiative_transfer_input(lines: list[str], photon_energy_grid: np.ndarray) -> dict[str, Any]:
    return {
        "number_of_frequency_groups": _extract_first(lines, r"Number of frequency groups\s*=\s*(\d+)", int),
        "radiation_transport_model": _extract_first(lines, r"Radiation transport model:\s*(.+)", lambda value: value.strip()),
        "frequency_gridding": _parse_frequency_gridding(lines),
        "photon_energy_grid": photon_energy_grid,
    }


def _parse_radiation_boundary_fluxes(lines: list[str], n_regions: int) -> dict[str, Any]:
    start = _find_line_index(lines, "Radiation Cooling Rates and Boundary Fluxes")
    if start < 0:
        return {}
    cooling = np.full(n_regions, np.nan, dtype=np.float64)
    flux = np.full(n_regions, np.nan, dtype=np.float64)
    ion_flux = np.full(n_regions, np.nan, dtype=np.float64)
    ele_flux = np.full(n_regions, np.nan, dtype=np.float64)
    terminal = np.nan
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if np.isfinite(cooling).any() or np.isfinite(terminal):
                break
            continue
        if "Energy Conservation Check" in stripped:
            break
        tokens = _extract_number_tokens(line)
        if not tokens:
            continue
        if re.fullmatch(r"\d+", tokens[0]) and len(tokens) >= 5:
            region = int(tokens[0])
            if 1 <= region <= n_regions:
                cooling[region - 1] = float(tokens[1])
                flux[region - 1] = float(tokens[2])
                ion_flux[region - 1] = float(tokens[3])
                ele_flux[region - 1] = float(tokens[4])
            continue
        if len(tokens) == 1:
            terminal = float(tokens[0])
    return {
        "region_net_cooling_rate": cooling,
        "region_net_flux_at_boundary": flux,
        "region_ion_flux_at_boundary": ion_flux,
        "region_electron_flux_at_boundary": ele_flux,
        "terminal_net_flux_at_boundary": terminal,
    }


def _parse_energy_summary(lines: list[str]) -> dict[str, Any]:
    start = _find_line_index(lines, "ENERGY SUMMARY:")
    if start < 0:
        return {}
    summary = {"initial": {}, "current": {}}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if summary["initial"]:
                break
            continue
        if "ENERGY EXCHANGE SUMMARY:" in stripped:
            break
        match = re.match(r"^\s*(.+?):\s*([0-9.E+-]+)\s+([0-9.E+-]+)\s*$", line)
        if not match:
            continue
        name = _canonical_diagnostic_name(match.group(1))
        summary["initial"][name] = float(_extract_number_tokens(match.group(2))[0])
        summary["current"][name] = float(_extract_number_tokens(match.group(3))[0])
    return summary


def _parse_energy_exchange(lines: list[str]) -> dict[str, Any]:
    start = _find_line_index(lines, "ENERGY EXCHANGE SUMMARY:")
    if start < 0:
        return {}
    section_map = {
        "Energy Sources To Plasma:": "sources_to_plasma",
        "Energy Lost From Grid:": "lost_from_grid",
        "Energy Exchanged Between Components:": "between_components",
    }
    exchange = {value: {} for value in section_map.values()}
    current_section: str | None = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "ENERGY BALANCE SUMMARY:" in stripped:
            break
        if stripped in section_map:
            current_section = section_map[stripped]
            continue
        match = re.match(r"^\s*(.+?):\s*([0-9.E+-]+)\s+([0-9.E+-]+)\s*$", line)
        if current_section is None or not match:
            continue
        name = _canonical_diagnostic_name(match.group(1))
        exchange[current_section][name] = {
            "total": float(_extract_number_tokens(match.group(2))[0]),
            "last_step": float(_extract_number_tokens(match.group(3))[0]),
        }
    return exchange


def _parse_energy_balance(lines: list[str]) -> dict[str, Any]:
    start = _find_line_index(lines, "ENERGY BALANCE SUMMARY:")
    if start < 0:
        return {}
    balance = {"initial": {}, "current": {}, "initial_plus_gains_minus_losses": {}}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if balance["initial"]:
                break
            continue
        matches = list(NUMBER_TOKEN_RE.finditer(line))
        if "=" not in line or len(matches) != 3:
            continue
        labels = []
        previous_end = 0
        for match in matches:
            labels.append(line[previous_end:match.start()].split("=")[0].strip())
            previous_end = match.end()
        values = [float(match.group(0).replace("D", "E")) for match in matches]
        keys = [_canonical_diagnostic_name(label) for label in labels]
        balance["initial"][keys[0]] = values[0]
        balance["current"][keys[1]] = values[1]
        balance["initial_plus_gains_minus_losses"][keys[2]] = values[2]
    return balance


def _parse_snapshot_diagnostics(lines: list[str], n_regions: int) -> dict[str, Any]:
    unit = _extract_first(lines, r"Energy Conservation Check -- units are \(([^)]+)\)", lambda value: value.strip())
    diagnostics: dict[str, Any] = {}
    boundary_fluxes = _parse_radiation_boundary_fluxes(lines, n_regions)
    if boundary_fluxes:
        diagnostics["radiation_boundary_fluxes"] = boundary_fluxes
    energy_summary = _parse_energy_summary(lines)
    if energy_summary:
        diagnostics["energy_summary"] = energy_summary
    energy_exchange = _parse_energy_exchange(lines)
    if any(energy_exchange.values()):
        diagnostics["energy_exchange"] = energy_exchange
    energy_balance = _parse_energy_balance(lines)
    if any(energy_balance.values()):
        diagnostics["energy_balance"] = energy_balance
    if unit and diagnostics:
        diagnostics["units"] = {"energy": unit, "radiation_boundary_fluxes": "J/s/cm2"}
    return diagnostics


def _consume_until_table_header_in_block(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index].lstrip().startswith("#"):
            return index
    return None


def _read_table_data_lines_in_block(lines: list[str], start_index: int) -> tuple[list[str], int]:
    data_lines: list[str] = []
    saw_numeric_line = False
    index = start_index
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped:
            lead = stripped[0]
            if lead.isdigit() or lead in "+-.":
                saw_numeric_line = True
                data_lines.append(line)
                index += 1
                continue
            if saw_numeric_line:
                break
            index += 1
            continue
        if not saw_numeric_line:
            index += 1
            continue
        break
    return data_lines, index


class HeaderSemanticParser:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER
        self._emitted_coordinate_issue_codes: set[str] = set()

    def _emit_coordinate_issue_once(self, source_name: str, issue: CoordinateValidationIssue) -> None:
        if issue.code in self._emitted_coordinate_issue_codes:
            return
        self._emitted_coordinate_issue_codes.add(issue.code)
        self.logger.warning("Coordinate validation issue in %s: %s", source_name, issue.message)

    def parse(self, source: Path, header_text: str) -> HeliosHeader:
        header_lines = header_text.splitlines()
        section_names = tuple(
            _normalize_label(line)
            for line in header_lines
            if line.strip().endswith("PARAMETERS:") or "INPUT PARAMETERS FOR HELIOS CALCULATION" in line
        )
        material_rows = _parse_simple_table(
            header_lines,
            "Material Equation of State parameters:",
            re.compile(r"^\s*(\d+)\s+(\S+)\s+(.+?)\s*$"),
            ("index", "eos_model", "file_path"),
        )
        opacity_rows = _parse_simple_table(
            header_lines,
            "Material Opacity parameters:",
            re.compile(r"^\s*(\d+)\s+(\S+)\s+(.+?)\s*$"),
            ("index", "opacity_model", "file_path"),
        )
        spatial_regions = _parse_simple_table(
            header_lines,
            "Spatial Region parameters:",
            re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s*$"),
            ("region_index", "min_index", "max_index", "material_index", "atomic_weight", "temperature", "mass_density", "ion_density", "velocity"),
        )
        laser_power = _parse_simple_table(
            header_lines,
            "Laser power table:",
            re.compile(r"^\s*(\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s*$"),
            ("index", "time", "power"),
        )
        zone_mass_rows = _parse_simple_table(
            header_lines,
            "Zone masses:",
            re.compile(r"^\s*(\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s*$"),
            ("index", "radius", "width", "mass", "atomic_weight"),
        )
        region_masses = _parse_simple_table(
            header_lines,
            "Zone masses per spatial region:",
            re.compile(r"^\s*(\d+)\s+([0-9.E+-]+)\s*$"),
            ("region_index", "mass"),
        )

        geometry = _extract_first(header_lines, r"Geometry type\s*=\s*(\S+)", str)
        coordinate_name = coordinate_name_for_geometry(geometry)
        n_zones = _extract_first(header_lines, r"Number of zones\s*=\s*(\d+)", int, 0)
        zone_ids = np.arange(1, n_zones + 1, dtype=np.int32)
        zone_mass_rows.sort(key=lambda row: int(row["index"]))
        outer_edges = np.full(n_zones, np.nan, dtype=np.float64)
        zone_width = np.full(n_zones, np.nan, dtype=np.float64)
        zone_mass = np.full(n_zones, np.nan, dtype=np.float64)
        atomic_weight = np.full(n_zones, np.nan, dtype=np.float64)
        for row in zone_mass_rows:
            index = int(row["index"]) - 1
            if index < 0 or index >= n_zones:
                continue
            outer_edges[index] = float(row["radius"])
            zone_width[index] = float(row["width"])
            zone_mass[index] = float(row["mass"])
            atomic_weight[index] = float(row["atomic_weight"])
        coordinate_issues: list[CoordinateValidationIssue] = []
        coordinate_edge, coordinate_center = build_coordinate_model(
            outer_edges,
            zone_width,
            geometry=geometry,
            issues=coordinate_issues,
        )

        zone_region_id = np.zeros(n_zones, dtype=np.int32)
        zone_material_index = np.zeros(n_zones, dtype=np.int32)
        for region in spatial_regions:
            start = int(region["min_index"]) - 1
            end = int(region["max_index"])
            zone_region_id[start:end] = int(region["region_index"])
            zone_material_index[start:end] = int(region["material_index"])

        photon_energy_grid = _parse_float_block(header_lines, "Photon energy grid:", "Zone masses:")
        regions, region_units = _rows_to_region_arrays(spatial_regions, region_masses)
        materials, material_units = _merge_material_tables(material_rows, opacity_rows)
        input_parameters = {
            "hydro": _parse_hydro_input(header_lines),
            "radiative_transfer": _parse_radiative_transfer_input(header_lines, photon_energy_grid),
            "radiation_source": _parse_radiation_source_input(header_lines),
            "laser_source": _parse_laser_input(header_lines, laser_power),
            "time_control": _parse_time_control_input(header_lines),
        }
        metadata = {
            "geometry": geometry,
            "helios_version": _extract_first(header_lines, r"\(ver\.\s*([^)]+)\)", str),
            "calculation_datetime": _extract_first(header_lines, r"Date and Time of Calculation:([0-9/:\.\s]+)", lambda value: value.strip()),
            "n_regions": _extract_first(header_lines, r"Number of regions\s*=\s*(\d+)", int, 0),
            "n_materials": _extract_first(header_lines, r"Number of materials\s*=\s*(\d+)", int, 0),
            "n_zones": n_zones,
            "plasma_model": input_parameters["hydro"].get("plasma_model"),
            "laser_origin_zone": input_parameters["laser_source"].get("origin_zone_index"),
            "eos_models": material_rows,
            "opacity_models": opacity_rows,
            "spatial_regions": spatial_regions,
            "laser_power_table": laser_power,
            "zone_masses_per_region": region_masses,
            "total_mass": _extract_first(header_lines, r"Total Mass \(g/cm\*\*X\)\s*=\s*([0-9.E+-]+)", float),
            "photon_energy_grid": photon_energy_grid,
            "openmp_note": _extract_first(header_lines, r"^(OpenMP.+)$", lambda value: value.strip()),
            "coordinate_model": {
                "coordinate_name": coordinate_name,
                "source_column_label": "Radius",
                "static_center_dataset": "coordinate_center",
                "static_edge_dataset": "coordinate_edge",
                "dynamic_center_dataset": "dynamic_coordinate_center",
                "dynamic_edge_dataset": "dynamic_coordinate_edge",
                "width_dataset": "zone_width",
                "legacy_static_center_alias": "x",
                "legacy_dynamic_center_alias": "radius",
            },
        }
        if coordinate_issues:
            metadata["coordinate_validation_issues"] = [
                {"code": issue.code, "message": issue.message, "severity": issue.severity}
                for issue in coordinate_issues
            ]
            for issue in coordinate_issues:
                self._emit_coordinate_issue_once(source.name, issue)

        header = HeliosHeader(
            source=source,
            simulation_name=source.stem,
            code_version=metadata["helios_version"],
            calculation_datetime=metadata["calculation_datetime"],
            header_sections=section_names,
            block_delimiter=BLOCK_DELIMITER,
            n_regions=metadata["n_regions"],
            n_materials=metadata["n_materials"],
            n_zones=n_zones,
            grid={
                "coordinate_center": coordinate_center,
                "coordinate_edge": coordinate_edge,
                "x": coordinate_center,
                "zone_id": zone_ids,
                "zone_mass": zone_mass,
                "zone_width": zone_width,
                "atomic_weight": atomic_weight,
                "zone_region_id": zone_region_id,
                "zone_material_index": zone_material_index,
            },
            grid_units={
                "coordinate_center": "cm",
                "coordinate_edge": "cm",
                "x": "cm",
                "zone_id": "",
                "zone_mass": "g/cm**X",
                "zone_width": "cm",
                "atomic_weight": "amu",
                "zone_region_id": "",
                "zone_material_index": "",
            },
            regions=regions,
            region_units=region_units,
            materials=materials,
            material_units=material_units,
            input_parameters=input_parameters,
            metadata=metadata,
        )
        if len(zone_mass_rows) != n_zones:
            self.logger.warning("Header declares %s zones but zone-mass table contains %s rows.", n_zones, len(zone_mass_rows))
        return header


class SnapshotBlockSemanticParser:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER
        self._emitted_coordinate_issue_codes: set[str] = set()

    def _emit_coordinate_issue_once(self, cycle: int, issue: dict[str, str]) -> None:
        code = str(issue.get("code", "")).strip() or "unknown_coordinate_issue"
        if code in self._emitted_coordinate_issue_codes:
            return
        self._emitted_coordinate_issue_codes.add(code)
        self.logger.warning(
            "Coordinate validation issue first observed at snapshot cycle %s: %s",
            cycle,
            issue.get("message", ""),
        )

    def parse(self, block_text: str, header: HeliosHeader) -> Snapshot:
        lines = block_text.splitlines()
        cycle_line_index = next((i for i, line in enumerate(lines) if CYCLE_HEADER_RE.match(line)), None)
        if cycle_line_index is None or cycle_line_index + 1 >= len(lines):
            raise ValueError("Snapshot block is missing cycle metadata.")

        cycle_tokens = _extract_number_tokens(lines[cycle_line_index + 1])
        if len(cycle_tokens) < 3:
            raise ValueError(f"Malformed cycle line: {lines[cycle_line_index + 1]}")
        cycle = int(float(cycle_tokens[0]))
        time_value = float(cycle_tokens[1])
        time_step = float(cycle_tokens[2])
        control_text = lines[cycle_line_index + 1].strip()

        first_header_index = _consume_until_table_header_in_block(lines, cycle_line_index + 2)
        if first_header_index is None or first_header_index + 1 >= len(lines):
            raise ValueError(f"Snapshot at cycle {cycle} is missing the first field table.")
        first_header = lines[first_header_index]
        first_units = lines[first_header_index + 1]
        primary_lines, next_index = _read_table_data_lines_in_block(lines, first_header_index + 2)
        primary_fields, primary_units, raw_field_map, primary_extras = _parse_field_table_rows(
            primary_lines,
            first_header,
            first_units,
            header.n_zones,
            self.logger,
            geometry=header.metadata.get("geometry"),
        )

        second_header_index = _consume_until_table_header_in_block(lines, next_index)
        if second_header_index is None or second_header_index + 1 >= len(lines):
            raise ValueError(f"Snapshot at cycle {cycle} is missing the second field table.")
        second_header = lines[second_header_index]
        second_units = lines[second_header_index + 1]
        secondary_lines, diagnostics_index = _read_table_data_lines_in_block(lines, second_header_index + 2)
        secondary_fields, secondary_units, secondary_map, secondary_extras = _parse_field_table_rows(
            secondary_lines,
            second_header,
            second_units,
            header.n_zones,
            self.logger,
            geometry=header.metadata.get("geometry"),
        )
        primary_row_count = int(primary_extras.get("parsed_row_count", header.n_zones))
        primary_expected = int(primary_extras.get("expected_row_count", header.n_zones))
        primary_padded = int(primary_extras.get("padded_row_count", 0))
        for issue in primary_extras.get("coordinate_issues", ()):
            if isinstance(issue, dict):
                self._emit_coordinate_issue_once(cycle, issue)
        if primary_row_count != primary_expected or primary_padded > 0:
            raise ValueError(
                f"Snapshot at cycle {cycle} has an incomplete primary field table "
                f"({primary_row_count}/{primary_expected} rows, padded_rows={primary_padded})."
            )
        secondary_row_count = int(secondary_extras.get("parsed_row_count", header.n_zones))
        secondary_expected = int(secondary_extras.get("expected_row_count", header.n_zones))
        secondary_padded = int(secondary_extras.get("padded_row_count", 0))
        if secondary_row_count != secondary_expected or secondary_padded > 0:
            raise ValueError(
                f"Snapshot at cycle {cycle} has an incomplete secondary field table "
                f"({secondary_row_count}/{secondary_expected} rows, padded_rows={secondary_padded})."
            )
        diagnostics = _parse_snapshot_diagnostics(lines[diagnostics_index:], header.n_regions)

        fields = {**primary_fields, **secondary_fields}
        field_units = {**primary_units, **secondary_units}
        raw_field_map.update(secondary_map)
        if all(name in fields for name in ("pressure_i", "pressure_e", "pressure_radiation")):
            fields["pressure"] = fields["pressure_i"] + fields["pressure_e"] + fields["pressure_radiation"]
            field_units["pressure"] = field_units["pressure_i"]
            raw_field_map["pressure"] = "Ion press + Elec press + Rad press"

        return Snapshot(
            cycle=cycle,
            time=time_value,
            time_step=time_step,
            time_step_control=control_text,
            fields=fields,
            field_units=field_units,
            raw_field_map=raw_field_map,
            coordinate_name=coordinate_name_for_geometry(header.metadata.get("geometry")),
            coordinate_center=np.asarray(primary_extras.get("coordinate_center"), dtype=np.float64)
            if primary_extras.get("coordinate_center") is not None
            else None,
            coordinate_edge=np.asarray(primary_extras.get("coordinate_edge"), dtype=np.float64)
            if primary_extras.get("coordinate_edge") is not None
            else None,
            diagnostics=diagnostics,
        )


class HeliosParser:
    def __init__(self, logger: logging.Logger | None = None, *, access_mode: str = "mmap") -> None:
        self.logger = logger or LOGGER
        self.access_mode = access_mode
        self.tokenizer = DEFAULT_TOKENIZER
        self.header_parser = HeaderSemanticParser(self.logger)
        self.snapshot_parser = SnapshotBlockSemanticParser(self.logger)

    def open_document(self, path: str | Path, *, access_mode: str | None = None):
        from .document import open_document
        from .indexing import StructuralIndexer

        return open_document(
            path,
            access_mode=access_mode or self.access_mode,
            indexer=StructuralIndexer(self.tokenizer),
            header_parser=self.header_parser,
            snapshot_parser=self.snapshot_parser,
        )

    def inspect(self, path: str | Path, *, access_mode: str | None = None) -> HeliosHeader:
        with timed_block("parser.inspect", logger=LOGGER, details=Path(path).name):
            with self.open_document(path, access_mode=access_mode) as document:
                return document.inspect()

    def preview(self, path: str | Path, *, access_mode: str | None = None) -> HeliosPreview:
        with timed_block("parser.preview", logger=LOGGER, details=Path(path).name):
            with self.open_document(path, access_mode=access_mode) as document:
                return document.preview()

    def iter_snapshots(
        self,
        path: str | Path,
        header: HeliosHeader | None = None,
        *,
        access_mode: str | None = None,
    ) -> Iterator[Snapshot]:
        def generator() -> Iterator[Snapshot]:
            with self.open_document(path, access_mode=access_mode) as document:
                yield from document.iter_snapshots(header=header)

        return generator()

    def iter_snapshots_streaming(
        self,
        path: str | Path,
        header: HeliosHeader | None = None,
        *,
        access_mode: str | None = None,
    ) -> Iterator[Snapshot]:
        def generator() -> Iterator[Snapshot]:
            with self.open_document(path, access_mode=access_mode) as document:
                yield from document.iter_snapshots_streaming(header=header)

        return generator()

    def count_snapshots(self, path: str | Path, *, access_mode: str | None = None) -> int:
        with self.open_document(path, access_mode=access_mode) as document:
            return document.parsed_snapshot_count()

    def parse(self, path: str | Path, *, access_mode: str | None = None) -> Simulation:
        with timed_block("parser.parse_full", logger=LOGGER, details=Path(path).name):
            with self.open_document(path, access_mode=access_mode) as document:
                return document.parse_full()
