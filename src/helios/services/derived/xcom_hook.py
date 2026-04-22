"""Optional cold-attenuation backend seam for XCOM-like integrations.

This module stays intentionally narrow:

- it packages the current HELIOS snapshot subset into stable cold-attenuation
  requests
- it resolves material identity from already-parsed HELIOS material metadata
- it probes the optional backend honestly without breaking normal startup
- it provides a bounded persistent cache for successful cold-attenuation
  requests

For this codebase the parsed EOS / opacity metadata is the authoritative
material-identity seed unless a better explicit formula/composition field is
already present in the parsed payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import threading
import time
from typing import Any, Protocol

import numpy as np

from helios.platform.archive_utils import extract_archive
from helios.runtime import RunContext
from helios.services.derived.models import DerivedRunData
from helios.services.derived.selection import build_analysis_mask, path_length_cm


_CACHE_SCHEMA_VERSION = 2
_CACHE_MAX_SIZE_BYTES = 64 * 1024 * 1024
_GENERIC_IDENTITY_TOKENS = frozenset(
    {
        "",
        "eosopa",
        "sesame",
        "opacity",
        "material",
        "materials",
        "propaceos",
        "table",
        "tables",
        "unknown",
        "none",
    }
)
_EXPLICIT_METADATA_KEYS = (
    "chemical_formula",
    "chemical_formula_text",
    "formula",
    "composition",
    "material_formula",
    "material_name",
    "name",
    "label",
)
_FORMULA_CASE_MAP = {
    "al": "Al",
    "al2o3": "Al2O3",
    "c": "C",
    "c2h4o": "C2H4O",
    "c10h8o4": "C10H8O4",
    "c22h10n2o5": "C22H10N2O5",
    "ch": "CH",
    "ch2": "CH2",
    "cu": "Cu",
    "fe": "Fe",
    "si": "Si",
    "sio2": "SiO2",
}
_FRIENDLY_DISPLAY_NAMES = {
    "al2o3": "sapphire",
    "c2h4o": "epoxy",
    "c10h8o4": "mylar",
    "c22h10n2o5": "kapton",
    "sio2": "silica",
}
_CURATED_ALIASES = {
    "adhesive": ("C2H4O", "epoxy"),
    "aluminum": ("Al", "Al"),
    "aluminium": ("Al", "Al"),
    "copper": ("Cu", "Cu"),
    "epoxy": ("C2H4O", "epoxy"),
    "glue": ("C2H4O", "epoxy"),
    "iron": ("Fe", "Fe"),
    "kapton": ("C22H10N2O5", "kapton"),
    "mylar": ("C10H8O4", "mylar"),
    "polyethylene": ("CH2", "polyethylene"),
    "quartz": ("SiO2", "silica"),
    "resin": ("C2H4O", "epoxy"),
    "sapphire": ("Al2O3", "sapphire"),
    "silica": ("SiO2", "silica"),
    "silicon": ("Si", "Si"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _wrapper_archive_path() -> Path:
    return _repo_root() / "helios_xcom_integration.zip"


def _source_archive_path() -> Path:
    return _repo_root() / "XCOM.tar.gz"


def _fallback_table_path() -> Path:
    return _repo_root() / "x-com_fallback" / "xcom_fallback_1keV_12keV_extended.json"


def _app_data_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "HeliosViewer" / "HELIOS Parse View" / "derived" / "xcom"


def _archive_fingerprint(path: Path) -> str:
    stats = path.stat()
    return f"{path.name}:{int(stats.st_size)}:{int(stats.st_mtime_ns)}"


def _stamp_path(root: Path) -> Path:
    return root / ".archive_stamp.json"


def _ensure_wrapper_root() -> Path:
    archive = _wrapper_archive_path()
    if not archive.exists():
        raise FileNotFoundError(f"Missing optional XCOM wrapper archive: {archive}")
    destination = _app_data_root() / "wrapper"
    marker = _stamp_path(destination)
    expected = {"fingerprint": _archive_fingerprint(archive)}
    package_root = destination / "helios_xcom_integration"
    if marker.exists() and package_root.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        if current == expected:
            return destination
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, destination)
    marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return destination


def _normalized_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _candidate_from_value(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).stem.strip() if ("\\" in text or "/" in text or "." in Path(text).name) else text
    candidate = candidate.strip()
    if not candidate:
        return None
    normalized = _normalized_token(candidate)
    if not normalized or normalized in _GENERIC_IDENTITY_TOKENS:
        return None
    if not any(character.isalpha() for character in candidate):
        return None
    return candidate


def _canonical_formula(candidate: str) -> str | None:
    normalized = _normalized_token(candidate)
    if normalized in _FORMULA_CASE_MAP:
        return _FORMULA_CASE_MAP[normalized]
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", str(candidate).strip()):
        return str(candidate).strip()
    return None


def _friendly_display_name(formula_or_label: str) -> str:
    normalized = _normalized_token(formula_or_label)
    return _FRIENDLY_DISPLAY_NAMES.get(normalized, str(formula_or_label))


@dataclass(frozen=True, slots=True)
class MaterialResolution:
    material_id: int
    backend_label: str | None
    display_label: str
    status: str
    canonical_key: str | None = None
    source_kind: str | None = None
    source_value: str | None = None
    notes: tuple[str, ...] = ()


def _format_material_resolution_label(resolution: MaterialResolution) -> str:
    status = str(resolution.status or "unresolved")
    source_kind = str(resolution.source_kind or "")
    display = str(resolution.display_label or f"material {int(resolution.material_id)}")
    if status == "resolved_from_explicit_metadata":
        return display
    if status == "resolved_from_eos":
        return display if display == str(resolution.backend_label or display) else f"{display} (from EOS)"
    if status == "resolved_from_opacity":
        return display if display == str(resolution.backend_label or display) else f"{display} (from opacity)"
    if status == "resolved_from_alias":
        if source_kind == "eos":
            return f"{display} (from EOS)"
        if source_kind == "opacity":
            return f"{display} (from opacity)"
        return f"{display} (alias)"
    if status == "guessed":
        if source_kind == "eos":
            return f"{display} (guessed from EOS)"
        if source_kind == "opacity":
            return f"{display} (guessed from opacity)"
        return f"{display} (guessed)"
    return f"unknown material {int(resolution.material_id)}"


def _resolution_summary_with_status(resolution: MaterialResolution) -> str:
    return f"{_format_material_resolution_label(resolution)} [{resolution.status}]"


def _series_value(series: object, index: int) -> object | None:
    if series is None:
        return None
    try:
        return series[index]
    except Exception:
        return None


def _guess_resolution(candidate: str, *, material_id: int, source_kind: str, source_value: str | None) -> MaterialResolution | None:
    normalized = _normalized_token(candidate)
    for alias_key, (backend_label, display_label) in _CURATED_ALIASES.items():
        if alias_key in normalized:
            return MaterialResolution(
                material_id=int(material_id),
                backend_label=str(backend_label),
                display_label=str(display_label),
                status="guessed",
                canonical_key=normalize_material_key(backend_label),
                source_kind=source_kind,
                source_value=source_value,
                notes=(f"Guessed from {source_kind} metadata token {candidate!r}.",),
            )
    for formula_key, canonical in _FORMULA_CASE_MAP.items():
        if normalized.startswith(formula_key) and len(formula_key) >= 2:
            return MaterialResolution(
                material_id=int(material_id),
                backend_label=str(canonical),
                display_label=_friendly_display_name(canonical),
                status="guessed",
                canonical_key=normalize_material_key(canonical),
                source_kind=source_kind,
                source_value=source_value,
                notes=(f"Guessed from {source_kind} metadata token {candidate!r}.",),
            )
    return None


def _resolve_from_candidate(
    candidate: str,
    *,
    material_id: int,
    source_kind: str,
    source_value: str | None,
    explicit: bool = False,
) -> MaterialResolution | None:
    formula = _canonical_formula(candidate)
    if formula is not None:
        return MaterialResolution(
            material_id=int(material_id),
            backend_label=str(formula),
            display_label=_friendly_display_name(formula),
            status="resolved_from_explicit_metadata" if explicit else f"resolved_from_{source_kind}",
            canonical_key=normalize_material_key(formula),
            source_kind=source_kind,
            source_value=source_value,
        )
    alias = _CURATED_ALIASES.get(_normalized_token(candidate))
    if alias is not None:
        backend_label, display_label = alias
        return MaterialResolution(
            material_id=int(material_id),
            backend_label=str(backend_label),
            display_label=str(display_label),
            status="resolved_from_alias",
            canonical_key=normalize_material_key(backend_label),
            source_kind=source_kind,
            source_value=source_value,
        )
    return _guess_resolution(candidate, material_id=int(material_id), source_kind=source_kind, source_value=source_value)


def resolve_material_identity(materials: dict[str, Any], material_id: int) -> MaterialResolution:
    material_index = np.asarray(materials.get("index", ()), dtype=np.int32)
    matches = np.flatnonzero(np.abs(material_index) == abs(int(material_id)))
    if matches.size == 0:
        return MaterialResolution(
            material_id=int(material_id),
            backend_label=None,
            display_label=f"material {int(material_id)}",
            status="unresolved",
            canonical_key=None,
        )
    row_index = int(matches[0])

    for key in _EXPLICIT_METADATA_KEYS:
        candidate = _candidate_from_value(_series_value(materials.get(key), row_index))
        if candidate is None:
            continue
        resolved = _resolve_from_candidate(
            candidate,
            material_id=int(material_id),
            source_kind="explicit_metadata",
            source_value=str(_series_value(materials.get(key), row_index) or ""),
            explicit=True,
        )
        if resolved is not None:
            return resolved

    eos_candidate = _candidate_from_value(_series_value(materials.get("eos_file_path"), row_index))
    if eos_candidate is not None:
        resolved = _resolve_from_candidate(
            eos_candidate,
            material_id=int(material_id),
            source_kind="eos",
            source_value=str(_series_value(materials.get("eos_file_path"), row_index) or ""),
        )
        if resolved is not None:
            return resolved

    eos_model_candidate = _candidate_from_value(_series_value(materials.get("eos_model"), row_index))
    if eos_model_candidate is not None:
        guessed = _guess_resolution(
            eos_model_candidate,
            material_id=int(material_id),
            source_kind="eos",
            source_value=str(_series_value(materials.get("eos_model"), row_index) or ""),
        )
        if guessed is not None:
            return guessed

    opacity_candidate = _candidate_from_value(_series_value(materials.get("opacity_file_path"), row_index))
    if opacity_candidate is not None:
        resolved = _resolve_from_candidate(
            opacity_candidate,
            material_id=int(material_id),
            source_kind="opacity",
            source_value=str(_series_value(materials.get("opacity_file_path"), row_index) or ""),
        )
        if resolved is not None:
            return resolved

    opacity_model_candidate = _candidate_from_value(_series_value(materials.get("opacity_model"), row_index))
    if opacity_model_candidate is not None:
        guessed = _guess_resolution(
            opacity_model_candidate,
            material_id=int(material_id),
            source_kind="opacity",
            source_value=str(_series_value(materials.get("opacity_model"), row_index) or ""),
        )
        if guessed is not None:
            return guessed

    return MaterialResolution(
        material_id=int(material_id),
        backend_label=None,
        display_label="unknown material",
        status="unresolved",
        canonical_key=None,
    )


def resolve_material_identities(materials: dict[str, Any]) -> dict[int, MaterialResolution]:
    material_ids = np.asarray(materials.get("index", ()), dtype=np.int32)
    resolutions: dict[int, MaterialResolution] = {}
    for material_id in material_ids.tolist():
        resolutions[int(abs(material_id))] = resolve_material_identity(materials, int(material_id))
    return resolutions


def material_display_labels_by_id(materials: dict[str, Any]) -> dict[int, str]:
    return {material_id: _format_material_resolution_label(resolution) for material_id, resolution in resolve_material_identities(materials).items()}


def _material_resolution_summary(
    request: "ColdAttenuationRequest",
) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...]]:
    grouped: dict[str, float] = {}
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for zone in request.zones:
        if zone.material_label:
            material_key = str(zone.material_canonical_key or zone.material_label)
            grouped[material_key] = float(grouped.get(material_key, 0.0) + float(zone.density_g_cm3) * float(zone.path_length_cm))
            resolved.add(
                f"{zone.material_display_label or zone.material_label} [{zone.material_resolution_status}]"
                if zone.material_resolution_status
                else str(zone.material_display_label or zone.material_label)
            )
        else:
            unresolved.add(
                f"{zone.material_display_label} [{zone.material_resolution_status}]"
                if zone.material_display_label
                else f"material {int(zone.material_id)} [{zone.material_resolution_status}]"
            )
    return grouped, tuple(sorted(resolved)), tuple(sorted(unresolved))


def _cache_file_path() -> Path:
    return _app_data_root() / "multizone_request_cache.sqlite3"


_CACHE_LOCK = threading.Lock()
_XCOM_MODULE_LOCK = threading.Lock()
_BACKEND_STATUS_LOCK = threading.Lock()
_FALLBACK_TABLE_LOCK = threading.Lock()
_XCOM_MODULE: object | None = None
_XCOM_BACKEND: "HeliosXcomBackend | None" = None
_XCOM_TABLE_BACKEND: "PrecomputedColdTableBackend | None" = None
_XCOM_BACKEND_BASIC_STATUS: "ColdAttenuationBackendStatus | None" = None
_XCOM_BACKEND_FULL_STATUS: "ColdAttenuationBackendStatus | None" = None
_XCOM_BACKEND_STATUS_FINGERPRINT: str | None = None
_PERSISTENT_CACHE: "PersistentColdAttenuationCache | None" = None
_FALLBACK_TABLE_CACHE: dict[str, object] | None = None
_FALLBACK_TABLE_DEFAULT_QUANTITY = "mu_rho_total_cm2_g"
_FALLBACK_TABLE_MIN_KEV = 1.0
_FALLBACK_TABLE_MAX_KEV = 12.0
_FALLBACK_TABLE_EXTRAP_FACTOR = 1.2
_CANONICAL_MATERIAL_ALIASES = {
    "al": "al",
    "aluminum": "al",
    "aluminium": "al",
    "cu": "cu",
    "copper": "cu",
    "fe": "fe",
    "iron": "fe",
    "si": "si",
    "silicon": "si",
    "ti": "ti",
    "titanium": "ti",
    "au": "au",
    "gold": "au",
    "be": "be",
    "beryllium": "be",
    "c": "c",
    "carbon": "c",
    "diamond": "c",
    "graphite": "c",
    "ch": "ch",
    "plastic_ch": "ch",
    "ch_plastic": "ch",
    "epoxy": "c2h4o",
    "epoxy_c2h4o": "c2h4o",
    "epoxyc2h4o": "c2h4o",
    "c2h4o": "c2h4o",
    "glass": "sio2",
    "glass_sio2": "sio2",
    "glasssio2": "sio2",
    "silica": "sio2",
    "sio2": "sio2",
    "kapton": "c22h10n2o5",
    "polyimide": "c22h10n2o5",
    "kapton_c22h10n2o5": "c22h10n2o5",
    "kaptonc22h10n2o5": "c22h10n2o5",
    "c22h10n2o5": "c22h10n2o5",
}
_CANONICAL_TO_FALLBACK_TABLE_KEY = {
    "al": "al",
    "cu": "cu",
    "fe": "fe",
    "si": "si",
    "ti": "ti",
    "au": "au",
    "be": "be",
    "c": "c",
    "ch": "ch",
    "c2h4o": "epoxy_c2h4o",
    "c22h10n2o5": "kapton_c22h10n2o5",
    "sio2": "sio2",
}


@dataclass(frozen=True, slots=True)
class ColdAttenuationZone:
    """Single-zone attenuation input for an optional cold backend."""

    zone_index: int
    region_id: int
    material_id: int
    material_label: str | None
    density_g_cm3: float
    path_length_cm: float
    material_display_label: str | None = None
    material_resolution_status: str = "unresolved"
    material_canonical_key: str | None = None


@dataclass(frozen=True, slots=True)
class ColdAttenuationRequest:
    """Structured snapshot request that a future XCOM-like backend can consume."""

    snapshot_index: int
    observation_side: str
    line_of_sight_cosine: float
    photon_energies_kev: tuple[float, ...]
    zones: tuple[ColdAttenuationZone, ...]


@dataclass(frozen=True, slots=True)
class ColdAttenuationResult:
    """Backend response for a future cold-baseline transmission service."""

    energies_kev: np.ndarray
    transmission: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ColdAttenuationBackendStatus:
    available: bool
    status: str
    message: str
    backend_name: str | None = None
    backend_fingerprint: str | None = None
    wrapper_present: bool = False
    import_ok: bool = False
    client_ok: bool = False
    compute_ok: bool = False
    checked_at: float | None = None


@dataclass(frozen=True, slots=True)
class ColdAttenuationTableLookup:
    original_label: str | None
    normalized_label: str
    material_key: str
    table_key: str
    energy_kev: float
    mu_rho_cm2_g: float
    interpolation_kind: str
    note: str | None = None


class OptionalColdAttenuationBackend(Protocol):
    """Protocol implemented by optional attenuation backends such as XCOM."""

    def compute_transmission(self, request: ColdAttenuationRequest) -> ColdAttenuationResult:
        """Return transmission for the supplied snapshot request."""


def _load_fallback_table() -> dict[str, object]:
    global _FALLBACK_TABLE_CACHE
    cached = _FALLBACK_TABLE_CACHE
    if cached is not None:
        return cached
    with _FALLBACK_TABLE_LOCK:
        cached = _FALLBACK_TABLE_CACHE
        if cached is not None:
            return cached
        path = _fallback_table_path()
        payload = json.loads(path.read_text(encoding="utf-8"))
        _FALLBACK_TABLE_CACHE = payload
        return payload


def fallback_table_available() -> bool:
    return _fallback_table_path().exists()


def _fallback_table_fingerprint() -> str | None:
    path = _fallback_table_path()
    if not path.exists():
        return None
    return f"precomputed_xcom_table:{_archive_fingerprint(path)}"


def _normalized_material_key_token(label: str | None) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_material_key(label: str | None) -> str:
    normalized = _normalized_material_key_token(label)
    compact = normalized.replace("_", "")
    for candidate in (normalized, compact):
        mapped = _CANONICAL_MATERIAL_ALIASES.get(candidate)
        if mapped is not None:
            return str(mapped)
    for token in ("c22h10n2o5", "c2h4o", "sio2", "ch"):
        if token in compact:
            return str(_CANONICAL_MATERIAL_ALIASES.get(token, token))
    return compact or normalized


def canonical_fallback_table_key(material: str | None) -> str:
    if material is None:
        raise KeyError("missing material key")
    table = _load_fallback_table()
    materials = table.get("materials", {})
    normalized = _normalized_material_key_token(material)
    canonical = normalize_material_key(material)
    table_key = _CANONICAL_TO_FALLBACK_TABLE_KEY.get(canonical, canonical)
    if table_key in materials:
        return str(canonical)
    expected = ", ".join(sorted(str(key) for key in materials.keys()))
    raise KeyError(
        "unknown fallback key: "
        f"{material} -> normalized: {normalized} -> canonical: {canonical} -> expected keys: [{expected}]"
    )


def _interp_loglog(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x0 <= 0.0 or x1 <= 0.0 or y0 <= 0.0 or y1 <= 0.0 or x0 == x1:
        return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    lx = math.log(float(x))
    lx0 = math.log(float(x0))
    lx1 = math.log(float(x1))
    ly0 = math.log(float(y0))
    ly1 = math.log(float(y1))
    ly = ly0 + (ly1 - ly0) * (lx - lx0) / (lx1 - lx0)
    return float(math.exp(ly))


def lookup_fallback_mu_rho(
    material: str | None,
    energy_kev: float,
    *,
    quantity: str = _FALLBACK_TABLE_DEFAULT_QUANTITY,
    allow_extrapolation: bool = True,
) -> ColdAttenuationTableLookup:
    table = _load_fallback_table()
    original_label = None if material is None else str(material)
    normalized_label = _normalized_material_key_token(material)
    material_key = canonical_fallback_table_key(material)
    table_key = _CANONICAL_TO_FALLBACK_TABLE_KEY.get(material_key, material_key)
    materials = table.get("materials", {})
    rows = materials[table_key]["rows"]
    energy_ev = float(energy_kev) * 1.0e3
    xs = [float(row["energy_eV"]) for row in rows]
    ys = [float(row[quantity]) for row in rows]
    if not xs:
        raise ValueError(f"Fallback XCOM table has no rows for {material_key}.")
    minimum_kev = float(xs[0]) * 1.0e-3
    maximum_kev = float(xs[-1]) * 1.0e-3
    lower_limit = minimum_kev / _FALLBACK_TABLE_EXTRAP_FACTOR
    upper_limit = maximum_kev * _FALLBACK_TABLE_EXTRAP_FACTOR
    if float(energy_kev) < lower_limit or float(energy_kev) > upper_limit:
        raise ValueError(
            f"Energy {float(energy_kev):.4g} keV is outside the allowed fallback-table range "
            f"{lower_limit:.4g}–{upper_limit:.4g} keV."
        )
    interpolation_kind = "exact"
    note = None
    index = 0
    while index < len(xs) and xs[index] < energy_ev:
        index += 1
    if index < len(xs) and xs[index] == energy_ev:
        mu_rho = ys[index]
    elif index == 0:
        if not allow_extrapolation or len(xs) < 2:
            raise ValueError("Energy is below the fallback-table range.")
        mu_rho = _interp_loglog(energy_ev, xs[0], ys[0], xs[1], ys[1])
        interpolation_kind = "extrapolated"
        note = (
            f"Precomputed XCOM table extrapolated below {minimum_kev:.4g} keV; "
            f"use with caution outside the native 1–12 keV range."
        )
    elif index >= len(xs):
        if not allow_extrapolation or len(xs) < 2:
            raise ValueError("Energy is above the fallback-table range.")
        mu_rho = _interp_loglog(energy_ev, xs[-2], ys[-2], xs[-1], ys[-1])
        interpolation_kind = "extrapolated"
        note = (
            f"Precomputed XCOM table extrapolated above {maximum_kev:.4g} keV; "
            f"use with caution outside the native 1–12 keV range."
        )
    else:
        mu_rho = _interp_loglog(energy_ev, xs[index - 1], ys[index - 1], xs[index], ys[index])
        interpolation_kind = "interpolated"
        note = "Precomputed XCOM table interpolated in log-log space across energy."
    return ColdAttenuationTableLookup(
        original_label=original_label,
        normalized_label=normalized_label,
        material_key=str(material_key),
        table_key=str(table_key),
        energy_kev=float(energy_kev),
        mu_rho_cm2_g=float(mu_rho),
        interpolation_kind=interpolation_kind,
        note=note,
    )


class PrecomputedColdTableBackend:
    """Deterministic cold attenuation fallback based on precomputed XCOM tables."""

    backend_name = "XCOM table"

    @property
    def backend_fingerprint(self) -> str:
        return str(_fallback_table_fingerprint() or "precomputed_xcom_table")

    def compute_transmission(self, request: ColdAttenuationRequest) -> ColdAttenuationResult:
        if not request.zones:
            raise ValueError("Cold attenuation request does not include any zones.")
        energies_kev = np.asarray(request.photon_energies_kev, dtype=np.float64)
        if energies_kev.size == 0:
            raise ValueError("Cold attenuation request must include at least one photon energy.")
        total_tau = np.zeros(energies_kev.shape, dtype=np.float64)
        grouped_budgets: dict[str, dict[str, object]] = {}
        interpolation_kinds: set[str] = set()
        interpolation_notes: set[str] = set()
        for zone in request.zones:
            material_key = str(zone.material_canonical_key or canonical_fallback_table_key(zone.material_label))
            areal_density = float(zone.density_g_cm3) * float(zone.path_length_cm)
            zone_tau = np.zeros(energies_kev.shape, dtype=np.float64)
            for energy_index, energy_kev in enumerate(energies_kev.tolist()):
                lookup = lookup_fallback_mu_rho(material_key, float(energy_kev))
                interpolation_kinds.add(str(lookup.interpolation_kind))
                if lookup.note:
                    interpolation_notes.add(str(lookup.note))
                zone_tau[energy_index] = float(lookup.mu_rho_cm2_g) * areal_density
            total_tau += zone_tau
            existing = grouped_budgets.get(
                material_key,
                {
                    "label": material_key,
                    "areal_density_g_cm2": 0.0,
                    "optical_depth": [0.0 for _ in energies_kev.tolist()],
                },
            )
            existing["areal_density_g_cm2"] = float(existing["areal_density_g_cm2"]) + areal_density
            for energy_index, value in enumerate(zone_tau.tolist()):
                existing["optical_depth"][energy_index] = float(existing["optical_depth"][energy_index]) + float(value)
            grouped_budgets[material_key] = existing
        transmission = np.exp(-np.clip(total_tau, a_min=None, a_max=700.0))
        interpolation_mode = "exact"
        if "extrapolated" in interpolation_kinds:
            interpolation_mode = "extrapolated"
        elif "interpolated" in interpolation_kinds:
            interpolation_mode = "interpolated"
        metadata = {
            "backend_name": self.backend_name,
            "backend_fingerprint": self.backend_fingerprint,
            "attenuation_mode": "total_with_coherent",
            "optical_depth": [float(value) for value in total_tau.tolist()],
            "material_budgets": [
                {
                    "label": str(label),
                    "areal_density_g_cm2": float(payload["areal_density_g_cm2"]),
                    "optical_depth": [float(value) for value in payload["optical_depth"]],
                }
                for label, payload in sorted(grouped_budgets.items())
            ],
            "source": "precomputed_xcom_table",
            "interpolation_mode": interpolation_mode,
            "interpolation_note": " ".join(sorted(interpolation_notes)).strip(),
            "quantity": _FALLBACK_TABLE_DEFAULT_QUANTITY,
            "quantity_type": "mu_rho_cm2_g",
        }
        return ColdAttenuationResult(
            energies_kev=energies_kev,
            transmission=np.asarray(transmission, dtype=np.float64),
            metadata=metadata,
        )


def load_precomputed_cold_backend() -> OptionalColdAttenuationBackend | None:
    global _XCOM_TABLE_BACKEND
    if not fallback_table_available():
        return None
    if _XCOM_TABLE_BACKEND is None:
        try:
            _load_fallback_table()
            _XCOM_TABLE_BACKEND = PrecomputedColdTableBackend()
        except Exception:
            return None
    return _XCOM_TABLE_BACKEND


class PersistentColdAttenuationCache:
    """Persistent bounded cache for successful multizone cold-attenuation requests."""

    def __init__(self, path: Path | None = None, *, max_size_bytes: int = _CACHE_MAX_SIZE_BYTES) -> None:
        self.path = _cache_file_path() if path is None else Path(path)
        self.max_size_bytes = max(1024, int(max_size_bytes))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                cache_key TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                size_bytes INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_last_access ON entries(last_access ASC)")
        self._conn.commit()
        self._index: dict[str, dict[str, float | int]] = {}
        self._load_index()

    def _load_index(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT cache_key, created_at, last_access, size_bytes FROM entries"
            ).fetchall()
            self._index = {
                str(row["cache_key"]): {
                    "created_at": float(row["created_at"]),
                    "last_access": float(row["last_access"]),
                    "size_bytes": int(row["size_bytes"]),
                }
                for row in rows
            }

    def _serialized_payloads(
        self,
        *,
        request_payload: dict[str, object],
        result_payload: dict[str, object],
    ) -> tuple[str, str, int]:
        request_json = json.dumps(request_payload, sort_keys=True, separators=(",", ":"))
        result_json = json.dumps(result_payload, sort_keys=True, separators=(",", ":"))
        size_bytes = len(request_json.encode("utf-8")) + len(result_json.encode("utf-8"))
        return request_json, result_json, size_bytes

    def _total_size_bytes(self) -> int:
        return int(sum(int(meta["size_bytes"]) for meta in self._index.values()))

    def _evict_if_needed(self) -> None:
        total_size = self._total_size_bytes()
        if total_size <= self.max_size_bytes:
            return
        ordered = sorted(self._index.items(), key=lambda item: (float(item[1]["last_access"]), str(item[0])))
        for cache_key, meta in ordered:
            self._conn.execute("DELETE FROM entries WHERE cache_key = ?", (str(cache_key),))
            total_size -= int(meta["size_bytes"])
            self._index.pop(str(cache_key), None)
            if total_size <= self.max_size_bytes:
                break
        self._conn.commit()

    def get(self, cache_key: str) -> dict[str, object] | None:
        key = str(cache_key)
        with self._lock:
            if key not in self._index:
                return None
            row = self._conn.execute(
                "SELECT created_at, last_access, size_bytes, request_json, result_json FROM entries WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                self._index.pop(key, None)
                return None
            now = time.time()
            self._conn.execute("UPDATE entries SET last_access = ? WHERE cache_key = ?", (now, key))
            self._conn.commit()
            self._index[key]["last_access"] = now
            return {
                "cache_key": key,
                "created_at": float(row["created_at"]),
                "last_used_at": now,
                "size_bytes": int(row["size_bytes"]),
                "request": json.loads(str(row["request_json"])),
                "result": json.loads(str(row["result_json"])),
            }

    def put(self, cache_key: str, *, request_payload: dict[str, object], result_payload: dict[str, object]) -> None:
        key = str(cache_key)
        request_json, result_json, size_bytes = self._serialized_payloads(
            request_payload=request_payload,
            result_payload=result_payload,
        )
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entries(cache_key, created_at, last_access, size_bytes, request_json, result_json)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    last_access = excluded.last_access,
                    size_bytes = excluded.size_bytes,
                    request_json = excluded.request_json,
                    result_json = excluded.result_json
                """,
                (key, now, now, int(size_bytes), request_json, result_json),
            )
            self._conn.commit()
            self._index[key] = {"created_at": now, "last_access": now, "size_bytes": int(size_bytes)}
            self._evict_if_needed()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()
            self._index.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entry_count": len(self._index),
                "total_size_bytes": self._total_size_bytes(),
                "max_size_bytes": int(self.max_size_bytes),
            }

    def debug_entries(self) -> list[dict[str, object]]:
        with self._lock:
            ordered = sorted(self._index.items(), key=lambda item: (-float(item[1]["last_access"]), str(item[0])))
            return [
                {
                    "cache_key": cache_key,
                    "created_at": float(meta["created_at"]),
                    "last_access": float(meta["last_access"]),
                    "size_bytes": int(meta["size_bytes"]),
                }
                for cache_key, meta in ordered
            ]


def persistent_cold_attenuation_cache(
    path: Path | None = None,
    *,
    max_size_bytes: int = _CACHE_MAX_SIZE_BYTES,
) -> PersistentColdAttenuationCache:
    global _PERSISTENT_CACHE
    if path is not None:
        return PersistentColdAttenuationCache(path=path, max_size_bytes=max_size_bytes)
    if _PERSISTENT_CACHE is None or _PERSISTENT_CACHE.path != _cache_file_path():
        _PERSISTENT_CACHE = PersistentColdAttenuationCache(max_size_bytes=max_size_bytes)
    return _PERSISTENT_CACHE


def clear_persistent_cold_attenuation_cache(path: Path | None = None) -> None:
    global _PERSISTENT_CACHE
    cache = persistent_cold_attenuation_cache(path=path) if path is not None else persistent_cold_attenuation_cache()
    cache.clear()
    if path is None:
        _PERSISTENT_CACHE = None


def cold_attenuation_cache_key(
    request: ColdAttenuationRequest,
    *,
    backend_fingerprint: str,
    attenuation_mode: str = "total_with_coherent",
) -> tuple[str, dict[str, object]]:
    grouped, _resolved, unresolved = _material_resolution_summary(request)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "backend_fingerprint": str(backend_fingerprint),
        "attenuation_mode": str(attenuation_mode),
        "energies_kev": [round(float(value), 9) for value in request.photon_energies_kev],
        "materials": [
            {"label": label, "areal_density_g_cm2": round(float(grouped[label]), 12)}
            for label in sorted(grouped)
        ],
        "unresolved_materials": list(unresolved),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload


def _load_helios_xcom_module() -> object:
    global _XCOM_MODULE
    if _XCOM_MODULE is not None:
        return _XCOM_MODULE
    with _XCOM_MODULE_LOCK:
        if _XCOM_MODULE is not None:
            return _XCOM_MODULE
        wrapper_root = _ensure_wrapper_root() / "helios_xcom_integration"
        if str(wrapper_root) not in sys.path:
            sys.path.insert(0, str(wrapper_root))
        _XCOM_MODULE = importlib.import_module("helios_xcom")
        return _XCOM_MODULE


def _current_backend_identity_fingerprint() -> str | None:
    wrapper_archive = _wrapper_archive_path()
    if wrapper_archive.exists():
        return _archive_fingerprint(wrapper_archive)
    source_archive = _source_archive_path()
    if source_archive.exists():
        return _archive_fingerprint(source_archive)
    return None


def _reset_backend_status_if_needed() -> None:
    global _XCOM_BACKEND_BASIC_STATUS, _XCOM_BACKEND_FULL_STATUS, _XCOM_BACKEND_STATUS_FINGERPRINT, _XCOM_BACKEND
    fingerprint = _current_backend_identity_fingerprint()
    if fingerprint == _XCOM_BACKEND_STATUS_FINGERPRINT:
        return
    _XCOM_BACKEND_BASIC_STATUS = None
    _XCOM_BACKEND_FULL_STATUS = None
    _XCOM_BACKEND_STATUS_FINGERPRINT = fingerprint
    _XCOM_BACKEND = None


def _probe_optional_cold_backend(*, probe_compute: bool) -> ColdAttenuationBackendStatus:
    checked_at = time.time()
    wrapper_archive = _wrapper_archive_path()
    source_archive = _source_archive_path()
    wrapper_present = bool(wrapper_archive.exists())
    backend_name = "XCOM"
    backend_fingerprint = _archive_fingerprint(wrapper_archive) if wrapper_present else (_archive_fingerprint(source_archive) if source_archive.exists() else None)
    if not wrapper_present:
        if source_archive.exists():
            return ColdAttenuationBackendStatus(
                available=False,
                status="source_only",
                message="Vendor XCOM source bundle is present, but the Python wrapper archive is not available.",
                backend_name=backend_name,
                backend_fingerprint=backend_fingerprint,
                wrapper_present=False,
                checked_at=checked_at,
            )
        return ColdAttenuationBackendStatus(
            available=False,
            status="unavailable",
            message="Optional XCOM backend is not installed.",
            backend_name=backend_name,
            backend_fingerprint=None,
            wrapper_present=False,
            checked_at=checked_at,
        )
    try:
        module = _load_helios_xcom_module()
    except Exception as exc:
        return ColdAttenuationBackendStatus(
            available=False,
            status="import_failed",
            message=f"XCOM wrapper import failed: {exc}",
            backend_name=backend_name,
            backend_fingerprint=backend_fingerprint,
            wrapper_present=True,
            import_ok=False,
            checked_at=checked_at,
        )
    try:
        client = module.default_client()
        client_backend = getattr(client, "backend", None)
        backend_fingerprint = str(getattr(client_backend, "backend_fingerprint", backend_fingerprint or backend_name))
        backend_name = str(getattr(client_backend, "backend_name", backend_name) or backend_name)
    except Exception as exc:
        return ColdAttenuationBackendStatus(
            available=False,
            status="client_failed",
            message=f"XCOM client initialization failed: {exc}",
            backend_name=backend_name,
            backend_fingerprint=backend_fingerprint,
            wrapper_present=True,
            import_ok=True,
            client_ok=False,
            checked_at=checked_at,
        )
    if not probe_compute:
        return ColdAttenuationBackendStatus(
            available=False,
            status="client_ok_unchecked",
            message="XCOM wrapper imported and client initialized; compute health is checked on explicit XCOM requests.",
            backend_name=backend_name,
            backend_fingerprint=backend_fingerprint,
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=False,
            checked_at=checked_at,
        )
    try:
        curve = module.compute_multizone_transmission(
            client,
            material_specs=["Fe"],
            density_g_cm3=[1.0e-6],
            path_length_cm=[1.0e-6],
            energies_kev=[8.0],
        )
        transmission = np.asarray(getattr(curve, "transmission", ()), dtype=np.float64)
        if transmission.size == 0 or not np.all(np.isfinite(transmission)):
            raise RuntimeError("Smoke compute returned no finite transmission values.")
    except Exception as exc:
        return ColdAttenuationBackendStatus(
            available=False,
            status="compute_failed",
            message=f"XCOM smoke compute failed: {exc}",
            backend_name=backend_name,
            backend_fingerprint=backend_fingerprint,
            wrapper_present=True,
            import_ok=True,
            client_ok=True,
            compute_ok=False,
            checked_at=checked_at,
        )
    return ColdAttenuationBackendStatus(
        available=True,
        status="active",
        message="XCOM wrapper imported and smoke compute succeeded for this session.",
        backend_name=backend_name,
        backend_fingerprint=backend_fingerprint,
        wrapper_present=True,
        import_ok=True,
        client_ok=True,
        compute_ok=True,
        checked_at=checked_at,
    )


def describe_optional_cold_backend(*, probe_compute: bool = False, force: bool = False) -> ColdAttenuationBackendStatus:
    global _XCOM_BACKEND_BASIC_STATUS, _XCOM_BACKEND_FULL_STATUS
    with _BACKEND_STATUS_LOCK:
        _reset_backend_status_if_needed()
        cached = _XCOM_BACKEND_FULL_STATUS if probe_compute else (_XCOM_BACKEND_FULL_STATUS or _XCOM_BACKEND_BASIC_STATUS)
        if cached is not None and not force:
            return cached
    status = _probe_optional_cold_backend(probe_compute=probe_compute)
    with _BACKEND_STATUS_LOCK:
        _reset_backend_status_if_needed()
        if probe_compute:
            _XCOM_BACKEND_FULL_STATUS = status
            if _XCOM_BACKEND_BASIC_STATUS is None:
                _XCOM_BACKEND_BASIC_STATUS = status
        else:
            _XCOM_BACKEND_BASIC_STATUS = status
    return status


class HeliosXcomBackend:
    """Optional cold-attenuation backend implemented through helios_xcom."""

    backend_name = "XCOM"

    def __init__(self, *, module: object | None = None, client: object | None = None) -> None:
        self._module = _load_helios_xcom_module() if module is None else module
        self._client = self._module.default_client() if client is None else client

    @property
    def backend_fingerprint(self) -> str:
        return str(self._client.backend.backend_fingerprint)

    def compute_transmission(self, request: ColdAttenuationRequest) -> ColdAttenuationResult:
        material_specs: list[str] = []
        density_values: list[float] = []
        path_values: list[float] = []
        for zone in request.zones:
            label = str(zone.material_label or "").strip()
            if not label:
                raise ValueError(
                    f"XCOM refinement requires a resolvable material label for material {int(zone.material_id)} "
                    f"({zone.material_display_label or 'unknown'})."
                )
            material_specs.append(label)
            density_values.append(float(zone.density_g_cm3))
            path_values.append(float(zone.path_length_cm))
        curve = self._module.compute_multizone_transmission(
            self._client,
            material_specs=material_specs,
            density_g_cm3=density_values,
            path_length_cm=path_values,
            energies_kev=request.photon_energies_kev,
        )
        grouped_budgets: dict[str, dict[str, object]] = {}
        for budget in getattr(curve, "per_material", ()):
            label = str(getattr(budget, "label", "") or "").strip() or "unknown"
            existing = grouped_budgets.get(
                label,
                {
                    "label": label,
                    "areal_density_g_cm2": 0.0,
                    "optical_depth": [0.0 for _ in getattr(budget, "optical_depth", ())],
                },
            )
            existing["areal_density_g_cm2"] = float(existing["areal_density_g_cm2"]) + float(getattr(budget, "areal_density_g_cm2", 0.0))
            for index, value in enumerate(getattr(budget, "optical_depth", ())):
                existing["optical_depth"][index] = float(existing["optical_depth"][index]) + float(value)
            grouped_budgets[label] = existing
        metadata = {
            "backend_name": self.backend_name,
            "backend_fingerprint": self.backend_fingerprint,
            "attenuation_mode": str(getattr(curve, "attenuation_mode", "total_with_coherent")),
            "optical_depth": [float(value) for value in getattr(curve, "optical_depth", ())],
            "material_budgets": [
                {
                    "label": label,
                    "areal_density_g_cm2": float(payload["areal_density_g_cm2"]),
                    "optical_depth": [float(value) for value in payload["optical_depth"]],
                }
                for label, payload in sorted(grouped_budgets.items())
            ],
        }
        return ColdAttenuationResult(
            energies_kev=np.asarray(getattr(curve, "energies_kev", ()), dtype=np.float64),
            transmission=np.asarray(getattr(curve, "transmission", ()), dtype=np.float64),
            metadata=metadata,
        )


def load_optional_cold_backend(*, require_compute_ok: bool = False) -> OptionalColdAttenuationBackend | None:
    global _XCOM_BACKEND
    status = describe_optional_cold_backend(probe_compute=bool(require_compute_ok))
    if require_compute_ok and not status.available:
        return None
    if _XCOM_BACKEND is None:
        try:
            _XCOM_BACKEND = HeliosXcomBackend()
        except Exception:
            return None
    return _XCOM_BACKEND


def build_cold_attenuation_request(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters,
    geometry,
    photon_energies_kev: tuple[float, ...],
) -> ColdAttenuationRequest:
    """Build the optional-backend request from the active HELIOS snapshot subset.

    Material identity is resolved from the parsed material tables. Explicit
    formula/composition metadata wins when present; otherwise EOS metadata is
    the authoritative practical identity seed, with opacity metadata as the
    secondary fallback.
    """

    mask, _, _ = build_analysis_mask(
        dataset,
        context,
        snapshot_index=snapshot_index,
        geometry=geometry,
        reuse_viewer_subset=parameters.reuse_viewer_subset,
        derived_region_ids=parameters.derived_region_ids,
        derived_material_ids=parameters.derived_material_ids,
        exclude_entry_region=parameters.exclude_entry_region,
        exclude_low_density=parameters.exclude_low_density,
        min_density_g_cm3=parameters.min_density_g_cm3,
        exclude_opposite_velocity=parameters.exclude_opposite_velocity,
        zone_index_lower=parameters.zone_index_lower,
        zone_index_upper=parameters.zone_index_upper,
        weighting_mode="path_integrated",
    )

    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
    path = np.asarray(path_length_cm(dataset, snapshot_index, geometry), dtype=np.float64)
    material_resolutions = resolve_material_identities(dataset.materials if isinstance(dataset.materials, dict) else {})

    zones: list[ColdAttenuationZone] = []
    for zone_index in np.flatnonzero(mask):
        material_id = int(abs(dataset.zone_material_index[zone_index]))
        resolution = material_resolutions.get(
            material_id,
            MaterialResolution(
                material_id=material_id,
                backend_label=None,
                display_label=f"material {material_id}",
                status="unresolved",
            ),
        )
        zones.append(
            ColdAttenuationZone(
                zone_index=int(zone_index + 1),
                region_id=int(dataset.zone_region_id[zone_index]),
                material_id=material_id,
                material_label=resolution.backend_label,
                density_g_cm3=float(density[zone_index]),
                path_length_cm=float(path[zone_index]),
                material_display_label=_format_material_resolution_label(resolution),
                material_resolution_status=str(resolution.status),
                material_canonical_key=resolution.canonical_key,
            )
        )

    return ColdAttenuationRequest(
        snapshot_index=int(snapshot_index),
        observation_side=str(geometry.observation_side),
        line_of_sight_cosine=float(geometry.line_of_sight_cosine),
        photon_energies_kev=tuple(float(value) for value in photon_energies_kev),
        zones=tuple(zones),
    )
