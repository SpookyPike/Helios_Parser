"""Canonical neutral-element mass fractions for XCOM coefficient queries."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import hashlib
import json
import re


_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()[:100]
_SYMBOL_TO_Z = {symbol: index for index, symbol in enumerate(_SYMBOLS, 1)}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def element_symbols() -> tuple[str, ...]:
    """Return symbols supported by the bundled XCOM atomic database (Z=1–100)."""
    return tuple(_SYMBOLS)


@dataclass(frozen=True, slots=True)
class CanonicalComposition:
    """Composition normalized to sorted elemental mass fractions."""

    mass_fractions: tuple[tuple[str, float], ...]
    atomic_weight_fingerprint: str

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple((symbol, float(weight).hex()) for symbol, weight in self.mass_fractions)


@lru_cache(maxsize=1)
def atomic_weights() -> tuple[dict[str, float], str]:
    resource = resources.files("helios.services.derived.resources").joinpath("xcom_atomic_weights.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    weights = {str(symbol): float(value) for symbol, value in payload["weights"].items()}
    fingerprint = hashlib.sha256(resource.read_bytes()).hexdigest()
    if set(weights) != set(_SYMBOL_TO_Z) or any(value <= 0 for value in weights.values()):
        raise ValueError("Packaged XCOM atomic-weight resource is incomplete or invalid.")
    return weights, fingerprint


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple element/count formula (e.g. CH, SiO2, C22H10N2O5)."""
    text = str(formula).strip()
    if not text:
        raise ValueError("Material formula is empty.")
    counts: dict[str, int] = {}
    cursor = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != cursor:
            raise ValueError(f"Unsupported formula syntax near {text[cursor:]!r}.")
        symbol, raw_count = match.groups()
        if symbol not in _SYMBOL_TO_Z:
            raise ValueError(f"Unsupported XCOM element symbol {symbol!r}; supported atomic numbers are 1–100.")
        count = int(raw_count or "1")
        if count < 1 or count > 10000:
            raise ValueError("Formula atom counts must be between 1 and 10,000.")
        counts[symbol] = counts.get(symbol, 0) + count
        cursor = match.end()
    if cursor != len(text) or not counts:
        raise ValueError(f"Unsupported formula syntax in {text!r}.")
    return counts


def composition_from_formula(formula: str) -> CanonicalComposition:
    counts = parse_formula(formula)
    weights, fingerprint = atomic_weights()
    masses = {symbol: count * weights[symbol] for symbol, count in counts.items()}
    total = sum(masses.values())
    return CanonicalComposition(
        tuple(sorted((symbol, mass / total) for symbol, mass in masses.items())),
        fingerprint,
    )


def composition_from_mass_fractions(values: dict[str, float]) -> CanonicalComposition:
    """Validate and normalize a user-supplied elemental mass-fraction vector."""
    if not values:
        raise ValueError("At least one elemental mass fraction is required.")
    normalized: dict[str, float] = {}
    for raw_symbol, raw_value in values.items():
        symbol = str(raw_symbol).strip().capitalize()
        if symbol not in _SYMBOL_TO_Z:
            raise ValueError(f"Unsupported XCOM element symbol {raw_symbol!r}; supported atomic numbers are 1–100.")
        value = float(raw_value)
        if not (value >= 0.0 and value < float("inf")):
            raise ValueError("Mass fractions must be finite and nonnegative.")
        normalized[symbol] = normalized.get(symbol, 0.0) + value
    total = sum(normalized.values())
    if total <= 0.0 or not (0.99999999 <= total <= 1.00000001):
        raise ValueError("Elemental mass fractions must sum to 1.0 (within 1e-8).")
    weights, fingerprint = atomic_weights()
    return CanonicalComposition(
        tuple(sorted((symbol, value / total) for symbol, value in normalized.items() if value > 0.0)),
        fingerprint,
    )
