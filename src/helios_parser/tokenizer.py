from __future__ import annotations

import re
from dataclasses import dataclass, field


FIELD_ALIAS_MAP = {
    "radius": "radius",
    "zone_width": "zone_width",
    "mass_dens": "density",
    "velocity": "velocity",
    "rad_temp": "temperature_radiation",
    "ion_temp": "temperature_i",
    "elec_temp": "temperature_e",
    "ion_press": "pressure_i",
    "elec_press": "pressure_e",
    "rad_press": "pressure_radiation",
    "compression": "compression",
    "compressio": "compression",
    "elec_dens": "electron_density",
    "n_elec_dens": "electron_density",
    "mean_chg": "mean_charge",
    "art_visc": "artificial_viscosity",
    "art_vis": "artificial_viscosity",
    "ion_energy": "ion_energy",
    "ele_energy": "electron_energy",
    "ionheatcap": "ion_heat_capacity",
    "eleheatcap": "electron_heat_capacity",
    "rad_energy": "radiation_energy",
    "kin_energy": "kinetic_energy",
    "radheating": "radiation_heating",
    "radcooling": "radiation_cooling",
    "radsink": "radiation_sink",
    "radnetheat": "radiation_net_heating",
    "lasersrc": "laser_source",
    "laserdep": "laser_deposition",
}


@dataclass(slots=True)
class RegexTokenizer:
    number_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:E[-+]?\d+|[-+]\d{2,3})?")
    )
    malformed_exponent_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"(?<![A-Za-z])([+-]?(?:\d+\.\d*|\.\d+))([+-]\d{2,3})(?!\d)")
    )
    cycle_header_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"^\s*Cycle\s+Time\s+\(s\)", re.MULTILINE)
    )
    field_alias_map: dict[str, str] = field(default_factory=lambda: dict(FIELD_ALIAS_MAP))
    _identifier_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _field_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def normalize_label(self, label: str) -> str:
        return re.sub(r"\s+", " ", label.strip().lstrip("#")).strip()

    def normalize_unit(self, unit: str) -> str:
        normalized = re.sub(r"\s+", " ", unit.strip())
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        return normalized

    def normalize_identifier(self, label: str) -> str:
        cached = self._identifier_cache.get(label)
        if cached is not None:
            return cached
        normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        self._identifier_cache[label] = normalized
        return normalized

    def canonical_field_name(self, label: str) -> str:
        cached = self._field_cache.get(label)
        if cached is not None:
            return cached
        normalized = self.normalize_identifier(label)
        canonical = self.field_alias_map.get(normalized, normalized)
        self._field_cache[label] = canonical
        return canonical

    def canonical_diagnostic_name(self, label: str) -> str:
        normalized = label.replace("->", " to ").replace("+", " ").replace("/", " ")
        normalized = normalized.replace("(", " ").replace(")", " ")
        return self.normalize_identifier(normalized)

    def normalize_number_token(self, token: str) -> str:
        if "D" in token or "d" in token:
            token = token.replace("D", "E").replace("d", "E")
        if "E" in token or "e" in token:
            return token
        for index in range(len(token) - 1, 0, -1):
            char = token[index]
            if char not in "+-":
                continue
            exponent = token[index + 1 :]
            if len(exponent) < 2 or len(exponent) > 3 or not exponent.isdigit():
                continue
            mantissa = token[:index]
            if not mantissa or mantissa in {"+", "-"} or "." not in mantissa:
                continue
            return f"{mantissa}E{char}{exponent}"
        return token

    def normalize_numeric_text(self, text: str) -> str:
        normalized = self.malformed_exponent_pattern.sub(r"\1E\2", text)
        if "D" in normalized or "d" in normalized:
            normalized = normalized.replace("D", "E").replace("d", "E")
        return normalized

    def split_table_row_tokens(self, line: str) -> list[str]:
        parts = line.split()
        if not parts:
            return []
        return [self.normalize_number_token(part) for part in parts]

    def extract_number_tokens(self, line: str) -> list[str]:
        tokens = self.number_pattern.findall(line)
        normalized: list[str] = []
        for token in tokens:
            normalized.append(self.normalize_number_token(token))
        return normalized
