from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Literal

TABLE_PATH = Path(__file__).with_name("xcom_fallback_1keV_12keV_extended.json")

ALIASES = {
    "aluminium": "al",
    "aluminum": "al",
    "copper": "cu",
    "iron": "fe",
    "silicon": "si",
    "titanium": "ti",
    "gold": "au",
    "beryllium": "be",
    "carbon": "c",
    "diamond": "c",
    "graphite": "c",
    "ch_plastic": "ch",
    "plastic_ch": "ch",
    "epoxy": "epoxy_c2h4o",
    "c2h4o": "epoxy_c2h4o",
    "kapton": "kapton_c22h10n2o5",
    "polyimide": "kapton_c22h10n2o5",
    "sio2": "sio2",
    "silica": "sio2",
    "glass": "glass_sio2",
    "glass_sio2": "glass_sio2",
}

Quantity = Literal["mu_rho_total_cm2_g", "mu_rho_no_coherent_cm2_g"]


def load_table(path: str | Path = TABLE_PATH) -> dict:
    return json.loads(Path(path).read_text())


def canonical_material_key(material: str, *, path: str | Path = TABLE_PATH) -> str:
    key = material.strip().lower()
    data = load_table(path)
    if key in data["materials"]:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise KeyError(f"unknown material key/alias: {material}")


def _interp_loglog(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    lx = math.log(x)
    lx0 = math.log(x0)
    lx1 = math.log(x1)
    ly0 = math.log(y0)
    ly1 = math.log(y1)
    if lx1 == lx0:
        return y0
    ly = ly0 + (ly1 - ly0) * (lx - lx0) / (lx1 - lx0)
    return math.exp(ly)


def lookup_mu_rho(
    material: str,
    energy_eV: float,
    quantity: Quantity = "mu_rho_total_cm2_g",
    *,
    allow_extrapolation: bool = True,
    path: str | Path = TABLE_PATH,
) -> float:
    table = load_table(path)
    key = canonical_material_key(material, path=path)
    data = table["materials"][key]["rows"]
    xs = [row["energy_eV"] for row in data]
    ys = [row[quantity] for row in data]
    i = bisect_left(xs, energy_eV)
    if i < len(xs) and xs[i] == energy_eV:
        return ys[i]
    if i == 0:
        if not allow_extrapolation:
            raise ValueError("energy below table range")
        return _interp_loglog(energy_eV, xs[0], ys[0], xs[1], ys[1])
    if i == len(xs):
        if not allow_extrapolation:
            raise ValueError("energy above table range")
        return _interp_loglog(energy_eV, xs[-2], ys[-2], xs[-1], ys[-1])
    return _interp_loglog(energy_eV, xs[i - 1], ys[i - 1], xs[i], ys[i])


def transmission_from_areal_density(
    material: str,
    energy_eV: float,
    areal_density_g_cm2: float,
    quantity: Quantity = "mu_rho_total_cm2_g",
    *,
    allow_extrapolation: bool = True,
    path: str | Path = TABLE_PATH,
) -> float:
    mu_rho = lookup_mu_rho(material, energy_eV, quantity, allow_extrapolation=allow_extrapolation, path=path)
    tau = mu_rho * areal_density_g_cm2
    return math.exp(-tau)


def transmission_from_density_and_thickness(
    material: str,
    energy_eV: float,
    density_g_cm3: float,
    thickness_um: float,
    quantity: Quantity = "mu_rho_total_cm2_g",
    *,
    allow_extrapolation: bool = True,
    path: str | Path = TABLE_PATH,
) -> float:
    areal_density_g_cm2 = density_g_cm3 * thickness_um * 1.0e-4
    return transmission_from_areal_density(
        material,
        energy_eV,
        areal_density_g_cm2,
        quantity,
        allow_extrapolation=allow_extrapolation,
        path=path,
    )
