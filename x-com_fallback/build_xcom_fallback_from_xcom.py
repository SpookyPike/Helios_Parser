"""Rebuild the cold 1–35 keV table from the bundled NIST XCOM 3.1 source.

Requires gfortran at generation time only. Uses binary64 arithmetic, full-precision
text output, explicit two-sided absorption edges, and adaptive log-log sampling.
The interpolation tolerance is numerical; it is not a physical accuracy claim.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import random
import subprocess
import tarfile
import tempfile

MATERIALS = [
    ("al", "Al", "element", "Al", "aluminium"),
    ("cu", "Cu", "element", "Cu", "copper"),
    ("fe", "Fe", "element", "Fe", "iron"),
    ("si", "Si", "element", "Si", "silicon"),
    ("ti", "Ti", "element", "Ti", "titanium"),
    ("au", "Au", "element", "Au", "gold"),
    ("be", "Be", "element", "Be", "beryllium"),
    ("c", "C", "element", "C", "carbon"),
    ("ch", "CH", "compound", "CH", "idealized CH plastic"),
    ("epoxy_c2h4o", "C2H4O", "compound", "C2H4O", "epoxy surrogate assumed as C2H4O"),
    ("kapton_c22h10n2o5", "C22H10N2O5", "compound", "C22H10N2O5", "Kapton/polyimide surrogate"),
    ("sio2", "SiO2", "compound", "SiO2", "silicon dioxide"),
]
COLUMNS = (
    "coherent_cm2_g", "incoherent_cm2_g", "photoelectric_cm2_g",
    "pair_nuclear_cm2_g", "pair_electron_cm2_g",
    "mu_rho_total_cm2_g", "mu_rho_no_coherent_cm2_g",
)
NUMBER = re.compile(r"[-+]?\d+\.\d+[EeDd][-+]\d+")
_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()[:100]
COMPILE_FLAGS = ["-std=legacy", "-O2", "-fdefault-real-8", "-fdefault-double-8"]
MIN_EV, MAX_EV = 1000.0, 35000.0
BATCH_SIZE = 64


def parse_output(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        matches = list(NUMBER.finditer(line))
        if len(matches) != 8:
            continue
        values = [float(m.group().replace("D", "E")) for m in matches]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("XCOM returned invalid coefficients")
        row = {"energy_eV": round(values[0] * 1e6, 8), **dict(zip(COLUMNS, values[1:]))}
        label = line[:matches[0].start()].strip()
        if label:
            if not rows or row["energy_eV"] != rows[-1]["energy_eV"]:
                raise ValueError(f"Unpaired absorption edge: {label}")
            rows[-1].update(edge_side="below", edge_label=label)
            row.update(edge_side="above", edge_label=label)
        rows.append(row)
    return rows


def precision_source(source: str) -> str:
    """Change serialization/legacy declarations, preserving the vendor formulas."""
    replacements = {
        "FORMAT(7X,1PE10.3,1P7E10.3)": "FORMAT(7X,1PE24.16,1P7E24.16)",
        "FORMAT(I3,1X,A2,1X,1PE10.3,1P7E10.3)": "FORMAT(I3,1X,A2,1X,1PE24.16,1P7E24.16)",
        "PARAMETER (MEA=1200)": "PARAMETER (MEA=600)",
        "DIMENSION X(1),F(1),A(1),B(1),C(1),D(1)": "DIMENSION X(*),F(*),A(*),B(*),C(*),D(*)",
        "DIMENSION X(1),A(1),B(1),C(1),D(1)": "DIMENSION X(*),A(*),B(*),C(*),D(*)",
        "DIMENSION X(1000),Y(1000)": "DIMENSION X(*),Y(*)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise ValueError(f"Unexpected vendor source: missing {old}")
        # Keep the 72-column fixed-form limit; SMS annotations occupy later columns.
        lines = []
        for line in source.splitlines():
            if old in line and not line.startswith("C"):
                line = line[:72].replace(old, new).rstrip()
                if len(line) > 72:
                    raise ValueError("Precision patch exceeds fixed-form line length")
            lines.append(line)
        source = "\n".join(lines) + "\n"
    return source


def read_atomic_weights(path: Path, archive_hash: str) -> dict[str, object]:
    values: list[float] = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        if fields and fields[0].isdigit() and len(fields) > 1:
            values.extend(float(value) for value in re.findall(r"\d+(?:\.\d+)?", fields[1]))
    symbols = _ELEMENT_SYMBOLS
    if len(values) != len(symbols):
        raise ValueError(f"Expected {len(symbols)} XCOM atomic weights; found {len(values)}.")
    return {
        "source": "XCOM 3.1 bundled ATWTS.DAT",
        "source_archive_sha256": archive_hash,
        "weights": dict(zip(symbols, values)),
    }


class NativeXcom:
    def __init__(self, directory: Path, compiler: str):
        self.directory = directory
        command = [compiler, *COMPILE_FLAGS, "XCOM.f", "-o", "xcom"]
        proc = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        self.executable = directory / "xcom"
        if len(list(directory.glob("MDATX3.*"))) != 100:
            raise ValueError("Incomplete XCOM element database")

    def query(self, mode: str, identifier: str, energies: list[float] | None) -> list[dict]:
        inputs = [identifier]
        inputs += ["2", identifier, "3"] if mode == "element" else ["3", identifier]
        if energies is None:
            inputs += ["1"]
        else:
            if not 1 <= len(energies) <= BATCH_SIZE or energies != sorted(set(energies)):
                raise ValueError("Expected a sorted unique batch of 1–64 energies")
            inputs += ["3", "1", str(len(energies)), *(f"{e / 1e6:.16e}" for e in energies), "N"]
        inputs += ["out.txt", "1"]
        out = self.directory / "out.txt"
        out.unlink(missing_ok=True)
        proc = subprocess.run([str(self.executable)], input="\n".join(inputs) + "\n",
                              cwd=self.directory, text=True, capture_output=True, timeout=30)
        if proc.returncode or not out.exists():
            raise RuntimeError(proc.stdout + proc.stderr)
        rows = parse_output(out.read_text())
        if not rows:
            raise ValueError("XCOM produced no numerical rows")
        if energies is not None:
            if len(rows) != len(energies):
                raise ValueError("Native output row count differs from requested energies")
            for row, energy in zip(rows, energies):
                if not math.isclose(row["energy_eV"], energy, rel_tol=1e-12, abs_tol=1e-7):
                    raise ValueError("Native output energy order differs from the request")
                row["energy_eV"] = energy
        return rows


def log_interp(energy, left, right, column):
    fraction = math.log(energy / left["energy_eV"]) / math.log(right["energy_eV"] / left["energy_eV"])
    a, b = left[column], right[column]
    return math.exp(math.log(a) + fraction * math.log(b / a)) if a > 0 and b > 0 else a + fraction * (b-a)


def build_material(native: NativeXcom, mode: str, identifier: str, tolerance: float) -> tuple[list[dict], dict]:
    nodes: dict[float, list[dict]] = {}
    for row in native.query(mode, identifier, None):
        if MIN_EV <= row["energy_eV"] <= MAX_EV:
            nodes.setdefault(row["energy_eV"], []).append(row)
    sampled: dict[float, dict] = {}

    def sample(energies):
        missing = sorted(set(energies) - sampled.keys())
        for start in range(0, len(missing), BATCH_SIZE):
            for row in native.query(mode, identifier, missing[start:start + BATCH_SIZE]):
                sampled[row["energy_eV"]] = row

    seeds = [float(e) for e in range(1000, int(MAX_EV) + 1, 500) if float(e) not in nodes]
    sample(seeds)
    nodes.update({e: [sampled[e]] for e in seeds})
    for iteration in range(14):
        checks = []
        energies = sorted(nodes)
        for a, b in zip(energies, energies[1:]):
            left, right = nodes[a][-1], nodes[b][0]
            for fraction in (.25, .5, .75):
                energy = math.exp(math.log(a) + fraction * math.log(b / a))
                checks.append((energy, left, right))
        sample(e for e, _, _ in checks)
        additions, maximum_error = set(), 0.0
        for energy, left, right in checks:
            actual = sampled[energy]
            error = max(abs(log_interp(energy, left, right, c) - actual[c]) / actual[c]
                        if actual[c] > 0 else abs(log_interp(energy, left, right, c)) for c in COLUMNS)
            maximum_error = max(maximum_error, error)
            if error > tolerance:
                additions.add(energy)
        if not additions:
            rows = [row for energy in sorted(nodes) for row in nodes[energy]]
            return rows, {"max_probe_relative_error": maximum_error,
                          "validation_probes": len(checks), "native_samples": len(sampled),
                          "refinement_iterations": iteration}
        nodes.update({e: [sampled[e]] for e in additions})
    raise RuntimeError(f"Adaptive interpolation tolerance not reached for {identifier}")


def verify_material(native: NativeXcom, material: dict, tolerance: float) -> dict:
    """Independent deterministic holdouts, never used to choose table nodes."""
    rng = random.Random(20261020)
    rows = material["rows"]
    xs = [r["energy_eV"] for r in rows]
    probes = {math.exp(rng.uniform(math.log(MIN_EV), math.log(MAX_EV))) for _ in range(200)}
    for row in rows:
        if row.get("edge_side") == "above":
            for delta in (-.01, .01):
                if MIN_EV < row["energy_eV"] + delta < MAX_EV:
                    probes.add(row["energy_eV"] + delta)
    energies = sorted(probes)
    maximum = 0.0
    for start in range(0, len(energies), BATCH_SIZE):
        actual_rows = native.query(material["mode"], material["identifier"], energies[start:start+BATCH_SIZE])
        for actual in actual_rows:
            energy = actual["energy_eV"]
            i = bisect_left(xs, energy)
            if i < len(xs) and xs[i] == energy:
                predicted = rows[i]
            else:
                predicted = {c: log_interp(energy, rows[i-1], rows[i], c) for c in COLUMNS}
            maximum = max(maximum, *(abs(predicted[c] / actual[c] - 1) if actual[c] > 0
                                     else abs(predicted[c]) for c in COLUMNS))
    if maximum > tolerance:
        raise ValueError(f"Independent native holdout error {maximum:.6g} exceeds {tolerance:.6g}")
    return {"holdout_probes": len(energies), "max_holdout_relative_error": maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--verify-only", action="store_true", help="Check the committed table against independent native holdouts without writing it")
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.relative_tolerance) or not 0 < args.relative_tolerance <= 1e-3:
        parser.error("relative tolerance must be finite and within (0, 0.001]")
    here = Path(__file__).resolve().parent
    repo = here.parent
    archive = repo / "XCOM.tar.gz"
    source_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    materials = {}
    atomic_weight_payload = None
    # Compile and execute on a Linux-native filesystem; the repo may live on NTFS.
    with tempfile.TemporaryDirectory(prefix="helios-xcom-table-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as data:
            members = [m for m in data.getmembers() if '/.rsrc/' not in m.name and not m.name.endswith('/.rsrc')]
            data.extractall(root, members=members, filter="data")
        source = root / "XCOM" / "XCOM.f"
        patched = precision_source(source.read_text())
        source.write_text(patched)
        atomic_weight_payload = read_atomic_weights(root / "XCOM" / "ATWTS.DAT", source_hash)
        native = NativeXcom(source.parent, args.compiler)
        if args.verify_only:
            payload = json.loads((here / "xcom_fallback_1keV_12keV_extended.json").read_text())
            expected_range = [MIN_EV, MAX_EV]
            if payload["metadata"].get("energy_range_eV") != expected_range:
                raise ValueError("Committed table energy range does not match this generator")
            tolerance = float(payload["metadata"]["relative_interpolation_tolerance"])
            validation = {}
            for key, material in payload["materials"].items():
                if material["mode"] == "compound_alias":
                    continue
                validation[key] = verify_material(native, material, tolerance)
                print(f"{key}: {validation[key]}", flush=True)
            if args.validation_report:
                args.validation_report.parent.mkdir(parents=True, exist_ok=True)
                args.validation_report.write_text(json.dumps(validation, indent=2) + "\n")
            return
        for key, display, mode, identifier, note in MATERIALS:
            rows, validation = build_material(native, mode, identifier, args.relative_tolerance * .5)
            validation.update(verify_material(native, {"mode": mode, "identifier": identifier, "rows": rows}, args.relative_tolerance))
            materials[key] = {"display_name": display, "mode": mode, "identifier": identifier,
                              "note": note, "validation": validation, "rows": rows}
            print(f"{key}: {len(rows)} rows; max probe error {validation['max_probe_relative_error']:.3g}", flush=True)
    materials["glass_sio2"] = {**materials["sio2"], "display_name": "glass ~ SiO2",
                               "mode": "compound_alias", "note": "glass surrogate assumed as SiO2; identical numbers to sio2"}
    payload = {"metadata": {
        "source": "NIST XCOM Version 3.1, local Fortran source/data",
        "schema_version": 2, "generator_version": 2,
        "source_archive_sha256": source_hash,
        "precision_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "compiler_flags": COMPILE_FLAGS,
        "arithmetic": "IEEE binary64; 17 significant digits serialized",
        "energy_range_eV": [MIN_EV, MAX_EV],
        "sampling": "adaptive log-log with explicit below/above absorption-edge pairs",
        "relative_interpolation_tolerance": args.relative_tolerance,
        "refinement_probe_tolerance": args.relative_tolerance * .5,
        "validation": "native quarter/midpoint probes plus 200 independent seeded holdouts per material and both sides of each edge",
        "exact_edge_policy": "first row (below edge)",
        "precision_note": "Interpolation tolerance and storage precision do not represent physical cross-section accuracy.",
        "quantity_units": "cm^2/g", "default_quantity": "mu_rho_total_cm2_g",
    }, "materials": materials}
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    name = "xcom_fallback_1keV_12keV_extended"
    (here / f"{name}.json").write_text(text, encoding="utf-8")
    (repo / "src/helios/services/derived/resources" / f"{name}.json").write_text(text, encoding="utf-8")
    atomic_weight_text = json.dumps(atomic_weight_payload, indent=2, allow_nan=False) + "\n"
    (here / "xcom_atomic_weights.json").write_text(atomic_weight_text, encoding="utf-8")
    (repo / "src/helios/services/derived/resources" / "xcom_atomic_weights.json").write_text(
        atomic_weight_text, encoding="utf-8"
    )
    with (here / f"{name}.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["material_key", "display_name", "mode", "identifier", "note", "energy_eV", *COLUMNS, "edge_side", "edge_label"]
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, material in materials.items():
            for row in material["rows"]:
                writer.writerow({"material_key": key, **{k: material[k] for k in fields[1:5]}, **row})


if __name__ == "__main__":
    main()
