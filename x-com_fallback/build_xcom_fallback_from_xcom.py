from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

LINE_RE = re.compile(
    r"^\s*([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)"
)

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

ENERGIES_EV = list(range(1000, 12000 + 1, 500))
ENERGIES_MEV = [e / 1.0e6 for e in ENERGIES_EV]


def parse_output(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        vals = [float(x) for x in m.groups()]
        rows.append(
            {
                "energy_eV": int(round(vals[0] * 1.0e6)),
                "coherent_cm2_g": vals[1],
                "incoherent_cm2_g": vals[2],
                "photoelectric_cm2_g": vals[3],
                "pair_nuclear_cm2_g": vals[4],
                "pair_electron_cm2_g": vals[5],
                "mu_rho_total_cm2_g": vals[6],
                "mu_rho_no_coherent_cm2_g": vals[7],
            }
        )
    return rows


def prepare_run_dir(src_dir: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        target = run_dir / p.name
        if target.exists():
            continue
        try:
            target.symlink_to(p)
        except Exception:
            if p.is_file():
                shutil.copy2(p, target)


def run_xcom(run_dir: Path, material_key: str, display: str, mode: str, ident: str, energies_mev: list[float]) -> str:
    exe = run_dir / "xcom"
    outname = f"out_{material_key}.txt"
    outpath = run_dir / outname
    if outpath.exists():
        outpath.unlink()
    inputs = [display]
    if mode == "element":
        inputs += ["2", ident, "3"]
    elif mode == "compound":
        inputs += ["3", ident]
    else:
        raise ValueError(mode)
    inputs += ["3", "1", str(len(energies_mev))]
    inputs += [f"{e:.6f}" for e in energies_mev]
    inputs += ["N", outname, "1"]
    proc = subprocess.run(
        [str(exe)],
        input=("\n".join(inputs) + "\n").encode(),
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.decode(errors="replace"))
    return outpath.read_text(errors="replace")


def main() -> None:
    here = Path(__file__).resolve().parent
    src_dir = here / "XCOM"
    run_dir = here / "XCOM_run"
    prepare_run_dir(src_dir, run_dir)

    materials = {}
    for key, display, mode, ident, note in MATERIALS:
        text = run_xcom(run_dir, key, display, mode, ident, ENERGIES_MEV)
        rows = parse_output(text)
        if len(rows) != len(ENERGIES_EV):
            raise RuntimeError(f"{key}: expected {len(ENERGIES_EV)} rows, got {len(rows)}")
        materials[key] = {
            "display_name": display,
            "mode": mode,
            "identifier": ident,
            "note": note,
            "rows": rows,
        }

    materials["glass_sio2"] = {
        "display_name": "glass ~ SiO2",
        "mode": "compound_alias",
        "identifier": "SiO2",
        "note": "glass surrogate assumed as SiO2; identical numbers to sio2",
        "rows": [dict(r) for r in materials["sio2"]["rows"]],
    }

    json_path = here / "xcom_fallback_1keV_12keV_extended.json"
    csv_path = here / "xcom_fallback_1keV_12keV_extended.csv"

    payload = {
        "metadata": {
            "source": "NIST XCOM Version 3.1 computed locally from uploaded Fortran source/data",
            "energy_range_eV": [1000, 12000],
            "energy_step_eV": 500,
            "quantity_units": "cm^2/g",
            "default_quantity": "mu_rho_total_cm2_g",
        },
        "materials": materials,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "material_key",
                "display_name",
                "mode",
                "identifier",
                "note",
                "energy_eV",
                "coherent_cm2_g",
                "incoherent_cm2_g",
                "photoelectric_cm2_g",
                "pair_nuclear_cm2_g",
                "pair_electron_cm2_g",
                "mu_rho_total_cm2_g",
                "mu_rho_no_coherent_cm2_g",
            ]
        )
        for key, mat in materials.items():
            for row in mat["rows"]:
                w.writerow(
                    [
                        key,
                        mat["display_name"],
                        mat["mode"],
                        mat["identifier"],
                        mat["note"],
                        row["energy_eV"],
                        row["coherent_cm2_g"],
                        row["incoherent_cm2_g"],
                        row["photoelectric_cm2_g"],
                        row["pair_nuclear_cm2_g"],
                        row["pair_electron_cm2_g"],
                        row["mu_rho_total_cm2_g"],
                        row["mu_rho_no_coherent_cm2_g"],
                    ]
                )


if __name__ == "__main__":
    main()
