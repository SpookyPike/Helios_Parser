"""Article-facing plasmon reference inputs.

The ambient/driven dispersion series now live in explicit JSON files with
provenance metadata instead of being hidden as scaffold-quality Python dicts.
Some spectrum references remain compact in-code helpers because they are not yet
part of the primary article benchmark harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DATA_DIR = Path(__file__).resolve().parent / "reference_data" / "plasmon"


def _load_reference_json(name: str) -> dict[str, Any]:
    path = _DATA_DIR / str(name)
    return json.loads(path.read_text(encoding="utf-8"))


GAWNE_2024_AMBIENT_AL_REFERENCE = {
    "source": "Gawne et al., Phys. Rev. B 109, L241112 (2024), Supplemental Fig. 1",
    "doi": "10.1103/PhysRevB.109.L241112",
    "photon_energy_kev": 8.31,
    "material": "Al",
    "rho_g_cm3": 2.70,
    "te_ev": 0.30,
    "provenance": {
        "quality": "manual_digitization_v1",
        "notes": [
            "Spectrum shapes remain a coarse manual extraction for overlay-only validation.",
            "Use the dispersion JSON references as the primary article-facing benchmark inputs."
        ]
    },
    "curves": {
        0.25: {"energy_ev": [5.0, 7.0, 9.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 23.0, 25.0], "intensity": [0.00, 0.01, 0.05, 0.15, 0.42, 0.70, 0.92, 1.00, 0.78, 0.47, 0.21, 0.06, 0.01, 0.00]},
        0.55: {"energy_ev": [5.0, 7.0, 9.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 23.0, 25.0], "intensity": [0.00, 0.01, 0.04, 0.12, 0.35, 0.61, 0.86, 0.98, 0.84, 0.56, 0.28, 0.10, 0.02, 0.00]},
        0.92: {"energy_ev": [5.0, 7.0, 9.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 23.0, 25.0], "intensity": [0.00, 0.01, 0.03, 0.10, 0.28, 0.51, 0.74, 0.93, 0.89, 0.66, 0.38, 0.14, 0.03, 0.00]},
        1.26: {"energy_ev": [5.0, 7.0, 9.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 23.0, 25.0], "intensity": [0.00, 0.01, 0.03, 0.08, 0.22, 0.41, 0.63, 0.84, 0.94, 0.79, 0.54, 0.22, 0.06, 0.01]}
    },
}


GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5 = _load_reference_json("ambient_al_dispersion_figs5.json")
USER_DRIVEN_AL_DISPERSION_REFERENCE = _load_reference_json("driven_al_dispersion_article.json")
