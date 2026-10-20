from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(os.environ.get("HELIOS_PERF_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios.services.derived import xcom_hook
from helios.services.derived.xcom_hook import (
    ColdAttenuationRequest,
    ColdAttenuationZone,
    PrecomputedColdTableBackend,
    lookup_fallback_mu_rho,
)


def benchmark(zones_count: int, repeats: int) -> dict[str, object]:
    energies = (2.0, 3.0, 5.0, 8.0)
    materials = ((1, "Cu", "cu"), (2, "Al", "al"), (3, "CH", "ch"))
    zones = tuple(
        ColdAttenuationZone(
            zone_index=index,
            region_id=1 + index % len(materials),
            material_id=materials[index % len(materials)][0],
            material_label=materials[index % len(materials)][1],
            material_canonical_key=materials[index % len(materials)][2],
            density_g_cm3=1.5 + (index % 7) * 0.1,
            path_length_cm=1.0e-4 + (index % 5) * 1.0e-6,
        )
        for index in range(zones_count)
    )
    request = ColdAttenuationRequest(
        snapshot_index=0,
        observation_side="front",
        line_of_sight_cosine=1.0,
        photon_energies_kev=energies,
        zones=zones,
    )

    # Independent scalar reference deliberately resolves a coefficient per zone.
    reference_tau = np.zeros(len(energies), dtype=np.float64)
    for zone in zones:
        for energy_index, energy in enumerate(energies):
            coefficient = lookup_fallback_mu_rho(zone.material_canonical_key, energy).mu_rho_cm2_g
            reference_tau[energy_index] += coefficient * zone.density_g_cm3 * zone.path_length_cm

    backend = PrecomputedColdTableBackend()
    original_lookup = xcom_hook.lookup_fallback_mu_rho
    elapsed: list[float] = []
    lookup_calls = 0
    max_abs_error = 0.0
    for _ in range(repeats):
        with mock.patch.object(xcom_hook, "lookup_fallback_mu_rho", wraps=original_lookup) as lookup_spy:
            started = time.perf_counter()
            result = backend.compute_transmission(request)
            elapsed.append((time.perf_counter() - started) * 1000.0)
            lookup_calls += lookup_spy.call_count
        actual_tau = np.asarray(result.metadata["optical_depth"], dtype=np.float64)
        max_abs_error = max(max_abs_error, float(np.max(np.abs(actual_tau - reference_tau))))
    return {
        "zones": zones_count,
        "unique_material_energy_pairs": len(materials) * len(energies),
        "repeats": repeats,
        "median_elapsed_ms": statistics.median(elapsed),
        "elapsed_samples_ms": elapsed,
        "lookup_calls_per_run": lookup_calls / repeats,
        "max_abs_optical_depth_error": max_abs_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark grouped XCOM work against a scalar reference.")
    parser.add_argument("--zones", type=int, default=1300)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.zones < 1 or args.repeats < 1:
        parser.error("--zones and --repeats must be positive")
    print(json.dumps(benchmark(args.zones, args.repeats), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
