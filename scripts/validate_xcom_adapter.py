from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile

from dataclasses import asdict

from _validation_common import REPO_ROOT, VALIDATION_ROOT, build_registry, preferred_hdf5_records, save_json

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from helios.platform.archive_utils import extract_archive, inspect_archive
from helios.services.derived import DerivedAnalysisParameters, build_cold_attenuation_request, load_run_data
from helios.services.derived.selection import build_analysis_geometry
from helios.runtime import RunContext


OUTPUT_DIR = VALIDATION_ROOT / "xcom_adapter"


def _request_preview(record_path: Path) -> dict[str, object]:
    dataset = load_run_data(record_path)
    all_region_ids = tuple(int(value) for value in sorted(set(dataset.zone_region_id.tolist())))
    all_material_ids = tuple(int(value) for value in sorted(set(abs(int(value)) for value in dataset.zone_material_index.tolist())))
    context = RunContext(
        path=dataset.path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=(),
        diagnostics=(),
        time_values=dataset.time_s.copy(),
        static_x_values=dataset.static_x_cm.copy(),
        zone_region_id=dataset.zone_region_id.copy(),
        zone_material_index=dataset.zone_material_index.copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=0,
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        selected_region_ids=all_region_ids,
        selected_material_ids=all_material_ids,
    )
    parameters = DerivedAnalysisParameters()
    geometry = build_analysis_geometry(
        dataset,
        context,
        observation_side=parameters.observation_side,
        line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
        profile_coordinate_mode=parameters.profile_coordinate_mode,
    )
    request = build_cold_attenuation_request(
        dataset,
        context,
        snapshot_index=0,
        parameters=parameters,
        geometry=geometry,
        photon_energies_kev=(8.0, 10.0, 12.0),
    )
    return {
        "dataset": record_path.name,
        "zones": len(request.zones),
        "observation_side": request.observation_side,
        "los_cosine": request.line_of_sight_cosine,
        "energies_kev": list(request.photon_energies_kev),
        "zone_preview": [
            {
                "zone_index": zone.zone_index,
                "region_id": zone.region_id,
                "material_id": zone.material_id,
                "material_label": zone.material_label,
                "density_g_cm3": zone.density_g_cm3,
                "path_length_cm": zone.path_length_cm,
            }
            for zone in request.zones[:5]
        ],
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    archives = {}
    for archive_name in ("helios_xcom_integration.zip", "XCOM.tar.gz"):
        archive_path = REPO_ROOT / archive_name
        if archive_path.exists():
            inspection = inspect_archive(archive_path, max_members=20)
            archives[archive_name] = {
                "archive_type": inspection.archive_type,
                "member_count": inspection.member_count,
                "top_level_entries": list(inspection.top_level_entries),
                "sample_members": [asdict(member) for member in inspection.members[:10]],
            }
    payload["archives"] = archives

    temp_root = Path(tempfile.mkdtemp(prefix="helios_xcom_validate_"))
    extracted_zip = extract_archive(REPO_ROOT / "helios_xcom_integration.zip", temp_root / "zip") if (REPO_ROOT / "helios_xcom_integration.zip").exists() else None
    extracted_tar = extract_archive(REPO_ROOT / "XCOM.tar.gz", temp_root / "tar") if (REPO_ROOT / "XCOM.tar.gz").exists() else None
    payload["extraction"] = {
        "zip_root": str(extracted_zip) if extracted_zip is not None else None,
        "tar_root": str(extracted_tar) if extracted_tar is not None else None,
    }

    wrapper_status: dict[str, object] = {"import_ok": False, "query_ok": False}
    if extracted_zip is not None:
        wrapper_root = extracted_zip / "helios_xcom_integration"
        sys.path.insert(0, str(wrapper_root))
        try:
            helios_xcom = importlib.import_module("helios_xcom")
            wrapper_status["import_ok"] = True
            wrapper_status["module_path"] = str(Path(helios_xcom.__file__).resolve())
            default_client = getattr(helios_xcom, "default_client")
            element_spec = getattr(helios_xcom, "ElementSpec")
            compute_multizone = getattr(helios_xcom, "compute_multizone_transmission")
            wrapper_status["exports"] = [
                "default_client",
                "ElementSpec",
                "compute_multizone_transmission",
            ]
            try:
                client = default_client()
                query = client.query(element_spec(symbol="Fe"), energies_kev=[8.0, 10.0, 12.0])
                wrapper_status["query_ok"] = True
                wrapper_status["query_preview"] = {
                    "energies_kev": list(getattr(query, "energies_kev", [])),
                    "mu_over_rho_cm2_g": list(getattr(query, "mu_over_rho_cm2_g", [])),
                }
                curve = compute_multizone(
                    client,
                    material_specs=["Fe", "Fe", "CH"],
                    density_g_cm3=[7.874, 7.874, 1.1],
                    path_length_cm=[1.0e-4, 2.0e-4, 5.0e-4],
                    energies_kev=[8.0, 10.0],
                )
                wrapper_status["stress_ok"] = True
                wrapper_status["stress_preview"] = {
                    "energies_kev": list(getattr(curve, "energies_kev", [])),
                    "transmission": list(getattr(curve, "transmission", [])),
                }
            except Exception as exc:
                wrapper_status["query_error"] = str(exc)
        except Exception as exc:
            wrapper_status["import_error"] = str(exc)
    payload["wrapper_status"] = wrapper_status

    registry = build_registry()
    preferred = preferred_hdf5_records(registry)
    if preferred:
        payload["derived_request_preview"] = _request_preview(Path(preferred[0].path))

    save_json(OUTPUT_DIR / "summary.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
