"""Shared loader and helper routines for Derived / Analysis services."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

import numpy as np

from helios.cache import get_session_raw_data_cache, run_identity_for_path
from helios.instrumentation import increment_counter, timed_block
from helios.runtime import RunContext
from helios.services.derived.models import (
    DerivedFieldCapabilities,
    DerivedFieldConsistency,
    DerivedRunData,
    DerivedWarning,
    WavePhysicsCapabilities,
)
from helios.services.geometry.coordinates import build_zone_property_from_regions, infer_laser_entry, subset_mask
from helios.services.units.conversions import weighted_mean
from helios_parser import HeliosRun

LOGGER = logging.getLogger(__name__)

_SHARED_RAW_CACHE = get_session_raw_data_cache()
_RUN_META_BUCKET = _SHARED_RAW_CACHE.bucket("session_run_meta", max_items=48)
_RUN_ARRAY_BUCKET = _SHARED_RAW_CACHE.bucket("session_run_arrays", max_items=192)

_OPTIONAL_FIELD_DTYPES: dict[str, object] = {
    "temperature_radiation": np.float64,
    "pressure_i": np.float64,
    "pressure_e": np.float64,
    "pressure_radiation": np.float64,
    "pressure": np.float64,
    "artificial_viscosity": np.float64,
    "ion_energy": np.float64,
    "electron_energy": np.float64,
    "radiation_energy": np.float64,
    "kinetic_energy": np.float64,
    "ion_heat_capacity": np.float64,
    "electron_heat_capacity": np.float64,
    "radiation_heating": np.float64,
    "radiation_cooling": np.float64,
    "radiation_sink": np.float64,
    "radiation_net_heating": np.float64,
    "laser_source": np.float64,
    "laser_deposition": np.float64,
}

_OPTIONAL_FIELD_ATTRS: dict[str, str] = {
    "temperature_radiation": "temperature_radiation_ev",
    "pressure_i": "pressure_i_j_cm3",
    "pressure_e": "pressure_e_j_cm3",
    "pressure_radiation": "pressure_radiation_j_cm3",
    "pressure": "pressure_total_j_cm3",
    "artificial_viscosity": "artificial_viscosity_j_cm3",
    "ion_energy": "ion_energy_j_g",
    "electron_energy": "electron_energy_j_g",
    "radiation_energy": "radiation_energy_j_g",
    "kinetic_energy": "kinetic_energy_j_g",
    "ion_heat_capacity": "ion_heat_capacity_j_g_ev",
    "electron_heat_capacity": "electron_heat_capacity_j_g_ev",
    "radiation_heating": "radiation_heating_j_g_s",
    "radiation_cooling": "radiation_cooling_j_g_s",
    "radiation_sink": "radiation_sink_j_g_s",
    "radiation_net_heating": "radiation_net_heating_j_g_s",
    "laser_source": "laser_source_j_g_s",
    "laser_deposition": "laser_deposition_j_g_s",
}


def _shared_array_view(values: np.ndarray | list[float] | tuple[float, ...], dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    view = array.view()
    view.setflags(write=False)
    return view


def _share_meta(run_key: tuple[object, ...], name: str, value: object) -> object:
    key = (run_key, str(name))
    _RUN_META_BUCKET[key] = value
    return value


def _share_array(run_key: tuple[object, ...], name: str, values: np.ndarray | list[float] | tuple[float, ...], dtype) -> np.ndarray:
    key = (run_key, str(name))
    shared = _shared_array_view(values, dtype)
    _RUN_ARRAY_BUCKET[key] = shared
    return shared


def _cached_meta(run_key: tuple[object, ...], name: str) -> object | None:
    value = _RUN_META_BUCKET.get((run_key, str(name)))
    if value is None:
        increment_counter("shared_raw_cache.meta.miss")
    else:
        increment_counter("shared_raw_cache.meta.hit")
    return value


def _cached_array(run_key: tuple[object, ...], name: str) -> np.ndarray | None:
    value = _RUN_ARRAY_BUCKET.get((run_key, str(name)))
    if value is None:
        increment_counter("shared_raw_cache.array.miss")
        return None
    increment_counter("shared_raw_cache.array.hit")
    return np.asarray(value)


def shared_raw_cache_stats() -> dict[str, object]:
    return {
        "meta": _RUN_META_BUCKET.stats(),
        "arrays": _RUN_ARRAY_BUCKET.stats(),
    }


def publish_open_run_payload(
    path: str | Path,
    *,
    summary: dict[str, object],
    metadata: dict[str, object],
    regions: dict[str, object],
    materials: dict[str, object],
    fields: Iterable[str],
    diagnostics: Iterable[str],
    time_values: np.ndarray,
    static_x_center: np.ndarray,
    static_x_edge: np.ndarray,
    zone_region_id: np.ndarray,
    zone_material_index: np.ndarray,
    has_dynamic_radius: bool,
    run_status: dict[str, object] | None = None,
    visar_support_metadata: dict[str, object] | None = None,
) -> tuple[object, ...]:
    """Publish lightweight run-open data into the shared session cache."""

    run_key = run_identity_for_path(path)
    _share_meta(run_key, "summary", dict(summary))
    _share_meta(run_key, "metadata", dict(metadata))
    _share_meta(run_key, "regions", dict(regions))
    _share_meta(run_key, "materials", dict(materials))
    _share_meta(run_key, "fields", tuple(str(name) for name in fields))
    _share_meta(run_key, "diagnostics", tuple(str(name) for name in diagnostics))
    _share_meta(run_key, "has_dynamic_radius", bool(has_dynamic_radius))
    if run_status is not None:
        _share_meta(run_key, "run_status", dict(run_status))
    if visar_support_metadata is not None:
        _share_meta(run_key, "visar_support_metadata", dict(visar_support_metadata))
    _share_array(run_key, "time_s", time_values, np.float64)
    _share_array(run_key, "static_x_center_cm", static_x_center, np.float64)
    _share_array(run_key, "static_x_edge_cm", static_x_edge, np.float64)
    _share_array(run_key, "zone_region_id", zone_region_id, np.int32)
    _share_array(run_key, "zone_material_index", zone_material_index, np.int32)
    return run_key


def publish_field_payload(
    path: str | Path,
    *,
    field_name: str,
    data: np.ndarray,
    edge_data: np.ndarray | None = None,
) -> tuple[object, ...]:
    """Publish a viewer-loaded field into the shared session cache."""

    run_key = run_identity_for_path(path)
    _share_array(run_key, f"field:{field_name}", data, np.float64)
    if edge_data is not None:
        _share_array(run_key, f"field:{field_name}:edge", edge_data, np.float64)
    return run_key


def _finite_allclose(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float) -> bool | None:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(valid):
        return None
    return bool(np.allclose(lhs[valid], rhs[valid], rtol=rtol, atol=atol))


def _specific_kinetic_energy_from_velocity_j_g(velocity_cm_s: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity_cm_s, dtype=np.float64)
    return 0.5 * np.square(velocity) * 1.0e-7


def _build_field_capabilities(
    *,
    available_fields: tuple[str, ...],
    optional_field_values: dict[str, np.ndarray | None],
    has_dynamic_radius: bool,
    run_status: dict[str, object] | None,
    visar_support_metadata: dict[str, object] | None,
    velocity: np.ndarray,
) -> DerivedFieldCapabilities:
    optional_available = tuple(sorted(name for name, values in optional_field_values.items() if values is not None))
    missing_optional = tuple(sorted(name for name, values in optional_field_values.items() if values is None))
    notes: list[str] = []
    total_pressure_matches: bool | None = None
    radiation_net_matches: bool | None = None
    kinetic_matches: bool | None = None

    pressure_i = optional_field_values.get("pressure_i")
    pressure_e = optional_field_values.get("pressure_e")
    pressure_radiation = optional_field_values.get("pressure_radiation")
    pressure_total = optional_field_values.get("pressure")
    if pressure_i is not None and pressure_e is not None and pressure_radiation is not None and pressure_total is not None:
        total_pressure_matches = _finite_allclose(
            pressure_total,
            np.asarray(pressure_i, dtype=np.float64) + np.asarray(pressure_e, dtype=np.float64) + np.asarray(pressure_radiation, dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        if total_pressure_matches is False:
            notes.append("pressure != pressure_i + pressure_e + pressure_radiation within tolerance")

    radiation_heating = optional_field_values.get("radiation_heating")
    radiation_cooling = optional_field_values.get("radiation_cooling")
    radiation_net = optional_field_values.get("radiation_net_heating")
    if radiation_heating is not None and radiation_cooling is not None and radiation_net is not None:
        radiation_sink = optional_field_values.get("radiation_sink")
        reconstructed_net = np.asarray(radiation_heating, dtype=np.float64) - np.asarray(radiation_cooling, dtype=np.float64)
        if radiation_sink is not None:
            reconstructed_net = reconstructed_net - np.asarray(radiation_sink, dtype=np.float64)
        radiation_net_matches = _finite_allclose(
            radiation_net,
            reconstructed_net,
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        if radiation_net_matches is False:
            notes.append("radiation_net_heating differs from radiation_heating - radiation_cooling - radiation_sink")

    kinetic_energy = optional_field_values.get("kinetic_energy")
    if kinetic_energy is not None:
        kinetic_matches = _finite_allclose(
            kinetic_energy,
            _specific_kinetic_energy_from_velocity_j_g(velocity),
            rtol=1.0e-5,
            atol=1.0e-10,
        )
        if kinetic_matches is False:
            notes.append("kinetic_energy is not consistent with 0.5*v^2 in J/g within tolerance")

    consistency = DerivedFieldConsistency(
        total_pressure_matches_components=total_pressure_matches,
        radiation_net_heating_matches_components=radiation_net_matches,
        kinetic_energy_matches_velocity=kinetic_matches,
        notes=tuple(notes),
    )
    return DerivedFieldCapabilities(
        available_fields=tuple(str(name) for name in available_fields),
        optional_available_fields=optional_available,
        missing_optional_fields=missing_optional,
        dynamic_radius_available=bool(has_dynamic_radius),
        run_status_available=run_status is not None,
        visar_support_available=visar_support_metadata is not None,
        pressure_components_available=all(optional_field_values.get(name) is not None for name in ("pressure_i", "pressure_e", "pressure_radiation")),
        total_pressure_available=optional_field_values.get("pressure") is not None,
        radiation_components_available=all(optional_field_values.get(name) is not None for name in ("radiation_heating", "radiation_cooling")),
        radiation_net_heating_available=optional_field_values.get("radiation_net_heating") is not None,
        kinetic_energy_available=optional_field_values.get("kinetic_energy") is not None,
        consistency=consistency,
    )


def _build_wave_physics_capabilities(field_capabilities: DerivedFieldCapabilities) -> WavePhysicsCapabilities:
    notes: list[str] = []
    pressure_support_level = "components" if field_capabilities.pressure_components_available else ("total_only" if field_capabilities.total_pressure_available else "unavailable")
    viscosity_support_level = "available" if "artificial_viscosity" in field_capabilities.optional_available_fields else "unavailable"
    if field_capabilities.radiation_components_available and field_capabilities.radiation_net_heating_available:
        radiation_support_level = "components+net"
    elif field_capabilities.radiation_components_available:
        radiation_support_level = "components"
    elif field_capabilities.radiation_net_heating_available:
        radiation_support_level = "net_only"
    else:
        radiation_support_level = "unavailable"
    if pressure_support_level == "unavailable":
        notes.append("Total-pressure wave evidence will have to degrade because no pressure fields are available.")
    if viscosity_support_level == "unavailable":
        notes.append("Shock-like evidence can only use pressure/density/velocity hooks until artificial viscosity is available.")
    if not field_capabilities.dynamic_radius_available:
        notes.append("Interface-event geometry is limited to static edge grids.")
    return WavePhysicsCapabilities(
        shock_evidence_supported=bool(
            field_capabilities.total_pressure_available
            or field_capabilities.pressure_components_available
        )
        and "density" in field_capabilities.available_fields
        and "velocity" in field_capabilities.available_fields,
        release_evidence_supported=bool(
            field_capabilities.total_pressure_available
            or field_capabilities.pressure_components_available
        )
        and "density" in field_capabilities.available_fields
        and "velocity" in field_capabilities.available_fields,
        contact_evidence_supported=bool(
            "density" in field_capabilities.available_fields
            and "velocity" in field_capabilities.available_fields
        ),
        interface_event_supported=True,
        preheat_supported=bool(
            radiation_support_level != "unavailable"
            or "temperature_e" in field_capabilities.available_fields
            or "temperature_i" in field_capabilities.available_fields
        ),
        pressure_support_level=pressure_support_level,
        viscosity_support_level=viscosity_support_level,
        radiation_support_level=radiation_support_level,
        notes=tuple(notes),
    )


def load_run_data(path: str | Path) -> DerivedRunData:
    """Load the field subset needed by the current Derived mode.

    This preserves the current lightweight load path for existing modules while
    also promoting richer optional hydro fields, run-status metadata, and
    VISAR-readiness support into one capability-driven payload for future
    modules.
    """

    source = Path(path)
    run_key = run_identity_for_path(source)
    with timed_block("hdf5.read.derived_run_data", logger=LOGGER):
        summary = _cached_meta(run_key, "summary")
        metadata = _cached_meta(run_key, "metadata")
        regions = _cached_meta(run_key, "regions")
        materials = _cached_meta(run_key, "materials")
        fields = _cached_meta(run_key, "fields")
        run_status = _cached_meta(run_key, "run_status")
        visar_support_metadata = _cached_meta(run_key, "visar_support_metadata")
        has_dynamic_radius = _cached_meta(run_key, "has_dynamic_radius")
        time_s = _cached_array(run_key, "time_s")
        static_x = _cached_array(run_key, "static_x_center_cm")
        static_x_edges = _cached_array(run_key, "static_x_edge_cm")
        zone_region_id = _cached_array(run_key, "zone_region_id")
        zone_material_index = _cached_array(run_key, "zone_material_index")
        density = _cached_array(run_key, "field:density")
        velocity = _cached_array(run_key, "field:velocity")
        temperature_e = _cached_array(run_key, "field:temperature_e")
        temperature_i = _cached_array(run_key, "field:temperature_i")
        temperature_r = _cached_array(run_key, "field:temperature_radiation")
        electron_density = _cached_array(run_key, "field:electron_density")
        mean_charge = _cached_array(run_key, "field:mean_charge")
        zone_width = _cached_array(run_key, "field:zone_width")
        radius = _cached_array(run_key, "field:radius")
        radius_edges = _cached_array(run_key, "field:radius:edge")
        optional_fields = {
            field_name: _cached_array(run_key, f"field:{field_name}")
            for field_name in _OPTIONAL_FIELD_DTYPES
        }

        needs_open = (
            summary is None
            or metadata is None
            or regions is None
            or materials is None
            or fields is None
            or has_dynamic_radius is None
            or time_s is None
            or static_x is None
            or static_x_edges is None
            or zone_region_id is None
            or zone_material_index is None
            or density is None
            or velocity is None
            or temperature_e is None
            or temperature_i is None
            or electron_density is None
            or mean_charge is None
        )
        if needs_open:
            increment_counter("shared_raw_cache.run.miss")
            with HeliosRun(source) as run:
                if summary is None:
                    summary = _share_meta(run_key, "summary", run.summary())
                if metadata is None:
                    metadata = _share_meta(run_key, "metadata", run.get_metadata())
                if regions is None:
                    regions = _share_meta(run_key, "regions", run.get_regions())
                if materials is None:
                    materials = _share_meta(run_key, "materials", run.get_materials())
                if fields is None:
                    fields = _share_meta(run_key, "fields", tuple(run.list_fields()))
                if has_dynamic_radius is None:
                    has_dynamic_radius = _share_meta(run_key, "has_dynamic_radius", bool(run.has_dynamic_coordinate()))
                if run_status is None:
                    run_status = _share_meta(run_key, "run_status", run.get_run_status())
                if visar_support_metadata is None:
                    visar_support = run.get_visar_support_metadata()
                    visar_support_metadata = _share_meta(
                        run_key,
                        "visar_support_metadata",
                        {
                            "velocity_field_name": visar_support.velocity_field_name,
                            "time_axis_name": visar_support.time_axis_name,
                            "static_coordinate_name": visar_support.static_coordinate_name,
                            "dynamic_coordinate_field_name": visar_support.dynamic_coordinate_field_name,
                            "boundary_indexing_consistent": visar_support.boundary_indexing_consistent,
                            "candidate_boundaries": tuple(visar_support.candidate_boundaries),
                            "event_timing_source": visar_support.event_timing_source,
                            "notes": tuple(visar_support.notes),
                        },
                    )
                if time_s is None:
                    time_s = _share_array(run_key, "time_s", run.get_time(), np.float64)
                if static_x is None:
                    static_x = _share_array(run_key, "static_x_center_cm", run.get_static_coordinate(location="center"), np.float64)
                if static_x_edges is None:
                    static_x_edges = _share_array(run_key, "static_x_edge_cm", run.get_static_coordinate(location="edge"), np.float64)
                if zone_region_id is None:
                    zone_region_id = _share_array(run_key, "zone_region_id", run.get_grid("zone_region_id"), np.int32)
                if zone_material_index is None:
                    zone_material_index = _share_array(run_key, "zone_material_index", run.get_grid("zone_material_index"), np.int32)

                field_names = set(str(name) for name in fields)
                if density is None:
                    density = _share_array(run_key, "field:density", run.get_field("density"), np.float64)
                if velocity is None:
                    velocity = _share_array(run_key, "field:velocity", run.get_field("velocity"), np.float64)
                if temperature_e is None:
                    temperature_e = _share_array(run_key, "field:temperature_e", run.get_field("temperature_e"), np.float64)
                if temperature_i is None:
                    temperature_i = _share_array(run_key, "field:temperature_i", run.get_field("temperature_i"), np.float64)
                if temperature_r is None and "temperature_radiation" in field_names:
                    temperature_r = _share_array(
                        run_key,
                        "field:temperature_radiation",
                        run.get_field("temperature_radiation"),
                        np.float64,
                    )
                if electron_density is None:
                    electron_density = _share_array(run_key, "field:electron_density", run.get_field("electron_density"), np.float64)
                if mean_charge is None:
                    mean_charge = _share_array(run_key, "field:mean_charge", run.get_field("mean_charge"), np.float64)
                if zone_width is None:
                    if "zone_width" in field_names:
                        zone_width = _share_array(run_key, "field:zone_width", run.get_field("zone_width"), np.float64)
                    else:
                        static_width = np.asarray(run.get_grid("zone_width"), dtype=np.float64)
                        zone_width = _share_array(
                            run_key,
                            "field:zone_width",
                            np.broadcast_to(static_width[None, :], (time_s.size, static_width.size)),
                            np.float64,
                        )
                if radius is None and run.has_dynamic_coordinate():
                    dynamic_center = run.get_dynamic_coordinate(location="center")
                    if dynamic_center is not None:
                        radius = _share_array(run_key, "field:radius", dynamic_center, np.float64)
                if radius_edges is None and run.has_dynamic_coordinate():
                    dynamic_edge = run.get_dynamic_coordinate(location="edge")
                    if dynamic_edge is not None:
                        radius_edges = _share_array(run_key, "field:radius:edge", dynamic_edge, np.float64)
                for field_name, dtype in _OPTIONAL_FIELD_DTYPES.items():
                    if optional_fields[field_name] is None and field_name in field_names:
                        optional_fields[field_name] = _share_array(
                            run_key,
                            f"field:{field_name}",
                            run.get_field(field_name),
                            dtype,
                        )
        else:
            increment_counter("shared_raw_cache.run.hit")

        assert summary is not None and metadata is not None and regions is not None and materials is not None
        if run_status is None and isinstance(summary, dict):
            summary_status = summary.get("run_status")
            if isinstance(summary_status, dict):
                run_status = dict(summary_status)
        assert time_s is not None and static_x is not None and static_x_edges is not None
        assert zone_region_id is not None and zone_material_index is not None
        assert density is not None and velocity is not None and temperature_e is not None and temperature_i is not None
        assert electron_density is not None and mean_charge is not None
        available_fields = tuple(str(name) for name in fields)
        n_zones = int(summary["n_zones"])
        zone_atomic_weight = build_zone_property_from_regions(
            regions,
            np.asarray(regions["atomic_weight"], dtype=np.float64),
            n_zones,
        )
        zone_initial_density = build_zone_property_from_regions(
            regions,
            np.asarray(regions["initial_mass_density"], dtype=np.float64),
            n_zones,
        )
        zone_initial_temperature = build_zone_property_from_regions(
            regions,
            np.asarray(regions["initial_temperature"], dtype=np.float64),
            n_zones,
        )
        laser_entry = infer_laser_entry(
            metadata=metadata,
            n_zones=n_zones,
            zone_region_id=zone_region_id,
            regions=regions,
        )
        capability_summary = _build_field_capabilities(
            available_fields=available_fields,
            optional_field_values=optional_fields,
            has_dynamic_radius=bool(has_dynamic_radius),
            run_status=None if run_status is None else dict(run_status),
            visar_support_metadata=None if visar_support_metadata is None else dict(visar_support_metadata),
            velocity=velocity,
        )
        wave_physics_capabilities = _build_wave_physics_capabilities(capability_summary)
    return DerivedRunData(
        path=source,
        summary=summary,
        metadata=metadata,
        regions=regions,
        materials=materials,
        time_s=time_s,
        static_x_cm=static_x,
        static_x_edge_cm=static_x_edges,
        zone_width_cm=zone_width,
        density_g_cm3=density,
        velocity_cm_s=velocity,
        temperature_e_ev=temperature_e,
        temperature_i_ev=temperature_i,
        temperature_radiation_ev=temperature_r,
        electron_density_cm3=electron_density,
        mean_charge=mean_charge,
        pressure_i_j_cm3=optional_fields["pressure_i"],
        pressure_e_j_cm3=optional_fields["pressure_e"],
        pressure_radiation_j_cm3=optional_fields["pressure_radiation"],
        pressure_total_j_cm3=optional_fields["pressure"],
        artificial_viscosity_j_cm3=optional_fields["artificial_viscosity"],
        ion_energy_j_g=optional_fields["ion_energy"],
        electron_energy_j_g=optional_fields["electron_energy"],
        radiation_energy_j_g=optional_fields["radiation_energy"],
        kinetic_energy_j_g=optional_fields["kinetic_energy"],
        ion_heat_capacity_j_g_ev=optional_fields["ion_heat_capacity"],
        electron_heat_capacity_j_g_ev=optional_fields["electron_heat_capacity"],
        radiation_heating_j_g_s=optional_fields["radiation_heating"],
        radiation_cooling_j_g_s=optional_fields["radiation_cooling"],
        radiation_sink_j_g_s=optional_fields["radiation_sink"],
        radiation_net_heating_j_g_s=optional_fields["radiation_net_heating"],
        laser_source_j_g_s=optional_fields["laser_source"],
        laser_deposition_j_g_s=optional_fields["laser_deposition"],
        radius_cm=radius,
        radius_edge_cm=radius_edges,
        zone_region_id=zone_region_id,
        zone_material_index=zone_material_index,
        zone_atomic_weight=zone_atomic_weight,
        zone_initial_density_g_cm3=zone_initial_density,
        zone_initial_temperature_ev=zone_initial_temperature,
        laser_entry=laser_entry,
        run_status=None if run_status is None else dict(run_status),
        visar_support_metadata=None if visar_support_metadata is None else dict(visar_support_metadata),
        field_capabilities=capability_summary,
        wave_physics_capabilities=wave_physics_capabilities,
    )


def context_zone_mask(context: RunContext, dataset: DerivedRunData) -> np.ndarray:
    if not context.has_run:
        return np.zeros(int(dataset.summary["n_zones"]), dtype=bool)
    return subset_mask(
        zone_region_id=dataset.zone_region_id,
        zone_material_index=dataset.zone_material_index,
        selected_region_ids=context.selected_region_ids,
        selected_material_ids=context.selected_material_ids,
    )


def snapshot_weights(dataset: DerivedRunData, snapshot_index: int, mask: np.ndarray) -> np.ndarray:
    widths = np.asarray(dataset.zone_width_cm[int(snapshot_index)], dtype=np.float64)
    weights = np.where(mask, np.maximum(widths, 0.0), 0.0)
    return weights


def weighted_snapshot_mean(values: np.ndarray, dataset: DerivedRunData, snapshot_index: int, mask: np.ndarray) -> float:
    return weighted_mean(np.asarray(values[int(snapshot_index)], dtype=np.float64), snapshot_weights(dataset, snapshot_index, mask))


def region_mask(dataset: DerivedRunData, region_id: int, base_mask: np.ndarray | None = None) -> np.ndarray:
    mask = np.asarray(dataset.zone_region_id, dtype=np.int32) == int(region_id)
    if base_mask is not None:
        mask &= np.asarray(base_mask, dtype=bool)
    return mask


def zone_coordinate_for_snapshot(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray:
    if dataset.radius_cm is not None:
        return np.asarray(dataset.radius_cm[int(snapshot_index)], dtype=np.float64)
    return np.asarray(dataset.static_x_cm, dtype=np.float64)


def field_capability_summary(dataset: DerivedRunData) -> dict[str, object]:
    """Return a compact field-availability summary for future modules."""

    capabilities = dataset.field_capabilities
    return {
        "available_fields": tuple(capabilities.available_fields),
        "optional_available_fields": tuple(capabilities.optional_available_fields),
        "missing_optional_fields": tuple(capabilities.missing_optional_fields),
        "dynamic_radius_available": bool(capabilities.dynamic_radius_available),
        "run_status_available": bool(capabilities.run_status_available),
        "visar_support_available": bool(capabilities.visar_support_available),
        "pressure_components_available": bool(capabilities.pressure_components_available),
        "total_pressure_available": bool(capabilities.total_pressure_available),
        "radiation_components_available": bool(capabilities.radiation_components_available),
        "radiation_net_heating_available": bool(capabilities.radiation_net_heating_available),
        "kinetic_energy_available": bool(capabilities.kinetic_energy_available),
        "consistency_notes": tuple(capabilities.consistency.notes),
        "total_pressure_matches_components": capabilities.consistency.total_pressure_matches_components,
        "radiation_net_heating_matches_components": capabilities.consistency.radiation_net_heating_matches_components,
        "kinetic_energy_matches_velocity": capabilities.consistency.kinetic_energy_matches_velocity,
    }


def laser_pulse_duration_s(dataset: DerivedRunData) -> float | None:
    metadata = dataset.metadata.get("input_parameters", {})
    if not isinstance(metadata, dict):
        return None
    laser_source = metadata.get("laser_source", {})
    if not isinstance(laser_source, dict):
        return None
    power_table = laser_source.get("power_table", {})
    if not isinstance(power_table, dict):
        return None
    try:
        times = np.asarray(power_table.get("time", ()), dtype=np.float64).reshape(-1)
        power = np.asarray(power_table.get("power", ()), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if times.size == 0 or power.size == 0 or times.size != power.size:
        return None
    valid = np.isfinite(times) & np.isfinite(power)
    active = valid & (power > 0.0)
    if np.count_nonzero(active) >= 2:
        active_times = times[active]
        return float(active_times[-1] - active_times[0])
    if np.count_nonzero(valid) >= 2:
        valid_times = times[valid]
        return float(valid_times[-1] - valid_times[0])
    return None


def picosecond_drive_warning(module_name: str, dataset: DerivedRunData, message: str) -> DerivedWarning | None:
    duration_s = laser_pulse_duration_s(dataset)
    if duration_s is None or not np.isfinite(duration_s):
        return None
    # Conservative regime flag: these quick looks were validated primarily on
    # ns-scale hydrodynamic runs. ps-scale drives are still parsed correctly,
    # but the same effective-state interpretation is less mature there.
    if duration_s <= 1.0e-9:
        return DerivedWarning(module_name, message, severity="caution")
    return None


def aggregate_warnings(*warning_groups: Iterable[DerivedWarning]) -> tuple[DerivedWarning, ...]:
    ordered: list[DerivedWarning] = []
    seen: set[tuple[str, str, str]] = set()
    for group in warning_groups:
        for warning in group:
            key = (warning.source, warning.message, warning.severity)
            if key in seen:
                continue
            ordered.append(warning)
            seen.add(key)
    return tuple(ordered)
