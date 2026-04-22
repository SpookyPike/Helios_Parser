"""Internal execution contract for derived modules with lazy time plots.

The existing analysis backend computes snapshot-local products for every module,
but time traces can now be requested lazily. This contract keeps the orchestration
explicit and reviewable for future heavier modules without changing the public
result objects consumed by the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Generic, TypeVar


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class DerivedModuleContract(Generic[ResultT]):
    """Reusable backend seam for one derived module."""

    name: str
    compute: Callable[..., ResultT]
    validate: Callable[[ResultT], None]
    required_capabilities: tuple[str, ...] = ()
    supports_lazy_time_plots: bool = True

    def capabilities_met(self, dataset: object) -> bool:
        field_capabilities = getattr(dataset, "field_capabilities", None)
        if field_capabilities is None:
            return len(self.required_capabilities) == 0
        available = set(getattr(field_capabilities, "available_fields", ()))
        optional_available = set(getattr(field_capabilities, "optional_available_fields", ()))
        flags = {
            "pressure_components": bool(getattr(field_capabilities, "pressure_components_available", False)),
            "total_pressure": bool(getattr(field_capabilities, "total_pressure_available", False)),
            "radiation_components": bool(getattr(field_capabilities, "radiation_components_available", False)),
            "radiation_net_heating": bool(getattr(field_capabilities, "radiation_net_heating_available", False)),
            "kinetic_energy": bool(getattr(field_capabilities, "kinetic_energy_available", False)),
            "dynamic_radius": bool(getattr(field_capabilities, "dynamic_radius_available", False)),
            "run_status": bool(getattr(field_capabilities, "run_status_available", False)),
            "visar_support": bool(getattr(field_capabilities, "visar_support_available", False)),
        }
        for capability in self.required_capabilities:
            normalized = str(capability)
            if normalized in flags:
                if not flags[normalized]:
                    return False
                continue
            if normalized not in available and normalized not in optional_available:
                return False
        return True

    @staticmethod
    def time_plots_loaded(module_result: object) -> bool:
        return bool(getattr(module_result, "time_plots", ()))

    @staticmethod
    def merge_time_plots(base_result: ResultT, updated_result: ResultT) -> ResultT:
        base_time_plots = tuple(getattr(base_result, "time_plots", ()))
        updated_time_plots = tuple(getattr(updated_result, "time_plots", ()))
        if updated_time_plots:
            return updated_result
        return replace(updated_result, time_plots=base_time_plots)
