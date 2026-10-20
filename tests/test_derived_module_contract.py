from __future__ import annotations

from types import SimpleNamespace
import unittest

from helios.services.derived.common import _optional_zone_material_property
from helios.services.derived.module_contract import DerivedModuleContract


class DerivedModuleContractTests(unittest.TestCase):
    def test_missing_material_property_is_unavailable_without_fabricating_values(self) -> None:
        self.assertIsNone(_optional_zone_material_property({}, "atomic_weight", 3))
        contract = DerivedModuleContract(
            name="spectroscopy",
            compute=lambda: None,
            validate=lambda result: None,
            required_material_properties=("zone_atomic_weight",),
        )
        dataset = SimpleNamespace(field_capabilities=SimpleNamespace(available_fields=()))
        self.assertFalse(contract.capabilities_met(dataset))

    def test_material_and_field_requirements_are_checked_together(self) -> None:
        contract = DerivedModuleContract(
            name="transmission",
            compute=lambda: None,
            validate=lambda result: None,
            required_capabilities=("density", "electron_density"),
            required_material_properties=("zone_initial_density_g_cm3",),
        )
        fields = SimpleNamespace(
            available_fields=("density", "electron_density"),
            optional_available_fields=(),
        )
        self.assertFalse(contract.capabilities_met(SimpleNamespace(field_capabilities=fields, zone_initial_density_g_cm3=None)))
        self.assertTrue(
            contract.capabilities_met(
                SimpleNamespace(field_capabilities=fields, zone_initial_density_g_cm3=object())
            )
        )


if __name__ == "__main__":
    unittest.main()
