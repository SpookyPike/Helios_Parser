"""Focused validation of XCOM composition canonicalization."""
import unittest

from helios.services.derived.xcom_composition import (
    composition_from_formula,
    composition_from_mass_fractions,
    parse_formula,
)


class XcomCompositionTests(unittest.TestCase):
    def test_formula_parsing_and_mass_weights(self):
        self.assertEqual({"C": 22, "H": 10, "N": 2, "O": 5}, parse_formula("C22H10N2O5"))
        composition = composition_from_formula("SiO2")
        self.assertEqual(("O", "Si"), tuple(symbol for symbol, _ in composition.mass_fractions))
        self.assertAlmostEqual(1.0, sum(value for _, value in composition.mass_fractions))
        self.assertEqual(composition_from_formula("SiO2").key, composition.key)

    def test_equivalent_elemental_mass_mixture_has_same_identity(self):
        from_formula = composition_from_formula("CH")
        from_input = composition_from_mass_fractions(dict(from_formula.mass_fractions))
        self.assertEqual(from_formula.key, from_input.key)

    def test_rejects_unsupported_or_malformed_formulas(self):
        for formula in ("", "Ca3(PO4)2", "CuSO4·5H2O", "Cu+", "Xx2", "U0", "U10001"):
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                composition_from_formula(formula)

    def test_rejects_invalid_mass_fraction_vectors(self):
        for values in ({}, {"Cu": -0.1, "Al": 1.1}, {"Cu": 0.7}, {"Xx": 1.0},
                       {"Cu": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                composition_from_mass_fractions(values)


if __name__ == "__main__":
    unittest.main()
