from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from helios_parser.tokenizer import RegexTokenizer


class NumericNotationTests(unittest.TestCase):
    def test_complete_tokens_accept_e_and_d_exponents(self) -> None:
        tokenizer = RegexTokenizer()
        tokens = tokenizer.extract_number_tokens("1.25E+03 -2.5e-04 .75D+2 3.0d-1")
        self.assertEqual(tokens, ["1.25E+03", "-2.5e-04", ".75E+2", "3.0E-1"])

    def test_omitted_e_notation_is_preserved(self) -> None:
        tokenizer = RegexTokenizer()
        self.assertEqual(tokenizer.extract_number_tokens("1.25-03 -2.50+02"), ["1.25E-03", "-2.50E+02"])


if __name__ == "__main__":
    unittest.main()
