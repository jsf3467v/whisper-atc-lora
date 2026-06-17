"""Unit tests for the shared text normalizer (Datasets/normalize.py).

The normalizer operates on both references and hypotheses before each metric calculation, ensuring 
consistent behavior in score dependencies. This includes processes like digit expansion, converting 
decimals to 'point", ICAO spelling standardization, removal of brackets or non-speech segments, and 
the digits=False escape hatch.
"""
import pytest

from normalize import normalize


def test_empty_and_none_return_empty_string():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_lowercases():
    assert normalize("DELTA") == "delta"


def test_digit_runs_are_spelled_out():
    assert normalize("Delta 470") == "delta four seven zero"
    assert normalize("FL090") == "fl zero nine zero"


def test_decimals_become_point():
    assert normalize("contact 118.5") == "contact one one eight point five"


def test_icao_spelling_variants_canonicalised():
    assert normalize("niner tree fife fower") == "nine three five four"
    assert normalize("alfa juliett whisky") == "alpha juliet whiskey"
    assert normalize("decimal") == "point"


def test_brackets_and_non_speech_markup_removed():
    assert normalize("[noise] cleared to land") == "cleared to land"
    assert normalize("turn left unintelligible heading") == "turn left heading"


def test_punctuation_stripped_apostrophes_kept():
    assert normalize("turn right, heading 270.") == "turn right heading two seven zero"
    assert normalize("don't") == "don't"


def test_whitespace_collapsed():
    assert normalize("  delta   four  ") == "delta four"


def test_digits_false_leaves_numbers_intact():
    assert normalize("delta 470", digits=False) == "delta 470"


@pytest.mark.parametrize("raw,expected", [
    ("Lufthansa 123", "lufthansa one two three"),
    ("CSA 1DZ", "csa one dz"),
    ("climb FL 350", "climb fl three five zero"),
])
def test_table_of_realistic_atc_lines(raw, expected):
    assert normalize(raw) == expected
