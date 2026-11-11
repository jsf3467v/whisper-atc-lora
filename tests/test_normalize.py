"""Unit tests for the shared text normalizer in Datasets/normalize.py."""

import pytest
from normalize import normalize


def test_empty():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_lowercase():
    assert normalize("DELTA") == "delta"


def test_digits():
    assert normalize("Delta 470") == "delta four seven zero"
    assert normalize("FL090") == "fl zero nine zero"


def test_decimals():
    assert normalize("contact 118.5") == "contact one one eight point five"


def test_spelling_variants():
    assert normalize("niner tree fife fower") == "nine three five four"
    assert normalize("alfa juliett whisky") == "alpha juliet whiskey"
    assert normalize("decimal") == "point"


def test_markup():
    assert normalize("[noise] cleared to land") == "cleared to land"
    assert normalize("turn left <unk> heading") == "turn left heading"
    assert normalize("speaker two noise abatement") == "speaker two noise abatement"


def test_punctuation():
    assert normalize("turn right, heading 270.") == "turn right heading two seven zero"
    assert normalize("we'll call you") == "we ll call you"
    assert normalize("'roger'") == "roger"


def test_xray():
    assert normalize("X-ray tango") == "xray tango"
    assert normalize("x ray") == normalize("xray") == "xray"


def test_non_ascii():
    assert normalize("\u0663 alpha") == "\u0663 alpha"


def test_whitespace():
    assert normalize("  delta   four  ") == "delta four"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Lufthansa 123", "lufthansa one two three"),
        ("CSA 1DZ", "csa one dz"),
        ("climb FL 350", "climb fl three five zero"),
    ],
)
def test_atc_lines(raw, expected):
    assert normalize(raw) == expected
