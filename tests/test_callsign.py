"""Unit tests for the callsign extractor (Evaluation/scoring.py).


The extractor interprets the ATC structure: a 1-2 word operator with its digit or 
phonetic sequence (airline format), or a leading phonetic sequence (registration format). 
It detects both formats, handles empty cases, and respects the documented leading-span limitation, 
making potential issues visible for future correction rather than ignoring them.

"""
import pytest

from scoring import callsign


def test_airline_form_operator_plus_digits():
    assert callsign("Delta 470") == ["delta", "four", "seven", "zero"]
    assert callsign("Lufthansa 456") == ["lufthansa", "four", "five", "six"]


def test_registration_form_leading_phonetic_run():
    assert callsign("hotel golf echo") == ["hotel", "golf", "echo"]


def test_mixed_operator_and_phonetic_suffix():
    assert callsign("November 123 Alpha") == ["november", "one", "two", "three", "alpha"]


def test_pure_value_run_has_no_callsign():
    # only digits, nothing name-like -> not a callsign
    assert callsign("two seven zero") == []


def test_empty_input():
    assert callsign("") == []


def test_operator_capped_at_two_words():
    # more than two leading non-value/non-phonetic words: only the first two
    # are taken as the operator, and with no value/phonetic run after them the
    # span is just those two tokens.
    assert callsign("climb to flight level") == ["climb", "to"]


@pytest.mark.xfail(reason="documented limitation: callsign after the instruction "
                          "(pilot readback) is not captured by the leading-span rule",
                   strict=True)
def test_trailing_readback_callsign_is_a_known_gap():
    # "... csa one delta zulu" sits at the end; the leading-span extractor cannot
    # reach it. When readback handling is added, this should start passing.
    assert callsign("descending flight level one hundred csa one delta zulu") == \
        ["csa", "one", "delta", "zulu"]
