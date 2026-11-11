"""Unit tests for the callsign extractor in Evaluation/scoring.py."""

from scoring import callsign


def test_airline():
    assert callsign("United 470") == ["united", "four", "seven", "zero"]
    assert callsign("Lufthansa 456") == ["lufthansa", "four", "five", "six"]


def test_operator_pair():
    assert callsign("air berlin 298 climb") == ["air", "berlin", "two", "nine", "eight"]


def test_registration():
    assert callsign("hotel golf echo") == ["hotel", "golf", "echo"]
    assert callsign("November 123 Alpha") == ["november", "one", "two", "three", "alpha"]


def test_mixed_code():
    assert callsign("CSA 1 delta zulu descend") == ["csa", "one", "delta", "zulu"]


def test_span_cap():
    span = callsign("lufthansa 1234567 climb")
    assert len(span) == 6 and span[-1] == "five"


def test_values_only():
    assert callsign("two seven zero") == []


def test_greetings():
    assert callsign("good morning lufthansa 456") == ["lufthansa", "four", "five", "six"]


def test_station():
    assert callsign("Praha Radar CSA 123") == ["csa", "one", "two", "three"]
    assert callsign("ruzyne tower good day lufthansa 4") == ["lufthansa", "four"]


def test_instructions():
    assert callsign("heading 270") == []
    assert callsign("descend 4000 feet") == []
    assert callsign("contact tower 118.1") == []
    assert callsign("QNH 1013") == []
    assert callsign("traffic 2 o'clock") == []
    assert callsign("FL330 climb") == []


def test_no_code_run():
    assert callsign("climb to flight level") == []
    assert callsign("roger") == []


def test_empty():
    assert callsign("") == []


def test_readback():
    assert callsign("descending flight level one hundred csa one delta zulu") == [
        "csa", "one", "delta", "zulu",
    ]  # fmt: skip
    assert callsign("cleared to land air berlin 298") == ["air", "berlin", "two", "nine", "eight"]


def test_readback_registration():
    assert callsign("cleared for takeoff oscar kilo echo lima alpha") == [
        "oscar", "kilo", "echo", "lima", "alpha",
    ]  # fmt: skip
    assert callsign("number two papa golf papa") == ["papa", "golf", "papa"]


def test_trailing_values_rejected():
    assert callsign("cleared to land runway 27") == []
    assert callsign("squawk 6746") == []
    assert callsign("wind 270 degrees 7 knots") == []
    assert callsign("contact zurich 133.4") == []


def test_trailing_routes_rejected():
    assert callsign("taxi to holding point via alpha charlie") == []
    assert callsign("via charlie zulu hotel echo") == []
    assert callsign("ruzyne tower information lima") == []


def test_leading_has_priority():
    assert callsign("lufthansa 456 climb flight level 340 csa 12") == [
        "lufthansa", "four", "five", "six",
    ]  # fmt: skip
