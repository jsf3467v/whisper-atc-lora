"""Unit tests for the per-utterance statistics and bootstrap in Evaluation/scoring.py."""

import numpy as np
import pytest
from scoring import (
    METRICS,
    change,
    corpus,
    edits,
    estimate,
    normalize,
    ratios,
    repetition,
    resampled,
    statistics,
    utterance,
)


def scored(ref, hyp):
    return utterance(normalize(ref), normalize(hyp))


def metric(row, name):
    k = METRICS.index(name)
    return row[0][k], row[1][k]


def test_edits():
    assert edits("a b c d", "a x c") == (1, 1, 0, 4)
    assert edits("a b", "a b c") == (0, 0, 1, 2)
    assert edits("a b", "") == (0, 2, 0, 2)


def test_counts():
    row = scored("Delta 470", "delta four seven one")
    assert metric(row, "wer") == (1, 4)
    assert metric(row, "sub") == (1, 4)
    assert metric(row, "ser") == (True, 1)
    assert metric(row, "cer") == (4, len("delta four seven zero"))


def test_grouped_numbers():
    row = scored("flight level one hundred", "flight level 100")
    assert metric(row, "wer")[0] == 2
    assert metric(row, "wer_numeric") == (0, 5)


def test_digit_granularity():
    row = scored("lufthansa 456", "lufthansa 457")
    assert metric(row, "wer_numeric") == metric(row, "wer") == (1, 4)


def test_shared_conventions():
    assert metric(scored("we won t", "we won't"), "wer_numeric")[0] == 0
    assert metric(scored("X-ray two nine zero", "xray 290"), "wer_numeric")[0] == 0


def test_empty_reference():
    num, den, kept, texts = statistics(["[noise]", "roger"], ["hello", "roger"])
    assert kept == [1] and texts == [("roger", "roger")]
    assert num.shape == den.shape == (1, len(METRICS))


def test_callsign_scores():
    exact = scored("lufthansa 456 climb", "lufthansa 456 climb")
    wrong = scored("lufthansa 456 climb", "lufthansa 457 climb")
    spurious = scored("roger", "united 12")
    assert metric(exact, "callsign_precision") == (True, True)
    assert metric(wrong, "callsign_recall") == (False, True)
    assert metric(wrong, "callsign_f1") == (0, 2)
    assert metric(wrong, "callsign_wer") == (1, 4)
    assert metric(spurious, "callsign_false_alarm") == (True, True)
    assert metric(spurious, "coverage") == (False, 1)


def test_callsign_order():
    row = scored("lufthansa 456", "lufthansa 654")
    assert metric(row, "callsign_recall")[0] == 0
    assert metric(row, "callsign_wer") == (2, 4)


def test_repetition():
    assert repetition("roger " * 12)
    assert not repetition("lufthansa four five six climb flight level three four zero")
    assert not repetition("")


def test_pooling():
    num, den, _, _ = statistics(["a b c d", "e f"], ["a x c", "e f"])
    wer = corpus(num, den)[METRICS.index("wer")]
    assert wer == pytest.approx(2 / 6)


def test_zero_denominator():
    assert np.isnan(ratios(np.array([1.0]), np.array([0.0]))[0])


def test_interval():
    rng = np.random.default_rng(1)
    refs = [" ".join(["a"] * int(n)) for n in rng.integers(3, 9, 120)]
    hyps = [" ".join(["a"] * (len(r.split()) - int(rng.integers(0, 2)))) for r in refs]
    num, den, _, _ = statistics(refs, hyps)
    boots = resampled([(num, den)], n_boot=1500, chunk=400)
    wer = estimate(corpus(num, den), boots[0])["wer"]
    assert wer["lo"] <= wer["value"] <= wer["hi"]
    assert wer["resamples"] == 1500 and boots.shape == (1, 1500, len(METRICS))


def test_single_group():
    num, den, _, _ = statistics(["a b c", "a b", "a"], ["a x c", "a b", ""])
    boots = resampled([(num, den)], groups=np.array(["r"] * 3), n_boot=200)
    wer = estimate(corpus(num, den), boots[0])["wer"]
    assert wer["lo"] == pytest.approx(wer["value"]) == pytest.approx(wer["hi"])


def test_strata():
    num, den, _, _ = statistics(["a b"] * 40, ["a b"] * 20 + ["x y"] * 20)
    strata = np.array(["clean"] * 20 + ["noisy"] * 20)
    free = estimate(corpus(num, den), resampled([(num, den)], n_boot=2000)[0])["wer"]
    fixed = estimate(corpus(num, den), resampled([(num, den)], strata=strata, n_boot=2000)[0])[
        "wer"
    ]
    assert fixed["lo"] == pytest.approx(0.5) == pytest.approx(fixed["hi"])
    assert free["hi"] - free["lo"] > 0


def test_grouping():
    refs, hyps = ["a b"] * 40, ["a b"] * 20 + ["x y"] * 20
    num, den, _, _ = statistics(refs, hyps)
    groups = np.array(["good"] * 20 + ["bad"] * 20)
    single = estimate(corpus(num, den), resampled([(num, den)], n_boot=2000)[0])["wer"]
    grouped = estimate(corpus(num, den), resampled([(num, den)], groups, n_boot=2000)[0])["wer"]
    assert grouped["hi"] - grouped["lo"] > single["hi"] - single["lo"]


def test_shared_generator():
    num, den, _, _ = statistics(["a b c", "a b", "a"], ["a x c", "a b", ""])
    rng = np.random.default_rng(0)
    first, second = (resampled([(num, den)], n_boot=50, seed=rng) for _ in range(2))
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, resampled([(num, den)], n_boot=50, seed=0))


def test_no_change():
    num, den, _, _ = statistics(["lufthansa 4 5", "csa 1 2"], ["lufthansa 4 6", "csa 1 2"])
    boots = resampled([(num, den), (num, den)], n_boot=300)
    point = corpus(num, den)
    c = change(point, point, boots[0], boots[1])
    assert c["wer"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_improve": 0.0, "p_tie": 1.0}
    assert "coverage" not in c


def test_direction():
    refs = ["lufthansa 4 5 climb"] * 30
    base = statistics(refs, ["lufthansa 4 6 descend"] * 30)[:2]
    model = statistics(refs, ["lufthansa 4 5 climb"] * 30)[:2]
    boots = resampled([base, model], n_boot=300)
    c = change(corpus(*base), corpus(*model), boots[0], boots[1])
    assert c["wer"]["delta"] < 0 and c["wer"]["p_improve"] == 1.0
    assert c["callsign_f1"]["delta"] == 1.0 and c["callsign_f1"]["p_improve"] == 1.0


def test_bad_input():
    empty = np.empty((0, len(METRICS)))
    with pytest.raises(ValueError):
        resampled([(empty, empty)])
    one = np.ones((1, len(METRICS)))
    two = np.ones((2, len(METRICS)))
    with pytest.raises(ValueError):
        resampled([(one, one), (two, two)])
