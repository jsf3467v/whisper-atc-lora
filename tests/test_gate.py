"""Tests for the empirical-Bernstein risk-control gate."""

import numpy as np
import pytest

from src.gate import (
    accepted_groups,
    mondrian_thresholds,
    risk_controlled_threshold,
)


def calibration_pair(n=400, seed=0):
    """Sorted scores with error that rises with score."""
    rng = np.random.default_rng(seed)
    scores = np.sort(rng.uniform(0.0, 1.0, size=n))
    errors = np.clip(scores * 0.3 + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)
    return scores, errors


# risk_controlled_threshold

def test_threshold_is_an_observed_score_or_neg_inf():
    scores, errors = calibration_pair()
    t = risk_controlled_threshold(scores, errors, epsilon=0.2, alpha=0.1)
    assert t == float("-inf") or np.isclose(scores, t).any()


def test_accepted_empirical_risk_within_tolerance():
    """The bound dominates the empirical mean, so accepted risk stays at or below epsilon."""
    scores, errors = calibration_pair()
    epsilon = 0.2
    t = risk_controlled_threshold(scores, errors, epsilon=epsilon, alpha=0.1)
    assert t > float("-inf")
    accepted = scores <= t
    assert accepted.any()
    assert errors[accepted].mean() <= epsilon + 1e-9


def test_threshold_monotonic_in_epsilon():
    """Looser tolerance never lowers the threshold."""
    scores, errors = calibration_pair()
    last = float("-inf")
    for epsilon in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        t = risk_controlled_threshold(scores, errors, epsilon=epsilon, alpha=0.1)
        assert t >= last - 1e-12
        last = t


def test_accepts_everything_when_error_free_and_lenient():
    scores = np.linspace(0.0, 1.0, 300)
    errors = np.zeros_like(scores)
    t = risk_controlled_threshold(scores, errors, epsilon=0.2, alpha=0.1)
    assert np.isclose(t, scores.max())
    assert (scores <= t).all()


def test_escalates_when_tolerance_unreachable():
    scores, errors = calibration_pair()
    t = risk_controlled_threshold(scores, errors, epsilon=1e-9, alpha=0.1)
    assert t == float("-inf")


def test_unsorted_input_matches_sorted():
    scores, errors = calibration_pair(seed=3)
    t_sorted = risk_controlled_threshold(scores, errors, epsilon=0.2, alpha=0.1)
    perm = np.random.default_rng(1).permutation(len(scores))
    t_shuffled = risk_controlled_threshold(
        scores[perm], errors[perm], epsilon=0.2, alpha=0.1
    )
    assert t_sorted == t_shuffled


def test_tighter_alpha_is_not_more_permissive():
    """Smaller alpha widens the bound, so it accepts no more."""
    scores, errors = calibration_pair(seed=5)
    t_loose = risk_controlled_threshold(scores, errors, epsilon=0.2, alpha=0.2)
    t_strict = risk_controlled_threshold(scores, errors, epsilon=0.2, alpha=0.01)
    assert t_strict <= t_loose + 1e-12


# mondrian_thresholds

def test_mondrian_is_per_group_and_string_keyed():
    scores, errors = calibration_pair()
    groups = np.where(np.arange(len(scores)) % 2 == 0, "T1", "T2")
    thresholds = mondrian_thresholds(
        scores, errors, groups, epsilon=0.2, alpha=0.1, minimum=10
    )
    assert set(thresholds) == {"T1", "T2"}
    assert all(isinstance(name, str) for name in thresholds)


def test_mondrian_small_group_escalates():
    scores = np.concatenate([np.linspace(0, 1, 200), np.linspace(0, 1, 3)])
    errors = np.concatenate([np.zeros(200), np.zeros(3)])
    groups = np.array(["big"] * 200 + ["tiny"] * 3)
    thresholds = mondrian_thresholds(
        scores, errors, groups, epsilon=0.2, alpha=0.1, minimum=10
    )
    assert thresholds["tiny"] == float("-inf")
    assert thresholds["big"] > float("-inf")


def test_mondrian_matches_per_group_single_call():
    scores, errors = calibration_pair(seed=7)
    groups = np.where(scores < 0.5, "low", "high")
    thresholds = mondrian_thresholds(
        scores, errors, groups, epsilon=0.3, alpha=0.1, minimum=5
    )
    for name in ("low", "high"):
        pick = groups == name
        expected = risk_controlled_threshold(
            scores[pick], errors[pick], epsilon=0.3, alpha=0.1
        )
        assert thresholds[name] == expected


# accepted_groups

def test_accepted_mask_uses_group_thresholds():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    groups = np.array(["A", "A", "B", "B"])
    thresholds = {"A": 0.4, "B": 0.5}
    mask = accepted_groups(scores, groups, thresholds)
    assert mask.tolist() == [True, True, False, False]


def test_accepted_unknown_group_is_rejected():
    scores = np.array([0.1, 0.2])
    groups = np.array(["A", "Z"])
    mask = accepted_groups(scores, groups, {"A": 1.0})
    assert mask.tolist() == [True, False]


def test_accepted_neg_inf_threshold_rejects_group():
    scores = np.array([0.0, -5.0, 0.3])
    groups = np.array(["g", "g", "g"])
    mask = accepted_groups(scores, groups, {"g": float("-inf")})
    assert not mask.any()


def test_full_pipeline_respects_per_group_risk():
    """Fitted thresholds keep each accepted contrast at or below the tolerance."""
    rng = np.random.default_rng(11)
    n = 600
    groups = rng.choice(["T1", "T2", "FLAIR"], size=n)
    scores = rng.uniform(0, 1, size=n)
    base = {"T1": 0.1, "T2": 0.2, "FLAIR": 0.5}
    errors = np.clip(
        scores * np.array([base[g] for g in groups]) + rng.normal(0, 0.03, n),
        0.0, 1.0,
    )
    epsilon = 0.15
    thresholds = mondrian_thresholds(
        scores, errors, groups, epsilon=epsilon, alpha=0.1, minimum=20
    )
    mask = accepted_groups(scores, groups, thresholds)
    for name in np.unique(groups):
        sel = mask & (groups == name)
        if sel.any():
            assert errors[sel].mean() <= epsilon + 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
