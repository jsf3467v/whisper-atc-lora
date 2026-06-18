"""Tests for pixel metrics and rank-based selective metrics.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("pandas")

import torch  # noqa: E402

from src.metrics import (  # noqa: E402
    aurc,
    auroc,
    nmse,
    psnr,
    risk_coverage,
    ssim,
)


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 64, 64, generator=g)


# pixel metrics

def test_nmse_zero_when_identical():
    img = random_batch()
    assert nmse(img, img).abs().max() < 1e-6


def test_ssim_one_when_identical():
    img = random_batch()
    assert (ssim(img, img) - 1).abs().max() < 1e-4


def test_nmse_grows_with_error():
    img = random_batch()
    noisy = img + 0.1 * random_batch(seed=1)
    assert (nmse(img, noisy) > nmse(img, img)).all()


def test_psnr_drops_with_error():
    img = random_batch()
    noisy = img + 0.1 * random_batch(seed=2)
    assert (psnr(img, noisy) < psnr(img, img)).all()


def test_metrics_are_per_sample():
    img = random_batch()
    assert ssim(img, img).shape == (2,)
    assert psnr(img, img).shape == (2,)
    assert nmse(img, img).shape == (2,)


# auroc

def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0


def test_auroc_inverted_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 0.0


def test_auroc_undefined_with_single_class():
    scores = np.array([0.1, 0.5, 0.9])
    assert np.isnan(auroc(scores, np.zeros(3)))
    assert np.isnan(auroc(scores, np.ones(3)))


def test_auroc_stays_in_unit_range():
    rng = np.random.default_rng(4)
    for _ in range(20):
        scores = rng.normal(size=40)
        labels = rng.integers(0, 2, size=40)
        if labels.min() == labels.max():
            continue
        value = auroc(scores, labels)
        assert 0.0 <= value <= 1.0


# risk_coverage and aurc

def test_risk_coverage_full_coverage_is_mean_error():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    errors = np.array([0.0, 0.2, 0.4, 0.6])
    coverage, risk = risk_coverage(scores, errors)
    assert coverage[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(errors.mean())


def test_risk_coverage_coverage_is_uniform_grid():
    scores = np.linspace(0, 1, 5)
    errors = np.zeros(5)
    coverage, _ = risk_coverage(scores, errors)
    assert np.allclose(coverage, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))


def test_aurc_lower_when_scores_order_errors_well():
    """A residual that ranks errors correctly scores below a shuffled one."""
    errors = np.linspace(0.0, 1.0, 50)
    good_scores = errors.copy()
    rng = np.random.default_rng(0)
    bad_scores = rng.permutation(errors)
    assert aurc(good_scores, errors) < aurc(bad_scores, errors)


def test_aurc_equals_mean_of_risk_curve():
    scores = np.array([0.1, 0.5, 0.2, 0.9])
    errors = np.array([0.0, 0.5, 0.1, 0.9])
    _, risk = risk_coverage(scores, errors)
    assert aurc(scores, errors) == pytest.approx(risk.mean())


def test_aurc_empty_is_nan():
    assert np.isnan(aurc(np.array([]), np.array([])))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
