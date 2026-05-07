"""Tests for evaluation metrics and bootstrap CIs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from proteinlm_bench.metrics import (
    bootstrap_metric,
    calibration_curve,
    compute_metrics,
    compute_metrics_with_ci,
    mae,
    pearson,
    rmse,
    spearman,
    top_k_enrichment,
)


def test_perfect_correlation():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(y, y) == pytest.approx(1.0)
    assert pearson(y, y) == pytest.approx(1.0)
    assert rmse(y, y) == pytest.approx(0.0)
    assert mae(y, y) == pytest.approx(0.0)


def test_anticorrelation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yp = -y
    assert spearman(y, yp) == pytest.approx(-1.0)
    assert pearson(y, yp) == pytest.approx(-1.0)


def test_top_k_enrichment_perfect():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yp = y.copy()
    assert top_k_enrichment(y, yp, fraction=0.4) == pytest.approx(1.0)


def test_top_k_enrichment_random_order():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    yp = rng.standard_normal(200)
    enr = top_k_enrichment(y, yp, fraction=0.1)
    assert 0.0 <= enr <= 0.5


def test_top_k_enrichment_invalid_fraction():
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        top_k_enrichment(y, y, fraction=0.0)
    with pytest.raises(ValueError):
        top_k_enrichment(y, y, fraction=1.5)


def test_compute_metrics_returns_all_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = compute_metrics(y, y)
    for key in ("spearman", "pearson", "rmse", "mae", "top_k_enrichment"):
        assert key in out


def test_bootstrap_metric_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    yp = y + rng.standard_normal(100) * 0.1
    point, lo, hi = bootstrap_metric(spearman, y, yp, n_samples=200, seed=0)
    assert math.isfinite(point) and math.isfinite(lo) and math.isfinite(hi)
    # Point estimate should sit inside the CI for a smooth metric like spearman
    # on a stable signal.
    assert lo - 1e-6 <= point <= hi + 1e-6


def test_compute_metrics_with_ci_disabled():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = compute_metrics_with_ci(y, y, bootstrap_samples=0)
    for v in out.values():
        assert v.ci_low is None
        assert v.ci_high is None


def test_calibration_curve_shapes():
    rng = np.random.default_rng(0)
    n = 50
    y = rng.standard_normal(n)
    yp = y + rng.standard_normal(n) * 0.1
    var = np.abs(rng.standard_normal(n))
    out = calibration_curve(y, yp, var, n_bins=5)
    assert out["predicted_std"].shape == out["empirical_rmse"].shape
    assert out["bin_counts"].sum() == n
