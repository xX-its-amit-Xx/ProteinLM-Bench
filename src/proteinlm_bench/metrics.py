"""Evaluation metrics for mutation effect prediction.

Includes the four headline regression metrics (Spearman, Pearson, RMSE, MAE),
a top-k enrichment score for retrieval-style evaluation, and bootstrap-based
confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class MetricResult:
    """A single metric value with optional bootstrap CI."""

    name: str
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def _as_float_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def top_k_enrichment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraction: float = 0.1,
) -> float:
    """Fraction of top-fraction-by-prediction variants that are also top by truth.

    A value of 1.0 means perfect retrieval of the top fitness variants;
    values near ``fraction`` indicate random ranking. This metric is
    particularly relevant for protein engineering, where one cares more about
    retrieving the best variants than fitting the bulk of the distribution.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"top-k fraction must be in (0, 1], got {fraction}")
    k = max(1, int(round(n * fraction)))
    top_pred = set(np.argsort(-y_pred)[:k].tolist())
    top_true = set(np.argsort(-y_true)[:k].tolist())
    return len(top_pred & top_true) / float(k)


_METRIC_FUNCS = {
    "spearman": spearman,
    "pearson": pearson,
    "rmse": rmse,
    "mae": mae,
    # top_k_enrichment takes an extra argument and is handled separately.
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: Iterable[str] = ("spearman", "pearson", "rmse", "mae", "top_k_enrichment"),
    top_k_fraction: float = 0.1,
) -> Dict[str, float]:
    """Compute a dictionary of named metrics in one shot."""
    out: Dict[str, float] = {}
    for name in metrics:
        if name == "top_k_enrichment":
            out[name] = top_k_enrichment(y_true, y_pred, fraction=top_k_fraction)
        elif name in _METRIC_FUNCS:
            out[name] = _METRIC_FUNCS[name](y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {name!r}")
    return out


def bootstrap_metric(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_samples: int = 200,
    ci: float = 0.95,
    seed: int = 0,
    **kwargs,
) -> Tuple[float, float, float]:
    """Bootstrap a metric to obtain ``(point_estimate, ci_low, ci_high)``."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred, **kwargs))
    if n_samples <= 0 or n < 2:
        return point, float("nan"), float("nan")

    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(metric_fn(y_true[idx], y_pred[idx], **kwargs))
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return point, lo, hi


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: Iterable[str] = ("spearman", "pearson", "rmse", "mae", "top_k_enrichment"),
    top_k_fraction: float = 0.1,
    bootstrap_samples: int = 200,
    bootstrap_ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, MetricResult]:
    """Compute metrics with bootstrap confidence intervals.

    Set ``bootstrap_samples=0`` to disable CI computation (returns
    ``MetricResult`` objects with only the point estimate populated).
    """
    results: Dict[str, MetricResult] = {}
    for name in metrics:
        if name == "top_k_enrichment":
            point, lo, hi = bootstrap_metric(
                top_k_enrichment,
                y_true,
                y_pred,
                n_samples=bootstrap_samples,
                ci=bootstrap_ci,
                seed=seed,
                fraction=top_k_fraction,
            )
        elif name in _METRIC_FUNCS:
            point, lo, hi = bootstrap_metric(
                _METRIC_FUNCS[name],
                y_true,
                y_pred,
                n_samples=bootstrap_samples,
                ci=bootstrap_ci,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown metric: {name!r}")
        results[name] = MetricResult(
            name=name,
            value=point,
            ci_low=(None if not np.isfinite(lo) else lo),
            ci_high=(None if not np.isfinite(hi) else hi),
        )
    return results


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_var: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Bin predictions by predicted standard deviation and report empirical RMSE.

    A well-calibrated model has empirical RMSE per bin that tracks the bin's
    predicted standard deviation roughly along the y = x diagonal.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    y_var = _as_float_array(y_var)
    y_std = np.sqrt(np.maximum(y_var, 0.0))
    if len(y_true) == 0:
        empty = np.array([], dtype=np.float64)
        return {"predicted_std": empty, "empirical_rmse": empty, "bin_counts": empty}

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_std, quantiles))
    if len(edges) < 2:
        edges = np.array([y_std.min(), y_std.max() + 1e-9])
    bin_idx = np.clip(np.digitize(y_std, edges[1:-1]), 0, len(edges) - 2)

    pred_std = np.zeros(len(edges) - 1, dtype=np.float64)
    emp_rmse = np.zeros(len(edges) - 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            pred_std[b] = float("nan")
            emp_rmse[b] = float("nan")
            continue
        pred_std[b] = float(y_std[mask].mean())
        emp_rmse[b] = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
    return {
        "predicted_std": pred_std,
        "empirical_rmse": emp_rmse,
        "bin_counts": counts.astype(np.float64),
    }


__all__ = [
    "MetricResult",
    "spearman",
    "pearson",
    "rmse",
    "mae",
    "top_k_enrichment",
    "compute_metrics",
    "compute_metrics_with_ci",
    "bootstrap_metric",
    "calibration_curve",
]
