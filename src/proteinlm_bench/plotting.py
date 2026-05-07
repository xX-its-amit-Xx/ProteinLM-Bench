"""Plotting helpers for benchmark outputs.

All functions accept fully-computed arrays and write a figure to disk; none
re-run the model or the metric code. This keeps plots cheap to regenerate
from saved predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import PathLike, ensure_dir, get_logger  # noqa: E402

logger = get_logger(__name__)

try:  # seaborn purely for nicer defaults; the rest of the code only uses mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
except ImportError:  # pragma: no cover
    pass


def _save(fig: plt.Figure, out_path: PathLike, dpi: int = 150) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_predicted_vs_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: PathLike,
    *,
    title: str = "Predicted vs observed fitness",
    dpi: int = 150,
) -> Path:
    """Scatter of observed vs predicted fitness with the y=x reference line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7, edgecolor="white", linewidth=0.4)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, alpha=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed fitness")
    ax.set_ylabel("Predicted fitness")
    ax.set_title(title)
    return _save(fig, out_path, dpi=dpi)


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: PathLike,
    *,
    title: str = "Residual distribution",
    bins: int = 20,
    dpi: int = 150,
) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=bins, edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (observed - predicted)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return _save(fig, out_path, dpi=dpi)


def plot_embedding_umap(
    embeddings: np.ndarray,
    fitness: np.ndarray,
    out_path: PathLike,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "Embedding UMAP",
    dpi: int = 150,
    seed: int = 0,
) -> Optional[Path]:
    """2D UMAP of embeddings, colored by fitness.

    Falls back to PCA if ``umap-learn`` is not installed, so the rest of the
    benchmark always succeeds.
    """
    n = len(embeddings)
    if n < 4:
        logger.warning("Skipping UMAP plot: need at least 4 points, got %d", n)
        return None

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, max(2, n - 1)),
            min_dist=min_dist,
            random_state=seed,
        )
        coords = reducer.fit_transform(embeddings)
        method = "UMAP"
    except (ImportError, ValueError) as exc:
        logger.warning("UMAP unavailable (%s); falling back to PCA.", exc)
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        method = "PCA"

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=fitness, cmap="viridis",
        s=25, alpha=0.85, edgecolor="white", linewidth=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Fitness")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(title)
    return _save(fig, out_path, dpi=dpi)


def plot_model_comparison(
    metrics_table: Mapping[str, Mapping[str, float]],
    out_path: PathLike,
    *,
    metric: str = "spearman",
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Bar chart comparing one metric across models.

    ``metrics_table`` is expected to be ``{model_name: {metric_name: value}}``.
    """
    models = list(metrics_table.keys())
    values = [float(metrics_table[m].get(metric, float("nan"))) for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, values, edgecolor="white")
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
    ax.set_ylabel(metric)
    ax.set_title(title or f"Model comparison: {metric}")
    return _save(fig, out_path, dpi=dpi)


def plot_uncertainty_calibration(
    predicted_std: np.ndarray,
    empirical_rmse: np.ndarray,
    out_path: PathLike,
    *,
    title: str = "Uncertainty calibration",
    dpi: int = 150,
) -> Path:
    """Plot predicted std vs empirical RMSE per uncertainty bin (y=x is ideal)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    mask = np.isfinite(predicted_std) & np.isfinite(empirical_rmse)
    ax.scatter(predicted_std[mask], empirical_rmse[mask], s=40, color="C0")
    if mask.any():
        lo = float(min(predicted_std[mask].min(), empirical_rmse[mask].min()))
        hi = float(max(predicted_std[mask].max(), empirical_rmse[mask].max()))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Predicted std (ensemble)")
    ax.set_ylabel("Empirical RMSE per bin")
    ax.set_title(title)
    return _save(fig, out_path, dpi=dpi)


__all__ = [
    "plot_predicted_vs_observed",
    "plot_residual_distribution",
    "plot_embedding_umap",
    "plot_model_comparison",
    "plot_uncertainty_calibration",
]
