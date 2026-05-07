"""End-to-end benchmark runner for ProteinLM-Bench.

Loads a YAML config, generates embeddings (real PLM or mock), trains the
configured downstream regressors, evaluates them with bootstrap CIs and
ensemble-variance uncertainty, runs epistasis analysis on multi-mutants, and
writes metrics + plots to disk.

Example
-------
    python scripts/run_benchmark.py --config configs/default.yaml --mock-embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Allow `python scripts/run_benchmark.py` to work without an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proteinlm_bench.data import (  # noqa: E402
    load_mutation_csv,
    parse_mutation_code,
    summarize_dataset,
    train_test_split_variants,
)
from proteinlm_bench.embeddings import embedder_from_dict  # noqa: E402
from proteinlm_bench.epistasis import analyze_epistasis, epistasis_summary  # noqa: E402
from proteinlm_bench.metrics import (  # noqa: E402
    calibration_curve,
    compute_metrics_with_ci,
)
from proteinlm_bench.models import (  # noqa: E402
    MLPRegressor,
    RandomForestRegressorWrapper,
    RidgeRegressor,
    build_models_from_config,
    fit_ensemble,
)
from proteinlm_bench.plotting import (  # noqa: E402
    plot_embedding_umap,
    plot_model_comparison,
    plot_predicted_vs_observed,
    plot_residual_distribution,
    plot_uncertainty_calibration,
)
from proteinlm_bench.utils import ensure_dir, get_logger, load_config, set_seed  # noqa: E402

logger = get_logger("run_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a YAML benchmark config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Force the mock embedding backend, ignoring config.embeddings.backend. "
        "Use this for fast CPU smoke tests with no model downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config.experiment.output_dir.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override config.data.csv_path. Plug in your own ProteinGym/DMS CSV here.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.experiment.seed.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation (useful for headless CI).",
    )
    return parser.parse_args()


def _ensemble_factory(model_name: str, base_cfg: Dict[str, Any]):
    """Return a callable ``factory(seed)`` that produces a fresh regressor."""
    if model_name == "ridge":
        alpha = float(base_cfg.get("alpha", 1.0))
        return lambda seed: RidgeRegressor(alpha=alpha)
    if model_name == "random_forest":
        n_est = int(base_cfg.get("n_estimators", 200))
        max_depth = base_cfg.get("max_depth")
        return lambda seed: RandomForestRegressorWrapper(
            n_estimators=n_est, max_depth=max_depth, random_state=seed
        )
    if model_name == "mlp":
        return lambda seed: MLPRegressor(
            hidden_dims=base_cfg.get("hidden_dims", [128, 64]),
            dropout=float(base_cfg.get("dropout", 0.1)),
            lr=float(base_cfg.get("lr", 1e-3)),
            weight_decay=float(base_cfg.get("weight_decay", 1e-4)),
            epochs=int(base_cfg.get("epochs", 50)),
            batch_size=int(base_cfg.get("batch_size", 32)),
            seed=seed,
        )
    raise ValueError(f"No ensemble factory for model {model_name!r}")


def _mutation_positions(df: pd.DataFrame) -> List[List[int]]:
    out: List[List[int]] = []
    for code in df["mutation_code"]:
        try:
            mutations = parse_mutation_code(str(code))
        except ValueError:
            out.append([])
            continue
        out.append([m.position for m in mutations])
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.data is not None:
        cfg.setdefault("data", {})["csv_path"] = args.data

    seed = int(cfg.get("experiment", {}).get("seed", 0))
    set_seed(seed)
    output_dir = ensure_dir(cfg.get("experiment", {}).get("output_dir", "outputs/run"))
    figures_dir = ensure_dir(Path(output_dir) / "figures")

    logger.info("Loading dataset: %s", cfg["data"]["csv_path"])
    df = load_mutation_csv(cfg["data"]["csv_path"])
    summary = summarize_dataset(df)
    logger.info("Dataset summary: %s", summary)

    train_df, test_df = train_test_split_variants(
        df,
        test_size=float(cfg["data"].get("test_size", 0.25)),
        seed=seed,
        group_by_protein=bool(cfg["data"].get("group_by_protein", False)),
    )
    logger.info("Train/test split: %d / %d", len(train_df), len(test_df))

    # ---- Embeddings ---------------------------------------------------------
    embedder = embedder_from_dict(
        cfg.get("embeddings", {}),
        mock_override=args.mock_embeddings,
    )
    logger.info(
        "Embedder: backend=%s dim=%d",
        type(embedder).__name__,
        getattr(embedder, "embedding_dim", -1),
    )

    train_positions = _mutation_positions(train_df)
    test_positions = _mutation_positions(test_df)

    X_train = embedder.embed_sequences(
        train_df["mutant_sequence"].tolist(),
        mutation_positions=train_positions,
    )
    X_test = embedder.embed_sequences(
        test_df["mutant_sequence"].tolist(),
        mutation_positions=test_positions,
    )
    y_train = train_df["fitness"].to_numpy(dtype=np.float64)
    y_test = test_df["fitness"].to_numpy(dtype=np.float64)

    # ---- Models -------------------------------------------------------------
    models = build_models_from_config(cfg.get("models", {}), seed=seed)
    if not models:
        logger.error("No models enabled in config. Nothing to evaluate.")
        return 1

    eval_cfg = cfg.get("evaluation", {})
    metric_names = eval_cfg.get(
        "metrics", ["spearman", "pearson", "rmse", "mae", "top_k_enrichment"]
    )
    top_k_fraction = float(eval_cfg.get("top_k_fraction", 0.1))
    bootstrap_samples = int(eval_cfg.get("bootstrap_samples", 200))
    bootstrap_ci = float(eval_cfg.get("bootstrap_ci", 0.95))
    ensemble_size = int(eval_cfg.get("ensemble_size", 5))

    metrics_table: Dict[str, Dict[str, float]] = {}
    full_results: Dict[str, Any] = {"dataset_summary": summary, "models": {}}

    predictions_per_model: Dict[str, np.ndarray] = {}
    variance_per_model: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions_per_model[name] = preds

        metric_results = compute_metrics_with_ci(
            y_test, preds,
            metrics=metric_names,
            top_k_fraction=top_k_fraction,
            bootstrap_samples=bootstrap_samples,
            bootstrap_ci=bootstrap_ci,
            seed=seed,
        )
        flat = {k: v.value for k, v in metric_results.items()}
        metrics_table[name] = flat
        logger.info(
            "  %s -> %s",
            name,
            ", ".join(f"{k}={v:.3f}" for k, v in flat.items() if np.isfinite(v)),
        )

        # Ensemble-variance uncertainty
        if ensemble_size > 1:
            mcfg = cfg.get("models", {}).get(name, {})
            factory = _ensemble_factory(name, mcfg)
            ens = fit_ensemble(
                factory, X_train, y_train, X_test,
                n_members=ensemble_size, base_seed=seed,
            )
            variance_per_model[name] = ens.variance
        else:
            variance_per_model[name] = np.zeros_like(preds)

        full_results["models"][name] = {
            "metrics": {k: v.to_dict() for k, v in metric_results.items()},
            "predictions": preds.tolist(),
            "ensemble_variance": variance_per_model[name].tolist(),
        }

    # ---- Epistasis analysis -------------------------------------------------
    epi_cfg = cfg.get("epistasis", {})
    if epi_cfg.get("enabled", True):
        logger.info("Running epistasis analysis on multi-mutant test variants.")
        for name, preds in predictions_per_model.items():
            epi_df = analyze_epistasis(
                train_df, test_df, preds,
                outlier_sd_threshold=float(epi_cfg.get("outlier_sd_threshold", 2.0)),
                reference_df=df,
            )
            epi_path = Path(output_dir) / f"epistasis_{name}.csv"
            epi_df.to_csv(epi_path, index=False)
            full_results["models"][name]["epistasis"] = {
                "summary": epistasis_summary(epi_df),
                "csv_path": str(epi_path),
            }
            logger.info(
                "  [%s] epistasis: %s",
                name,
                full_results["models"][name]["epistasis"]["summary"],
            )

    # ---- Plots --------------------------------------------------------------
    plot_cfg = cfg.get("plotting", {})
    if (not args.no_plots) and plot_cfg.get("enabled", True):
        dpi = int(plot_cfg.get("dpi", 150))
        for name, preds in predictions_per_model.items():
            plot_predicted_vs_observed(
                y_test, preds,
                figures_dir / f"pred_vs_obs_{name}.png",
                title=f"Predicted vs observed ({name})",
                dpi=dpi,
            )
            plot_residual_distribution(
                y_test, preds,
                figures_dir / f"residuals_{name}.png",
                title=f"Residuals ({name})",
                dpi=dpi,
            )
            cal = calibration_curve(y_test, preds, variance_per_model[name])
            plot_uncertainty_calibration(
                cal["predicted_std"], cal["empirical_rmse"],
                figures_dir / f"calibration_{name}.png",
                title=f"Calibration ({name})",
                dpi=dpi,
            )

        plot_embedding_umap(
            X_test, y_test,
            figures_dir / "embedding_umap.png",
            n_neighbors=int(plot_cfg.get("umap_n_neighbors", 15)),
            min_dist=float(plot_cfg.get("umap_min_dist", 0.1)),
            seed=seed,
            dpi=dpi,
        )

        for metric_name in ("spearman", "pearson", "rmse", "mae", "top_k_enrichment"):
            if metric_name in next(iter(metrics_table.values())):
                plot_model_comparison(
                    metrics_table,
                    figures_dir / f"model_comparison_{metric_name}.png",
                    metric=metric_name,
                    dpi=dpi,
                )

    # ---- Persist results ----------------------------------------------------
    metrics_path = Path(output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(full_results, fh, indent=2)

    summary_rows = []
    for name, flat in metrics_table.items():
        row = {"model": name, **flat}
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(Path(output_dir) / "metrics_summary.csv", index=False)

    logger.info("Wrote results to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
