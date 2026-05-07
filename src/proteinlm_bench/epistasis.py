"""Epistasis analysis for multi-mutant variants.

The additive (non-epistatic) null model predicts that the fitness effect of a
multi-mutant variant equals the *sum* of the constituent single-mutant effects:

    f_add(m1, m2, ...) = f_WT + sum_i (f_mi - f_WT)

Real proteins frequently violate this — interactions between residues mean the
combined effect is larger or smaller than the sum of its parts. We quantify
this as the **epistasis residual**: the difference between the observed
multi-mutant fitness and the additive expectation. The benchmark also reports
the difference between predicted and additive expectation, which captures how
well a model recovers genuine epistatic structure rather than memorising
single-mutant effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import parse_mutation_code


@dataclass
class EpistasisRecord:
    """Per-variant epistasis breakdown (multi-mutants only)."""

    protein_id: str
    mutation_code: str
    n_mutations: int
    observed_fitness: float
    predicted_fitness: float
    additive_expectation: float
    observed_epistasis: float          # observed - additive (true epistasis)
    predicted_epistasis: float         # predicted - additive
    residual: float                    # observed - predicted
    is_outlier: bool


def _build_single_mutant_lookup(
    df: pd.DataFrame,
    *,
    fitness_col: str = "fitness",
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Index wild-type and single-mutant fitness values per protein.

    Returns ``(wt_lookup, single_lookup)`` where ``wt_lookup[protein_id]``
    is the WT fitness (typically 0.0) and ``single_lookup[(protein_id, code)]``
    is the fitness of the single-mutant ``code`` on ``protein_id``.
    """
    wt_lookup: Dict[str, float] = {}
    single_lookup: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        code = str(row["mutation_code"]).strip()
        protein = str(row["protein_id"])
        try:
            mutations = parse_mutation_code(code)
        except ValueError:
            continue
        if len(mutations) == 0:
            wt_lookup[protein] = float(row[fitness_col])
        elif len(mutations) == 1:
            single_lookup[(protein, str(mutations[0]))] = float(row[fitness_col])
    return wt_lookup, single_lookup


def additive_expectation(
    protein_id: str,
    mutations: List[str],
    wt_lookup: Dict[str, float],
    single_lookup: Dict[Tuple[str, str], float],
) -> Optional[float]:
    """Compute the additive (non-epistatic) expectation for a multi-mutant.

    Returns ``None`` if any constituent single-mutant or the WT is missing
    from the lookup tables, since we cannot construct the prediction without
    those reference points.
    """
    if protein_id not in wt_lookup:
        return None
    f_wt = wt_lookup[protein_id]
    total = f_wt
    for m in mutations:
        key = (protein_id, m)
        if key not in single_lookup:
            return None
        total += single_lookup[key] - f_wt
    return float(total)


def analyze_epistasis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    fitness_col: str = "fitness",
    outlier_sd_threshold: float = 2.0,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score multi-mutant test variants against the additive null.

    Single-mutant references are sourced from ``reference_df`` if provided —
    the typical pattern is to pass the *full* dataset there so the additive
    expectation can use single-mutant ground truth even when those variants
    happened to land in the test split. If only ``train_df`` is available
    (the strictest setting), pass ``reference_df=train_df``.

    Parameters
    ----------
    outlier_sd_threshold:
        Variants with absolute model residual exceeding this many standard
        deviations of the residual distribution are flagged as ``is_outlier``.
        These are candidates for genuinely hard-to-predict epistatic effects.
    """
    if reference_df is None:
        reference_df = pd.concat([train_df, test_df], ignore_index=True)

    wt_lookup, single_lookup = _build_single_mutant_lookup(
        reference_df, fitness_col=fitness_col
    )

    records: List[EpistasisRecord] = []
    residuals: List[float] = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        try:
            mutations = parse_mutation_code(str(row["mutation_code"]))
        except ValueError:
            continue
        if len(mutations) < 2:
            continue
        protein = str(row["protein_id"])
        mut_strs = [str(m) for m in mutations]
        f_add = additive_expectation(protein, mut_strs, wt_lookup, single_lookup)
        if f_add is None:
            continue
        observed = float(row[fitness_col])
        predicted = float(predictions[i])
        records.append(
            EpistasisRecord(
                protein_id=protein,
                mutation_code=str(row["mutation_code"]),
                n_mutations=len(mutations),
                observed_fitness=observed,
                predicted_fitness=predicted,
                additive_expectation=f_add,
                observed_epistasis=observed - f_add,
                predicted_epistasis=predicted - f_add,
                residual=observed - predicted,
                is_outlier=False,
            )
        )
        residuals.append(observed - predicted)

    if not records:
        return pd.DataFrame(
            columns=[
                "protein_id", "mutation_code", "n_mutations",
                "observed_fitness", "predicted_fitness", "additive_expectation",
                "observed_epistasis", "predicted_epistasis", "residual", "is_outlier",
            ]
        )

    residual_arr = np.asarray(residuals, dtype=np.float64)
    sd = float(residual_arr.std()) if len(residual_arr) > 1 else 0.0
    threshold = sd * outlier_sd_threshold if sd > 0 else float("inf")
    for rec in records:
        rec.is_outlier = abs(rec.residual) > threshold

    return pd.DataFrame([rec.__dict__ for rec in records])


def epistasis_summary(epi_df: pd.DataFrame) -> Dict[str, float]:
    """One-line summary statistics for the epistasis DataFrame."""
    if epi_df.empty:
        return {
            "n_multi_mutants": 0,
            "mean_observed_epistasis": float("nan"),
            "mean_abs_observed_epistasis": float("nan"),
            "mean_abs_residual": float("nan"),
            "fraction_outliers": float("nan"),
            "epistasis_correlation": float("nan"),
        }
    obs = epi_df["observed_epistasis"].to_numpy()
    pred = epi_df["predicted_epistasis"].to_numpy()
    if obs.std() > 0 and pred.std() > 0 and len(obs) >= 2:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(obs, pred)
        epi_corr = float(rho)
    else:
        epi_corr = float("nan")
    return {
        "n_multi_mutants": int(len(epi_df)),
        "mean_observed_epistasis": float(obs.mean()),
        "mean_abs_observed_epistasis": float(np.mean(np.abs(obs))),
        "mean_abs_residual": float(np.mean(np.abs(epi_df["residual"].to_numpy()))),
        "fraction_outliers": float(epi_df["is_outlier"].mean()),
        "epistasis_correlation": epi_corr,
    }


__all__ = [
    "EpistasisRecord",
    "additive_expectation",
    "analyze_epistasis",
    "epistasis_summary",
]
