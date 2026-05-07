"""Tests for epistasis analysis and the additive null model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proteinlm_bench.data import apply_mutations, parse_mutation_code
from proteinlm_bench.epistasis import (
    additive_expectation,
    analyze_epistasis,
    epistasis_summary,
)


WT = "ACDEFGHIKLMN"


def _row(code: str, fitness: float) -> dict:
    if code == "WT":
        seq = WT
    else:
        seq = apply_mutations(WT, parse_mutation_code(code))
    return {
        "protein_id": "p",
        "wildtype_sequence": WT,
        "mutation_code": code,
        "mutant_sequence": seq,
        "fitness": fitness,
        "n_mutations": 0 if code == "WT" else len(parse_mutation_code(code)),
    }


def _make_dataset():
    rows = [
        _row("WT", 0.0),
        _row("A1G", -0.5),
        _row("D3K", 0.3),
        _row("F5L", -0.2),
        # Multi-mutants with known additive expectations:
        _row("A1G:D3K", -0.2),         # purely additive (expectation = -0.2)
        _row("A1G:F5L", -1.5),         # strongly negative epistasis (expectation = -0.7)
        _row("D3K:F5L", 0.1),          # near-additive (expectation = 0.1)
    ]
    return pd.DataFrame(rows)


def test_additive_expectation_basic():
    wt_lookup = {"p": 0.0}
    single = {
        ("p", "A1G"): -0.5,
        ("p", "D3K"): 0.3,
        ("p", "F5L"): -0.2,
    }
    assert additive_expectation("p", ["A1G", "D3K"], wt_lookup, single) == pytest.approx(-0.2)
    assert additive_expectation("p", ["A1G", "F5L"], wt_lookup, single) == pytest.approx(-0.7)


def test_additive_expectation_missing_returns_none():
    wt_lookup = {"p": 0.0}
    single = {("p", "A1G"): -0.5}
    assert additive_expectation("p", ["A1G", "X9X"], wt_lookup, single) is None
    assert additive_expectation("missing", ["A1G"], wt_lookup, single) is None


def test_analyze_epistasis_filters_to_multi_mutants():
    df = _make_dataset()
    # Pretend the predictor is the additive expectation (so residual = obs - add).
    additive_predictions = []
    for _, row in df.iterrows():
        if row["n_mutations"] >= 2:
            mutations = parse_mutation_code(row["mutation_code"])
            single_lookup = {
                "A1G": -0.5, "D3K": 0.3, "F5L": -0.2,
            }
            f_add = sum(single_lookup[str(m)] for m in mutations)
            additive_predictions.append(f_add)
        else:
            additive_predictions.append(row["fitness"])

    epi_df = analyze_epistasis(
        train_df=df, test_df=df,
        predictions=np.array(additive_predictions),
        outlier_sd_threshold=1.5,
        reference_df=df,
    )
    assert (epi_df["n_mutations"] >= 2).all()
    # Three multi-mutants in the dataset
    assert len(epi_df) == 3

    # Predicted - additive should be ~0 since predictions == additive
    assert np.allclose(epi_df["predicted_epistasis"].to_numpy(), 0.0, atol=1e-9)

    # observed - additive: A1G:D3K -> -0.2 - (-0.2) = 0; A1G:F5L -> -1.5 - (-0.7) = -0.8;
    # D3K:F5L -> 0.1 - 0.1 = 0
    obs_epi = epi_df.set_index("mutation_code")["observed_epistasis"].to_dict()
    assert obs_epi["A1G:D3K"] == pytest.approx(0.0, abs=1e-9)
    assert obs_epi["A1G:F5L"] == pytest.approx(-0.8, abs=1e-9)
    assert obs_epi["D3K:F5L"] == pytest.approx(0.0, abs=1e-9)


def test_outlier_flagging():
    df = _make_dataset()
    # Predictions: one has a large residual, the others are perfect.
    preds = df["fitness"].to_numpy().astype(float).copy()
    # Find index of A1G:F5L and inject a big prediction error
    idx = df.index[df["mutation_code"] == "A1G:F5L"].tolist()[0]
    preds[idx] += 5.0  # large residual

    epi_df = analyze_epistasis(
        train_df=df, test_df=df,
        predictions=preds,
        outlier_sd_threshold=1.0,
        reference_df=df,
    )
    flagged = epi_df.loc[epi_df["mutation_code"] == "A1G:F5L", "is_outlier"].iloc[0]
    assert bool(flagged) is True


def test_summary_handles_empty():
    empty = pd.DataFrame(columns=[
        "protein_id", "mutation_code", "n_mutations",
        "observed_fitness", "predicted_fitness", "additive_expectation",
        "observed_epistasis", "predicted_epistasis", "residual", "is_outlier",
    ])
    summary = epistasis_summary(empty)
    assert summary["n_multi_mutants"] == 0
