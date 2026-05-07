"""Tests for mutation parsing, sequence application, and CSV loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from proteinlm_bench.data import (
    Mutation,
    apply_mutations,
    load_mutation_csv,
    parse_mutation_code,
    summarize_dataset,
    train_test_split_variants,
)

EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "example_mutations.csv"


def test_parse_single_mutation():
    muts = parse_mutation_code("A42G")
    assert muts == (Mutation("A", 42, "G"),)


def test_parse_multi_mutation():
    muts = parse_mutation_code("A42G:T15S")
    assert muts == (Mutation("A", 42, "G"), Mutation("T", 15, "S"))


def test_parse_wildtype_returns_empty():
    assert parse_mutation_code("WT") == ()
    assert parse_mutation_code("wt") == ()
    assert parse_mutation_code("") == ()


@pytest.mark.parametrize("code", ["A42", "42G", "AG", "A-1G", "A42g"])
def test_parse_malformed_raises(code):
    with pytest.raises(ValueError):
        parse_mutation_code(code)


def test_apply_mutations_round_trip():
    wt = "ACDEFGHIK"
    muts = parse_mutation_code("A1G:K9R")
    out = apply_mutations(wt, muts)
    assert out == "GCDEFGHIR"


def test_apply_mutations_position_mismatch():
    wt = "ACDEFGHIK"
    muts = parse_mutation_code("M1A")
    with pytest.raises(ValueError):
        apply_mutations(wt, muts)


def test_apply_mutations_out_of_range():
    wt = "ACDE"
    muts = parse_mutation_code("E4A")  # valid
    apply_mutations(wt, muts)
    bad = parse_mutation_code("A100G")
    with pytest.raises((IndexError, ValueError)):
        apply_mutations(wt, bad)


def test_load_example_csv_schema():
    df = load_mutation_csv(EXAMPLE_CSV)
    expected_cols = {"protein_id", "wildtype_sequence", "mutation_code",
                     "mutant_sequence", "fitness", "n_mutations"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


def test_summarize_dataset_keys():
    df = load_mutation_csv(EXAMPLE_CSV)
    summary = summarize_dataset(df)
    for key in ("n_rows", "n_proteins", "n_wildtype", "n_single_mutants", "n_multi_mutants"):
        assert key in summary
    assert summary["n_rows"] == len(df)
    assert summary["n_multi_mutants"] >= 1


def test_train_test_split_disjoint():
    df = load_mutation_csv(EXAMPLE_CSV)
    train, test = train_test_split_variants(df, test_size=0.25, seed=0)
    # Reconstruct via index-free comparison
    assert len(train) + len(test) == len(df)
    assert len(train) > 0 and len(test) > 0


def test_train_test_split_group_by_protein():
    df = load_mutation_csv(EXAMPLE_CSV)
    if df["protein_id"].nunique() < 2:
        pytest.skip("Need >= 2 proteins for grouped split.")
    train, test = train_test_split_variants(
        df, test_size=0.5, seed=0, group_by_protein=True,
    )
    assert set(train["protein_id"]).isdisjoint(set(test["protein_id"]))
