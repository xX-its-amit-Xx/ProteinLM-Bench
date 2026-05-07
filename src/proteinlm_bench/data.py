"""Mutation dataset loading, parsing, and train/test splitting.

The expected CSV schema is:

    protein_id,wildtype_sequence,mutation_code,mutant_sequence,fitness

`mutation_code` follows the standard ``<wt><pos><mut>`` notation (e.g. ``A42G``)
and supports multi-mutants joined by ``":"`` (e.g. ``A42G:T15S``). The literal
string ``WT`` denotes the wild-type reference. Positions are 1-indexed, matching
the convention used in deep mutational scanning datasets and ProteinGym.

Users wishing to plug in real datasets (e.g. ProteinGym substitution
benchmarks, DMS data from Fowler & Fields, or in-house screens) only need to
produce a CSV with these columns; nothing in the rest of the pipeline assumes
synthetic data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import PathLike

REQUIRED_COLUMNS = (
    "protein_id",
    "wildtype_sequence",
    "mutation_code",
    "mutant_sequence",
    "fitness",
)

# Single-mutation regex: <WT-AA><pos><MUT-AA>, e.g. "A42G".
_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Mutation:
    """A single point mutation in 1-indexed protein coordinates."""

    wt_aa: str
    position: int
    mut_aa: str

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"


@dataclass
class Variant:
    """A protein variant with one or more point mutations."""

    protein_id: str
    wildtype_sequence: str
    mutant_sequence: str
    mutations: Tuple[Mutation, ...]
    fitness: float
    raw_code: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_wildtype(self) -> bool:
        return len(self.mutations) == 0

    @property
    def n_mutations(self) -> int:
        return len(self.mutations)


def parse_mutation_code(code: str) -> Tuple[Mutation, ...]:
    """Parse a mutation code such as ``"A42G"`` or ``"A42G:T15S"``.

    Returns an empty tuple for the wild-type sentinel ``"WT"`` (case-insensitive).
    Raises ``ValueError`` for malformed codes so callers can decide whether to
    drop or surface them.
    """
    if code is None:
        raise ValueError("mutation_code is None")
    code = str(code).strip()
    if code == "" or code.upper() == "WT":
        return ()

    parts = [p.strip() for p in code.split(":") if p.strip()]
    mutations = []
    for part in parts:
        match = _MUTATION_RE.match(part)
        if not match:
            raise ValueError(f"Malformed mutation token: {part!r}")
        wt_aa, pos, mut_aa = match.group(1), int(match.group(2)), match.group(3)
        if pos < 1:
            raise ValueError(f"Mutation position must be 1-indexed: {part!r}")
        mutations.append(Mutation(wt_aa=wt_aa, position=pos, mut_aa=mut_aa))
    return tuple(mutations)


def apply_mutations(wildtype: str, mutations: Iterable[Mutation]) -> str:
    """Apply a set of point mutations to a wild-type sequence.

    Validates that the wild-type residue at each position matches the mutation
    code. Useful both for sanity-checking input CSVs and for synthesising
    `mutant_sequence` when the dataset only provides mutation codes.
    """
    seq = list(wildtype)
    for mut in mutations:
        idx = mut.position - 1
        if idx < 0 or idx >= len(seq):
            raise IndexError(
                f"Mutation {mut} position {mut.position} out of range for "
                f"sequence of length {len(seq)}"
            )
        if seq[idx] != mut.wt_aa:
            raise ValueError(
                f"WT residue mismatch at position {mut.position}: "
                f"sequence has {seq[idx]!r} but mutation code expects {mut.wt_aa!r}"
            )
        seq[idx] = mut.mut_aa
    return "".join(seq)


def load_mutation_csv(
    path: PathLike,
    *,
    drop_invalid: bool = True,
    verify_sequences: bool = True,
) -> pd.DataFrame:
    """Load a mutation CSV and validate its schema.

    Parameters
    ----------
    path:
        Path to the CSV file.
    drop_invalid:
        If True, silently drop rows with malformed mutation codes. If False,
        raise on the first malformed row.
    verify_sequences:
        If True, re-derive `mutant_sequence` from `wildtype_sequence` +
        `mutation_code` and check that it matches the value in the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required columns: {missing}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    parsed_mutations: List[Optional[Tuple[Mutation, ...]]] = []
    for code in df["mutation_code"]:
        try:
            parsed_mutations.append(parse_mutation_code(code))
        except ValueError:
            parsed_mutations.append(None)

    df = df.copy()
    df["_parsed_mutations"] = parsed_mutations
    invalid_mask = df["_parsed_mutations"].isna()

    if invalid_mask.any():
        if drop_invalid:
            df = df.loc[~invalid_mask].copy()
        else:
            bad = df.loc[invalid_mask, "mutation_code"].tolist()
            raise ValueError(f"Found {len(bad)} malformed mutation codes: {bad[:5]}...")

    df["n_mutations"] = df["_parsed_mutations"].apply(len)

    if verify_sequences:
        for idx, row in df.iterrows():
            mutations = row["_parsed_mutations"]
            if not mutations:
                continue
            try:
                derived = apply_mutations(row["wildtype_sequence"], mutations)
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Row {idx} failed sequence verification "
                    f"(protein={row['protein_id']}, code={row['mutation_code']}): {exc}"
                ) from exc
            if derived != row["mutant_sequence"]:
                raise ValueError(
                    f"Row {idx} mutant_sequence does not match wildtype + "
                    f"mutation_code (protein={row['protein_id']}, "
                    f"code={row['mutation_code']})"
                )

    df = df.drop(columns=["_parsed_mutations"])
    return df.reset_index(drop=True)


def to_variants(df: pd.DataFrame) -> List[Variant]:
    """Convert a validated mutation dataframe into a list of Variant dataclasses."""
    variants: List[Variant] = []
    for _, row in df.iterrows():
        mutations = parse_mutation_code(row["mutation_code"])
        variants.append(
            Variant(
                protein_id=str(row["protein_id"]),
                wildtype_sequence=str(row["wildtype_sequence"]),
                mutant_sequence=str(row["mutant_sequence"]),
                mutations=mutations,
                fitness=float(row["fitness"]),
                raw_code=str(row["mutation_code"]),
            )
        )
    return variants


def train_test_split_variants(
    df: pd.DataFrame,
    test_size: float = 0.25,
    *,
    seed: int = 0,
    group_by_protein: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random or grouped train/test split.

    With ``group_by_protein=True``, entire proteins are held out — the
    appropriate setting when measuring generalisation across protein families
    rather than across mutations within a protein.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)

    if group_by_protein:
        proteins = df["protein_id"].unique()
        rng.shuffle(proteins)
        n_test = max(1, int(round(len(proteins) * test_size)))
        test_proteins = set(proteins[:n_test])
        test_mask = df["protein_id"].isin(test_proteins)
    else:
        n = len(df)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        test_idx = set(idx[:n_test].tolist())
        test_mask = df.index.isin(test_idx)

    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    return train_df, test_df


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return a small dict of descriptive statistics for logging / reporting."""
    n_singles = int((df["n_mutations"] == 1).sum()) if "n_mutations" in df.columns else 0
    n_multi = int((df["n_mutations"] > 1).sum()) if "n_mutations" in df.columns else 0
    n_wt = int((df["n_mutations"] == 0).sum()) if "n_mutations" in df.columns else 0
    return {
        "n_rows": int(len(df)),
        "n_proteins": int(df["protein_id"].nunique()),
        "n_wildtype": n_wt,
        "n_single_mutants": n_singles,
        "n_multi_mutants": n_multi,
        "fitness_mean": float(df["fitness"].mean()) if len(df) else float("nan"),
        "fitness_std": float(df["fitness"].std()) if len(df) else float("nan"),
        "fitness_min": float(df["fitness"].min()) if len(df) else float("nan"),
        "fitness_max": float(df["fitness"].max()) if len(df) else float("nan"),
    }


__all__ = [
    "Mutation",
    "Variant",
    "REQUIRED_COLUMNS",
    "parse_mutation_code",
    "apply_mutations",
    "load_mutation_csv",
    "to_variants",
    "train_test_split_variants",
    "summarize_dataset",
]
