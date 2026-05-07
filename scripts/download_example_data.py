"""Generate (or refresh) the bundled synthetic example dataset.

The file shipped at ``data/example_mutations.csv`` is a small synthetic dataset
designed so the benchmark runs end-to-end on CPU without any external
downloads. Real benchmarks should plug in a public DMS dataset instead — see
the pointers at the bottom of this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from proteinlm_bench.data import apply_mutations, parse_mutation_code  # noqa: E402
from proteinlm_bench.utils import ensure_dir, get_logger, set_seed  # noqa: E402

logger = get_logger("download_example_data")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _random_protein(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(AMINO_ACIDS, size=length))


def _random_single_mutations(seq: str, n: int, rng: np.random.Generator) -> list:
    positions = rng.choice(len(seq), size=n, replace=False) + 1  # 1-indexed
    codes = []
    for p in positions:
        wt = seq[p - 1]
        mut = wt
        while mut == wt:
            mut = rng.choice(AMINO_ACIDS)
        codes.append(f"{wt}{int(p)}{mut}")
    return codes


def synthesize_dataset(
    n_proteins: int = 2,
    seq_length: int = 60,
    n_singles_per_protein: int = 25,
    n_doubles_per_protein: int = 8,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthesize a small, realistic-looking mutation/fitness dataset.

    Per-residue fitness effects are drawn once at the start (so each position
    has a stable "preference"), and the multi-mutant fitness is the additive
    sum of single-mutant effects plus a small epistasis term — giving the
    epistasis analysis something non-trivial to find.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p_idx in range(n_proteins):
        protein_id = f"demo_protein_{chr(ord('a') + p_idx)}"
        wt = _random_protein(seq_length, rng)
        # Per-position-and-residue fitness effect table
        effect = rng.normal(0, 0.3, size=(seq_length, len(AMINO_ACIDS)))
        aa_to_idx = {a: i for i, a in enumerate(AMINO_ACIDS)}

        rows.append(
            {
                "protein_id": protein_id,
                "wildtype_sequence": wt,
                "mutation_code": "WT",
                "mutant_sequence": wt,
                "fitness": 0.0,
            }
        )

        single_codes = _random_single_mutations(wt, n_singles_per_protein, rng)
        single_effects = {}
        for code in single_codes:
            mutations = parse_mutation_code(code)
            mut = mutations[0]
            f = float(effect[mut.position - 1, aa_to_idx[mut.mut_aa]])
            single_effects[str(mut)] = f
            rows.append(
                {
                    "protein_id": protein_id,
                    "wildtype_sequence": wt,
                    "mutation_code": code,
                    "mutant_sequence": apply_mutations(wt, mutations),
                    "fitness": round(f, 3),
                }
            )

        for _ in range(n_doubles_per_protein):
            pair = rng.choice(single_codes, size=2, replace=False)
            if pair[0] == pair[1]:
                continue
            code = ":".join(pair)
            mutations = parse_mutation_code(code)
            try:
                mut_seq = apply_mutations(wt, mutations)
            except (ValueError, IndexError):
                continue
            additive = sum(single_effects[str(m)] for m in mutations)
            epistasis = float(rng.normal(0, 0.2))
            f = additive + epistasis
            rows.append(
                {
                    "protein_id": protein_id,
                    "wildtype_sequence": wt,
                    "mutation_code": code,
                    "mutant_sequence": mut_seq,
                    "fitness": round(f, 3),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=str, default=str(ROOT / "data/example_mutations.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-proteins", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--n-singles", type=int, default=25)
    parser.add_argument("--n-doubles", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    df = synthesize_dataset(
        n_proteins=args.n_proteins,
        seq_length=args.seq_length,
        n_singles_per_protein=args.n_singles,
        n_doubles_per_protein=args.n_doubles,
        seed=args.seed,
    )
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return 0


# -----------------------------------------------------------------------------
# Plug-in points for real datasets
# -----------------------------------------------------------------------------
# To benchmark against real data:
#
#   1. ProteinGym (https://proteingym.org/): download the substitution
#      benchmark CSVs and adapt to the schema:
#        protein_id, wildtype_sequence, mutation_code, mutant_sequence, fitness
#      ProteinGym already provides `mutant`, `mutated_sequence`, and
#      `DMS_score` columns — just rename them and join in the WT sequence.
#
#   2. Deep mutational scanning datasets (e.g. those archived at MaveDB,
#      https://www.mavedb.org/): export the CSV, parse the `hgvs_pro` column
#      into our colon-joined mutation code format, and supply the WT.
#
#   3. Custom in-house screens: produce a CSV in the schema above and pass
#      it via `python scripts/run_benchmark.py --data path/to/your.csv`.
#
# Larger datasets are out-of-scope for this script's defaults — synthetic
# data exists only so the demo is self-contained.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
