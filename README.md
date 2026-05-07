# ProteinLM-Bench

A reproducible benchmarking framework for evaluating protein language models
(PLMs) on **mutation effect prediction** — the task of predicting the effect
of one or more amino-acid substitutions on a protein's function or fitness.

The framework is intentionally small and modular: it loads a CSV of
mutant sequences and experimental fitness scores, generates frozen
embeddings from a pretrained PLM, trains lightweight regression heads on top,
and reports a battery of regression and retrieval metrics with bootstrap
confidence intervals, ensemble-based uncertainty, and an epistasis breakdown
for multi-mutant variants.

> **Status:** early-stage research tool. The bundled example dataset is small
> and synthetic so the demo runs end-to-end on CPU in seconds. The intended
> production workflow is to plug in a real dataset (e.g. ProteinGym, MaveDB
> deep mutational scans, or an in-house screen).

---

## Why mutation effect prediction?

Predicting how amino-acid substitutions change a protein's stability, binding
affinity, or catalytic activity is a foundational problem in computational
protein engineering and variant interpretation. Pretrained protein language
models — ESM2, ProtBert, ProtT5, Ankh, and others — have demonstrated
non-trivial zero-shot signal on these tasks, and frozen-embedding +
lightweight-head pipelines remain a strong and cheap baseline against which
more elaborate methods should be measured.

ProteinLM-Bench focuses on three things that often get short shrift in PLM
demos:

1. **Apples-to-apples model comparison**, on the same splits, with the same
   downstream heads, and the same metric definitions.
2. **Honest uncertainty**, via bootstrap CIs on the metrics themselves and
   ensemble variance on per-variant predictions.
3. **Epistasis-aware evaluation**, separating signal that comes from
   memorising single-mutant effects from signal that captures genuine
   higher-order interactions between residues.

---

## Repository layout

```
proteinlm-bench/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   └── example_mutations.csv         # small synthetic demo dataset
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── scripts/
│   ├── run_benchmark.py              # main CLI entry point
│   └── download_example_data.py      # regenerates the synthetic CSV
├── src/proteinlm_bench/
│   ├── data.py                       # CSV loading, mutation parsing, splitting
│   ├── embeddings.py                 # mock + HuggingFace PLM backends
│   ├── models.py                     # ridge / RF / MLP heads, ensembling
│   ├── metrics.py                    # Spearman, Pearson, RMSE, MAE, top-k, CIs
│   ├── epistasis.py                  # additive null, observed/predicted epistasis
│   ├── plotting.py                   # diagnostic figures
│   └── utils.py                      # config + RNG + logging helpers
└── tests/
    ├── test_data.py
    ├── test_metrics.py
    └── test_epistasis.py
```

---

## Installation

The framework targets Python 3.9+. The minimum runtime stack (numpy, pandas,
scikit-learn, scipy, matplotlib, PyTorch, PyYAML) is installed via:

```bash
pip install -r requirements.txt
```

To use real protein language models, install the optional `transformers`
extra (already in `requirements.txt`):

```bash
pip install transformers tokenizers
```

For an editable install:

```bash
pip install -e ".[all]"
```

The HuggingFace backend is **optional**: the benchmark ships a `--mock-embeddings`
mode that produces deterministic hash-based vectors so the entire pipeline
can run without downloading model weights. This is the path taken by the
test suite and by CI.

---

## Quickstart

```bash
python scripts/run_benchmark.py --config configs/default.yaml --mock-embeddings
```

This will:

1. Load the example CSV at `data/example_mutations.csv`.
2. Split into train / test (single-protein default, group-by-protein optional).
3. Embed mutant sequences with the mock backend.
4. Train ridge, random-forest, and shallow-MLP heads.
5. Compute Spearman / Pearson / RMSE / MAE / top-k enrichment with bootstrap CIs.
6. Estimate per-variant uncertainty via a 5-member bootstrap ensemble.
7. Run the epistasis analysis on multi-mutant test variants.
8. Write metrics, predictions, epistasis CSVs, and figures to `outputs/default_run/`.

To run with real ESM2 embeddings instead, drop `--mock-embeddings`:

```bash
python scripts/run_benchmark.py --config configs/default.yaml
```

The default config points at `facebook/esm2_t6_8M_UR50D`; switch to
`Rostlab/prot_bert` (or any other compatible HuggingFace AutoModel) by editing
`configs/default.yaml`.

---

## Plugging in your own data

The framework expects a CSV with the schema:

| column              | description                                              |
| ------------------- | -------------------------------------------------------- |
| `protein_id`        | identifier for the parent protein                        |
| `wildtype_sequence` | reference sequence (1-indexed positions)                 |
| `mutation_code`     | `A42G`, or `A42G:T15S` for multi-mutants, or `WT`        |
| `mutant_sequence`   | the protein sequence with the mutation(s) applied        |
| `fitness`           | experimental score (any monotonic fitness proxy is fine) |

Three good places to source real datasets:

* **[ProteinGym](https://proteingym.org/)** — a curated benchmark of zero-shot
  variant effect tasks. Rename the `mutant`, `mutated_sequence`, and
  `DMS_score` columns and join in the WT, and the CSV is ready to go.
* **[MaveDB](https://www.mavedb.org/)** — a public archive of multiplexed
  assays of variant effect (deep mutational scans). Parse the `hgvs_pro`
  column into our `:`-joined notation.
* **In-house screens** — emit a CSV in the schema above and pass it via
  `python scripts/run_benchmark.py --data path/to/your.csv`.

---

## Benchmark metrics

| metric              | what it measures                                                                       |
| ------------------- | -------------------------------------------------------------------------------------- |
| `spearman`          | rank correlation between predicted and observed fitness                                |
| `pearson`           | linear correlation                                                                     |
| `rmse`              | root mean squared error                                                                |
| `mae`               | mean absolute error                                                                    |
| `top_k_enrichment`  | fraction of top-fraction-by-prediction variants that are also top by truth (retrieval) |

Spearman is the most common headline number in the variant-effect literature,
because experimental fitness scales are typically non-linear and only
monotonically related to the underlying biophysical quantity. RMSE and MAE
are reported as well for users whose downstream task cares about absolute
fitness values. The top-k enrichment metric reflects the protein engineering
use case, where retrieving the best few hundred variants out of thousands
matters more than fitting the bulk of the distribution.

All metrics are reported with bootstrap confidence intervals
(default: 200 samples, 95% CI), so the same number is harder to overinterpret.

### Uncertainty estimation

* **Bootstrap CIs** on every reported metric.
* **Ensemble variance** on per-variant predictions, obtained by training
  `evaluation.ensemble_size` regressors on bootstrap-resampled training data.
  The variance is then binned and plotted against empirical RMSE in
  `figures/calibration_<model>.png`; a well-calibrated model has bins that
  follow the y = x diagonal.

---

## Epistasis analysis

For every multi-mutant variant in the test set, ProteinLM-Bench computes the
**additive (non-epistatic) expectation**:

```
f_add(m1, m2, ...) = f_WT + Σ_i (f_mi - f_WT)
```

where `f_WT` is the wild-type fitness and `f_mi` is the experimental fitness
of single-mutant `mi`. This is the prediction you would make if the protein
had zero residue-residue interactions.

Two derived quantities are then reported per variant:

* **Observed epistasis** = `observed_fitness − f_add`. This is the genuine
  experimental epistatic effect, sourced entirely from the data.
* **Predicted epistasis** = `predicted_fitness − f_add`. This is what your
  model thinks the epistatic effect is.

Their correlation (`epistasis_correlation` in the summary output) tells you
whether the model is recovering real epistatic structure or merely
memorising single-mutant effects: a model that only memorises singles will
predict `f_add` and score zero on this metric, regardless of how well it
does on the headline Spearman number.

Variants whose model residual exceeds `epistasis.outlier_sd_threshold`
standard deviations are flagged as `is_outlier`. These are the candidates
worth looking at by hand — they tend to be where structural context matters
most and where simple frozen-embedding heads struggle.

---

## Outputs

After a run, `outputs/<experiment.name>/` contains:

```
metrics.json                     # full results: per-model metrics, predictions, epistasis
metrics_summary.csv              # flat one-row-per-model table
epistasis_<model>.csv            # per-multi-mutant epistasis breakdown
figures/
  pred_vs_obs_<model>.png        # scatter, with y=x reference
  residuals_<model>.png          # residual histogram
  calibration_<model>.png        # predicted std vs empirical RMSE per bin
  embedding_umap.png             # UMAP of test-set embeddings, colored by fitness
  model_comparison_<metric>.png  # bar chart across models
```

---

## Testing

```bash
pytest
```

The bundled tests cover mutation parsing, sequence application, metric
correctness on synthetic inputs, bootstrap CI sanity, additive-expectation
arithmetic, and outlier flagging. They do not require any model downloads
and run in well under a second.

---

## Limitations

* The example CSV is **synthetic and tiny**. Numbers from running on it have
  no scientific meaning — they exist only to verify that the pipeline works
  end-to-end. Real benchmarks need real data.
* The default split is random. For protein-level generalisation, set
  `data.group_by_protein: true`. For mutation-position-level generalisation
  (e.g. holding out specific positions, as ProteinGym does), you will need to
  extend `data.train_test_split_variants` — the current splitter is
  deliberately simple.
* Only frozen-embedding + linear/MLP heads are evaluated. Methods that
  fine-tune the PLM, that condition on structure (ESMFold, AlphaFold), or
  that use evolutionary couplings (EVE, MSA Transformer) are out of scope
  for the v0.1 baseline and would each warrant their own benchmark module.
* The mock embedding backend is a deterministic hash and produces a
  weak-but-non-zero signal on the toy dataset; do not interpret mock-mode
  numbers as PLM performance.
* Multi-mutant epistasis analysis requires that all constituent
  single-mutants are present in the dataset (this is the convention in deep
  mutational scans). Variants without a complete set of singles are simply
  excluded from the epistasis report.

---

## Future work

* **More heads**: gradient-boosted trees, GP regression, calibrated quantile
  heads.
* **More backbones**: ProtT5, Ankh, ESM2-650M / 3B, and structure-aware
  models like SaProt.
* **Zero-shot protocols**: log-likelihood ratios from masked language
  modelling, which sidestep the need for a downstream head entirely.
* **Stratified evaluation**: per-position, per-protein, and per-conservation
  bucket reporting, so we can see *where* embeddings fail rather than just
  *how often*.
* **Active learning loops**: use the ensemble variance to drive variant
  selection in a closed-loop protein engineering setting.
* **Native ProteinGym integration**: a one-command importer for the
  ProteinGym substitution and indel benchmarks.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use ProteinLM-Bench in academic work, please cite the repository
and the underlying protein language model whose embeddings you evaluate.
