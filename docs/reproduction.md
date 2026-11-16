# Reproduction Guide

This guide states precisely what can be reproduced, from what inputs, and
what the boundaries are.

## Three levels of claim

| Level | Meaning | Requires |
|---|---|---|
| **Demo execution** | the public code path runs end to end | nothing beyond this repo (offline) |
| **Experiment reproduction** | the paper's methods/protocols/analyses execute on a dataset | your authorized data in the documented format |
| **Numerical reproduction** | the paper's reported values re-emerge | the original authorized enterprise dataset + its frozen splits + comparable run conditions |

The bundled synthetic demo supports demo execution **only**. Reporting
synthetic metrics as results misrepresents what was run.

## What the paper's numbers are tied to

* Dataset: a private power-grid export (3,035 table records, 142,500 column
  records; 88 question–SQL pairs; 25 originally annotated target tables).
* Populations: 88 run records; 76 questions with complete verifiable gold;
  65 gold-budget-feasible at ten columns (58 at five columns). These are
  **properties of that dataset**; this implementation recomputes every
  population from the supplied gold (`terag_release.runner.gold_populations`,
  cross-validated by `evaluate`).
* Result version: TE-RAG rows/cases come from the corrected replay
  (`transfer_v3_target_cap`) under unchanged gold, folds, and parameters;
  cost/efficiency figures come from the **original** runs. Both statements
  carry over here: the algorithm identity is unchanged, and
  `terag_release benchmark` produces *new* measurements only.
* Splits: five fixed question folds; five target-table folds with AST-based
  training exclusions. Fold construction is deterministic (no RNG); the
  algorithm is reimplemented in `terag_release.protocols` and a frozen
  manifest can be supplied instead.

## End-to-end on your data (experiment reproduction)

1. **Prepare inputs** (formats in data_format.md): field schema CSV, table
   schema CSV, QA CSV (question, sql, original table/field annotation).
2. **Build gold** — `python -m terag_release build-gold ...` (AST via
   sqlglot; statuses `verified_complete` / `needs_review` / `parse_failed`;
   unresolved or ambiguous references are recorded, never forced into gold).
3. **Build folds** — `python -m terag_release make-folds ...` (question
   folds with duplicate-group safety; table folds with regex v1 split +
   AST v2 training exclusion; all invariants asserted).
4. **Validate** — `python -m terag_release validate-data ...`.
5. **Run** — `run-experiments` (offline first; `--allow-network` for
   Dense/DenseDeepSeek/adapted methods; `DEEPSEEK_API_KEY` from the
   environment only; temperature 0.1, one transient retry, no fallback —
   failures remain in the denominator).
6. **Evaluate offline** — `evaluate --families both` recomputes every
   metric from saved predictions; later prediction files replace earlier
   *complete* method/protocol groups (never per-question patching).
7. **Analyze** — `analyze --analysis saved-prefix|heldout|transfer|isolation`;
   `plot`; `benchmark` (new measurements).

## What is explicitly NOT recoverable

* Original per-transfer traces and per-query score components — never saved
  by the original runs. New `--diagnostics` exports are new-computation
  evidence with declared identity, not recovered history.
* The paper's efficiency table — original code, original machine, original
  measurement boundary.
* Held-out results for methods not run on that protocol — must remain
  absent, never zero-filled.
* The enterprise data itself, and therefore exact numerical reproduction
  without the owner's authorization.

## Statistical protocol (for reference)

* Question-paired Wilcoxon signed-rank: two-sided, `zero_method='wilcox'`,
  no continuity correction, `method='auto'` (SciPy).
* Cluster bootstrap: clusters = gold `original_table`; 10,000 cluster
  resamples; RNG `numpy.random.default_rng(42)`; percentile 2.5/97.5 CI,
  unadjusted.
* Families (Bonferroni applied by the caller): k=6 Covered@5 family
  {noBC, noAnnIdx, noAnnHit, Dense, DenseDeepSeek, ExampleRRF}; k=2 adapted
  family {LinkAlign-SingleDB-Adapted, KaSLA-Adapted}; always vs
  `question_v1/TERAG-Full`.
* Zero differences → no p-value; incomplete family → `not_computed` with
  missing members listed.

## Regression expectations on the demo

The test suite (`tests/`) pins: protocol invariants, exact ablation toggles,
metric semantics (including the FGCS normalization correction and
error-in-denominator handling), transfer caps (≤3 per normalized physical
target across duplicate schema rows, ≤8 texts per table, call-local state),
config preflight, English-comment policy, and package-content safety. The
end-to-end demo test additionally checks that offline re-evaluation of saved
predictions matches the runner's own summary.
