# Reproduction Guide

This guide states precisely what can be reproduced, from what inputs, and
what the boundaries are.

## Three levels of claim

| Level | Meaning | Requires |
|---|---|---|
| **Demo execution** | the public code path runs end to end | nothing beyond this repo (offline) |
| **BIRD numerical reproduction** | the paper's BIRD values re-emerge | nothing beyond this repo (offline; all inputs + frozen configs + reference predictions shipped) |
| **Experiment reproduction** | the paper's methods/protocols/analyses execute on a dataset | your authorized data in the documented format |
| **Grid numerical reproduction** | the paper's power-grid values re-emerge | the original authorized enterprise dataset + its frozen splits + comparable run conditions |

The bundled synthetic demo supports demo execution **only**. Reporting
synthetic metrics as results misrepresents what was run. The BIRD
experiments support full numerical reproduction offline (below).

## BIRD: full reproduction path (offline)

1. **Verify the shipped inputs** — `python -m unittest tests.test_bird_data`
   checks the raw-file SHA256 pins, the 55/50 cell counts, the 1530-question
   verified population (4 unverified dev questions excluded and recorded in
   the manifests), the 69-database Train development tree, and the absence
   of regenerable caches.
2. **Recompute every paper value from the reference predictions** —
   `python -m terag_release evaluate-bird-reference --output runs/refcheck`
   (no model, no API; metrics recomputed with the frozen implementation and
   compared to the shipped per-question and aggregate files; any mismatch
   fails).
3. **Re-execute methods from code** — `python -m terag_release run-bird
   --experiment <name> --output runs/...` (experiments in §"BIRD" of the
   README: three-stage, H-opt, fixed/optimized ablations, baselines, full
   matrix). Engines are rebuilt from the frozen inputs (PYTHONHASHSEED=0 is
   enforced); each question carries execution evidence for the declared
   mechanisms; a declared-but-not-executed `field_rerank`/`hg` layer is a
   hard error.
4. **Statistics** — `python -m terag_release bird-statistics --run RUN
   --output OUT` recomputes the db-cluster exact sign-flip / bootstrap over
   the frozen declared families (Bonferroni over the declared K; Delta =
   Full − Ablation; post-hoc rows labeled).
5. **Optional online baselines** — Dense (pinned MiniLM revision download)
   and the three DeepSeek API methods run only with `--allow-network`, your
   `DEEPSEEK_API_KEY`, and an explicit `--budget-cap-cny`. These are new
   measurements at your cost; they are not guaranteed to match the
   historical reference predictions, and no historical budget is reused.

Boundaries kept visible on purpose: the Q gate trades recall/FGCS for
precision (+1.91pp F1 corrected-significant, ColR −1.11 / FGCS −3.33); the
independently optimized BIRD Full−noBIndex comparison is negative (−0.51pp,
p=0.031); C shows no verifiable independent gain; H-opt's edge over the
metadata baselines does not survive declared correction; empty H folds are
recorded as empty, never fabricated.

## Grid: what the paper's numbers are tied to

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
7. **Statistics** — `statistics --run ... --gold ...` writes the paired
   variant−Full differences (Wilcoxon + cluster bootstrap, seed 42,
   families separate) that the figure step consumes.
8. **Analyze and plot** — `analyze --analysis saved-prefix|heldout|transfer|
   isolation`; `plot` (auto-detects the statistics output for whiskers);
   `benchmark` (new measurements).

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
