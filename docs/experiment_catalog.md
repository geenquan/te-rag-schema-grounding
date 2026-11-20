# Experiment Catalog

This catalog maps every experiment, table, and figure reported in the paper
*TE-RAG: Historical Domain Annotations for Enterprise Schema Grounding*
(DASFAA 2027 submission, paper tag `paper-dasfaa2027-v1.0`, commit
`25aa5e807d5cd6f8d5a3911e253ef89424d743e4`) to its implementation, configuration,
and execution entry point in this repository.

Reading conventions:

* **Verified (paper run)** — the number/result was produced by the private,
  authorized evaluation over the enterprise dataset. This repository contains
  the exact implementation, but not that dataset; reproducing the *numbers*
  requires the authorized data, fixed splits, and comparable run conditions.
* **Demo execution** — runnable here on the bundled synthetic dataset; it
  verifies that the code path works, **not** that it reproduces paper numbers.
* **Experiment reproduction** — with user-supplied data in the documented
  format, the same protocol can be executed end to end.

The paper's fixed evaluation constants (88 run records / 76 valid complete-gold
questions / 65 ten-column budget-feasible questions / 58 five-column feasible)
are properties of the private dataset. This implementation **never hardcodes
those denominators**: every population is recomputed from the supplied gold
(`terag_release evaluate`, `experiment/audit_saved_predictions.py`).

---

## 0. Final-paper experiment map (profile `paper_final_20261120`)

The submitted paper (docs/paper_with_appendix.pdf) reports the experiments
below. Every BIRD value is reproducible offline here; grid values need the
authorized enterprise dataset (engine + protocols + frozen parameters are
complete).

| Paper item | What it is | Reproduce with |
|---|---|---|
| Table 1 (datasets) | Power-grid / BIRD population counts | grid: private data (documented); BIRD: `data/bird/` counts verified by `tests/test_bird_data.py` |
| Table 2 `tab:qoverall` — Q overall (both datasets) | 11 power-grid methods + 12 BIRD methods, Covered@5/ColP/ColR/ColF1/FGCS | BIRD: `run-bird --experiment bird_full_matrix` (+ `bird_dense`/`bird_api` for the last three baselines, or read `results/reference/bird/aggregates/question_summary.csv`); grid: baselines via `run-experiments` on authorized data, Full via the grid engine replay |
| Figure 3 `fig:overall-performance` | Covered@5 vs column F1 scatter, 23 points (11 grid + 12 BIRD) | computed from the same aggregate files (no experiments run by the figure) |
| Table 3 `tab:qfixed` — Q fixed-config ablations | Grid fixed ablations (per-fold params) + BIRD `qB_rerank_gate_v1-no*` nine rows | BIRD: `run-bird --experiment bird_q_fixed_ablation`; grid: grid engine with `grid_selected_configurations.csv` |
| Figure 4 `fig:ablation-heatmap` | 8 ablations × 3 metrics × 2 datasets, direction **Ablation − Full** (opposite of the statistics convention) | same inputs as Table 3 |
| Table 4 `tab:gate` — BIRD Q field decomposition | Core / Rerank only / Gate (P/R/F1, FGCS, Avg. columns) | `run-bird --experiment bird_q_three_stage` |
| A1 `tab:qoverallpr` | Q overall P/R detail | `aggregates/question_summary.csv` (colp/colrec columns) |
| A2 `tab:qfixedpr` | Q fixed ablation P/R detail | `metrics/question/<method>_per_question.csv` |
| A3 `tab:qindependent` | Independently optimized Full and no* tracks (BIRD 29.02/29.53 identity) | `run-bird --experiment bird_optimized_replay` |
| A4 `tab:hall` | H overall (core 76.54/21.99/51.70; H-opt 82.29/22.98/57.97) | `run-bird --experiment bird_h_opt` (Core + `full_href_1_8`) |
| A5 `tab:hfixed` | H fixed ablations | same experiment, `full_href_1_8-no*` rows |
| A6 `tab:hindependent` | H independently optimized track | `bird_optimized_replay` (optimized_full_H / optimized_H_no*) |
| Efficiency table | original-run timings | `benchmark` produces NEW measurements only |
| Selector API cost (≈ CNY 30) | historical spend | documented boundary; never reused — online reruns use your own cap |

Statistical statements (Q gate +1.91pp F1 corrected p=0.0352; optimized
Full−noBIndex = −0.51pp p=0.031; noFieldGraph +0.59pp p=0.035) come from the
db-cluster exact sign-flip / bootstrap over the frozen declared families
(`configs/paper/final_20261120/bird_families.json`):
`python -m terag_release bird-statistics --run RUN --output OUT`. The
declared-family Bonferroni K is never shrunk; zero/cancel members keep
their slot; comparisons against `route*` rows are labeled post-hoc
exploratory. Direction convention: **Delta = Full − Ablation**; Figure 4's
heatmap is the opposite direction (Ablation − Full) and is labeled so in the
paper.

BIRD reference predictions (`results/reference/bird/`) allow every value
above to be recomputed without running anything:
`python -m terag_release evaluate-bird-reference --output runs/refcheck`.

---

---

## 1. Methods (11 reported in main/appendix tables)

| Paper method | Registry name | Implementation | Method config |
|---|---|---|---|
| BM25 | `BM25` | `retrievers/bm25_retriever.py` | `configs/methods/bm25.yaml` |
| TF-IDF | `Vector` | `retrievers/vector_retriever.py` (legacy class name `VectorRetriever` = TF-IDF, not dense) | `configs/methods/tfidf.yaml` |
| Hybrid | `Hybrid` | `retrievers/hybrid_retriever.py` (repository-specific lexical/sparse combination) | `configs/methods/hybrid.yaml` |
| MetadataMatched | `MatchedSchema` | `retrievers/matched_schema_retriever.py` (comment-enriched index, no history) | `configs/methods/metadata_matched.yaml` |
| ExampleRRF | `ExampleRAG` | `retrievers/example_rag_retriever.py` (TF-IDF-retrieved historical examples, RRF fusion) | `configs/methods/example_rrf.yaml` |
| CooccurRRF | `Cooccurrence` | `retrievers/cooccurrence_retriever.py` (token–element co-occurrence, RRF fusion) | `configs/methods/cooccur_rrf.yaml` |
| Dense | `Dense` | `retrievers/dense_retriever.py` (multilingual MiniLM; optional deps: torch/transformers) | `configs/methods/dense.yaml` |
| DenseDeepSeek | `DenseDeepSeek` | `retrievers/linkalign_retriever.py` (legacy file name; **custom** Dense top-20 tables + per-table top-10 columns + DeepSeek selection — *not* LinkAlign) | `configs/methods/dense_deepseek.yaml` |
| LinkAlign-SingleDB-Adapted | `LinkAlign-SingleDB-Adapted` | `retrievers/linkalign_official_retriever.py` + `retrievers/upstream/linkalign_prompts.py` | `configs/methods/linkalign_adapted.yaml` |
| KaSLA-Adapted | `KaSLA-Adapted` | `retrievers/kasla_retriever.py` + `retrievers/upstream/kasla_utils.py` | `configs/methods/kasla_adapted.yaml` |
| TE-RAG (Full) | `TE-RAG-V2` | `terag/terag_retriever_v2.py` via `TERAGV2Adapter` in `retrievers/retriever_factory.py`; algorithm identity `transfer_v3_target_cap` | `configs/methods/terag_full.yaml` |

Notes carried over from the paper:

* The two adapted methods keep the `-Adapted` suffix everywhere; they declare
  representation/model/scorer replacements (multilingual MiniLM for BGE,
  DeepSeek for Qwen/LLM, logistic scorers for CodeS) and do **not** reproduce
  the original checkpoints. They were run on Question-Level only; the paper
  reports no Held-Out-Table numbers for them and none may be imputed.
* `DenseDeepSeek` is an independent custom baseline, not LinkAlign.
* Historical names `LLM`, `TE-RAG` (v1), `Vector-as-dense` and SQL-accuracy
  results are **not** part of the paper's experiment matrix and are not
  shipped as reported methods.
* **BIRD (final paper)** runs the same ten baseline names through the frozen
  BIRD workspace retrievers (`terag/engines/bird_runtime/retrievers/`, wired
  in `terag/engines/bird/baselines.py`) — these differ from the legacy grid
  retrievers above, and the BIRD numbers come from them. The engine methods
  (`Core`, selected/fixed/optimized/adaptive rows) run through the frozen
  BIRD engines (`terag/engines/bird/`) and the unified entry
  (`terag/engines/bird/unified_entry.py`).

## 2. Formal protocol matrix (25 saved method/protocol groups)

Two protocols over the same question set (paper §4.1):

* **Question-Level (`question_v1`)** — five fixed folds; each question is
  tested exactly once; training supervision is the training-fold SQL
  (`full_sql` injection). Config: `configs/protocols/question_level.yaml`.
* **Held-Out-Table (`table_v2`)** — five target-table folds; training SQL
  never references the fold's held-out target tables (AST-based training
  exclusion, see §6 below). Config: `configs/protocols/heldout_table.yaml`.

Output budget for every method/protocol: at most **5 tables**, at most **10
columns per table** (k=5 tables for the main comparison; the adapted
comparisons use their declared pools but project to the same budget).

| # | Protocol | Group | Variant definition (exact toggles) |
|---|---|---|---|
| 1–11 | Q | BM25, TF-IDF, Hybrid, MetadataMatched, ExampleRRF, CooccurRRF, Dense, DenseDeepSeek, LinkAlign-SingleDB-Adapted, KaSLA-Adapted, TERAG-Full | Full = annotations `use_history=true`, `use_transfer=true`, `use_annotation_index=true`, graph `parse_mode=ast`, all remaining parameters at `configs/base.yaml` defaults |
| 12 | Q | TERAG-noBC | `annotations.use_history=false` (implies no transfer) |
| 13 | Q | TERAG-noAnnIdx | `annotations.use_annotation_index=false` (B/C kept out of the index; extraction and column TERM retained) |
| 14 | Q | TERAG-noAnnHit | `feature.field_weights.annotation_hit=0.0` (column TERM hit score disabled) |
| 15–20 | H | BM25, TF-IDF, Hybrid, MetadataMatched, ExampleRRF, CooccurRRF, TERAG-Full | same definitions |
| 21 | H | TERAG-noBC | as above |
| 22 | H | TERAG-noC | `annotations.use_transfer=false` (B only) |
| 23 | H | TERAG-noGraph | `ablation.use_graph_weight=false` — explicit table/column graph **scores** disabled; the bipartite graph is still built (pattern mining consumes it). This is *not* "graph never constructed". |
| 24 | H | TERAG-noPattern | `ablation.use_template_mining=false` — the independent keyword-pattern channel disabled |
| — | H | (TERAG-noAnnIdx / noAnnHit / adapted methods) | **Not run** in the paper for this protocol; absent groups must not be zero-filled |

Variant definitions live in `terag_release/variants.py` (single source of
truth, mirrored in `configs/methods/*.yaml`).

Execution:

```sh
# full matrix on user data (config references are NAME-or-PATH:
# "mydata" resolves under configs/datasets/, or pass an explicit file path)
python -m terag_release run-experiments --dataset mydata \
    --experiment full_matrix --output runs/full_matrix

# offline re-evaluation of saved predictions (denominators recomputed)
python -m terag_release evaluate --gold GOLD.jsonl --budget budget.json \
    --predictions PRED.jsonl [PRED2.jsonl ...] --output NEW_DIR --families both
```

## 3. Metrics (Table `tab:main`, `tab:adapted`, `tab:all_question`; §4.2)

* Covered@5 — all gold tables of the question appear in the top-5 returned
  tables. Column P/R/F1 — macro-averaged set metrics over normalized
  `table.column` keys. FGCS — full-gold coverage success: gold tables covered
  **and** every gold column returned; computed with the same normalization as
  column P/R/F1 (the paper's case-sensitivity correction is inherent in
  `eval/retrieval_metrics.py::schema_coverage_success`).
* Table Recall@1/@5 (appendix Table `tab:recall_details`) —
  `table_recall_at_k`.
* Errors and legal empty predictions stay in the denominator as failures
  (`eval/full_gold.py::full_gold_metrics`: `status == "error"` ⇒ empty output
  even if a record carries stale fields).

Implementation: `eval/retrieval_metrics.py` (single definition source),
`eval/full_gold.py` (full-gold population + aggregation). Entrances:
`terag_release evaluate` / `python -m experiment.audit_saved_predictions`.

Demo status: demo execution computes all of these on synthetic gold.

## 4. Column saved-prefix analysis (Table `tab:column_budget`)

Saved top-5 table outputs truncated to 5 or 10 raw column slots per table
(original column order kept; duplicate slots retained; metrics use normalized
sets). Gold-feasible populations (per-table gold column count ≤ budget and ≤5
gold tables) are recomputed from gold — never hardcoded 58/65.

Implementation: `analysis/saved_prefix.py` (port of the review-S3 analysis;
functions `eligible`, `crop`, per-question error rows, aggregation).
Entrance: `python -m terag_release analyze --analysis saved-prefix --run RUN_DIR ...`.

## 5. Gold completeness, budget feasibility, train/test isolation

* **Gold construction** — `scripts/build_full_gold.py`: AST (sqlglot,
  MySQL→Hive chain) resolution of every QA SQL against the real schema CSVs;
  statuses `verified_complete` / `needs_review` / `parse_failed`; original
  SELECT annotations preserved as a separate caliber (`original_fields`);
  unresolved/ambiguous items recorded, never forced into gold.
* **Budget feasibility** — derived purely from gold: ≤5 distinct gold tables
  and ≤ budget (10/5) gold columns per table. Recomputed and cross-validated
  by `terag_release evaluate` / `experiment/audit_saved_predictions.py`.
* **Isolation** — `scripts/verify_isolation.py`: independent regex reference
  extraction (FROM/JOIN) cross-checked against AST gold table sets; derives
  the Held-Out-Table v2 training exclusions (train questions whose SQL
  references a held-out target table are excluded; unparseable train SQL is
  conservatively excluded). The runner re-asserts zero target-table leaks per
  fold before fitting.

## 6. Paired statistics (§4.1 "Metrics and statistics")

* Question-paired Wilcoxon signed-rank (two-sided, `zero_method='wilcox'`,
  no continuity correction, `method='auto'`).
* Target-table cluster bootstrap: clusters = gold `original_table`; 10,000
  cluster resamples, seed 42, percentile 2.5/97.5 CI, unadjusted.
* Two declared families, corrected separately: **k=6** Covered@5 family =
  {noBC, noAnnIdx, noAnnHit, Dense, DenseDeepSeek, ExampleRRF} vs Full;
  **k=2** adapted family = {LinkAlign-SingleDB-Adapted, KaSLA-Adapted} vs
  Full. Bonferroni = min(1, p_raw × k) applied by the caller.
* Zero-difference comparisons get no fabricated p-value; insufficient samples
  are reported as not computable.

Implementation: `eval/full_gold.py::paired_covered`; family assembly in
`experiment/audit_saved_predictions.py` (`FAMILIES`) — imported and reused
by `analysis/statistics.py` so the two entry points cannot drift. The
fixed families are Question-Level comparisons; held-out comparisons are
exploratory. A family-corrected value requires the complete family in the
run (otherwise `not_computed` with missing members; raw paired values
remain available for plotting). Requires SciPy (optional dependency; the
CLI reports clearly when unavailable). Demo status: the deterministic
synthetic generator produces enough records/clusters to exercise the
flow; results on synthetic data are **not** paper conclusions. Zero
difference means identical per-question Covered@5 values, not identical
retrieval outputs (the paper's Full/noC boundary).

## 7. Transfer (C) evidence and diagnostics (§3.5, appendix `sec:transfer_audit`)

Counts and constraints of accepted one-hop transfers: ≤3 acceptances per
normalized physical target column (across duplicate schema rows), ≤8 distinct
C texts per table, one-hop only (C never becomes a source), direct-first
deduplication, rejection totals with bounded samples.

Implementation: `retrievers/annotation_transfer.py`
(`METHOD_VERSION = "transfer_v3_target_cap"`), consumed inside
`terag/terag_retriever_v2.py::fit`. Per-fold evidence (accepted records,
rejections, merged index inputs, column terms) is exportable via
`terag/diagnostics.py::DiagnosticExport` (off by default; enabling it does not
change outputs). Analysis entrance: `python -m terag_release analyze
--analysis transfer --run RUN_DIR`.

Paper caveats that remain true here: historical per-transfer traces and
per-query score components of the original runs were not saved; new exports
are new-computation evidence, not recovered originals.

## 8. Failure/case analysis (§4.6, Table `tab:cases`)

Held-out disagreement categories (both-success / Full-only / control-only /
both-fail), per-question gold-table ranks within saved top-5, case selection
rules, Full-vs-noC output comparison.

Implementation: `analysis/heldout_failures.py` (port of the review-S3
disagreement logic, parameterized method set; historical fixed case IDs are
not hardcoded). Entrance: `python -m terag_release analyze --analysis heldout ...`.

## 9. Efficiency (Table `tab:efficiency`, §4.5)

The paper's numbers are **original-run wall-clock measurements** (arithmetic
mean Fit over five folds; Query = mean of fold means; all 88 run records;
Python 3.11.3, macOS, x86-64). They are tied to the original code and machine
and are **not** reproduced by this repository.

This repository provides `python -m terag_release benchmark` — a fresh timing
harness over the current implementation with the same measurement boundaries
(fit wall-clock per fold; per-query latency averaged per fold then across
folds). Any timing it produces is a **new measurement** and must be labeled as
such; it must not be presented as a reproduction of the paper's efficiency
table.

## 10. Figures (Fig. `fig:overall_tradeoff`, Fig. `fig:ablation_boundary`)

Regenerable from saved evaluation outputs: (a) five question-level controls —
Covered@5 and column P–R/F1; (b) variant-minus-Full Covered@5 differences with
unadjusted cluster CIs (hollow markers where no CI exists: identical outputs);
(c) held-out Covered@5 versus macro **column recall** for run methods.
Framework diagram (Fig. `fig:framework`) is a schematic, not an experiment,
and is not regenerated.

Implementation: `analysis/statistics.py` (paired Wilcoxon + cluster bootstrap,
seed 42, 10,000 resamples, direction variant−Full with consistently flipped
CI endpoints; families k=6/k=2 kept separate; zero differences carry no
interval) and `analysis/plots.py`. Entrances:

```sh
python -m terag_release statistics --run RUN_DIR --gold GOLD.jsonl
# writes RUN_DIR/statistics/statistics.json (auto-detected by plot)
python -m terag_release plot --run RUN_DIR --output FIG_DIR
```

Interval whiskers come only from the statistics file; both steps are purely
offline over saved predictions. The paper embeds vector PDFs rendered from
the private results; these scripts re-render equivalent panels from any
conforming run directory and are not numerical reproductions.

---

## What is explicitly *not* in this repository

* The enterprise dataset, gold, folds, predictions, prompts/responses, API
  caches, indexes/graphs/annotation artifacts of the paper runs — including
  enterprise reference predictions.
* The BIRD SQLite database archives (schema grounding needs no database
  rows; official downloads linked in `data/bird/SOURCES.md`).
* Historical SQL-accuracy execution evaluation (`eval/sql_eval.py`); SQL
  generation/execution is out of scope for the paper. SQL is parsed (AST) for
  supervision/gold/isolation only — `SQL AST parsing is not database
  execution`.
* The private replay tooling that consumes prior private run artifacts
  (`experiment/replay_*.py`, `experiment/analyze_transfer_replay.py`,
  `experiment/diagnose_transfer_similarity.py`); the capabilities are covered
  by the fresh runner + offline evaluation + analysis modules.
