# TE-RAG (Public Release)

Public implementation of **TE-RAG: Historical Domain Annotations for
Enterprise Schema Grounding** — a retrieval framework that turns sparse
historical *question–SQL* pairs into reusable, source-linked domain
annotations, extends them by constrained one-hop transfer, and uses the
accepted text in schema indexing and column matching to ground a new
question to the physical tables and columns it needs.

The task is **schema grounding**: return at most 5 candidate tables and at
most 10 columns per table for a natural-language question. SQL *generation*
and *execution* are out of scope; SQL is parsed (AST) for supervision, gold
construction, and isolation auditing only.

## Version identity

| Item | Value |
|---|---|
| Public release | `public-v0.1.x` (see CHANGELOG) |
| Final-paper profile | `paper_final_20261120` — the frozen identities of the current submission (this README's BIRD paths; see below and docs/release_mapping.md) |
| Original paper baseline | tag `paper-dasfaa2027-v1.0`, commit `25aa5e807d5cd6f8d5a3911e253ef89424d743e4` |
| Algorithm identity (grid) | `transfer_v3_target_cap` scoring, final engine identity `graph_casefold_ids_v1+pattern_consumer_casefold_v1` (see docs/release_mapping.md) |
| BIRD engine identity | per-database frozen engines (`BirdEngine`/`VnextEngine`) executed through a unified entry; implementation ids recorded per run |
| Relationship | Same algorithm lineage, exported under an explicit file whitelist with English comments, parameterized paths, a unified CLI, layered configs, a synthetic demo, the complete public BIRD inputs, and offline analyses. Private enterprise data, predictions, prompts, and caches are **not** included. |

The paper with appendix is shipped as
[docs/paper_with_appendix.pdf](docs/paper_with_appendix.pdf)
(SHA256 `c426ba6a178456304396f043c2b5f2e7f9c3cd52d0bea6569c467e3081021c46`,
21 pages). No TeX sources, figures, or writing materials are included.

What this repository supports — stated precisely:

* **Demo execution** — the bundled synthetic dataset runs the full pipeline
  offline. It verifies that public code paths work. **It does not reproduce
  any paper number.**
* **BIRD reproduction (complete)** — the full public BIRD Train/Dev inputs,
  frozen five-fold manifests, gold, per-database schemas, and frozen engine
  configurations are included, so every BIRD table of the paper (overall Q,
  the Q three-stage Core/Rerank/Gate decomposition, H-opt, fixed and
  independently-optimized ablations, all offline baselines) can be
  re-executed end to end offline, and every reported value recomputed from
  the shipped reference predictions without any model or API.
* **Experiment reproduction on private data** — with your own authorized
  data in the documented format, every reported method, protocol, and
  analysis can be executed end to end (the power-grid final engine is
  included for exactly this purpose).
* **Grid numerical reproduction** — reproducing the power-grid *values*
  additionally requires the original authorized enterprise dataset, its
  fixed splits, and comparable run conditions. Those inputs are not
  redistributed here; the denominators (e.g. 88 runs / 76 valid / 65
  budget-feasible in the paper) are properties of that dataset and are
  always **recomputed** from your gold, never hardcoded.

## Method structure

```
training question–SQL pairs (current fold only)
  ├─ SQL AST role parsing ──► question–element bipartite graph
  │                            └─ element-associated questions ──► keyword patterns
  └─ direct annotation miner ──► B (TERM / TEMPLATE / PREDICATE / ROLE)
                                 └─ constrained one-hop transfer ──► C
B + C ──► annotation-enriched index, column TERM lookup
new question ──► BM25 + graph votes + pattern match (tables)
             ──► direct match + graph + pattern + training-rec + TERM hit (columns)
```

**Parameter status, by profile.** For the ORIGINAL paper baseline profile
(`paper-dasfaa2027-v1.0` / `transfer_v3_target_cap`, the 25-group matrix
below): all ranking weights, thresholds, and caps were fixed before
evaluation (see `configs/base.yaml` and `terag_release/paper_profile.py`)
and no parameter was tuned. For the FINAL paper profile
(`paper_final_20261120`, the BIRD/grid paths in this README): parameters
WERE selected on held-out development splits during the research
(development-set selection, recorded in
`configs/paper/final_20261120/*`); this repository REPLAYS those frozen
selections and never re-runs the search. Training partitions are still
consumed at fit time to build history features (graph, patterns, B/C
annotations, KaSLA's fold-trained logistic classifiers) — "no parameter
tuning at reproduction time" does not mean "nothing is fit". Four
annotation types are mined: TERM (business expression → column), TEMPLATE
(slot-normalized question templates), PREDICATE (typed WHERE conditions),
ROLE (historical SQL roles). Transfer C is one-hop only, gated by
similarity/typing/context constraints, capped per physical target column
(≤3 acceptances) and per table (≤8 texts).

## Supported methods and protocol matrix

**Grid/private protocol matrix (original paper baseline).** Eleven reported
methods: **BM25, TF-IDF, Hybrid, MetadataMatched, ExampleRRF, CooccurRRF,
Dense, DenseDeepSeek, LinkAlign-SingleDB-Adapted, KaSLA-Adapted,
TE-RAG-Full**; six legacy ablations: **noBC, noC, noAnnIdx, noAnnHit,
noGraph, noPattern**; the final-paper grid profile additionally exposes the
nine independent switch ablations (**noBIndex, noBHit, noB, noTableGraph,
noFieldGraph, noGraph, noPattern, noC**) with per-fold frozen parameters
(`configs/paper/final_20261120/grid_selected_configurations.csv`) through
the final engine `terag.engines.grid.GridEngine`.

Formal protocol matrix (exactly the original paper's 25 saved groups):

| Protocol | Groups |
|---|---|
| Question-Level (`question_v1`) | 14 = 11 methods + noBC + noAnnIdx + noAnnHit |
| Held-Out-Table (`table_v2`) | 11 = BM25/TF-IDF/Hybrid/MetadataMatched/ExampleRRF/CooccurRRF + Full + noBC + noC + noGraph + noPattern |

**BIRD matrix (final paper).** The same ten baselines plus the engine
methods of `configs/paper/final_20261120/bird_final_configs.json`: `Core`
(the DEFAULT-parameter full engine), the Q-selected `qB_rerank_gate_v1`
with its rerank-only and nine fixed-ablation members, the H-selected
`full_href_1_8` with its gate-off and fixed-ablation members, the
independently optimized `optimized_full_Q/H` tracks, adaptive variants, and
candidate rows — 54 question / 50 heldout method groups in total. Dense and
the three API methods ran the question protocol only, exactly as reported.

Dense, DenseDeepSeek, LinkAlign-SingleDB-Adapted and KaSLA-Adapted were run
on Question-Level only in the paper; absent groups are never zero-filled.
The two adapted methods keep the `-Adapted` suffix: they retain upstream
mechanisms with declared replacements (multilingual MiniLM for BGE, DeepSeek
for Qwen, logistic scorers for CodeS) and do **not** reproduce the original
checkpoints. `DenseDeepSeek` is an independent custom baseline, not
LinkAlign.

Full details: **docs/experiment_catalog.md** (every table/figure mapped to
its implementation, config, and entry point).

## Environments

Three notions, kept distinct (details in docs/configuration.md):

| Environment | What it is |
|---|---|
| Paper-run environment | Python 3.11.3, numpy 1.26.4, pandas 2.0.2, networkx 3.1, jieba 0.42.1, scikit-learn 1.3.0, PyYAML 6.0, sqlglot **30.30.0**, matplotlib 3.7.2 — as recorded for the accepted runs. `requirements.txt` pins these (PyYAML resolved to the 6.0.1 bugfix release). |
| Verified public environment | `requirements-verified.txt` — the exact versions this release's full test suite and README walkthrough were executed against (identical to the pins above, plus SciPy 1.17.1 for statistics). |
| Compatibility range | `pyproject.toml` `dependencies` (looser `>=` bounds). Combinations outside the verified set are untested; do not assume them. |

## Installation

Python ≥ 3.10. The supported installation is a **source checkout**
(editable or plain) — the CLI reads `configs/`, `demo/`, and `scripts/`
from the repository root.

Download the source ZIP from this repository page and extract it. For
anonymous review, use the download option on the anonymous repository
page. Open a terminal in the extracted directory containing
`pyproject.toml`, then run:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # paper-run pins (verified)
pip install -e .                         # terag-release CLI + packages
# optional groups:
pip install 'scipy>=1.11'                # paired statistics (quoted!)
pip install -r requirements-optional.txt # dense (torch/transformers) + api (openai)
```

A standard wheel also builds (`pip wheel .`); package discovery includes
the frozen engine subpackages (`terag.engines.*`, verified importable
outside the source tree) plus the pinned third-party sources/licenses, and
supports library use and the input-validation/evaluation commands
anywhere. The configuration-tree commands (`run-experiments`, `run-demo`,
`run-bird`, `run-grid-replay`, `benchmark`) resolve configs, frozen
parameter tables, and `data/bird` from the repository root and therefore
need the checkout; the wheel deliberately ships no datasets.

## Quick start — bundled demo (offline, no network, no API)

The repository ships the demo dataset (`demo/data/`, synthetic
"Aurora Logistics") with its gold already built — start from it:

```sh
# 0. check the inputs against the documented contract
python -m terag_release validate-data \
    --field-csv demo/data/field_schema.csv --table-csv demo/data/table_schema.csv \
    --qa-csv demo/data/qa_data.csv --gold demo/data/full_gold_v2.jsonl \
    --output runs/demo_validate

# 1. run the full offline demo matrix (both protocols, 5 folds, 10 methods)
python -m terag_release run-demo --output runs/demo

# 2. offline evaluation of the saved predictions (statistics families)
python -m terag_release evaluate \
    --gold demo/data/full_gold_v2.jsonl \
    --budget runs/demo/run_meta/budget_feasibility.json \
    --predictions runs/demo/predictions.jsonl \
    --output runs/demo_eval --families both

# 3. paired statistics (Wilcoxon + cluster bootstrap) for the figures
python -m terag_release statistics --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl          # writes runs/demo/statistics/

# 4. figures (auto-detects the statistics step's output)
python -m terag_release plot --run runs/demo --output runs/demo_figs

# 5. analyses + fresh timing (new measurement)
python -m terag_release analyze --analysis saved-prefix --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_prefix
python -m terag_release analyze --analysis heldout --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_heldout
python -m terag_release analyze --analysis isolation --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_iso
python -m terag_release benchmark --dataset demo --variants TERAG-Full BM25 \
    --output runs/demo_bench
```

To REGENERATE the synthetic dataset (deterministic; always into a NEW
directory — the bundled `demo/data` is tracked and never overwritten):

```sh
python -m terag_release generate-demo --output /tmp/demo_regenerated
python -m terag_release build-gold \
    --field-csv /tmp/demo_regenerated/field_schema.csv \
    --table-csv /tmp/demo_regenerated/table_schema.csv \
    --qa-csv /tmp/demo_regenerated/qa_data.csv \
    --output-dir /tmp/demo_regenerated --qid-prefix q
# then point a dataset config at /tmp/demo_regenerated (see below)
```

Demo numbers are synthetic. They demonstrate the machinery; do not report
them as results.

### Run status and provenance

Every execution command requires a **NEW output directory** (existing
directories are refused) and writes `run_meta/run_meta.json`: code hashes,
the fully resolved configuration **with its SHA256** (any value execution
actually consumes changes that hash), per-unit effective parameters, input
checksums, environment versions, planned/executed/skipped units, and
per-unit status/timing. Runs finish as `completed`, `partial` (single-fold
`--fold`), `completed_with_failures`, or `failed`; construction/fit
failures leave one `status=error` record per affected question, and
per-question RETRIEVAL exceptions degrade the unit to
`completed_with_errors` (or `failed` when every question fails) while
keeping the successful predictions, the executed timings, and the
evaluation denominators intact — the run then exits non-zero and no
success summary is written. Legal empty predictions count as successes. `validate-data`, `analyze`, `plot`, and
`statistics` validate their inputs and write a report/manifest; the full
run_meta provenance applies to the execution commands.

## BIRD — complete reproduction of the paper's BIRD experiments (offline)

The repository ships the **complete public BIRD retrieval inputs**
(`data/bird/`, CC BY-SA 4.0): the raw Train/Dev questions and table schemas,
the frozen Dev five-fold population (question 55 / heldout 50 database-fold
cells; 1530 verified evaluation questions of 1534 raw — the 4 exclusions are
recorded in the manifests), the derived gold, per-database schema CSVs, the
Train development partitions (9068 verified of 9428 raw), and the frozen
engine configurations of the final paper. SQLite database archives are not
needed for schema grounding and are not shipped (official links in
`data/bird/SOURCES.md`).

Key identities (percent; see docs/experiment_catalog.md for every table):

| Paper row | protocol / method | Covered@5 | Column F1 | FGCS |
|---|---|---:|---:|---:|
| Q Core | question / `Core` | 93.79 | 25.30 | 75.62 |
| Q Rerank only | question / `qB_rerank_gate_v1-field_filter_gate_off` | 93.79 | 25.97 | 81.24 |
| Q + Gate | question / `qB_rerank_gate_v1` | 93.79 | 27.87 | 77.91 |
| H core | heldout / `Core` | 76.54 | 21.99 | 51.70 |
| H-opt | heldout / `full_href_1_8` | 82.29 | 22.98 | 57.97 |

Recompute them from the shipped reference predictions (no model, no API,
seconds):

```sh
python -m terag_release evaluate-bird-reference --output runs/refcheck
```

Re-execute the methods from code (offline; engines are rebuilt from the
frozen inputs; `PYTHONHASHSEED=0` is enforced automatically):

```sh
# 1. question three-stage decomposition (Core / Rerank / Gate)
python -m terag_release run-bird --experiment bird_q_three_stage --output runs/bird_q3

# 2. H-opt and the fixed H ablations
python -m terag_release run-bird --experiment bird_h_opt --output runs/bird_h

# 3. fixed Q ablations (nine switch rows) and the optimized-track replay
python -m terag_release run-bird --experiment bird_q_fixed_ablation --output runs/bird_qabl
python -m terag_release run-bird --experiment bird_optimized_replay --output runs/bird_opt

# 4. the six offline baselines; the complete offline matrix; Core alone
python -m terag_release run-bird --experiment bird_baselines_offline --output runs/bird_base
python -m terag_release run-bird --experiment bird_full_matrix --output runs/bird_all
python -m terag_release run-bird --methods Core --protocols question --output runs/bird_core

# 5. paired statistics over a run's per-question metrics (db-cluster exact
#    sign-flip + bootstrap, frozen declared families, Bonferroni over the
#    declared K; Delta = Full - Ablation)
python -m terag_release bird-statistics --run runs/bird_q3 --output runs/bird_q3_stats
```

Every execution command writes a NEW output directory containing per-cell
predictions, per-question execution evidence (mechanism
enabled/executed/changed — a declared-but-not-executed `field_rerank`/`hg`
layer is a hard error, never a silent no-op), per-method recomputed metrics
with population/budget/db-scope gates, and a `run_meta/run_meta.json`
provenance block (code hashes, frozen-config SHA256, data identity, planned
and executed units, statuses, timings). Failed questions become error
records that stay in the denominator and make the run exit non-zero.

### Expected frozen failures (33 error records — disclosed, not hidden)

The frozen final evaluation contains **33 error records** (2 questions ×
methods whose candidate-detail list was empty, so the frozen `field_rerank`
layer returned the record without its applied-marker and the unified entry
then hard-errored `field_rerank declared but apply produced no marker`):
`superhero::821` (25 method/protocol groups) and
`debit_card_specializing::1486` (8 groups; the Q-protocol noGraph/
noTableGraph members and their H-protocol crossovers — the H-selected
`full_href_1_8` track itself has none). They exist identically in the
shipped reference predictions and in the research frozen evaluation, and
stay in every denominator as zero-scored error records.

* The machine-readable list is
  `results/reference/bird/EXPECTED_FROZEN_FAILURES.json` (33 entries, by
  protocol/method/qid).
* A `run-bird` command that reproduces any listed method therefore ends
  with `run_status=completed_with_failures` and **exits non-zero** — that
  is the documented frozen behavior, not a migration regression.
* `evaluate-bird-reference` reconciles the live error records against the
  list: the 33 expected failures are disclosed as such; any OTHER error
  record (and any disappearance of an expected one) is reported as a
  problem. Errors are never ignored wholesale, deleted, rewritten to
  success, or excluded from the denominators.

### Optional Dense and API baselines (network, your cost)

```sh
# Dense: downloads the pinned MiniLM revision on first use
python -m terag_release run-bird --experiment bird_dense --allow-network \
    --output runs/bird_dense

# DenseDeepSeek / KaSLA-Adapted / LinkAlign-SingleDB-Adapted (deepseek-flash,
# thinking disabled): your DEEPSEEK_API_KEY (env-only, never a file), an
# explicit budget cap, and explicit network opt-in
export DEEPSEEK_API_KEY=...
python -m terag_release run-bird --experiment bird_api --allow-network \
    --budget-cap-cny 5 --output runs/bird_api
```

The three API methods keep the frozen research call semantics (one
app-level retry, conservative cross-process budget broker with peak-price
reservations and unknown-bill accounting, durable HTTP response cache).
Online model responses are **not** guaranteed to match the historical
reference predictions bit-for-bit; compare against
`results/reference/bird/` for the paper's values. The historical runs spent
≈ CNY 30 in total on the selector API; that budget is never reused — you
must set your own cap. `LinkAlign-SingleDB-Adapted` and `KaSLA-Adapted`
remain adapted baselines (declared model/scorer replacements), not
reproductions of the original checkpoints; `DenseDeepSeek` is an independent
custom baseline.

## Running the full protocol on your own data (grid/private)

Configuration references accept a **name** (under `configs/<kind>/`) or an
explicit **file path** — e.g. `--dataset mydata` or
`--dataset /path/to/mydata.yaml`.

```sh
# 0. copy the dataset template and fill in paths + domain vocabulary
cp configs/datasets/private.example.yaml configs/datasets/mydata.yaml

# 1. build the full gold from your QA SQL (AST parsing; no execution)
python -m terag_release build-gold --field-csv F.csv --table-csv T.csv \
    --qa-csv Q.csv --output-dir derived/ --qid-prefix q

# 2. build the deterministic fold manifest (question folds + AST-isolated
#    table folds; dup groups never split, unparseable train SQL excluded)
python -m terag_release make-folds --qa-csv Q.csv \
    --gold derived/full_gold_v2.jsonl --output derived/folds

# 3. validate everything (manifests are RE-VALIDATED against your current
#    data; cached checks inside a manifest are never trusted)
python -m terag_release validate-data --field-csv F.csv --table-csv T.csv \
    --qa-csv Q.csv --gold derived/full_gold_v2.jsonl \
    --splits-manifest derived/folds/splits_manifest.json \
    --output /tmp/validate

# 4a. offline matrix (lexical/history baselines + TE-RAG + ablations)
python -m terag_release run-experiments --dataset mydata \
    --experiment full_matrix --offline-only --output runs/full_offline

# 4b. include model/API methods (downloads / billed DeepSeek API)
export DEEPSEEK_API_KEY=...        # never in files
python -m terag_release run-experiments --dataset mydata \
    --experiment full_matrix --allow-network --output runs/full_all

# 5. evaluate / statistics / analyze / plot as in the demo, pointing at runs/full_*
```

To reuse a frozen split manifest, set `data.splits_manifest` in your
dataset config (it is re-validated before anything runs; a configured but
missing manifest is an error, never a silent rebuild).

### Final grid engine replay (profile `paper_final_20261120`)

The commands above run the ORIGINAL 25-group matrix through the legacy
runner. The FINAL paper profile's grid engine
(`terag.engines.grid.GridEngine`, accepted scoring identity
`graph_casefold_ids_v1+pattern_consumer_casefold_v1`) has its own runnable
entry and replays the frozen per-fold parameters of
`configs/paper/final_20261120/grid_selected_configurations.csv` — no
parameter search is executed. On your authorized data (same input format as
above: schema CSVs, QA CSV, full gold, make-folds manifest):

```sh
# 0-3. prepare + validate inputs exactly as above (build-gold, make-folds,
#      validate-data), with a dataset config pointing at your data

# 4. final Full (frozen per-fold parameters)
python -m terag_release run-grid-replay --dataset mydata \
    --splits-manifest derived/folds/splits_manifest.json \
    --track full --output runs/grid_full

# 5. fixed-configuration ablation: the SAME per-fold Full parameters with
#    one module switch off (params identical, only the engine variant)
python -m terag_release run-grid-replay --dataset mydata \
    --splits-manifest derived/folds/splits_manifest.json \
    --track fixed --member noBIndex --output runs/grid_fixed_noBIndex
#    (member: noBIndex noBHit noB noTableGraph noFieldGraph noGraph
#     noPattern noC)

# 6. independently optimized configuration replay: the variant's OWN frozen
#    parameters for each fold (direct replay of the recorded selection)
python -m terag_release run-grid-replay --dataset mydata \
    --splits-manifest derived/folds/splits_manifest.json \
    --track optimized --member noBIndex --output runs/grid_opt_noBIndex
```

Each run writes per-cell `predictions.jsonl` (the common record shape:
qid/setting/fold/method/variant/status/retrieved_tables/
retrieved_column_keys/retrieved_columns_detail), per-cell
`per_question_metrics.json` recomputed from the final records against your
gold, `cell.json` with the effective parameters and their source (the
frozen table's `chosen` config id), and `run_meta/run_meta.json` provenance
(parameter-table SHA256, manifest SHA256, input checksums, execution plan,
statuses, timings). Before ANY engine is constructed, the raw manifest is
fully re-accepted on your current QA/gold/qids (`validate-data` semantics:
fold completeness/exclusivity, duplicate groups, train/test disjointness,
held-out target membership, AST isolation recomputed from your gold — the
manifest's cached checks are never trusted); tampered manifests are hard
errors and leave no run artifacts. Run statuses: `completed` (the full
two-setting five-fold plan, clean), `partial` (a legitimately narrowed
clean run such as `--fold N` — real outputs kept, explicitly labeled),
`completed_with_failures[_partial]` / `failed` (non-zero exit), and
`incomplete` (planned units missing — never reported as success). Invalid
folds (e.g. 99), unknown settings, and empty execution plans are rejected
up front instead of writing an empty success. `--settings question_v1` / `--settings table_v2` and `--fold N`
narrow the replay. `--qid-prefix` sets the positional qid prefix when the
QA csv has no qid column (documented default `q`).

Verification attribution (kept precise): the UNDERLYING GridEngine was
verified against the authorized research inputs in an earlier round —
10/10 folds, 1584 question-level comparisons, retrieved tables and columns
EXACTLY identical (see docs/release_mapping.md). This CLI entry itself was
verified end-to-end on the bundled SYNTHETIC fixture (construction, fit,
predict, per-cell evaluation, all three parameter-selection tracks, and
manifest/execution-plan guards); no full grid experiment was re-executed
for the entry's own wiring. The enterprise inputs themselves are not
redistributed, so grid numbers on public data do not exist; the legacy
25-group commands above remain the historical profile and are never
presented as the final engine.

### Profiles: paper vs custom — and what is a read-only mirror

Every configuration field falls into exactly one of three categories:

1. **Supported and wired** — the baseline retriever parameters (k1, b,
   top_k, tokenizer language) and the `terag` sections (weights,
   thresholds, annotation/ablation toggles). Under `--profile custom`
   these genuinely reach the constructors and change the configuration
   SHA256 (tests/test_config_binding.py); under the default `paper`
   profile a deviation from the paper constants is a startup error naming
   the parameter.
2. **Read-only mirrors, validated under EVERY profile** —
   `transfer_defaults` (checked against the actual `TransferConfig`
   defaults), method `model:`/`api:` sections (checked key-by-key against
   the constants in the retriever modules, typos included), protocol
   `folds: 5` / `supervision: full_sql`, and the retrieval `budget`
   (5 tables / 10 columns). The runner does not parameterize these, so a
   changed YAML value would be silently ignored — it is rejected instead,
   including under `profile=custom`.
3. **Rejected** — unknown keys, typos, and unsupported values anywhere.

There is deliberately no fourth category ("accepted, snapshotted,
ignored").

## Configuration

Layered YAML under `configs/`: `base.yaml` (fixed algorithm parameters),
`datasets/*.yaml` (paths, tokenizer language, domain keyword lists),
`protocols/*.yaml`, `methods/*.yaml` (one per method/ablation — consumed
by the runner, not documentation-only), `experiments/*.yaml`. Resolution
order: paper constants ← base ← dataset ← protocol ← method ← experiment ←
CLI; unknown keys are hard errors; credential-shaped values are rejected
at load time; API keys come from `DEEPSEEK_API_KEY` only (no file
fallback). Model/API settings (temperature 0.1, one transient retry, no
fallback, `deepseek-chat` requests) are documented in
`configs/methods/*.yaml` and validated against the code constants in
`terag_release/paper_profile.py`. Full reference:
docs/configuration.md.

## Outputs

A run directory contains:

```
runs/demo/
├── predictions.jsonl          # one record per (question, method, protocol):
│                              #   qid, setting, fold, variant, method, status,
│                              #   retrieved_tables (≤5), retrieved_column_keys,
│                              #   retrieved_columns_detail, query_time, metrics;
│                              #   construct/fit failures leave status=error
│                              #   records with error_stage (denominators intact)
├── summary_full_gold.csv      # complete runs only: tr@1/3/5, covered@5,
│                              # col P/R/F1, FGCS, feasible-subset FGCS, counts
├── run_meta/
│   ├── run_meta.json          # provenance: resolved config + SHA256, per-unit
│   │                          # effective params, input checksums, environment,
│   │                          # planned/executed/skipped units, run_status
│   ├── splits_manifest.json   # deterministic folds + AST exclusions + checks
│   └── budget_feasibility.json# gold-derived feasible population
├── statistics/statistics.json # (after the statistics step) paired diffs,
│                              # CIs, families, input fingerprints
├── diagnostics/               # (with --diagnostics) B/C evidence per fold
└── artifacts/, external_cache/# per-unit intermediates; API-response cache
                               # for the adapted methods lives in-run
```

## Evaluation and statistics

* Metrics: Covered@5, Table Recall@K, macro column P/R/F1 over normalized
  `table.column` keys, and FGCS (full-gold coverage success) — gold tables
  covered AND every gold column returned, with the same normalization as
  column P/R/F1. Errors and legal empty predictions stay in the denominator.
* `terag_release evaluate` recomputes everything from saved predictions
  (later files replace earlier ones per COMPLETE method/protocol group).
* `terag_release statistics` computes variant−Full Covered@5 differences
  with question-paired Wilcoxon (two-sided, Wilcox zero-method) and
  target-table cluster bootstrap (10,000 resamples, seed 42, percentile
  2.5–97.5, unadjusted), direction-consistent with the figures. The fixed
  k=6/k=2 families are QUESTION-LEVEL comparisons shared with
  `evaluate`: a family-corrected p is emitted only when the whole family
  is present (otherwise the entry records `not_computed` with its missing
  members — raw paired values are still reported for plotting). Held-out
  comparisons are labeled `exploratory_heldout`, never borrowed into the
  Question-Level families. A zero difference means identical per-question
  Covered@5 values — not identical full outputs. Duplicate prediction
  records are rejected before any statistics run.
* `plot` verifies the statistics file's CONTENT identity against the
  current run (predictions and gold digests, per-comparison n and mean
  consistency with the summary) before drawing intervals — another run's
  statistics are refused, while a legitimate copy of the same materials
  passes. Whiskers come only from that file; hollow markers mark absent
  intervals (identical per-question Covered@5 / not computable).

## Efficiency

The paper's efficiency table reflects the **original runs** (original code,
original machine). `terag_release benchmark` measures the current build
with the same measurement boundaries (mean fit wall-clock per fold; query =
mean of fold means; offline methods only). Any number it produces is a
**new measurement** and must be labeled as such.

## Tests

```sh
python -m unittest discover -s tests -v
```

227 tests cover: deterministic paper-parity fold construction and manifest
re-validation (leaks/duplication/misordering rejected before any fit),
qid↔question alignment, the 25-group matrix and exact ablation toggles,
KaSLA full-SQL supervision on the real classifiers, metric
normalization/denominators/budgets, statistics (Wilcoxon/bootstrap/
Bonferroni/zero-diff) wired into the drawn error bars, configuration
binding (parameter change → constructor change → hash change; frozen
parameters rejected under the paper profile), run-status semantics
(all-fail → non-zero exit + error records; partial labeled), credential
boundaries (no file fallback, ever), English-comment policy, and
package-content safety — plus the final-paper additions: BIRD input
identity (raw SHA pins, 55/50 cells, 1530/9068 verified populations),
ported-engine behavior (switch semantics, unified-entry hard errors on
declared-but-not-executed mechanisms, budget violations, metric parity on
crafted records, frozen config values), runner wiring (method resolution,
API budget/credential guards, families-sidecar tamper detection), grid
engine identity, frozen profile file identities (paper PDF SHA), and
reference-results integrity (completeness, sanitization, and the paper's
key values including the negative boundaries).

## Limitations and honest boundaries

* The enterprise (power-grid) dataset, its gold, folds, saved predictions,
  prompts/responses, API caches, indexes, graphs, and annotation artifacts
  of the paper runs are private and not redistributed; enterprise reference
  predictions are not published. The grid numbers can therefore be
  *executed* here only on authorized data — the engine, protocols, and
  frozen per-fold parameters are complete.
* BIRD inputs shipped here are the complete public retrieval inputs; the
  BIRD **SQLite databases** are not shipped (not needed for schema
  grounding; official links in data/bird/SOURCES.md).
* The API methods (DenseDeepSeek, KaSLA-Adapted,
  LinkAlign-SingleDB-Adapted) were **not** re-run online for this release:
  their historical reference predictions ship as sanitized per-question
  records (`results/reference/bird/`), and their online entry points are
  contract-tested offline. An online rerun is a new measurement at your own
  cost and is not guaranteed to match the historical responses.
* Historical per-transfer traces and per-query score components of the
  original grid runs were never saved; new diagnostic exports are
  new-computation evidence, not recovered originals.
* `noGraph` disables explicit graph **scores**; the graph is still built
  (pattern mining consumes it). `noPattern` disables the independent keyword
  pattern channel — distinct from TEMPLATE annotations. On BIRD the
  independently optimized Full/noBIndex comparison is **negative**
  (Full−noBIndex = −0.51pp F1, corrected p=0.031); C shows no verifiable
  gain; the H-opt advantage over metadata baselines does not survive
  declared correction. These boundaries are kept visible, not smoothed.
* Historical SQL-accuracy numbers and legacy experimental retrievers (a
  simulated LLM method, TE-RAG v1, RAT-SQL/RESDSQL/Seq2SQL/IRNet
  adaptations) are intentionally not shipped.

## Documentation

* [docs/experiment_catalog.md](docs/experiment_catalog.md) — every paper
  experiment → implementation/config/entry point
* [docs/reproduction.md](docs/reproduction.md) — reproduction guide and
  boundaries
* [docs/data_format.md](docs/data_format.md) — input contract (CSVs, gold,
  splits, predictions; the BIRD data tree)
* [docs/configuration.md](docs/configuration.md) — configuration reference
  (resolution order, profiles, fixed-parameter enforcement)
* [docs/release_mapping.md](docs/release_mapping.md) — paper tag ↔ public
  release mapping (including all v0.1.1 adaptations and the final-paper
  profile)
* [docs/paper_with_appendix.pdf](docs/paper_with_appendix.pdf) — the
  submitted paper with appendix (21 pages)
* [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) — third-party notices;
  BIRD data license: `data/bird/LICENSE-BIRD-DATA.txt` (CC BY-SA 4.0)

## License

TBD — the repository **code** license is pending the owner's decision
(deliberately not set at v0.1.0/v0.1.1). Until a LICENSE file is present, no
reuse rights are granted beyond inspection. Third-party components (LinkAlign
prompts, KaSLA utilities) carry their own MIT licenses (see
THIRD_PARTY_LICENSES.md). The BIRD **data** under `data/bird/` is licensed
CC BY-SA 4.0 from the BIRD benchmark authors — a separate license that
covers the raw and derived data files, independent of the code license.
