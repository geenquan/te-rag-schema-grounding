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
| Paper baseline | tag `paper-dasfaa2027-v1.0`, commit `25aa5e807d5cd6f8d5a3911e253ef89424d743e4` |
| Algorithm identity | `transfer_v3_target_cap` (unchanged; see docs/release_mapping.md) |
| Relationship | Same algorithm code, exported under an explicit file whitelist with English comments, parameterized paths, a unified CLI, layered configs, a synthetic demo, and offline analyses. Private data, predictions, prompts, and caches are **not** included. |

What this repository supports — stated precisely:

* **Demo execution** — the bundled synthetic dataset runs the full pipeline
  offline. It verifies that the public code path works. **It does not
  reproduce any paper number.**
* **Experiment reproduction** — with your own authorized data in the
  documented format, every reported method, protocol, and analysis can be
  executed end to end.
* **Numerical reproduction** — reproducing the paper's *values* additionally
  requires the original authorized enterprise dataset, its fixed splits, and
  comparable run conditions. Those inputs are not redistributed here; the
  denominators (e.g. 88 runs / 76 valid / 65 budget-feasible in the paper)
  are properties of that dataset and are always **recomputed** from your
  gold, never hardcoded.

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

All ranking weights, thresholds, and caps are **fixed** (see
`configs/base.yaml` and `terag_release/paper_profile.py`); nothing is
trained. Four annotation types are mined: TERM (business expression →
column), TEMPLATE (slot-normalized question templates), PREDICATE (typed
WHERE conditions), ROLE (historical SQL roles). Transfer C is one-hop only,
gated by similarity/typing/context constraints, capped per physical target
column (≤3 acceptances) and per table (≤8 texts).

## Supported methods and protocol matrix

Eleven reported methods: **BM25, TF-IDF, Hybrid, MetadataMatched,
ExampleRRF, CooccurRRF, Dense, DenseDeepSeek, LinkAlign-SingleDB-Adapted,
KaSLA-Adapted, TE-RAG-Full**; six ablations: **noBC, noC, noAnnIdx, noAnnHit,
noGraph, noPattern**.

Formal protocol matrix (exactly the paper's 25 saved groups):

| Protocol | Groups |
|---|---|
| Question-Level (`question_v1`) | 14 = 11 methods + noBC + noAnnIdx + noAnnHit |
| Held-Out-Table (`table_v2`) | 11 = BM25/TF-IDF/Hybrid/MetadataMatched/ExampleRRF/CooccurRRF + Full + noBC + noC + noGraph + noPattern |

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

```sh
git clone https://github.com/geenquan/te-rag-schema-grounding.git
cd te-rag-schema-grounding
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # paper-run pins (verified)
pip install -e .                         # terag-release CLI + packages
# optional groups:
pip install 'scipy>=1.11'                # paired statistics (quoted!)
pip install -r requirements-optional.txt # dense (torch/transformers) + api (openai)
```

A standard wheel also builds (`pip wheel .`); it ships the Python packages
plus the pinned third-party sources/licenses and supports library use and
the input-validation/evaluation commands anywhere. The configuration-tree
commands (`run-experiments`, `run-demo`, `benchmark`) resolve configs from
the repository root and therefore need the checkout.

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

## Running the full protocol on your own data

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

119 tests cover: deterministic paper-parity fold construction and manifest
re-validation (leaks/duplication/misordering rejected before any fit),
qid↔question alignment, the 25-group matrix and exact ablation toggles,
KaSLA full-SQL supervision on the real classifiers, metric
normalization/denominators/budgets, statistics (Wilcoxon/bootstrap/
Bonferroni/zero-diff) wired into the drawn error bars, configuration
binding (parameter change → constructor change → hash change; frozen
parameters rejected under the paper profile), run-status semantics
(all-fail → non-zero exit + error records; partial labeled), credential
boundaries (no file fallback, ever), English-comment policy, and
package-content safety.

## Limitations and honest boundaries

* The enterprise dataset, gold, folds, saved predictions, prompts/responses,
  API caches, indexes, graphs, and annotation artifacts of the paper runs
  are private and not redistributed.
* Historical per-transfer traces and per-query score components of the
  original runs were never saved; new diagnostic exports are new-computation
  evidence, not recovered originals.
* `noGraph` disables explicit graph **scores**; the graph is still built
  (pattern mining consumes it). `noPattern` disables the independent keyword
  pattern channel — distinct from TEMPLATE annotations.
* BIRD results, historical SQL-accuracy numbers, and legacy experimental
  retrievers (a simulated LLM method, TE-RAG v1, RAT-SQL/RESDSQL/Seq2SQL/
  IRNet adaptations) are intentionally not shipped.
* Dense/DeepSeek/adapted methods were not exercised against real models or
  the API in this release's verification (offline verification only);
  their entry points are mock-tested.

## Documentation

* [docs/experiment_catalog.md](docs/experiment_catalog.md) — every paper
  experiment → implementation/config/entry point
* [docs/reproduction.md](docs/reproduction.md) — reproduction guide and
  boundaries
* [docs/data_format.md](docs/data_format.md) — input contract (CSVs, gold,
  splits, predictions)
* [docs/configuration.md](docs/configuration.md) — configuration reference
  (resolution order, profiles, fixed-parameter enforcement)
* [docs/release_mapping.md](docs/release_mapping.md) — paper tag ↔ public
  release mapping (including all v0.1.1 adaptations)
* [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) — third-party notices

## License

TBD — the repository license is pending the owner's decision (deliberately
not set at v0.1.0/v0.1.1). Until a LICENSE file is present, no reuse rights
are granted beyond inspection. Third-party components (LinkAlign prompts,
KaSLA utilities) carry their own MIT licenses (see THIRD_PARTY_LICENSES.md).
