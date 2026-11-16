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
| Public release | `public-v0.1.0` |
| Paper baseline | tag `paper-dasfaa2027-v1.0`, commit `25aa5e807d5cd6f8d5a3911e253ef89424d743e4` |
| Algorithm identity | `transfer_v3_target_cap` (unchanged; see docs/release_mapping.md) |
| Relationship | Same algorithm code, exported under an explicit file whitelist with English comments, parameterized paths, a new unified CLI, layered configs, a synthetic demo, and offline analyses. Private data, predictions, prompts, and caches are **not** included. |

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
`configs/base.yaml`); nothing is trained. Four annotation types are mined:
TERM (business expression → column), TEMPLATE (slot-normalized question
templates), PREDICATE (typed WHERE conditions), ROLE (historical SQL roles).
Transfer C is one-hop only, gated by similarity/typing/context constraints,
capped per physical target column (≤3 acceptances) and per table (≤8 texts).

## Supported methods and protocol matrix

Eleven reported methods: **BM25, TF-IDF, Hybrid, MetadataMatched,
ExampleRRF, CooccurRRF, Dense, DenseDeepSeek,
LinkAlign-SingleDB-Adapted, KaSLA-Adapted, TE-RAG-Full**; six ablations:
**noBC, noC, noAnnIdx, noAnnHit, noGraph, noPattern**.

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

## Installation

Python ≥ 3.10.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core pipeline (offline methods)
pip install -e .                         # installs the terag-release CLI
# optional groups:
pip install scipy>=1.11                  # paired statistics
pip install -r requirements-optional.txt # dense (torch/transformers) + api (openai)
```

Core dependencies (pinned to the paper's recorded versions):
pandas 2.0.2, numpy 1.26.4, networkx 3.1, jieba 0.42.1, scikit-learn 1.3.0,
PyYAML, sqlglot 30.3.0, matplotlib 3.7.2.

## Offline demo (no network, no API)

```sh
# 1. generate the deterministic synthetic dataset and its AST gold
python -m terag_release generate-demo
python scripts/build_full_gold.py \
    --field-csv demo/data/field_schema.csv --table-csv demo/data/table_schema.csv \
    --qa-csv demo/data/qa_data.csv --output-dir demo/data --qid-prefix q

# 2. validate the inputs against the documented contract
python -m terag_release validate-data \
    --field-csv demo/data/field_schema.csv --table-csv demo/data/table_schema.csv \
    --qa-csv demo/data/qa_data.csv --gold demo/data/full_gold_v2.jsonl \
    --output /tmp/demo_validate

# 3. run the full offline demo matrix (both protocols, 5 folds, 10 methods)
python -m terag_release run-demo --output runs/demo

# 4. offline evaluation of the saved predictions (with statistics families)
python -m terag_release evaluate \
    --gold demo/data/full_gold_v2.jsonl \
    --budget runs/demo/run_meta/budget_feasibility.json \
    --predictions runs/demo/predictions.jsonl \
    --output runs/demo_eval --families both

# 5. analyses + figures + fresh timing (new measurement)
python -m terag_release analyze --analysis saved-prefix --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_prefix
python -m terag_release analyze --analysis heldout --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_heldout
python -m terag_release analyze --analysis isolation --run runs/demo \
    --gold demo/data/full_gold_v2.jsonl --output runs/demo_iso
python -m terag_release plot --run runs/demo --output runs/demo_figs
python -m terag_release benchmark --dataset demo --variants TERAG-Full BM25 \
    --output runs/demo_bench
```

Every command writes to a **new** output directory (existing directories are
refused) and drops a `run_meta/run_meta.json` provenance snapshot (code
hashes, resolved config + SHA256, input checksums, environment versions,
per-unit status and timing).

Demo numbers are synthetic. They demonstrate the machinery; do not report
them as results.

## Running the full protocol on your own data

```sh
# 0. copy the dataset template and fill in your paths + domain vocabulary
cp configs/datasets/private.example.yaml configs/datasets/mydata.yaml

# 1. build the full gold from your QA SQL (AST parsing; no execution)
python -m terag_release build-gold --field-csv F.csv --table-csv T.csv \
    --qa-csv Q.csv --output-dir derived/ --qid-prefix q

# 2. build the deterministic fold manifest (question folds + AST-isolated
#    table folds)
python -m terag_release make-folds --qa-csv Q.csv --gold derived/full_gold_v2.jsonl \
    --output derived/folds

# 3. validate everything
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

# 5. evaluate / analyze / plot as in the demo, pointing at runs/full_*
```

To reuse an existing frozen split manifest, set
`data.splits_manifest` in your dataset config (the manifest format is what
`make-folds` writes; see docs/data_format.md).

## Configuration

Layered YAML under `configs/`: `base.yaml` (fixed algorithm parameters),
`datasets/*.yaml` (paths, tokenizer language, domain keyword lists),
`protocols/*.yaml`, `methods/*.yaml` (one per method/ablation — auditable),
`experiments/*.yaml`. Files may `extends:`; the resolved config is exported
into every run's `run_meta` with a SHA256. Credential-shaped values are
rejected at load time; API keys come from `DEEPSEEK_API_KEY` only.

Model/API settings (temperature 0.1, one transient retry, no fallback,
`deepseek-chat` requests) are documented in `configs/methods/*.yaml` and
docs/configuration.md.

## Outputs

A run directory contains:

```
runs/demo/
├── predictions.jsonl          # one record per (question, method, protocol):
│                              #   qid, setting, fold, variant, method, status,
│                              #   retrieved_tables (≤5), retrieved_column_keys,
│                              #   retrieved_columns_detail [{table, score,
│                              #   columns [{key, score}]}], query_time, metrics
├── summary_full_gold.csv      # per (setting, variant): tr@1/3/5, covered@5,
│                              # col P/R/F1, FGCS, feasible-subset FGCS, counts
├── run_meta/
│   ├── run_meta.json          # provenance snapshot (see above)
│   ├── splits_manifest.json   # deterministic folds + AST exclusions + checks
│   └── budget_feasibility.json# gold-derived feasible population
├── diagnostics/               # (with --diagnostics) B/C evidence per fold
└── artifacts/                 # per-unit intermediate artifacts
```

## Evaluation, statistics, and what they mean

* Metrics: Covered@5, Table Recall@K, macro column P/R/F1 over normalized
  `table.column` keys, and FGCS (full-gold coverage success) — gold tables
  covered AND every gold column returned, with the same normalization as
  column P/R/F1. Errors and legal empty predictions stay in the denominator.
* Statistics (needs SciPy): question-paired Wilcoxon (two-sided, Wilcox
  zero-method) + target-table cluster bootstrap (10,000 resamples, seed 42,
  percentile 2.5–97.5, unadjusted); Bonferroni applied per **declared
  family** — k=6 {noBC, noAnnIdx, noAnnHit, Dense, DenseDeepSeek,
  ExampleRRF} and k=2 {LinkAlign-…, KaSLA-…} vs Full, separately. A family
  that is incomplete in the supplied predictions is reported
  `not_computed` with its missing members — never silently shrunk.
* Zero-difference comparisons get no fabricated p-value.

## Efficiency

The paper's efficiency table reflects the **original runs** (original code,
original machine). `terag_release benchmark` measures the current build with
the same measurement boundaries (mean fit wall-clock per fold; query = mean
of fold means). Any number it produces is a **new measurement** and must be
labeled as such.

## Tests

```sh
python -m unittest discover -s tests -v
```

67 tests cover: deterministic folds and AST isolation, the 25-group matrix
and exact ablation toggles, metric normalization/denominators/budgets,
statistics (Wilcoxon/bootstrap/Bonferroni/zero-diff), transfer cap semantics
on synthetic annotations, configuration loading + secret refusal + preflight
of every shipped config, the end-to-end demo pipeline incl. offline
re-evaluation roundtrip, English-comment checks, and package-content safety
(no credentials, private paths, or enterprise identifiers).

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

## Documentation

* [docs/experiment_catalog.md](docs/experiment_catalog.md) — every paper
  experiment → implementation/config/entry point
* [docs/reproduction.md](docs/reproduction.md) — reproduction guide and
  boundaries
* [docs/data_format.md](docs/data_format.md) — input contract (CSVs, gold,
  splits, predictions)
* [docs/configuration.md](docs/configuration.md) — configuration reference
* [docs/release_mapping.md](docs/release_mapping.md) — paper tag ↔ public
  release mapping
* [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) — third-party notices

## License

TBD — the repository license is pending the owner's decision. Until a
LICENSE file is present, no reuse rights are granted beyond inspection.
Third-party components (LinkAlign prompts, KaSLA utilities) carry their own
MIT licenses (see THIRD_PARTY_LICENSES.md).
