# Configuration Reference

## Layout and resolution

```
configs/
├── base.yaml                 fixed algorithm parameters (paper values)
├── datasets/demo.yaml        bundled synthetic dataset
├── datasets/private.example.yaml   template for your data
├── protocols/question_level.yaml   question_v1
├── protocols/heldout_table.yaml    table_v2
├── methods/*.yaml            one file per method/ablation (17) — CONSUMED
│                             by the runner, not documentation-only
└── experiments/*.yaml        method×protocol bundles (full_matrix, demo_offline, ...)
```

Override order (later wins):

```
paper constants (terag_release/paper_profile.py — the code is the truth)
  <- base.yaml
  <- dataset yaml
  <- protocol yaml (per protocol)
  <- method yaml (per method)
  <- experiment yaml
  <- CLI flags
```

Every reference accepts a **name** (resolved under `configs/<kind>/`, with
or without the `.yaml` suffix) or an explicit **file path**; paths are
never re-suffixed or re-joined onto the config directory. Unknown keys,
unknown parameters, and YAML values contradicting the code-defined variant
matrix are hard errors — nothing is silently ignored. The loader
**rejects credential-shaped values** (`api_key: <non-placeholder>` fails;
`${ENV_VAR}` / `<placeholder>` / empty pass), and every execution command
exports the fully resolved configuration plus its canonical SHA256 into
`run_meta/run_meta.json`. Any value execution actually consumes changing
changes that hash; per-unit `effective_params` record exactly what each
constructor received.

Relative dataset paths resolve against the repository root; absolute paths
pass through.

### Profiles

* `paper` (default): every parameter frozen by the paper (BM25 constants,
  ranking weights, feature internals, transfer gates/caps, protocol shape,
  budget) is validated against `paper_profile` after merging — a changed
  value is a startup error naming the parameter, the paper value, and the
  configured value. Method `model:` sections are validated against the
  constants imported from the retriever modules (tests pin the equality).
* `custom` (`--profile custom`, or `profile: custom` in an experiment or
  dataset config): deviations of the SUPPORTED surface (retriever
  parameters k1/b/top_k/tokenizer; terag sections — weights, thresholds,
  annotation/ablation toggles) are allowed, genuinely reach the
  retrievers, and are stamped into run_meta.

### Read-only mirrors (validated under EVERY profile)

These fields restate constants fixed in code; execution would silently
ignore a contradicting YAML value, so any deviation — or an unknown/
misspelled key inside them — is a hard startup error even under
`profile=custom`:

| Field | Validated against |
|---|---|
| `base.yaml: transfer_defaults` | the actual `TransferConfig` defaults in `retrievers/annotation_transfer.py` (imported at validation time) |
| method `model:` sections (incl. nested `classifier`) | `paper_profile.PAPER_MODEL_PARAMS`, itself pinned to the module constants by tests |
| method `api:` sections | fixed environment-variable boundary (`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / default base URL) |
| protocol `folds` / `supervision` | structural constants (5 folds, `full_sql`); the runner is not parameterized for anything else |
| `budget` (base and per-protocol) | fixed retrieval budget (5 tables / 10 columns per table) |

Protocol references accept a name or a file path, but the file only
LOCATES the config — the canonical `name` inside it (`question_v1` /
`table_v2`) is the execution identity, and a config without a recognized
name is rejected (no more empty-plan surprises from absolute paths).

## Fixed algorithm parameters (configs/base.yaml — do not change when
reproducing the paper protocol)

| Group | Parameters |
|---|---|
| BM25 | `k1=1.5`, `b=0.75` |
| Index document weights | table name 2.0 · table desc 1.5 · field name 1.0 · field desc 0.8 · annotation text 0.5; `include_field_desc=true` |
| Graph role weights | SELECT 1.2 · WHERE 2.0 · JOIN 1.8 · GROUP BY 1.5 · ORDER BY 1.3 · HAVING 1.4 · FROM 1.0 (max per edge, never summed); `parse_mode=ast` |
| Table ranking | BM25 0.35 · graph 0.35 · pattern 0.30 |
| Column ranking | direct 0.40 · graph 0.20 · pattern (`role_prior`) 0.15 · training-rec 0.25 · `annotation_hit` 1.0 |
| Feature internals | graph threshold 0.15, propagation weight 3.0; train-sim threshold 0.30, multiplier 1.5, cap 3.0; direct multiplier 1.5, substring bonus 0.4; field graph cap 5.0; pattern scale 0.6 |
| Pattern mining | `similarity_threshold=0.3`, `min_pattern_count=1`; mode `keywords` needs domain keyword lists (dataset config) |
| Seed | 42 (statistics RNG; fold construction is deterministic and uses no RNG) |

## Transfer admission (fixed; defaults of `TransferConfig` in
`retrievers/annotation_transfer.py`, documented in base.yaml)

`min_total_sim=0.5` · `schema_only_min=0.5` (separate gate when template
evidence is missing) · `context_min=0.3` (TERM table-context gate) ·
`max_sources_per_target=3` (κ_s) · `max_annotations_per_target=3` (κ_a,
cumulative across duplicate schema rows) · `max_texts_per_table=8` (κ_t).
One-hop only; C never becomes a source; direct (B) evidence wins conflicts.

## Method toggles (configs/methods/*.yaml ↔ terag_release/variants.py)

| Variant | Toggle |
|---|---|
| TERAG-Full | `use_history=true`, `use_transfer=true`, `use_annotation_index=true` |
| noBC | `use_history=false` |
| noC | `use_transfer=false` |
| noAnnIdx | `use_annotation_index=false` |
| noAnnHit | `feature.field_weights.annotation_hit=0.0` |
| noGraph | `ablation.use_graph_weight=false` (graph still built) |
| noPattern | `ablation.use_template_mining=false` |

## Dataset configuration

`data.field_csv/table_csv/qa_csv/gold` (paths), optional
`data.splits_manifest` (frozen folds), `terag.tokenizer.default`
(`chinese`=jieba / `english`=regex), `terag.pattern_mining.entity_keywords /
attribute_keywords / aggregation_keywords / time_patterns` — **your domain
vocabulary is algorithm input data**, written in the language of your
questions. The paper's Chinese power-grid keyword lists belong to the
private domain and are not redistributed; the demo config shows a synthetic
example.

## Model / API settings

| Method | Setting |
|---|---|
| Dense | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, mean pooling + L2 norm, 256-token truncation, first 30 column names in table docs |
| DenseDeepSeek | Dense top-20 tables × top-10 displayed columns + `deepseek-chat` selection, temperature 0.1, max_tokens 2000, timeout 60 s, ≤1 transient retry, no fallback |
| LinkAlign-SingleDB-Adapted | upstream pipeline prompts; 24-column chunks, 256-token embeddings, 20+20 retrieval chunks; temperature 0.1; max_tokens 32768 (isolation) / 8192 (other) |
| KaSLA-Adapted | upstream relevance/weight/knapsack; LogisticRegression(C=1.0, max_iter=300, seed 42); 20 shown tables, 20 columns/table; capacity from 30 most similar training questions |

API access: `DEEPSEEK_API_KEY` (required) and optional `DEEPSEEK_BASE_URL`
(default `https://api.deepseek.com`) from the **environment only**. Requests
name `deepseek-chat`; saved responses may name a server-side alias — the
code records what the server returned (`not_provided` when absent) and
never invents response metadata. Requests were recorded as
`deepseek-chat`/served by `deepseek-flash` in the paper's private runs;
future alias behavior is not assumed.

If a model or key is missing, the method errors clearly and is recorded as
`failed` — it is **never** degraded to random/BM25/simulated output under
the original method name.

## Outputs and safety

`runtime.forbid_overwrite=true`: every command requires a NEW output
directory. `run_meta` snapshots: code file hashes, resolved config + SHA256,
input checksums, dataset/gold populations, split-manifest digest,
environment versions, per-unit status/fit time/query means, and the timing
boundary note.

---

## BIRD and final-paper configuration (profile `paper_final_20261120`)

The BIRD commands bypass the legacy YAML layer deliberately: their frozen
parameters are **files, not YAML mirrors**, and are SHA-verified at every
run:

* `configs/paper/final_20261120/bird_final_configs.json` — every engine
  method's `engine` kind (`bird`/`vnext`), predict switch (`engine_variant`)
  and full parameter vector, including the `field_rerank` and `hg`
  mechanism namespaces. The runner loads it, refuses duplicate method
  names, and stamps `params_sha256` per method into run_meta.
* `configs/paper/final_20261120/bird_families.json` (+ `.sha256` sidecar) —
  the declared statistical families; the sidecar must match or the
  statistics command refuses (edited-after-declaration detection).
* `configs/paper/final_20261120/grid_selected_configurations.csv` — the
  grid per-fold Q/H frozen parameters replayed by the grid engine.
* `configs/datasets/bird.yaml` — documents the data tree and populations
  (the BIRD commands resolve data through `terag_release.bird_data`, which
  pins raw-file SHA256 identities).
* `configs/experiments/bird_*.yaml` — method selections for `run-bird`
  (`methods` + `protocols`; unknown names are hard errors; engine methods
  run through the unified entry, baselines through the frozen adapters).
* `configs/paper/final_20261120/grid_selected_configurations.csv` — the
  frozen per-(setting, fold, variant) grid parameters consumed by
  `run-grid-replay` (tracks full / fixed / optimized; read-only, SHA
  recorded in run_meta).

API method boundaries (enforced in code, mirrored here): credentials from
`DEEPSEEK_API_KEY`/`DEEPSEEK_BASE_URL` environment variables only (no file
fallback exists); the model is `deepseek-flash` with thinking disabled; the
budget cap is **always user-set** for the current run
(`--budget-cap-cny` / `TERAG_BIRD_BUDGET_CNY`) and must be a **finite
positive number** — NaN, ±inf, zero, negative, and boolean values are
rejected at every entry point (CLI, environment variable, `configure()`,
the runtime property, `require_budget_cap()`, and direct `BudgetBroker`
construction) before any network request or reservation; no historical cap
is reused; evidence, ledger and HTTP cache live under the current run
directory; the MiniLM encoder is pinned to revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42` with an optional
`--minilm-cache-dir` and downloads only under `--allow-network`.
