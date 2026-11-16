# Configuration Reference

## Layout and resolution

```
configs/
├── base.yaml                 fixed algorithm parameters (paper values)
├── datasets/demo.yaml        bundled synthetic dataset
├── datasets/private.example.yaml   template for your data
├── protocols/question_level.yaml   question_v1
├── protocols/heldout_table.yaml    table_v2
├── methods/*.yaml            one auditable file per method/ablation (17)
└── experiments/*.yaml        method×protocol bundles (full_matrix, demo_offline, ...)
```

Any file may declare `extends: <relative-path>`; children deep-merge over
parents (method files extend `terag_full.yaml` or `../base.yaml`). The
loader resolves the whole tree, **rejects credential-shaped values**
(`api_key: <non-placeholder>` fails; `${ENV_VAR}` / `<placeholder>` / empty
pass), and every run exports the resolved config plus its SHA256 into
`run_meta/run_meta.json`.

Relative dataset paths resolve against the repository root; absolute paths
pass through.

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
