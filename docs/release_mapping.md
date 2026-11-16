# Release Mapping: Paper Tag ↔ Public Release

This public release maps to the paper baseline **tag
`paper-dasfaa2027-v1.0` (commit
`25aa5e807d5cd6f8d5a3911e253ef89424d743e4)** of the private research
repository. The algorithm identity is unchanged: **`transfer_v3_target_cap`**.
The paper's PDF SHA256 is
`c7b246a1cebe5b913dcbd096865d6936794af2474d06056f5627fc7747da0721`.

## File-level mapping

### Exported unchanged (byte-identical to the tag)

| Public path | Notes |
|---|---|
| `retrievers/upstream/LinkAlign_LICENSE`, `KaSLA_LICENSE`, `kasla_utils.py`, `linkalign_prompts.py` | pinned third-party snapshot + MIT licenses; byte-identical |
| `retrievers/linkalign_official_retriever.py`, `retrievers/kasla_retriever.py` | already English; byte-identical |
| `eval/full_gold.py` | already English; byte-identical |
| `experiment/audit_saved_predictions.py` | one documented adaptation (below) |
| `terag/diagnostics.py` | already English; byte-identical |

### Exported with comments/docstrings/user-facing strings translated
(structure verified AST-identical to the tag; only strings that contained
Chinese were rewritten)

`terag/`: `tokenization.py`, `config.py`, `graph_builder.py`,
`sql_role_parser.py`, `pattern_miner.py`, `index_builder.py`,
`feature_extractor.py`, `ranker.py`, `code_mapper.py`,
`terag_retriever_v2.py` · `retrievers/`: `base_retriever.py`,
`retriever_factory.py`, `bm25_retriever.py`, `vector_retriever.py`,
`hybrid_retriever.py`, `matched_schema_retriever.py`,
`example_rag_retriever.py`, `cooccurrence_retriever.py`,
`dense_retriever.py`, `linkalign_retriever.py`, `annotation_miner.py`,
`annotation_transfer.py`, `sql_context.py` · `eval/retrieval_metrics.py` ·
`datasets/`: `base_adapter.py`, `gt_utils.py` · `scripts/`:
`build_full_gold.py`, `verify_isolation.py` (then adapted, see below)

Chinese text that is **algorithm input data** (keyword dictionaries, date
regexes, CJK class ranges, demo queries) was deliberately kept — translating
it would change runtime behavior.

### Adapted (documented behavior-neutral or interface-level changes)

| File | Change | Behavior impact |
|---|---|---|
| `retrievers/retriever_factory.py` | registry pruned to the 11 paper methods (legacy simulated-LLM/TE-RAG-v1/Graph/RESDSQL/RAT-SQL/Seq2SQL/IRNet registrations removed); stage-tag strings cleaned; `TERAGV2Adapter` requires an explicit `TERAGConfig` (no implicit `config.yaml` discovery, no silent default); adapter `fit()` forwards the opt-in `diagnostics` export | registry contains exactly the shipped methods; misconfiguration now errors clearly at construction; diagnostics unchanged in outputs |
| `terag/terag_retriever_v2.py` | unused `weight_learner` import removed (module not shipped, never used); private-path `main()` demo removed (replaced by `terag_release run-demo`) | none (import was dead; main() was private-path demo) |
| `terag/__init__.py`, `retrievers/__init__.py`, `datasets/__init__.py` | exports limited to shipped modules | none for shipped paths |
| `scripts/build_full_gold.py` | private default paths → required CLI args (`--field-csv/--table-csv/--qa-csv/--output-dir/--qid-prefix/--dialect`) | same computation on any dataset |
| `scripts/verify_isolation.py` | same parameterization (`--gold/--qa-csv/--qid-prefix/--out-dir`) | same computation on any dataset |
| `experiment/audit_saved_predictions.py` | an incomplete fixed statistics family is recorded `not_computed` with its missing members instead of raising (the paper's runs always supplied complete families) | no partial family is ever computed or shrunk |
| `retrievers/linkalign_retriever.py` (DenseDeepSeek) | v0.1.0 shipped the tag's original bytes — including its `config.yaml` key fallback, which the v0.1.0 mapping wrongly described as removed (corrected; see the R8 entry in the v0.1.1 section below for the actual removal) | none at v0.1.0; removed in v0.1.1 |

### Not exported (private or out of paper scope)

* `experiment/run_p1b.py`, `run_expansion.py` and the historical experiment
  framework — replaced by `terag_release` (fold construction, supervision,
  variant toggles, and prediction schema are faithful reimplementations of
  the tag's logic; see `terag_release/protocols.py`,
  `supervision.py`, `variants.py`, `runner.py`).
* `experiment/replay_*.py`, `analyze_transfer_replay.py`,
  `diagnose_transfer_similarity.py`, `analyze_review_s3.py` — they operate
  on private prior-run artifacts; their capabilities are provided by the
  fresh runner, `evaluate`, and `analysis/` (saved-prefix, heldout
  failures, transfer evidence — parameterized, no private paths).
* `terag/weight_learner.py`, `role_weight_optimizer.py`, `sql_generator.py`,
  `cli.py`; `retrievers/llm_retriever.py` (simulated), `terag_retriever.py`
  (v1), `graph_retriever.py`, `ratsql/resdsql/seq2sql/irnet_retriever.py`;
  `datasets/bird_adapter.py`, BIRD scripts and experiments; `eval/sql_eval.py`
  (SQL execution accuracy — out of paper scope); historical tests with
  enterprise fixtures.
* All private data, results, splits, caches, credentials, and internal
  reports.

## New public components (no paper-tag counterpart)

`terag_release/` (config, protocols, supervision, variants, runner,
validation, benchmark, reporting, cli), `analysis/` (saved_prefix,
heldout_failures, transfer, plots), `demo/` (generator + committed data),
`configs/` tree, `tests/`, `docs/`, README, CHANGELOG, THIRD_PARTY_LICENSES.

## v0.1.1 adaptations (independent-review fixes R1–R9)

Beyond the v0.1.0 export rules above, the following public-side changes were
made after independent review (private tag sources untouched; the algorithm
identity is unchanged):

| Area | Change | Behavior impact |
|---|---|---|
| KaSLA supervision (R1) | `terag_release/variants.py`: KaSLA-Adapted receives `full_sql` training supervision, matching the paper runner (`run_expansion.py` injects before the reviewer loop); LinkAlign correctly stays raw (its fit is schema-only) | classifier labels and capacity estimation now derive from the complete SQL gold, not the original SELECT annotation |
| Configuration binding (R2) | new `terag_release/paper_profile.py` + rewritten `config.py` resolver: override order paper<-base<-dataset<-protocol<-method<-experiment<-CLI, strict key/parameter validation, name-or-path references, per-unit effective parameters, resolved-config SHA256 covering every consumed value, paper/custom profiles with frozen-parameter enforcement | method/protocol YAMLs now genuinely reach constructors; silent no-op config changes are impossible; deviations require an explicit custom profile |
| Manifest re-validation (R3) | `protocols.validate_manifest` recomputes every invariant (coverage, exclusivity, dup groups, target-table single-fold membership, exact training membership, AST isolation with current gold, conservative unparseable exclusion) before any fit; cached manifest `checks` never trusted; missing configured manifest is an error; test frames ordered BY qid with per-record question verification | stale/edited/leaky manifests fail before execution; question/qid misalignment is impossible regardless of manifest ordering |
| Fold-helper parity (R4) | `_norm_tokens` restored to the exact paper semantics (no extra lowercase; large-magnitude CJK numeral not masked); parity pinned against an embedded reference in tests | duplicate grouping matches the paper protocol on boundary inputs |
| Statistics + figures (R5) | new `analysis/statistics.py` (paired Wilcoxon + cluster bootstrap, direction variant−Full with flipped CI endpoints, families kept separate) and `statistics` CLI; `analysis/plots.py` draws real whiskers from that file, hollow markers for absent intervals, and the held-out Covered@5 vs column-recall panel | the figure pipeline consumes genuine statistical artifacts with input fingerprints; no retrieval re-run |
| Failure semantics (R7) | runner preflight, planned/executed/skipped units, run_status (completed/partial/completed_with_failures/failed), per-question error records for construct/fit failures (no fabricated timings), coverage validation, non-zero exits, no-overwrite enforcement, benchmark refuses network methods | a failing run can no longer exit 0 or silently drop denominators; metadata always lands on disk |
| Credential boundary (R8) | `retrievers/linkalign_retriever.py`: the config.yaml key fallback was REMOVED — v0.1.0 shipped the original tag bytes here and the mapping wrongly claimed it was gone; environment variables are the only source | no repository file is ever read for credentials; empty env fails with an environment-pointing error |
| Environments (R9) | `requirements.txt` corrected to the paper-recorded pins (sqlglot **30.30.0**, pandas 2.0.2, ...); `requirements-verified.txt` records the exact verified environment; wheel packaging includes the pinned third-party sources/licenses and was verified importable outside the source tree | version claims are now accurate; binary distribution carries its licenses |

The runner additionally orders adapted-method API caches under the run
directory (path parameterization) and `run-demo` defaults to the bundled
dataset with regeneration always targeting a new directory (R6).

## Verification performed at release time

* AST-equivalence of every translated file against the tag (structure and
  all non-Chinese string constants identical; Chinese strings/docstrings
  rewritten only).
* 67 public tests green (protocols, matrix/toggles, metrics, statistics,
  transfer caps, config preflight, end-to-end demo + offline re-evaluation
  roundtrip, English-comment policy, package-content safety).
* Clean-environment run of: import + registry, generate-demo, build-gold,
  validate-data, run-demo (both protocols), evaluate (families), all four
  analyses, plot, benchmark.
* Package scan: no credentials, private absolute paths, enterprise
  identifiers, or private-run references.

Not verified here (impossible without private assets): numerical
reproduction of the paper's tables; real model/API methods (Dense encoder
download, DeepSeek API) — recorded as **not executed**.
