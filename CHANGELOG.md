# Changelog

## public-v0.1.1 (2026-11-17)

Fixes from the independent review of v0.1.0; no algorithm change
(`transfer_v3_target_cap` semantics untouched), no experiment scope change.

* **KaSLA supervision restored (R1)** — KaSLA-Adapted now receives `full_sql`
  training supervision exactly as the paper runner did, so its classifiers
  and capacity estimation are labeled by the complete SQL gold instead of
  the original SELECT annotation (LinkAlign remains raw — its fit is
  schema-only, as in the paper).
* **Configuration genuinely binds (R2)** — new paper-profile module +
  resolver: paper constants live in code as the single truth; YAML mirrors
  are validated against them under the default `paper` profile (deviations
  are startup errors), `custom` profile lets changes through for real;
  method/protocol YAML parameters actually reach the constructors; every
  unit records its effective parameters; the resolved-config SHA256 covers
  all consumed values; unknown keys and YAML/code contradictions are hard
  errors; name-or-path references never double-join.
* **Split manifests re-validated (R3)** — external manifests are rechecked
  against the CURRENT QA/gold (coverage, exclusivity, duplicate groups,
  target-table membership, exact training membership, AST isolation,
  conservative unparseable exclusion) before anything runs; cached checks
  are ignored; missing configured manifests are errors; test frames are
  ordered by qid and per-record question correspondence is verified, so
  reordered manifests can no longer mislabel predictions.
* **Fold helpers match the paper (R4)** — duplicate-group token
  normalization restored to the exact tag semantics (no extra lowercasing;
  the large-magnitude CJK numeral is not masked); parity pinned by tests
  against an embedded reference implementation.
* **Statistics and figures connected (R5)** — new `statistics` command
  (Wilcoxon + cluster bootstrap, variant−Full direction with correctly
  flipped CI endpoints, k=6/k=2 families separate, zero differences
  unclothed) and plots that draw whiskers from that artifact, hollow
  markers for absent intervals, and the held-out Covered@5 vs column-recall
  panel; inputs fingerprinted; fully offline.
* **Failure runs are failures (R7)** — preflight; planned/executed/skipped
  units; run statuses `completed`/`partial`/`completed_with_failures`/
  `failed`; per-question error records for construct/fit failures with no
  fabricated timings; coverage validation; non-zero exit codes; run
  metadata always written; no overwrites; benchmark refuses network
  methods.
* **Credential boundary enforced (R8)** — the DenseDeepSeek client's
  repository-file key fallback (shipped in v0.1.0 despite the mapping's
  claim) is removed; environment variables are the only source; tests pin
  that no repo file is read and no client is built without a key.
* **Environments and packaging accurate (R9)** — requirements corrected to
  the paper-recorded pins (notably sqlglot 30.30.0); a verified-environment
  lockfile records what the suite actually ran against; README separates
  paper/verified/compat environments; the wheel now packages the pinned
  third-party sources and licenses and imports correctly outside the source
  tree (config-tree commands still require the checkout — stated honestly).
* **README executable as written (R6)** — quick start uses the bundled
  demo; regeneration always targets a new directory; name-or-path config
  interface documented; quoted pip constraints; run-status/provenance
  claims match actual behavior; every command re-verified from a clean
  copy.

Second recheck round (same unreleased candidate):

* **Held-out target/test binding** — manifests are revalidated against
  the CURRENT QA target annotation: the declared target universe must
  reproduce it exactly (emptied/augmented/reshuffled assignments are
  rejected), each fold's test set must be exactly the questions whose
  original target table it declares, and the same table's questions may
  not scatter across folds. Counterexamples A (all targets emptied) and
  B (fold test sets swapped, train rebuilt) from the recheck are now
  regression tests, along with unknown/missing targets and a fit guard.
* **No accepted-but-ignored configuration** — every field is wired,
  a read-only mirror validated against the actual runtime constant under
  EVERY profile (transfer defaults vs the real `TransferConfig`, model/
  api sections key-by-key vs the module constants, protocol folds/
  supervision, retrieval budget), or rejected. Typos in nested model
  sections are errors. Protocol file paths only locate configs; the
  canonical `name` drives execution. All five recheck counterexamples
  are regression tests; the custom surface (retriever params, terag
  sections) is documented as the only customizable scope.
* **Retrieval errors change run status** — per-question outcomes are
  aggregated per unit (attempted/succeeded/errors/legal-empty); partial
  failures degrade the unit to `completed_with_errors`, total failures
  fail it; successes, executed timings, per-question error records and
  denominators are preserved; run status, CLI exit codes, and benchmark
  statuses propagate (no success summary over failed retrievals).
* **Statistics families and input integrity** — fixed families are
  Question-Level only and shared with `evaluate` (incomplete family ⇒
  `not_computed` + missing members, no corrected p; raw paired values
  still reported); held-out comparisons are exploratory; duplicate
  prediction records are rejected before indexing; population gaps are
  reported, never silently shrunk; `zero_covered_difference` is
  explicitly not a claim of identical outputs (Full/noC boundary).
* **Plot content-identity binding** — before drawing intervals the plot
  verifies the statistics file against the CURRENT run's content
  (predictions/gold digests; per-comparison metric/direction/n/mean
  consistency with the summary); another run's statistics or a doctored
  digest is refused, a legitimate copy of the same materials passes,
  point-only remains available when no statistics exist.
* **Version single-sourcing** — `terag_release.__version__` is the single
  source (pyproject reads it dynamically; `reporting` imports it);
  candidate version aligned at 0.1.1; code fingerprints now cover
  paper_profile, statistics/plots/analysis modules, and the actual
  model/API implementations.

## public-v0.1.0

Initial public release of the TE-RAG implementation and evaluation
protocols.

* Algorithm identity `transfer_v3_target_cap`, exported from paper tag
  `paper-dasfaa2027-v1.0` (commit `25aa5e80…`) under an explicit file
  whitelist; translated comments verified AST-equivalent to the tag.
* Full method set: BM25, TF-IDF (Vector), Hybrid, MetadataMatched
  (MatchedSchema), ExampleRRF, CooccurRRF, Dense, DenseDeepSeek,
  LinkAlign-SingleDB-Adapted, KaSLA-Adapted, TE-RAG-V2 (Full) + six
  ablations; 25-group protocol matrix (Question-Level ×14,
  Held-Out-Table ×11) with exact toggle semantics.
* New `terag_release` CLI: validate-data, generate-demo, build-gold,
  make-folds, run-demo, run-experiments, evaluate, analyze, plot,
  benchmark — all with run_meta provenance (resolved config + SHA256,
  input checksums, environment, per-unit status/timing), new-output-only
  writes, and credential rejection at config load.
* Deterministic synthetic demo ("Aurora Logistics", 17 tables / 49
  questions; full-scale generator variant) demonstrating schema loading,
  AST gold, B/C annotation generation with provenance (including accepted
  one-hop transfers), both protocols, evaluation, statistics machinery,
  analyses, plots, and fresh timing.
* Offline analyses: saved-prefix column budgets, held-out disagreement
  categories + case selection, transfer evidence (caps and training-fold
  containment verified), isolation reports.
* Layered configuration tree with fixed paper parameters, per-method
  auditable configs, dataset templates, and environment-only API
  credentials.
* Third-party notices: LinkAlign prompts and KaSLA utilities (pinned
  upstream snapshots, MIT).
* Known boundaries (as later corrected by the independent review and the
  v0.1.1 fixes above): KaSLA supervision, config binding, manifest
  re-validation, statistics-driven figures, failure semantics, the
  DenseDeepSeek key fallback, and dependency pins were all deficient in
  v0.1.0 and are addressed in v0.1.1. Enterprise data/predictions are not
  included; the paper efficiency table is not reproducible here
  (benchmark produces new measurements); model/API methods were not
  exercised against real services in either release's verification.
