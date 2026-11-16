# Changelog

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
* Known boundaries: enterprise data/predictions not included; paper
  efficiency table not reproducible here (benchmark produces new
  measurements); model/API methods not exercised in this release's
  verification.
