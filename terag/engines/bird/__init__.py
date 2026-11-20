"""BIRD engine package: the frozen final-paper BIRD retrieval engines and
their execution machinery.

Contents:
- bird_engine / vnext_engine: the per-database TE-RAG scoring engines
  (base and route-M fork) with unified-entry contracts.
- annotation_inputs / annotation_miner_bird / annotation_miner_bird_vnext
  / annotation_transfer_bird / feature_extractor_vnext: the B/C annotation
  mining and transfer machinery used by the engines.
- study_defaults: the frozen variant/parameter vector (VARIANTS, DEFAULT,
  switches, numeric_identity).
- unified_entry: the single authoritative execution path (mechanisms
  applied or hard error).
- field_rerank: the Q column-gate post-processing layer.
- legacy_ablations: legacy noBC/noAnnIdx/noAnnHit parameter-mask wrappers.
- config_factory: make_bird_config, the frozen per-database TERAGConfig.
- baselines: the offline baselines (BM25, MetadataMatched, ExampleRRF,
  TFIDF, Hybrid, CooccurRRF, Dense).
"""
from terag.engines.bird.bird_engine import BirdEngine

__all__ = ['BirdEngine']
