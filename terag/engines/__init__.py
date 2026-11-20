"""Frozen paper engines (public port of the final-paper retrieval stack).

Subpackages:
- ``bird_runtime``: the frozen TE-RAG runtime the BIRD paper runs executed
  against (terag core, retrievers, dataset adapters, retrieval metrics),
  re-rooted from the final research workspace.
- ``bird``: the BIRD per-database engines (BirdEngine/VnextEngine), the
  unified execution entry, field-rerank post-processing, study defaults,
  legacy ablations, the config factory and the offline baselines.
- ``hgate``: the H-search reliability gating layer (post-hoc score
  transforms over engine channel outputs).
- ``evalkit``: population-exact evaluation, metric core, prediction IO and
  the accepted statistical core (db-cluster sign-flip + bootstrap CI).
- ``api``: optional online baselines (API adapters; separate scope).
- ``grid``: the power-grid replay engine (separate scope).
"""
