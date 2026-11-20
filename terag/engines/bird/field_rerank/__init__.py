"""Agent B field-precision layer (direction B, V2).

Separable rerank/filter/supervised layers that consume frozen-engine trace
features (snapshots or live traces) and reshuffle or shrink the per-table
column lists WITHOUT touching table candidates or the 5x10 budget.

Design contract:
- Inputs: per-question table list (fixed, from the engine) with full column
  orderings and exact feature values (direct_match, graph_propagation,
  role_prior, train_recommend, BHit, CHit, name_only) — the same values the
  frozen engine's predict(trace=True) exposes.
- Outputs: a new record with identical qid/tables; per-table column lists
  reranked (<=10) or filtered (<=10, honest counts); metrics recomputed with
  the frozen eval.full_gold_metrics.
- With all layer knobs at identity, the output record is byte-identical to
  the engine control record (parity enforced by tests/test_field_parity.py).

Nothing here imports or modifies engine/*.py internals at runtime; the only
engine code touched is the frozen study_defaults.DEFAULT vector (read) and
the frozen transform() semantics replicated for offline replay of field-axis
parametric changes.
"""
from .replay import SnapshotQuestion, replay_record, transform_scores
from .rerank import rerank_question, RERANK_DEFAULTS
from .field_filter import filter_question, FILTER_DEFAULTS

__all__ = ['SnapshotQuestion', 'replay_record', 'transform_scores',
           'rerank_question', 'RERANK_DEFAULTS',
           'filter_question', 'FILTER_DEFAULTS']
