"""Evaluation kit: population-exact metrics, gates and the accepted
statistical core for the BIRD final evaluation.

Modules:
- metrics_core: exact per-question metric recomputation + budget/scope
  gates (METRIC_KEYS inlined).
- population: BirdPopulation over the frozen Dev five-fold inputs
  (data/bird/processed by default).
- predictions_io: prediction record loading/hashing and generic JSONL
  writing helpers.
- evaluate: collect/gate/score/summarize one method's saved predictions.
- stats_core: pure statistical core (verbatim accepted semantics).
- stats_v2: final statistics over the declared families (families file
  and metrics root passed explicitly).
"""
from terag.engines.evalkit import metrics_core, population, predictions_io
from terag.engines.evalkit.population import BirdPopulation
from terag.engines.evalkit.evaluate import (
    GateError, collect_cell_records, evaluate_records, summarize, write_results)
from terag.engines.evalkit.metrics_core import (
    METRIC_KEYS, actual_counts, gate_problems, metric_values)

__all__ = [
    'metrics_core', 'population', 'predictions_io', 'BirdPopulation',
    'GateError', 'collect_cell_records', 'evaluate_records', 'summarize',
    'write_results', 'METRIC_KEYS', 'actual_counts', 'gate_problems',
    'metric_values',
]
