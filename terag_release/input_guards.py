"""Shared input guards for BIRD statistics and reference-result checking.

Historical defects (independent review PROBES.json):
1. loading per-question rows into a dict silently KEPT THE LAST row when a
   qid appeared twice — duplicate records must be rejected before any
   dictionary is built;
2. a comparison between two methods that both lost the same questions (a
   common truncation) was computed and labeled ``computed``/``confirmatory``
   because only the two methods' qid sets were compared with each other —
   formal Dev statistics must verify the EXACT frozen OOF population per
   protocol, not method-vs-method equality;
3. metric values were never checked for finiteness or legal range.

This module provides guarded loaders (duplicate rejection, finite/range
validation) and population verification against the frozen manifests. The
population is derived from the shipped manifests via
``terag_release.bird_data`` (never hardcoded); callers may inject a small
legitimate population for tests.
"""

import csv
import gzip
import json
import math
from pathlib import Path


class DuplicateQidError(ValueError):
    """The same qid appears more than once in one per-question file."""


class InvalidMetricError(ValueError):
    """A metric value is not finite or outside its legal [0, 1] range."""


class PopulationMismatchError(ValueError):
    """The records do not cover exactly the expected frozen population."""


METRIC_KEYS = ['tr@1', 'tr@3', 'tr@5', 'covered@5', 'mrr',
               'colp', 'colrec', 'colf1', 'fgcs']
LEGAL_STATUSES = ('ok', 'error', 'timeout')


def load_metric_rows(path, metrics=METRIC_KEYS):
    """Read a per-question metrics CSV into {qid: row}.

    Rejects duplicate qids (before any dict assignment — the retained row
    can never be a silent last-write winner) and metric values that are not
    finite numbers in [0, 1] (the CSVs store fractions; percent-scale files
    are rejected rather than rescaled). Status values outside the legal set
    are rejected as unknown semantics.
    """
    rows = {}
    with Path(path).open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        has_status = 'status' in (reader.fieldnames or [])
        for lineno, row in enumerate(reader, 2):
            qid = (row.get('qid') or '').strip()
            if not qid:
                raise ValueError(f'{path}: line {lineno}: empty qid')
            if qid in rows:
                raise DuplicateQidError(
                    f'{path}: duplicate qid {qid!r} (lines '
                    f'{lineno} and earlier) — rejecting the file instead of '
                    'keeping a silent last-write winner')
            if has_status:
                # status is optional (the historical per-question metric
                # exports carry metrics only); when present it must be legal
                status = (row.get('status') or '').strip()
                if status not in LEGAL_STATUSES:
                    raise ValueError(
                        f'{path}: {qid}: unknown status {status!r} (legal: '
                        f'{list(LEGAL_STATUSES)})')
            for metric in metrics:
                raw = row.get(metric)
                if raw is None or raw == '':
                    raise InvalidMetricError(
                        f'{path}: {qid}: metric {metric} missing')
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise InvalidMetricError(
                        f'{path}: {qid}: metric {metric}={raw!r} is not a '
                        'number') from exc
                if not math.isfinite(value):
                    raise InvalidMetricError(
                        f'{path}: {qid}: metric {metric}={raw!r} is not '
                        'finite')
                if not 0.0 <= value <= 1.0:
                    raise InvalidMetricError(
                        f'{path}: {qid}: metric {metric}={raw!r} outside '
                        '[0, 1] (files store fractions; percent-scale '
                        'inputs are rejected, not rescaled)')
            rows[qid] = row
    return rows


def load_prediction_rows(path):
    """Read a (possibly gzipped) JSONL predictions file into {qid: record}.

    Same duplicate-qid rejection as load_metric_rows; records keep their
    full sanitized shape.
    """
    rows = {}
    p = Path(path)
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p, 'rt', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record.get('qid') or '').strip()
            if not qid:
                raise ValueError(f'{path}: line {lineno}: empty qid')
            if qid in rows:
                raise DuplicateQidError(
                    f'{path}: duplicate qid {qid!r} (line {lineno}) — '
                    'rejecting the file instead of keeping a silent '
                    'last-write winner')
            rows[qid] = record
    return rows


def expected_population(protocol, populations=None):
    """The exact frozen OOF qid set of a protocol.

    Derived from the shipped manifests (``terag_release.bird_data``) — the
    union of every applicable cell's test_qids; never a hardcoded constant.
    ``populations`` (tests only) injects {protocol: set/list of qids} for a
    small legitimate population.
    """
    if populations is not None:
        if protocol not in populations:
            raise KeyError(f'no injected population for {protocol!r}')
        return set(populations[protocol])
    from terag_release.bird_data import applicable_cells
    qids = set()
    for cell in applicable_cells(protocol):
        qids.update(cell['test_qids'])
    return qids


def population_problems(rows, protocol, populations=None, label=''):
    """Compare a loaded {qid: row} mapping against the frozen population.

    Returns a list of human-readable problems (empty = exact match). A
    common truncation of BOTH methods is a population problem per method —
    method-vs-method equality alone never establishes completeness.
    """
    expected = expected_population(protocol, populations)
    have = set(rows)
    problems = []
    where = f'{label or protocol}'
    missing = sorted(expected - have)
    extra = sorted(have - expected)
    if missing:
        problems.append(f'{where}: {len(missing)} of {len(expected)} frozen '
                        f'OOF qids missing (e.g. {missing[:5]}) — partial '
                        'input cannot support formal full-population '
                        'statistics')
    if extra:
        problems.append(f'{where}: {len(extra)} qids outside the frozen '
                        f'population (e.g. {extra[:5]})')
    return problems
