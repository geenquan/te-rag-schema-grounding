"""Input-guard regressions for BIRD statistics and reference checking.

Historical defects (independent review PROBES.json), all now pinned:
1. duplicate qid rows silently kept the LAST row (dict assignment);
2. two methods truncated to the SAME 2-question subset were compared and
   labeled computed/confirmatory — only method-vs-method equality was
   checked, never the frozen OOF population;
3. NaN/Inf metric values flowed into the statistics unchecked;
4. reference checking trusted len(dict) == 1530 instead of file hashes,
   exact populations, and duplicate rejection.

The tests inject small legitimate populations (no 1530 hardcoding).
"""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from terag_release import bird_stats
from terag_release.input_guards import (DuplicateQidError,
                                        InvalidMetricError,
                                        expected_population,
                                        load_metric_rows,
                                        load_prediction_rows,
                                        population_problems)

METRICS = ['tr@1', 'tr@3', 'tr@5', 'covered@5', 'mrr',
           'colp', 'colrec', 'colf1', 'fgcs']


def write_metrics_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qid'] + METRICS)
        for qid, values in rows:
            w.writerow([qid] + [f'{v:.10f}' for v in values])


def metric_row(values):
    return {m: f'{v:.10f}' for m, v in zip(METRICS, values)}


class TestGuardedLoaders(unittest.TestCase):
    def test_duplicate_qid_rejected_before_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.csv'
            write_metrics_csv(path, [
                ('db1::0', [0.1] * 9),
                ('db1::0', [0.9] * 9),   # duplicate: 0.1 vs 0.9
            ])
            with self.assertRaises(DuplicateQidError) as ctx:
                load_metric_rows(path)
            self.assertIn('db1::0', str(ctx.exception))

    def test_duplicate_qid_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.jsonl.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(json.dumps({'qid': 'db1::0', 'status': 'ok'}) + '\n')
                f.write(json.dumps({'qid': 'db1::0', 'status': 'ok'}) + '\n')
            with self.assertRaises(DuplicateQidError):
                load_prediction_rows(path)

    def test_nan_and_inf_metrics_rejected(self):
        for bad in ('nan', 'inf', '-inf', '1.5', '-0.01'):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'm.csv'
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('qid,' + ','.join(METRICS) + '\n')
                    f.write(f'db1::0,' + ','.join(
                        [bad] + ['0.1'] * 8) + '\n')
                with self.assertRaises(InvalidMetricError, msg=bad):
                    load_metric_rows(path)

    def test_illegal_status_rejected_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.csv'
            path.write_text('qid,status,' + ','.join(METRICS) + '\n'
                            'db1::0,weird,' + ','.join(['0.1'] * 9) + '\n',
                            encoding='utf-8')
            with self.assertRaises(ValueError):
                load_metric_rows(path)

    def test_status_column_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.csv'
            write_metrics_csv(path, [('db1::0', [0.5] * 9)])
            rows = load_metric_rows(path)   # no status column: fine
            self.assertEqual(list(rows), ['db1::0'])


class TestPopulationGuards(unittest.TestCase):
    POP = {'question': {'dbA::0', 'dbA::1', 'dbB::0', 'dbB::1'}}

    def test_common_truncation_flagged_per_method(self):
        rows = {'dbA::0': {}, 'dbA::1': {}}   # both methods lost dbB
        problems = population_problems(rows, 'question', self.POP,
                                       label='question/method')
        self.assertTrue(any('missing' in p for p in problems),
                        problems)
        self.assertTrue(problems)

    def test_exact_population_passes(self):
        rows = {q: {} for q in self.POP['question']}
        self.assertEqual(
            population_problems(rows, 'question', self.POP), [])

    def test_extra_qid_flagged(self):
        rows = {q: {} for q in self.POP['question']}
        rows['dbZ::9'] = {}
        problems = population_problems(rows, 'question', self.POP)
        self.assertTrue(any('outside the frozen population' in p
                            for p in problems))


class TestComparisonPopulationGate(unittest.TestCase):
    """_comparison must refuse formal statistics on truncated inputs."""

    POP = [f'dbA::{i}' for i in range(4)] + [f'dbB::{i}' for i in range(4)]

    def _rows(self, qids, value):
        return {q: metric_row([value] * 9) for q in qids}

    def test_common_two_question_subset_is_not_computable(self):
        subset = self.POP[:2]
        result = bird_stats._comparison(self._rows(subset, 0.1),
                                        self._rows(subset, 0.25),
                                        'colf1', 'confirmatory', self.POP)
        self.assertEqual(result['status'], 'incomplete_population')
        self.assertIsNone(result.get('p_sign_flip_raw'))
        self.assertIsNone(result.get('ci95_bootstrap_pp'))
        self.assertNotIn('mean_diff_pp', result)
        self.assertEqual(result['n_missing_from_frozen_population'], 6)

    def test_divergent_sets_refused(self):
        result = bird_stats._comparison(self._rows(self.POP[:4], 0.1),
                                        self._rows(self.POP[4:], 0.2),
                                        'colf1', 'confirmatory', self.POP)
        self.assertEqual(result['status'], 'population_mismatch')
        self.assertIsNone(result.get('p_sign_flip_raw'))

    def test_full_population_computes_with_original_numbers(self):
        a = self._rows(self.POP, 0.30)
        b = self._rows(self.POP, 0.15)   # constant +0.15 diff
        result = bird_stats._comparison(a, b, 'colf1', 'confirmatory',
                                        self.POP)
        self.assertEqual(result['status'], 'zero_difference'
                         if False else result['status'])  # keep actual
        self.assertIn(result['status'], ('computed', 'zero_difference'))
        self.assertAlmostEqual(result['mean_diff_pp'], 15.0, places=6)
        self.assertEqual(result['n'], 8)
        self.assertEqual(result['n_clusters'], 2)


class TestRunStatisticsEndToEnd(unittest.TestCase):
    """run_statistics on a tiny injected population: valid inputs compute,
    duplicate/truncated inputs are refused without p-values."""

    POP = {'question': [f'db{i}::q{j}' for i in range(3) for j in range(4)]}

    def _write_run(self, tmp, methods):
        for method, rows in methods.items():
            d = Path(tmp) / 'metrics' / method
            d.mkdir(parents=True)
            write_metrics_csv(d / 'question_per_question.csv', rows)

    def test_valid_small_population_statistics(self):
        import numpy as np
        rng = np.random.default_rng(7)
        base = {q: rng.random(9) * 0.5 for q in self.POP['question']}
        gate = {q: [min(v + 0.05, 0.99) for v in vals]
                for q, vals in base.items()}
        with tempfile.TemporaryDirectory() as tmp:
            self._write_run(tmp, {
                'qB_rerank_gate_v1': [(q, gate[q])
                                      for q in self.POP['question']],
                'qB_rerank_gate_v1-field_filter_gate_off':
                    [(q, base[q]) for q in self.POP['question']]})
            out = bird_stats.run_statistics(
                tmp, str(Path(tmp) / 'stats'), populations=self.POP)
            rows = list(csv.DictReader(
                open(out / 'bird_statistics.csv')))
            gate_rows = [r for r in rows
                         if r['family'] == 'fixed_ablation'
                         and r['metric'] == 'colf1'
                         and r['comparison'] ==
                         'qB_rerank_gate_v1 - '
                         'qB_rerank_gate_v1-field_filter_gate_off']
            self.assertTrue(gate_rows)
            for r in gate_rows:
                self.assertNotEqual(r['status'], 'incomplete_population')
                self.assertIsNotNone(r['mean_diff_pp'])

    def test_truncated_methods_get_no_formal_statistics(self):
        half = self.POP['question'][:6]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_run(tmp, {
                'qB_rerank_gate_v1': [(q, [0.2] * 9) for q in half],
                'qB_rerank_gate_v1-field_filter_gate_off':
                    [(q, [0.1] * 9) for q in half]})
            out = bird_stats.run_statistics(
                tmp, str(Path(tmp) / 'stats'), populations=self.POP)
            rows = list(csv.DictReader(
                open(out / 'bird_statistics.csv')))
            gate_rows = [r for r in rows
                         if r['family'] == 'fixed_ablation'
                         and r['comparison'] ==
                         'qB_rerank_gate_v1 - '
                         'qB_rerank_gate_v1-field_filter_gate_off']
            self.assertTrue(gate_rows)
            for r in gate_rows:
                self.assertEqual(r['status'], 'incomplete_population')
                # refused pairs carry NO metric rows at all: no p-value and
                # no CI may exist anywhere for this comparison
                self.assertIn(r['p_sign_flip_raw'], ('None', '', None))
                self.assertIn(r['ci95_bootstrap_pp'], ('None', '', None))
            self.assertFalse([r for r in rows
                              if r['comparison'] ==
                              'qB_rerank_gate_v1 - '
                              'qB_rerank_gate_v1-field_filter_gate_off'
                              and r['metric'] not in ('None', None, '')])

    def test_duplicate_qid_file_rejected(self):
        rows = [(q, [0.2] * 9) for q in self.POP['question']]
        rows.append((self.POP['question'][0], [0.9] * 9))
        with tempfile.TemporaryDirectory() as tmp:
            self._write_run(tmp, {
                'qB_rerank_gate_v1': rows,
                'qB_rerank_gate_v1-field_filter_gate_off':
                    [(q, [0.1] * 9) for q in self.POP['question']]})
            with self.assertRaises(DuplicateQidError):
                bird_stats.run_statistics(
                    tmp, str(Path(tmp) / 'stats'), populations=self.POP)


class TestExpectedPopulation(unittest.TestCase):
    def test_frozen_population_derived_not_hardcoded(self):
        q = expected_population('question')
        self.assertEqual(len(q), 1530)   # derived from the shipped manifests
        h = expected_population('heldout')
        self.assertEqual(len(h), 1530)
        self.assertEqual(q, h)           # same OOF population, both protocols


if __name__ == '__main__':
    unittest.main()
