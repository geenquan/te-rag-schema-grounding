"""Grid replay end-to-end regressions on a SYNTHETIC fixture.

The final grid engine is executed for real (construction, fit, predict,
per-cell evaluation, parameter selection from a frozen table) on the
bundled synthetic demo dataset in the documented private-data format — no
enterprise input is used or copied. The fixture parameter table carries
Full and noBIndex rows so all three tracks (full / fixed / optimized) and
the fixed-uses-Full-params rule are exercised.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from terag_release import grid_runner
from terag_release.config import REPO_ROOT

DEMO = REPO_ROOT / 'demo' / 'data'


def write_params_csv(path):
    """A tiny frozen parameter table with different Full/noBIndex params
    (so the fixed-vs-optimized distinction is observable)."""
    import csv
    full_params = {"table_weights": [0.4, 0.35, 0.25],
                   "field_weights": [0.4, 0.2, 0.15, 0.25],
                   "k1": 1.5, "b": 0.75, "theta_g": 0.15,
                   "table_normalization": "raw",
                   "field_normalization": "raw",
                   "direct_scale": 1.0, "description_scale": 1.0,
                   "field_graph_cap": 5.0, "min_total_sim": 0.5,
                   "source_topk": 3, "modules": None,
                   "b_index_enabled": True, "b_hit_enabled": True,
                   "table_graph_enabled": True, "field_graph_enabled": True,
                   "b_hit_weight": 1.0, "c_hit_weight": 1.0,
                   "b_text_norm": "none", "b_duplicate_penalty": "none",
                   "b_source_weight": "none", "b_match": "exact",
                   "c_match": "exact", "c_admission_subset": 0.5,
                   "c_evidence_mix": [0.3, 0.4, 0.3],
                   "table_graph_alpha": 0.0, "field_graph_alpha": 0.0,
                   "table_graph_norm": "none", "field_graph_norm": "none",
                   "table_graph_clip": 1000000.0, "table_graph_gamma": 3.0,
                   "field_graph_gamma": 1.0, "field_graph_threshold": 0.15,
                   "field_graph_neighbors": 0, "field_edge_power": 1.0,
                   "table_pattern_norm": "none",
                   "field_pattern_norm": "none",
                   "pattern_aggregation": "max", "pattern_smoothing": 0,
                   "b_index": 0.5, "c_index": 0.5,
                   "c_confidence": "constant"}
    opt_params = dict(full_params)
    opt_params.update({"table_weights": [0.55, 0.2, 0.25], "b": 0.9})
    rows = []
    for setting in ('question_v1', 'table_v2'):
        for fold in range(5):
            rows.append({'setting': setting, 'fold': str(fold),
                         'variant': 'Full', 'chosen': 'fx_full_00',
                         'params': json.dumps(full_params),
                         'outer_used': 'False'})
            rows.append({'setting': setting, 'fold': str(fold),
                         'variant': 'noBIndex', 'chosen': 'fx_nob_01',
                         'params': json.dumps(opt_params),
                         'outer_used': 'False'})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['setting', 'fold', 'variant',
                                          'chosen', 'params', 'outer_used'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestGridReplayEndToEnd(unittest.TestCase):
    run_counter = 0

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='grid_replay_test_')
        root = Path(cls.tmp.name)
        # synthetic private-format inputs = the bundled demo dataset
        cls.data = root / 'data'
        cls.data.mkdir()
        for name in ('field_schema.csv', 'table_schema.csv', 'qa_data.csv'):
            shutil.copy(DEMO / name, cls.data / name)
        cls.gold = cls.data / 'full_gold_v2.jsonl'
        shutil.copy(DEMO / 'full_gold_v2.jsonl', cls.gold)
        # frozen splits manifest via the documented make-folds path
        from terag_release import protocols
        from terag_release.runner import load_gold, load_rows
        rows = load_rows(str(cls.data / 'qa_data.csv'), qid_prefix='q')
        gold = load_gold(str(cls.gold))
        q_folds = protocols.build_question_folds(rows)
        t_v1 = protocols.build_table_folds(rows)
        ast_folds, exclusions = protocols.apply_ast_exclusion(
            t_v1, gold, rows['qid'].tolist())
        checks = protocols.verify_splits(rows, q_folds, t_v1)
        manifest = protocols.build_manifest(rows, q_folds, t_v1, ast_folds,
                                            exclusions, checks)
        cls.manifest = root / 'splits_manifest.json'
        cls.manifest.write_text(json.dumps(manifest, ensure_ascii=False),
                                encoding='utf-8')
        # fixture frozen parameter table
        cls.params_csv = root / 'selected_configurations.csv'
        write_params_csv(cls.params_csv)
        # dataset config pointing at the fixture data
        cls.dataset_yaml = root / 'fixture_dataset.yaml'
        cls.dataset_yaml.write_text(
            'name: grid_fixture\n\ndata:\n'
            f'  field_csv: {cls.data / "field_schema.csv"}\n'
            f'  table_csv: {cls.data / "table_schema.csv"}\n'
            f'  qa_csv: {cls.data / "qa_data.csv"}\n'
            f'  gold: {cls.gold}\n', encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, track, member=None, fold=0):
        TestGridReplayEndToEnd.run_counter += 1
        out = (Path(self.tmp.name)
               / f'run{TestGridReplayEndToEnd.run_counter}_'
                 f'{track}_{member or "x"}_{fold}')
        meta = grid_runner.run_grid_replay(
            dataset=str(self.dataset_yaml),
            splits_manifest=str(self.manifest),
            track=track, member=member,
            settings=['question_v1'], fold=fold, output=str(out),
            params_csv=str(self.params_csv), qid_prefix='q')
        return meta, out

    def test_full_track_runs_engine_predicts_and_evaluates(self):
        meta, out = self._run('full')
        self.assertEqual(meta['run_status'], 'completed')
        cell = out / 'question_v1' / 'fold0' / 'Full'
        records = [json.loads(l) for l in
                   (cell / 'predictions.jsonl').read_text().splitlines()
                   if l.strip()]
        self.assertTrue(records)
        for r in records:
            self.assertIn(r['status'], ('ok', 'error'))
            self.assertLessEqual(len(r['retrieved_tables']), 5)
            self.assertEqual(r['method'], 'Full')
        cell_meta = json.loads((cell / 'cell.json').read_text())
        self.assertEqual(cell_meta['params_source'], 'full:fx_full_00')
        self.assertEqual(cell_meta['effective_params']['b'], 0.75)
        per_q = json.loads(
            (cell / 'per_question_metrics.json').read_text())
        self.assertEqual(len(per_q), len(records))
        run_meta = json.loads((out / 'run_meta' / 'run_meta.json')
                              .read_text())
        self.assertEqual(run_meta['params_table']['sha256'],
                         grid_runner.sha256_file(self.params_csv))

    def test_fixed_track_uses_full_params_with_switch_off(self):
        _, out_full = self._run('full')
        _, out_fixed = self._run('fixed', member='noBIndex')
        full_meta = json.loads((out_full / 'question_v1' / 'fold0' / 'Full'
                                / 'cell.json').read_text())
        fixed_meta = json.loads((out_fixed / 'question_v1' / 'fold0'
                                 / 'noBIndex' / 'cell.json').read_text())
        # the fixed ablation must use the IDENTICAL Full parameters
        self.assertEqual(full_meta['effective_params'],
                         fixed_meta['effective_params'])
        self.assertEqual(fixed_meta['params_source'],
                         'fixed:fx_full_00')
        self.assertEqual(fixed_meta['method'], 'noBIndex')

    def test_optimized_track_uses_variant_frozen_params(self):
        _, out_opt = self._run('optimized', member='noBIndex')
        opt_meta = json.loads((out_opt / 'question_v1' / 'fold0' / 'noBIndex'
                               / 'cell.json').read_text())
        # the optimized replay uses the VARIANT's own frozen parameters
        self.assertEqual(opt_meta['params_source'], 'optimized:fx_nob_01')
        self.assertEqual(opt_meta['effective_params']['b'], 0.9)
        self.assertNotEqual(opt_meta['effective_params']['table_weights'],
                            [0.4, 0.35, 0.25])

    def test_missing_member_is_error(self):
        with self.assertRaises(ValueError):
            grid_runner.resolve_track_params('fixed', 'noSuchSwitch',
                                             'question_v1', 0,
                                             grid_runner.load_selected_params(
                                                 self.params_csv))

    def test_output_directory_is_new(self):
        first = self._run('full')[1]
        with self.assertRaises(Exception):
            grid_runner.run_grid_replay(
                dataset=str(self.dataset_yaml),
                splits_manifest=str(self.manifest),
                track='full', settings=['question_v1'], fold=0,
                output=str(first),   # an already-used directory
                params_csv=str(self.params_csv), qid_prefix='q')


if __name__ == '__main__':
    unittest.main()
