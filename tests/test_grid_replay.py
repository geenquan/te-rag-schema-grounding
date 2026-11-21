"""Grid replay end-to-end regressions on a SYNTHETIC fixture.

The final grid engine is executed for real (construction, fit, predict,
per-cell evaluation, parameter selection from a frozen table) on the
bundled synthetic demo dataset in the documented private-data format — no
enterprise input is used or copied. The fixture parameter table carries
Full and noBIndex rows so all three tracks (full / fixed / optimized) and
the fixed-uses-Full-params rule are exercised.

Guard regressions (independent-review probes): the raw manifest is fully
re-accepted BEFORE any engine is constructed (train/test overlap and
held-out-target leaks are rejected pre-fit), and invalid folds / empty
execution plans can never produce a success status — a legal single fold
is explicitly 'partial'.
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


def build_fixture(root):
    """Create a complete synthetic grid-replay fixture under ``root``.

    Returns (dataset_yaml, manifest, params_csv, data_dir, gold_path).
    Both test classes build their OWN fixture (temporary directories are
    cleaned per class, so a shared class attribute would dangle).
    """
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('field_schema.csv', 'table_schema.csv', 'qa_data.csv'):
        shutil.copy(DEMO / name, data / name)
    gold_path = data / 'full_gold_v2.jsonl'
    shutil.copy(DEMO / 'full_gold_v2.jsonl', gold_path)
    # frozen splits manifest via the documented make-folds path
    from terag_release import protocols
    from terag_release.runner import load_gold, load_rows
    rows = load_rows(str(data / 'qa_data.csv'), qid_prefix='q')
    gold = load_gold(str(gold_path))
    q_folds = protocols.build_question_folds(rows)
    t_v1 = protocols.build_table_folds(rows)
    ast_folds, exclusions = protocols.apply_ast_exclusion(
        t_v1, gold, rows['qid'].tolist())
    checks = protocols.verify_splits(rows, q_folds, t_v1)
    manifest_dict = protocols.build_manifest(rows, q_folds, t_v1, ast_folds,
                                             exclusions, checks)
    manifest = root / 'splits_manifest.json'
    manifest.write_text(json.dumps(manifest_dict, ensure_ascii=False),
                        encoding='utf-8')
    params_csv = root / 'selected_configurations.csv'
    write_params_csv(params_csv)
    dataset_yaml = root / 'fixture_dataset.yaml'
    dataset_yaml.write_text(
        'name: grid_fixture\n\ndata:\n'
        f'  field_csv: {data / "field_schema.csv"}\n'
        f'  table_csv: {data / "table_schema.csv"}\n'
        f'  qa_csv: {data / "qa_data.csv"}\n'
        f'  gold: {gold_path}\n', encoding='utf-8')
    return dataset_yaml, manifest, params_csv, data, gold_path


class TestGridReplayEndToEnd(unittest.TestCase):
    run_counter = 0

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='grid_replay_test_')
        (cls.dataset_yaml, cls.manifest, cls.params_csv, cls.data,
         cls.gold) = build_fixture(Path(cls.tmp.name))

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
        # single-fold request: a legitimately narrowed run, explicitly
        # labeled partial — never presented as the full five-fold run
        self.assertEqual(meta['run_status'], 'partial')
        self.assertTrue(meta['is_partial'])
        self.assertEqual(meta['n_planned_units'], 1)
        self.assertEqual(meta['n_executed_units'], 1)
        self.assertEqual(meta['missing_units'], [])
        self.assertEqual(meta['execution_plan'],
                         [{'setting': 'question_v1', 'fold': 0}])
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


class TestGridReplayGuards(unittest.TestCase):
    """Manifest re-acceptance + execution-plan guards (synthetic inputs).

    Every rejection test asserts BOTH the specific error and that the
    GridEngine constructor was never reached (the guard fires before any
    engine construction, not after a failed fit).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='grid_guard_test_')
        cls.root = Path(cls.tmp.name)
        (cls.dataset_yaml, cls.manifest, cls.params_csv, cls.data,
         cls.gold) = build_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- data accessors (fixture loading mirrors the runner's) ----
    @classmethod
    def _loaded_rows(cls):
        import pandas as pd
        rows = pd.read_csv(cls.data / 'qa_data.csv')
        if 'qid' not in rows.columns:
            rows['qid'] = [f'q{i:03d}' for i in range(len(rows))]
        rows['qid'] = rows['qid'].astype(str)
        return rows

    @classmethod
    def _gold(cls):
        gold = {}
        for line in cls.gold.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                gold[str(rec['qid'])] = rec
        return gold

    @classmethod
    def _qids(cls):
        return cls._loaded_rows()['qid'].tolist()

    def _patch_engine(self):
        """Patch GridEngine so tests can assert it was never constructed."""
        import unittest.mock as mock
        return mock.patch('terag.engines.grid.GridEngine')

    # ------------------------------------------------ acceptance of manifest
    def test_valid_manifest_passes_validation(self):
        from terag_release.grid_runner import validate_manifest_full
        manifest = validate_manifest_full(
            self.manifest, self._loaded_rows(), self._gold(),
            self._qids())
        self.assertIn('question_folds', manifest)

    def test_train_test_overlap_rejected_before_engine(self):
        from terag_release import protocols
        from terag_release.grid_runner import validate_manifest_full
        manifest = json.loads(
            Path(self.manifest).read_text(encoding='utf-8'))
        test_qid = manifest['ast_refined_table_folds']['0']['test'][0]
        manifest['ast_refined_table_folds']['0']['train'].append(test_qid)
        tampered = self.root / 'overlap_manifest.json'
        tampered.write_text(json.dumps(manifest, ensure_ascii=False),
                            encoding='utf-8')
        with self._patch_engine() as engine_cls:
            with self.assertRaises(protocols.ManifestValidationError) as ctx:
                validate_manifest_full(tampered, self._loaded_rows(),
                                       self._gold(), self._qids())
            self.assertIn('train/test overlap', str(ctx.exception))
            engine_cls.assert_not_called()

    def test_train_sql_referencing_heldout_target_rejected(self):
        from terag_release import protocols
        from terag_release.grid_runner import validate_manifest_full
        manifest = json.loads(
            Path(self.manifest).read_text(encoding='utf-8'))
        # move a fold-1 TEST question (whose gold references fold-1's
        # held-out target table) into fold-1's TRAIN side: a direct
        # held-out-target training leak
        fd = manifest['ast_refined_table_folds']['1']
        leak_qid = fd['test'][0]
        fd['test'] = [q for q in fd['test'] if q != leak_qid]
        fd['train'].append(leak_qid)
        tampered = self.root / 'leak_manifest.json'
        tampered.write_text(json.dumps(manifest, ensure_ascii=False),
                            encoding='utf-8')
        with self._patch_engine() as engine_cls:
            with self.assertRaises(protocols.ManifestValidationError) as ctx:
                validate_manifest_full(tampered, self._loaded_rows(),
                                       self._gold(), self._qids())
            self.assertIn('references held-out target table',
                          str(ctx.exception))
            engine_cls.assert_not_called()

    def test_missing_question_row_rejected(self):
        from terag_release import protocols
        from terag_release.grid_runner import validate_manifest_full
        manifest = json.loads(
            Path(self.manifest).read_text(encoding='utf-8'))
        manifest['question_folds']['0'] = manifest['question_folds']['0'][:-1]
        tampered = self.root / 'holed_manifest.json'
        tampered.write_text(json.dumps(manifest, ensure_ascii=False),
                            encoding='utf-8')
        with self._patch_engine() as engine_cls:
            with self.assertRaises(protocols.ManifestValidationError):
                validate_manifest_full(tampered, self._loaded_rows(),
                                       self._gold(), self._qids())
            engine_cls.assert_not_called()

    # ------------------------------------------------ execution-plan guards
    def test_invalid_fold_99_rejected_before_any_run(self):
        out = self.root / 'never_fold99'
        with self._patch_engine() as engine_cls:
            with self.assertRaises(ValueError) as ctx:
                grid_runner.run_grid_replay(
                    dataset=str(self.dataset_yaml),
                    splits_manifest=str(self.manifest),
                    track='full', settings=['question_v1'], fold=99,
                    output=str(out), params_csv=str(self.params_csv),
                    qid_prefix='q')
            self.assertIn('invalid fold 99', str(ctx.exception))
            engine_cls.assert_not_called()
        self.assertFalse(out.exists())   # no success artifacts written

    def test_negative_fold_rejected(self):
        with self._patch_engine() as engine_cls:
            with self.assertRaises(ValueError) as ctx:
                grid_runner.run_grid_replay(
                    dataset=str(self.dataset_yaml),
                    splits_manifest=str(self.manifest),
                    track='full', settings=['question_v1'], fold=-1,
                    output=str(self.root / 'never_neg'),
                    params_csv=str(self.params_csv), qid_prefix='q')
            self.assertIn('non-negative', str(ctx.exception))
            engine_cls.assert_not_called()

    def test_empty_execution_plan_rejected(self):
        with self._patch_engine() as engine_cls:
            with self.assertRaises(ValueError) as ctx:
                grid_runner.run_grid_replay(
                    dataset=str(self.dataset_yaml),
                    splits_manifest=str(self.manifest),
                    track='full', settings=[], fold=None,
                    output=str(self.root / 'never_empty'),
                    params_csv=str(self.params_csv), qid_prefix='q')
            self.assertIn('empty', str(ctx.exception).lower())
            engine_cls.assert_not_called()

    def test_unknown_setting_rejected(self):
        with self._patch_engine() as engine_cls:
            with self.assertRaises(ValueError) as ctx:
                grid_runner.run_grid_replay(
                    dataset=str(self.dataset_yaml),
                    splits_manifest=str(self.manifest),
                    track='full', settings=['question_v9'], fold=None,
                    output=str(self.root / 'never_unknown'),
                    params_csv=str(self.params_csv), qid_prefix='q')
            self.assertIn('unknown settings', str(ctx.exception))
            engine_cls.assert_not_called()

    # ------------------------------------------------ status semantics
    def test_full_plan_completes_as_completed(self):
        out = self.root / 'run_full_plan'
        meta = grid_runner.run_grid_replay(
            dataset=str(self.dataset_yaml),
            splits_manifest=str(self.manifest),
            track='full', settings=None, fold=None,   # the FULL plan
            output=str(out), params_csv=str(self.params_csv),
            qid_prefix='q')
        self.assertEqual(meta['run_status'], 'completed')
        self.assertFalse(meta['is_partial'])
        self.assertEqual(meta['n_executed_units'], 10)  # 2 settings x 5
        self.assertEqual(meta['n_planned_units'], 10)
        # real outputs on disk for every planned unit
        for setting in ('question_v1', 'table_v2'):
            for f in range(5):
                self.assertTrue((out / setting / f'fold{f}' / 'Full'
                                 / 'predictions.jsonl').is_file())

    def test_failed_units_exit_nonzero_at_cli(self):
        """A run whose units fail must exit non-zero (never a success)."""
        from terag_release import cli
        import unittest.mock as mock
        fake_meta = {'run_status': 'failed'}
        with mock.patch('terag_release.grid_runner.run_grid_replay',
                        return_value=fake_meta):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(['run-grid-replay', '--dataset', 'x',
                          '--splits-manifest', 'm.json', '--track', 'full',
                          '--output', '/tmp/never'])
            self.assertEqual(ctx.exception.code, 1)

    def test_partial_run_exits_zero_at_cli(self):
        """A clean partial run succeeds (exit 0) and is labeled partial."""
        from terag_release import cli
        import unittest.mock as mock
        import contextlib, io
        fake_meta = {'run_status': 'partial'}
        with mock.patch('terag_release.grid_runner.run_grid_replay',
                        return_value=fake_meta):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(['run-grid-replay', '--dataset', 'x',
                          '--splits-manifest', 'm.json', '--track', 'full',
                          '--output', '/tmp/never2'])
            # no SystemExit raised, and the status is reported as partial
            self.assertIn("run 'partial'", buf.getvalue())
