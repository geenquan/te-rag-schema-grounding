"""Unit tests for the ported frozen BIRD engine package (terag.engines.*).

Covers, without any dependency on the research repository:
1. study_defaults DEFAULT / switches semantics (variant masks and module
   mask validation).
2. unified_entry.run_one hard errors (missing field_rerank marker on a
   declared mechanism; budget violation) using a tiny stub engine.
3. metrics_core.metric_values against hand-computed values on crafted
   records, including the error-record zeroing rule.
4. BirdPopulation cell-count validation against a minimal fixture tree.
5. config_factory.make_bird_config exact frozen values.
6. import of every ported module.

Run from the repository root:
    python3 -m unittest tests.test_engine_port -v
"""
import importlib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PORTED_MODULES = [
    'terag.engines',
    'terag.engines.bird_runtime',
    'terag.engines.bird_runtime.code_mapper',
    'terag.engines.bird_runtime.config',
    'terag.engines.bird_runtime.feature_extractor',
    'terag.engines.bird_runtime.graph_builder',
    'terag.engines.bird_runtime.index_builder',
    'terag.engines.bird_runtime.pattern_miner',
    'terag.engines.bird_runtime.ranker',
    'terag.engines.bird_runtime.sql_role_parser',
    'terag.engines.bird_runtime.terag_retriever_v2',
    'terag.engines.bird_runtime.tokenization',
    'terag.engines.bird_runtime.weight_learner',
    'terag.engines.bird_runtime.eval.retrieval_metrics',
    'terag.engines.bird_runtime.datasets.base_adapter',
    'terag.engines.bird_runtime.datasets.bird_adapter',
    'terag.engines.bird_runtime.datasets.gt_utils',
    'terag.engines.bird_runtime.retrievers.annotation_miner',
    'terag.engines.bird_runtime.retrievers.annotation_transfer',
    'terag.engines.bird_runtime.retrievers.base_retriever',
    'terag.engines.bird_runtime.retrievers.bm25_retriever',
    'terag.engines.bird_runtime.retrievers.cooccurrence_retriever',
    'terag.engines.bird_runtime.retrievers.dense_retriever',
    'terag.engines.bird_runtime.retrievers.example_rag_retriever',
    'terag.engines.bird_runtime.retrievers.hybrid_retriever',
    'terag.engines.bird_runtime.retrievers.kasla_retriever',
    'terag.engines.bird_runtime.retrievers.linkalign_official_retriever',
    'terag.engines.bird_runtime.retrievers.linkalign_retriever',
    'terag.engines.bird_runtime.retrievers.matched_schema_retriever',
    'terag.engines.bird_runtime.retrievers.vector_retriever',
    'terag.engines.bird',
    'terag.engines.bird.annotation_inputs',
    'terag.engines.bird.annotation_miner_bird',
    'terag.engines.bird.annotation_miner_bird_vnext',
    'terag.engines.bird.annotation_transfer_bird',
    'terag.engines.bird.baselines',
    'terag.engines.bird.bird_engine',
    'terag.engines.bird.config_factory',
    'terag.engines.bird.feature_extractor_vnext',
    'terag.engines.bird.field_rerank',
    'terag.engines.bird.field_rerank.apply',
    'terag.engines.bird.field_rerank.field_filter',
    'terag.engines.bird.field_rerank.replay',
    'terag.engines.bird.field_rerank.rerank',
    'terag.engines.bird.field_rerank.supervised',
    'terag.engines.bird.legacy_ablations',
    'terag.engines.bird.study_defaults',
    'terag.engines.bird.unified_entry',
    'terag.engines.bird.vnext_engine',
    'terag.engines.hgate',
    'terag.engines.hgate.gating',
    'terag.engines.evalkit',
    'terag.engines.evalkit.evaluate',
    'terag.engines.evalkit.metrics_core',
    'terag.engines.evalkit.population',
    'terag.engines.evalkit.predictions_io',
    'terag.engines.evalkit.stats_core',
    'terag.engines.evalkit.stats_v2',
]


class TestModuleImports(unittest.TestCase):
    def test_every_ported_module_imports(self):
        for name in PORTED_MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestStudyDefaults(unittest.TestCase):
    def setUp(self):
        from terag.engines.bird import study_defaults
        self.sd = study_defaults

    def test_default_vector_frozen_values(self):
        d = self.sd.DEFAULT
        self.assertEqual(d['table_weights'], [.35, .35, .30])
        self.assertEqual(d['field_weights'], [.40, .20, .15, .25])
        self.assertEqual(d['k1'], 1.5)
        self.assertEqual(d['b'], .75)
        self.assertIsNone(d['modules'])
        self.assertEqual(
            self.sd.VARIANTS,
            ['Full', 'noBIndex', 'noBHit', 'noB', 'noTableGraph',
             'noFieldGraph', 'noGraph', 'noPattern', 'noC'])

    def test_switches_full(self):
        a = self.sd.switches(self.sd.DEFAULT, 'Full')
        for key in ('BIndex', 'BHit', 'TableGraph', 'FieldGraph', 'C',
                    'Pattern'):
            self.assertTrue(a[key], key)
        self.assertTrue(a['B'])
        self.assertTrue(a['Graph'])

    def test_switches_nobindex_kills_bindex_only(self):
        a = self.sd.switches(self.sd.DEFAULT, 'noBIndex')
        self.assertFalse(a['BIndex'])
        self.assertTrue(a['BHit'])     # B channel survives through hits
        self.assertTrue(a['B'])
        self.assertTrue(a['TableGraph'])
        self.assertTrue(a['FieldGraph'])
        self.assertTrue(a['C'])
        self.assertTrue(a['Pattern'])

    def test_switches_nograph_kills_both_graphs(self):
        a = self.sd.switches(self.sd.DEFAULT, 'noGraph')
        self.assertFalse(a['TableGraph'])
        self.assertFalse(a['FieldGraph'])
        self.assertFalse(a['Graph'])
        self.assertTrue(a['BIndex'])
        self.assertTrue(a['BHit'])
        self.assertTrue(a['C'])
        self.assertTrue(a['Pattern'])

    def test_switches_noc_kills_c_only(self):
        a = self.sd.switches(self.sd.DEFAULT, 'noC')
        self.assertFalse(a['C'])
        self.assertTrue(a['BIndex'])
        self.assertTrue(a['BHit'])
        self.assertTrue(a['TableGraph'])
        self.assertTrue(a['Pattern'])

    def test_switches_modules_mask_and_validation(self):
        p = dict(self.sd.DEFAULT)
        p['modules'] = {'C': False, 'Pattern': False}
        a = self.sd.switches(p, 'Full')
        self.assertFalse(a['C'])
        self.assertFalse(a['Pattern'])
        self.assertTrue(a['BIndex'])
        # A masked-off switch cannot be re-enabled by the mask.
        p2 = dict(self.sd.DEFAULT)
        p2['modules'] = {'TableGraph': False}
        self.assertFalse(self.sd.switches(p2, 'Full')['TableGraph'])
        # Unknown module names and non-bool values are hard errors.
        for bad in ({'NotAModule': True}, {'C': 1}, {'C': None}):
            p3 = dict(self.sd.DEFAULT)
            p3['modules'] = bad
            with self.assertRaises(RuntimeError):
                self.sd.switches(p3, 'Full')

    def test_switches_unknown_variant_rejected(self):
        with self.assertRaises(RuntimeError):
            self.sd.switches(self.sd.DEFAULT, 'noSuchVariant')


class _StubEngine:
    """Minimal engine satisfying unified_entry.run_one's contract."""

    def __init__(self, record):
        self._record = record

    def predict(self, qid, params, variant, trace=False):
        rec = dict(self._record)
        rec['qid'] = qid
        return rec, {}, {}


def _ok_record(n_tables=2, with_details=True):
    columns = [{'key': f't{i}.c{j}', 'score': 0.1}
               for j in range(3) for i in range(n_tables)]
    record = {
        'qid': 'q1', 'status': 'ok',
        'retrieved_tables': [f't{i}' for i in range(n_tables)],
        'retrieved_column_keys': [c['key'] for c in columns],
    }
    if with_details:
        record['retrieved_columns_detail'] = [
            {'table': f't{i}', 'table_score': 1.0 - i * 0.1,
             'columns': [{'key': f't{i}.c{j}', 'score': 0.1}
                         for j in range(3)]}
            for i in range(n_tables)]
    return record


GOLD = {'gold_tables': ['t0'], 'gold_columns': ['t0.c0']}


class TestUnifiedEntry(unittest.TestCase):
    def setUp(self):
        from terag.engines.bird import unified_entry
        self.ue = unified_entry

    def test_run_one_plain_record(self):
        engine = _StubEngine(_ok_record())
        rec, met, ev = self.ue.run_one(engine, 'q1', {}, 'Full', GOLD)
        self.assertEqual(rec['retrieved_tables'], ['t0', 't1'])
        self.assertTrue(ev['budget_ok'])
        self.assertFalse(ev['mechanisms']['field_rerank']['enabled'])
        self.assertFalse(ev['mechanisms']['hg']['enabled'])
        self.assertAlmostEqual(met['tr@5'], 1.0)

    def test_run_one_field_rerank_declared_but_marker_missing(self):
        # A record without retrieved_columns_detail: apply_config_to_record
        # returns it unchanged, so the applied marker is absent and run_one
        # must hard-error (declared mechanism produced no execution marker).
        engine = _StubEngine(_ok_record(with_details=False))
        params = {'field_rerank': {'k_cap': 5}}
        with self.assertRaises(AssertionError):
            self.ue.run_one(engine, 'q1', params, 'Full', GOLD)

    def test_run_one_field_rerank_applied(self):
        engine = _StubEngine(_ok_record())
        params = {'field_rerank': {'k_cap': 2}}
        rec, _met, ev = self.ue.run_one(engine, 'q1', params, 'Full', GOLD)
        self.assertIn('field_rerank_applied', rec)
        self.assertTrue(ev['mechanisms']['field_rerank']['executed'])
        self.assertEqual(ev['mechanisms']['field_rerank']['columns_after'],
                         4)  # 2 tables x k_cap 2

    def test_run_one_budget_violation_raises(self):
        engine = _StubEngine(_ok_record(n_tables=6))
        with self.assertRaises(RuntimeError) as ctx:
            self.ue.run_one(engine, 'q1', {}, 'Full', GOLD)
        self.assertIn('budget violated', str(ctx.exception))

    def test_run_one_per_table_column_budget_violation(self):
        record = _ok_record(n_tables=1)
        record['retrieved_columns_detail'][0]['columns'] = [
            {'key': f't0.c{j}', 'score': 0.1} for j in range(11)]
        record['retrieved_column_keys'] = [
            c['key'] for c in record['retrieved_columns_detail'][0]['columns']]
        engine = _StubEngine(record)
        with self.assertRaises(RuntimeError):
            self.ue.run_one(engine, 'q1', {}, 'Full', GOLD)


class TestMetricsCore(unittest.TestCase):
    def setUp(self):
        from terag.engines.evalkit import metrics_core
        self.mc = metrics_core

    def test_record_a_hand_computed(self):
        record = {'status': 'ok',
                  'retrieved_tables': ['t1', 't2'],
                  'retrieved_column_keys': ['t1.a', 't2.b']}
        gold = {'gold_tables': ['t1', 't3'],
                'gold_columns': ['t1.a', 't2.c']}
        m = self.mc.metric_values(record, gold)
        # tr@k = |gold & topK| / |gold|; gold has 2 tables, only t1 retrieved
        self.assertAlmostEqual(m['tr@1'], 0.5)
        self.assertAlmostEqual(m['tr@3'], 0.5)
        self.assertAlmostEqual(m['tr@5'], 0.5)
        self.assertAlmostEqual(m['covered@5'], 0.0)   # t3 never retrieved
        self.assertAlmostEqual(m['mrr'], 1.0)         # t1 at rank 1
        # columns: gold {t1.a, t2.c} vs pred {t1.a, t2.b}
        self.assertAlmostEqual(m['colp'], 0.5)
        self.assertAlmostEqual(m['colrec'], 0.5)
        self.assertAlmostEqual(m['colf1'], 0.5)
        self.assertAlmostEqual(m['fgcs'], 0.0)        # tables not covered

    def test_error_record_zeroes_every_metric(self):
        record = {'status': 'error',
                  'retrieved_tables': ['t1'],
                  'retrieved_column_keys': ['t1.a']}
        m = self.mc.metric_values(record, {'gold_tables': ['t1'],
                                           'gold_columns': ['t1.a']})
        for k in self.mc.METRIC_KEYS:
            self.assertEqual(m[k], 0.0, k)

    def test_record_c_hand_computed_with_c_prefix_normalization(self):
        record = {'status': 'ok',
                  'retrieved_tables': ['x', 'y', 'z'],
                  # 'C:' prefix is stripped by the frozen normalization
                  'retrieved_column_keys': ['C:x.p', 'x.q', 'y.r']}
        gold = {'gold_tables': ['y'], 'gold_columns': ['y.r', 'x.p']}
        m = self.mc.metric_values(record, gold)
        self.assertAlmostEqual(m['tr@1'], 0.0)   # top-1 is x, gold is y
        self.assertAlmostEqual(m['tr@3'], 1.0)
        self.assertAlmostEqual(m['tr@5'], 1.0)
        self.assertAlmostEqual(m['covered@5'], 1.0)
        self.assertAlmostEqual(m['mrr'], 0.5)    # y at rank 2
        # pred columns {x.p, x.q, y.r} (2 of 3 are gold)
        self.assertAlmostEqual(m['colp'], 2 / 3)
        self.assertAlmostEqual(m['colrec'], 1.0)
        self.assertAlmostEqual(m['colf1'], 0.8)
        self.assertAlmostEqual(m['fgcs'], 1.0)

    def test_metric_keys_frozen_list(self):
        self.assertEqual(
            self.mc.METRIC_KEYS,
            ['tr@1', 'tr@3', 'tr@5', 'covered@5', 'mrr',
             'colp', 'colrec', 'colf1', 'fgcs'])


class TestBirdPopulation(unittest.TestCase):
    FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'bird_population'

    def test_wrong_expected_cell_count_raises(self):
        from terag.engines.evalkit.population import BirdPopulation
        pop = BirdPopulation(root=self.FIXTURE,
                             expected_cells={'question': 2})  # fixture has 1
        with self.assertRaises(RuntimeError) as ctx:
            pop.cells('question')
        self.assertIn('population identity broken', str(ctx.exception))

    def test_correct_cell_count_and_accessors(self):
        from terag.engines.evalkit.population import BirdPopulation
        pop = BirdPopulation(root=self.FIXTURE,
                             expected_cells={'question': 1})
        cells = pop.cells('question')
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]['db'], 'demo')
        self.assertEqual(cells[0]['fold'], 0)
        self.assertEqual(cells[0]['test_qids'], ['q1', 'q2'])
        self.assertEqual(pop.test_qids('question'), ['q1', 'q2'])
        self.assertEqual(pop.schema_tables('demo'), {'t'})
        rows = pop.train_rows('question', 0, 'demo')
        self.assertEqual(rows[0]['qid'], 't1')
        gold = pop.gold()
        self.assertEqual(gold['q1']['gold_columns'], ['t.a'])

    def test_missing_train_rows_and_schema_raise(self):
        from terag.engines.evalkit.population import BirdPopulation
        pop = BirdPopulation(root=self.FIXTURE,
                             expected_cells={'question': 1})
        with self.assertRaises(FileNotFoundError):
            pop.train_rows('question', 3, 'demo')
        with self.assertRaises(FileNotFoundError):
            pop.schema_tables('no_such_db')

    def test_frozen_default_expectation(self):
        from terag.engines.evalkit.population import BirdPopulation
        self.assertEqual(BirdPopulation.FROZEN_EXPECTED_CELLS,
                         {'question': 55, 'heldout': 50})


class TestConfigFactory(unittest.TestCase):
    def test_make_bird_config_frozen_values(self):
        from terag.engines.bird.config_factory import make_bird_config
        schema_dir = Path('/tmp/anywhere/schema/somedb')
        artifacts = '/tmp/anywhere/cache'
        cfg = make_bird_config('somedb', schema_dir, artifacts)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.data.dataset, 'bird')
        self.assertEqual(cfg.data.field_csv,
                         str(schema_dir / 'field_schema.csv'))
        self.assertEqual(cfg.data.table_csv,
                         str(schema_dir / 'table_schema.csv'))
        self.assertEqual(cfg.data.qa_csv, '')
        self.assertEqual(cfg.index.bm25.k1, 1.5)
        self.assertEqual(cfg.index.bm25.b, 0.75)
        fw = cfg.index.field_weights
        self.assertEqual((fw.table_name, fw.table_desc, fw.field_name,
                          fw.field_desc, fw.pattern),
                         (2.0, 1.5, 1.0, 0.8, 0.5))
        self.assertTrue(cfg.index.include_field_desc)
        self.assertEqual(cfg.pattern_mining.mode, 'generic')
        self.assertTrue(cfg.annotations.use_history)
        self.assertTrue(cfg.annotations.use_transfer)
        self.assertTrue(cfg.annotations.use_annotation_index)
        self.assertEqual(cfg.tokenizer.language, 'english')
        self.assertEqual(cfg.sql_dialect, 'sqlite')
        bird = cfg.data.bird
        self.assertEqual(bird.sql_dialect, 'sqlite')
        self.assertFalse(bird.use_evidence)
        self.assertTrue(bird.use_database_description)
        self.assertTrue(bird.use_value_description)
        self.assertEqual(cfg.output.artifacts_dir, artifacts)
        self.assertFalse(cfg.output.save_graph)
        self.assertFalse(cfg.output.save_index)
        self.assertFalse(cfg.output.save_patterns)
        self.assertFalse(cfg.output.save_role_stats)

    def test_make_bird_config_dialect_language_params(self):
        from terag.engines.bird.config_factory import make_bird_config
        cfg = make_bird_config('db', Path('/s'), '/a', language='chinese',
                               dialect='mysql')
        self.assertEqual(cfg.tokenizer.language, 'chinese')
        self.assertEqual(cfg.data.bird.sql_dialect, 'mysql')


if __name__ == '__main__':
    unittest.main()
