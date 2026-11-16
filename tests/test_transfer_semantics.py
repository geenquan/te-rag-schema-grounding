"""Transfer (C) semantics on synthetic annotations: physical target budgets,
one-hop, direct priority. Ported from the paper tag's regression suite
(tests/test_transfer_target_cap.py, first class — the replay-dependent class
is not applicable to the public release)."""

import unittest
from collections import Counter

from retrievers.annotation_transfer import (SchemaInfo, TransferConfig,
                                            transfer_annotations)


class TestPhysicalTargetCap(unittest.TestCase):
    def fixture(self, names=('customer', 'customer')):
        column = {'field_name': 'customer', 'field_desc': 'customer',
                  'field_type': 'string'}
        schema = SchemaInfo({'s': [dict(column)],
                             't': [dict(column, field_name=n) for n in names]},
                            {'s': 'billing', 't': 'billing'})
        annotations = [dict(annotation_id=str(i), type='TERM',
                            target='s.customer', status='confirmed',
                            content=f'alias{i}', source_qids=['train0'])
                       for i in range(5)]
        return schema, annotations

    def test_duplicate_rows_share_one_physical_budget(self):
        schema, annotations = self.fixture()
        result = transfer_annotations(annotations, schema)
        self.assertEqual([r['adapted_content'] for r in result['accepted']],
                         ['alias0', 'alias1', 'alias2'])

    def test_case_variants_share_one_physical_budget(self):
        schema, annotations = self.fixture(('customer', 'CUSTOMER', 'customer'))
        result = transfer_annotations(annotations, schema)
        counts = Counter(r['target_element'].lower()
                         for r in result['accepted'])
        self.assertEqual(counts, {'t.customer': 3})

    def test_released_table_budget_can_reach_next_target(self):
        schema, annotations = self.fixture(('customer', 'customer', 'client'))
        result = transfer_annotations(
            annotations, schema,
            TransferConfig(min_total_sim=.3, max_texts_per_table=5))
        self.assertEqual(
            Counter(r['target_element'] for r in result['accepted']),
            {'t.customer': 3, 't.client': 2})
        self.assertEqual(
            len({r['adapted_content'] for r in result['accepted']}), 5)

    def test_custom_and_zero_target_budgets(self):
        schema, annotations = self.fixture()
        for limit in (0, 1, 2, 4):
            with self.subTest(limit=limit):
                result = transfer_annotations(
                    annotations, schema,
                    TransferConfig(max_annotations_per_target=limit))
                self.assertEqual(len(result['accepted']), limit)

    def test_budget_is_call_local(self):
        schema, annotations = self.fixture()
        first = transfer_annotations(annotations, schema)
        second = transfer_annotations(annotations, schema)
        self.assertEqual(first, second)
        self.assertEqual(len(second['accepted']), 3)

    def test_defaults_are_the_paper_values(self):
        cfg = TransferConfig()
        self.assertEqual(cfg.min_total_sim, 0.5)
        self.assertEqual(cfg.schema_only_min, 0.5)
        self.assertEqual(cfg.context_min, 0.3)
        self.assertEqual(cfg.max_sources_per_target, 3)
        self.assertEqual(cfg.max_annotations_per_target, 3)
        self.assertEqual(cfg.max_texts_per_table, 8)


if __name__ == '__main__':
    unittest.main()
