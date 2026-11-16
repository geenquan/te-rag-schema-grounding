"""Variant-matrix tests: the 25-group protocol matrix and exact toggles.

These pin the ablation semantics of the accepted `transfer_v3_target_cap`
configuration (see docs/experiment_catalog.md §2).
"""

import unittest

from terag_release import variants


class TestMatrix(unittest.TestCase):
    def test_matrix_sizes(self):
        self.assertEqual(len(variants.QUESTION_MATRIX), 14)
        self.assertEqual(len(variants.TABLE_MATRIX), 11)
        self.assertEqual(len(variants.FULL_MATRIX), 25)

    def test_adapted_methods_question_level_only(self):
        for name in ("LinkAlign-SingleDB-Adapted", "KaSLA-Adapted"):
            self.assertEqual(variants.VARIANTS[name]["protocols"],
                             ["question_v1"])

    def test_q_only_ablations(self):
        for name in ("TERAG-noAnnIdx", "TERAG-noAnnHit"):
            self.assertEqual(variants.VARIANTS[name]["protocols"],
                             ["question_v1"])

    def test_h_only_ablations(self):
        for name in ("TERAG-noC", "TERAG-noGraph", "TERAG-noPattern"):
            self.assertEqual(variants.VARIANTS[name]["protocols"],
                             ["table_v2"])


class TestToggles(unittest.TestCase):
    BASE = {"annotations": {}, "graph": {}, "feature": {}, "ablation": {}}

    def test_full_enables_all_channels(self):
        cfg = variants.variant_config(self.BASE, "TERAG-Full")
        ann = cfg["annotations"]
        self.assertTrue(ann["use_history"])
        self.assertTrue(ann["use_transfer"])
        self.assertTrue(ann["use_annotation_index"])
        self.assertEqual(cfg["graph"]["parse_mode"], "ast")

    def test_noBC_disables_history(self):
        ann = variants.variant_config(self.BASE, "TERAG-noBC")["annotations"]
        self.assertFalse(ann["use_history"])
        self.assertFalse(ann["use_transfer"])

    def test_noC_keeps_b(self):
        ann = variants.variant_config(self.BASE, "TERAG-noC")["annotations"]
        self.assertTrue(ann["use_history"])
        self.assertFalse(ann["use_transfer"])

    def test_noAnnIdx_keeps_extraction_not_index(self):
        ann = variants.variant_config(self.BASE, "TERAG-noAnnIdx")["annotations"]
        self.assertTrue(ann["use_history"])
        self.assertTrue(ann["use_transfer"])
        self.assertFalse(ann["use_annotation_index"])

    def test_noAnnHit_zeroes_weight_only(self):
        cfg = variants.variant_config(self.BASE, "TERAG-noAnnHit")
        self.assertEqual(cfg["feature"]["field_weights"]["annotation_hit"], 0.0)
        self.assertTrue(cfg["annotations"]["use_annotation_index"])

    def test_noGraph_disables_graph_scores_not_build(self):
        cfg = variants.variant_config(self.BASE, "TERAG-noGraph")
        self.assertFalse(cfg["ablation"]["use_graph_weight"])
        self.assertTrue(cfg["annotations"]["use_history"])

    def test_noPattern_disables_pattern_channel(self):
        self.assertFalse(
            variants.variant_config(self.BASE, "TERAG-noPattern")
            ["ablation"]["use_template_mining"])

    def test_base_mutation_safety(self):
        base = {"annotations": {"use_history": False},
                "graph": {}, "feature": {"field_weights": {}}, "ablation": {}}
        variants.variant_config(base, "TERAG-Full")
        self.assertFalse(base["annotations"]["use_history"],
                         "variant_config must not mutate the input")


class TestMethodIdentity(unittest.TestCase):
    def test_registry_names(self):
        self.assertEqual(variants.VARIANTS["TFIDF"]["factory"], "Vector")
        self.assertEqual(variants.VARIANTS["MetadataMatched"]["factory"],
                         "MatchedSchema")
        self.assertEqual(variants.VARIANTS["ExampleRRF"]["factory"],
                         "ExampleRAG")
        self.assertEqual(variants.VARIANTS["CooccurRRF"]["factory"],
                         "Cooccurrence")
        self.assertEqual(variants.VARIANTS["DenseDeepSeek"]["factory"],
                         "DenseDeepSeek")
        self.assertEqual(variants.VARIANTS["TERAG-Full"]["factory"],
                         "TE-RAG-V2")

    def test_supervision_targets(self):
        supervised = {"ExampleRRF", "CooccurRRF", "TERAG-Full", "TERAG-noBC"}
        for name in supervised:
            self.assertEqual(variants.VARIANTS[name]["supervision"],
                             "full_sql")
        for name in ("BM25", "TFIDF", "Hybrid", "MetadataMatched"):
            self.assertIsNone(variants.VARIANTS[name]["supervision"])

    def test_metadata_matched_has_no_history(self):
        terag = variants.VARIANTS["MetadataMatched"]["terag"]
        self.assertFalse(terag["annotations"]["use_history"])
        self.assertFalse(terag["annotations"]["use_transfer"])

    def test_network_flags(self):
        for name in ("Dense", "DenseDeepSeek", "LinkAlign-SingleDB-Adapted",
                     "KaSLA-Adapted"):
            self.assertTrue(variants.method_requires_network(name))
        for name in ("BM25", "TERAG-Full", "ExampleRRF"):
            self.assertFalse(variants.method_requires_network(name))


if __name__ == "__main__":
    unittest.main()
