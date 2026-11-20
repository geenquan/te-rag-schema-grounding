"""Grid engine unit tests: import surface, frozen construction values, and
the frozen-C parameter guard (no scoring execution here; the full parity
against the accepted run is a research-environment verification documented
in docs/release_mapping.md)."""

import unittest
from pathlib import Path

from terag_release.config import REPO_ROOT


class TestGridEngineImport(unittest.TestCase):
    def test_package_imports(self):
        from terag.engines.grid import GridEngine, make_grid_config
        self.assertEqual(
            GridEngine.implementation_id,
            "graph_casefold_ids_v1+pattern_consumer_casefold_v1")
        self.assertEqual(GridEngine.language, "chinese")

    def test_make_grid_config_frozen_values(self):
        from terag.engines.grid import make_grid_config
        cfg = make_grid_config(
            field_csv=Path("/tmp/f.csv"), table_csv=Path("/tmp/t.csv"),
            artifacts_dir=Path("/tmp/art"))
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.index.bm25.k1, 1.5)
        self.assertEqual(cfg.index.bm25.b, 0.75)
        self.assertEqual(cfg.index.field_weights.table_name, 2.0)
        self.assertEqual(cfg.index.field_weights.pattern, 0.5)
        self.assertTrue(cfg.index.include_field_desc)
        self.assertEqual(cfg.graph.parse_mode, "ast")
        self.assertEqual(cfg.pattern_mining.mode, "keywords")
        self.assertTrue(cfg.annotations.use_history)
        self.assertTrue(cfg.annotations.use_transfer)
        self.assertTrue(cfg.annotations.use_annotation_index)
        self.assertEqual(cfg.tokenizer.language, "chinese")
        self.assertFalse(cfg.output.save_index)

    def test_frozen_c_generation_params_guarded(self):
        """selected_transfer refuses anything but (0.5, 3): the frozen C
        generation parameters cannot be changed by configuration."""
        from terag.engines.grid.engine import GridEngine
        engine = object.__new__(GridEngine)  # no __init__ (no private data)
        engine.native_transfers = {}
        engine.cache_dir = None
        with self.assertRaises(RuntimeError):
            GridEngine.selected_transfer(
                engine, {"min_total_sim": 0.4, "source_topk": 3})


if __name__ == "__main__":
    unittest.main()
