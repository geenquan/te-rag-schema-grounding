"""Configuration tests: extends, resolution, secret refusal, fingerprints,
and preflight of every shipped method/protocol/experiment config."""

import unittest

from terag_release.config import (ConfigError, config_sha256, load_config,
                                  load_dataset, load_experiment, load_protocol,
                                  terag_sections, load_base, build_terag_config)

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLoading(unittest.TestCase):
    def test_base_loads_with_paper_values(self):
        base = load_base()
        terag = base["terag"]
        self.assertEqual(terag["index"]["bm25"]["k1"], 1.5)
        self.assertEqual(terag["index"]["bm25"]["b"], 0.75)
        self.assertEqual(terag["feature"]["table_weights"],
                         {"bm25_score": 0.35, "graph_score": 0.35,
                          "pattern_score": 0.30})
        self.assertEqual(terag["feature"]["field_weights"]["annotation_hit"],
                         1.0)
        self.assertEqual(terag["graph"]["role_weights"]["WHERE"], 2.0)
        self.assertEqual(base["transfer_defaults"]["min_total_sim"], 0.5)
        self.assertEqual(base["transfer_defaults"]["max_texts_per_table"], 8)

    def test_method_extends_chain(self):
        full = load_config(REPO_ROOT / "configs" / "methods" / "terag_full.yaml")
        self.assertTrue(full["terag"]["annotations"]["use_history"])
        noBC = load_config(REPO_ROOT / "configs" / "methods" / "terag_noBC.yaml")
        self.assertFalse(noBC["terag"]["annotations"]["use_history"])
        self.assertTrue(noBC["terag"]["annotations"]
                        ["use_annotation_index"])

    def test_dataset_paths_anchored_to_repo(self):
        demo = load_dataset("demo")
        self.assertTrue(demo["data"]["field_csv"].startswith(str(REPO_ROOT)))
        self.assertTrue(os.path.exists(demo["data"]["field_csv"]))
        self.assertTrue(os.path.exists(demo["data"]["gold"]))

    def test_secret_refusal(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as f:
            f.write("api_key: sk-realtokenvalue\nname: bad\n")
            path = f.name
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.unlink(path)

    def test_placeholder_secret_accepted(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as f:
            f.write("model:\n  api_key: ${DEEPSEEK_API_KEY}\n")
            path = f.name
        try:
            load_config(path)  # must not raise
        finally:
            os.unlink(path)

    def test_fingerprint_stable_and_sensitive(self):
        cfg = load_config(REPO_ROOT / "configs" / "methods" / "terag_full.yaml")
        self.assertEqual(config_sha256(cfg), config_sha256(cfg))
        cfg["terag"]["annotations"]["use_transfer"] = False
        self.assertNotEqual(
            config_sha256(cfg),
            config_sha256(load_config(
                REPO_ROOT / "configs" / "methods" / "terag_full.yaml")))


class TestPreflightAllConfigs(unittest.TestCase):
    """Every shipped config parses and every referenced method exists."""

    def test_all_method_configs(self):
        from terag_release import variants
        factories = {d["factory"] for d in variants.VARIANTS.values()}
        methods_dir = REPO_ROOT / "configs" / "methods"
        files = sorted(methods_dir.glob("*.yaml"))
        self.assertGreaterEqual(len(files), 17)
        for path in files:
            cfg = load_config(path)
            # Baseline configs name the factory/registry implementation;
            # TE-RAG configs additionally carry the paper variant name.
            self.assertIn(cfg.get("method"), factories | set(variants.VARIANTS),
                          f"{path.name}: unknown method {cfg.get('method')}")
            if "variant" in cfg:
                self.assertIn(cfg["variant"], variants.VARIANTS)

    def test_protocol_configs(self):
        for name in ("question_level", "heldout_table"):
            cfg = load_protocol(name)
            self.assertEqual(cfg["folds"], 5)
            self.assertEqual(cfg["supervision"], "full_sql")
            self.assertEqual(cfg["budget"]["max_tables"], 5)
            self.assertEqual(cfg["budget"]["max_columns_per_table"], 10)

    def test_experiment_configs_reference_known_methods(self):
        from terag_release import variants
        for name in ("full_matrix", "demo_offline",
                     "question_only_offline"):
            cfg = load_experiment(name)
            for method in cfg["methods"]:
                self.assertIn(method, variants.VARIANTS)

    def test_terag_config_build(self):
        base, dataset = load_base(), load_dataset("demo")
        sections = terag_sections(base, dataset)
        cfg = build_terag_config(
            sections, dataset["data"],
            artifacts_dir=tempfile.mkdtemp(prefix="terag_test_art_"))
        self.assertEqual(cfg.index.bm25.k1, 1.5)
        self.assertEqual(cfg.annotations.use_history, False)  # base default
        self.assertEqual(cfg.pattern_mining.entity_keywords[:2],
                         ["仓库", "网点"])
        self.assertTrue(os.path.isabs(cfg.data.field_csv))


if __name__ == "__main__":
    unittest.main()
