"""Configuration tests: extends, resolution, secret refusal, fingerprints,
paper-profile enforcement, and preflight of every shipped config."""

import os
import tempfile
import unittest
from pathlib import Path

from terag_release.config import (ConfigError, config_sha256, load_base,
                                  load_config, load_dataset, load_experiment,
                                  load_method, load_protocol, resolve_run,
                                  build_terag_config)
from terag_release import paper_profile

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

    def test_base_yaml_matches_paper_profile(self):
        """The YAML mirror may not drift from the code constants."""
        base_terag = load_base()["terag"]
        self.assertEqual(paper_profile.diff_against_paper(base_terag), [])

    def test_method_files_parse_and_match_matrix(self):
        from terag_release import variants
        for name, spec in variants.VARIANTS.items():
            cfg, path = load_method(name)  # raises on any contradiction
            self.assertEqual(path.stem, spec["config"])

    def test_dataset_paths_anchored_to_repo(self):
        dataset, _ = load_dataset("demo")
        self.assertTrue(dataset["data"]["field_csv"].startswith(str(REPO_ROOT)))
        self.assertTrue(os.path.exists(dataset["data"]["field_csv"]))
        self.assertTrue(os.path.exists(dataset["data"]["gold"]))

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
        resolved_a = resolve_run("demo", "demo_offline",
                                 method_names=["BM25"])
        resolved_b = resolve_run("demo", "demo_offline",
                                 method_names=["BM25"])
        self.assertEqual(config_sha256(resolved_a), config_sha256(resolved_b))
        resolved_b["methods"]["BM25"]["effective"]["retriever_params"]["k1"] \
            = 2.0
        self.assertNotEqual(config_sha256(resolved_a),
                            config_sha256(resolved_b))


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
            declared = cfg.get("method")
            self.assertIn(declared, factories | set(variants.VARIANTS),
                          f"{path.name}: unknown method {declared}")

    def test_protocol_configs(self):
        for name in ("question_level", "heldout_table"):
            cfg, _ = load_protocol(name)
            self.assertEqual(cfg["folds"], 5)
            self.assertEqual(cfg["supervision"], "full_sql")
            self.assertEqual(cfg["budget"]["max_tables"], 5)
            self.assertEqual(cfg["budget"]["max_columns_per_table"], 10)

    def test_experiment_configs_reference_known_methods(self):
        from terag_release import variants
        for name in ("full_matrix", "demo_offline",
                     "question_only_offline"):
            cfg, _ = load_experiment(name)
            for method in cfg["methods"]:
                self.assertIn(method, variants.VARIANTS)

    def test_full_resolution_preflight(self):
        from terag_release import variants
        resolved = resolve_run("demo", "full_matrix")
        self.assertEqual(resolved["profile"], "paper")
        for variant in variants.FULL_MATRIX:
            if variant in resolved["methods"]:
                self.assertIn("effective", resolved["methods"][variant])
        # network methods are resolvable but never runnable without the flag
        self.assertIn("Dense", resolved["methods"])

    def test_terag_config_build(self):
        resolved = resolve_run("demo", "demo_offline",
                               method_names=["TERAG-Full"])
        sections = resolved["methods"]["TERAG-Full"]["effective"][
            "terag_sections"]
        cfg = build_terag_config(
            sections, resolved["dataset"]["data"],
            artifacts_dir=tempfile.mkdtemp(prefix="terag_test_art_"))
        self.assertEqual(cfg.index.bm25.k1, 1.5)
        self.assertTrue(cfg.annotations.use_history)
        self.assertEqual(cfg.pattern_mining.entity_keywords[:2],
                         ["仓库", "网点"])
        self.assertTrue(os.path.isabs(cfg.data.field_csv))


if __name__ == "__main__":
    unittest.main()
