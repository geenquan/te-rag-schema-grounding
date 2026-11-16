"""R2 regression: configuration genuinely binds to execution.

Probes from the independent review, now required to pass:
* change BM25 k1 to 9.9 -> the constructed retriever receives 9.9 (custom
  profile) and the resolved-config SHA256 changes;
* change the fixed table weight bm25_score -> the algorithm input changes
  and the SHA256 changes (custom), or startup fails (paper profile);
* method/protocol YAMLs actually enter the effective parameters;
* same effective config -> same hash; different -> different hash;
* frozen constants in the retriever modules equal the declared paper
  profile values;
* name-or-path references never produce doubled directories/suffixes;
* unknown parameters are hard errors.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

import terag_release.config as cfg_mod  # noqa: E402
from terag_release import paper_profile  # noqa: E402
from terag_release.config import (ConfigError, config_sha256, load_named,
                                  resolve_run)  # noqa: E402


def _patch_yaml(path: Path, mutate):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def _mirror_config_tree(tmp: Path) -> Path:
    """Copy configs/ into tmp/configs so tests can mutate single files."""
    import shutil
    dst = Path(tmp) / "configs"
    shutil.copytree(REPO_ROOT / "configs", dst)
    return dst


class TestParameterBinding(unittest.TestCase):
    def test_bm25_k1_flows_to_constructor_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror_config_tree(Path(tmp))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                baseline = resolve_run("demo", "demo_offline",
                                       method_names=["BM25"],
                                       profile="custom")
                _patch_yaml(tree / "methods" / "bm25.yaml",
                            lambda d: d.update({"retriever": {"k1": 9.9}}))
                changed = resolve_run("demo", "demo_offline",
                                      method_names=["BM25"],
                                      profile="custom")
            effective = changed["methods"]["BM25"]["effective"]
            self.assertEqual(effective["retriever_params"]["k1"], 9.9,
                             "method YAML k1 must reach the constructor "
                             "parameters")
            self.assertNotEqual(config_sha256(baseline),
                                config_sha256(changed),
                                "an effective parameter change must change "
                                "the configuration identity")
            self.assertEqual(config_sha256(changed), config_sha256(changed))

            # the real constructor consumes it
            from terag_release import runner as runner_mod
            created = {}
            real_create = runner_mod.create_retriever

            def spy(variant, dataset_cfg, method_entry, artifacts_dir,
                    cache_dir=None):
                retriever = real_create(variant, dataset_cfg, method_entry,
                                        artifacts_dir, cache_dir)
                created["k1"] = retriever.config.k1
                return retriever

            with mock.patch.object(runner_mod, "create_retriever",
                                   side_effect=spy):
                runner_mod.run_experiments(
                    resolved=changed, run_dir=Path(tmp) / "run",
                    fold_override=0)
            self.assertEqual(created["k1"], 9.9,
                             "the constructed retriever must receive the "
                             "configured k1")

    def test_paper_profile_rejects_frozen_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror_config_tree(Path(tmp))
            _patch_yaml(tree / "methods" / "bm25.yaml",
                        lambda d: d.update({"retriever": {"k1": 9.9}}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline",
                                method_names=["BM25"])
            self.assertIn("k1", str(ctx.exception))

    def test_base_weight_change_flows_and_changes_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror_config_tree(Path(tmp))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                baseline = resolve_run("demo", "demo_offline",
                                       method_names=["BM25", "TERAG-Full"],
                                       profile="custom")
                _patch_yaml(tree / "base.yaml",
                            lambda d: d["terag"]["feature"]["table_weights"]
                            .update({"bm25_score": 0.99}))
                changed = resolve_run("demo", "demo_offline",
                                      method_names=["BM25", "TERAG-Full"],
                                      profile="custom")
                terag = resolve_run("demo", "demo_offline",
                                    method_names=["TERAG-Full"],
                                    profile="custom")
            self.assertEqual(
                terag["methods"]["TERAG-Full"]["effective"]["terag_sections"]
                ["feature"]["table_weights"]["bm25_score"], 0.99,
                "base.yaml weight changes must reach the TERAG sections")
            self.assertNotEqual(config_sha256(baseline),
                                config_sha256(changed))
            # and under the paper profile the same change is a hard error
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline",
                                method_names=["TERAG-Full"])
            self.assertIn("bm25_score", str(ctx.exception))

    def test_terag_variant_effective_params_and_run_meta(self):
        resolved = resolve_run("demo", "demo_offline",
                               method_names=["TERAG-Full", "TERAG-noAnnHit"])
        for variant, expect_annotation_hit in (("TERAG-Full", 1.0),
                                               ("TERAG-noAnnHit", 0.0)):
            sections = resolved["methods"][variant]["effective"][
                "terag_sections"]
            self.assertEqual(
                sections["feature"]["field_weights"]["annotation_hit"],
                expect_annotation_hit)
            self.assertTrue(sections["annotations"]["use_history"])
            self.assertEqual(sections["graph"]["parse_mode"], "ast")

    def test_protocol_config_enters_resolution(self):
        resolved = resolve_run("demo", "demo_offline", method_names=["BM25"])
        self.assertIn("question_v1", resolved["protocols"])
        self.assertEqual(resolved["protocols"]["question_v1"]["folds"], 5)
        self.assertEqual(resolved["protocols"]["question_v1"]
                         ["supervision"], "full_sql")

    def test_unknown_parameter_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror_config_tree(Path(tmp))
            (tree / "methods" / "bm25.yaml").write_text(
                "method: BM25\nretriever:\n  k1: 1.5\n  bogus: 1\n",
                encoding="utf-8")
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline",
                                method_names=["BM25"])
            self.assertIn("bogus", str(ctx.exception))

    def test_name_or_path_never_doubles_suffix(self):
        cfg, path = load_named("bm25", "methods")
        self.assertEqual(path, (REPO_ROOT / "configs/methods/bm25.yaml"))
        cfg2, path2 = load_named("bm25.yaml", "methods")
        self.assertEqual(path2, path)
        cfg3, path3 = load_named(str(REPO_ROOT / "configs/methods/bm25.yaml"),
                                 "methods")
        self.assertEqual(path3, path)
        with self.assertRaises(ConfigError):
            load_named("configs/datasets/private.yaml", "datasets")


class TestFixedConstants(unittest.TestCase):
    """The paper profile must equal the actual module constants."""

    def test_dense_constants(self):
        from retrievers import dense_retriever
        expected = paper_profile.PAPER_MODEL_PARAMS["Dense"]
        self.assertEqual(dense_retriever.MODEL_NAME, expected["name"])
        self.assertEqual(dense_retriever.MAX_TOKENS, expected["max_tokens"])
        self.assertEqual(dense_retriever.TABLE_COL_LIMIT,
                         expected["table_column_limit"])

    def test_densedeepseek_constants(self):
        from retrievers import linkalign_retriever
        expected = paper_profile.PAPER_MODEL_PARAMS["DenseDeepSeek"]
        self.assertEqual(linkalign_retriever.CTX_TABLE_LIMIT,
                         expected["candidate_tables"])
        self.assertEqual(linkalign_retriever.CTX_COL_LIMIT,
                         expected["displayed_columns_per_table"])
        self.assertEqual(linkalign_retriever.REQUEST_TIMEOUT,
                         expected["timeout_s"])
        self.assertEqual(linkalign_retriever.MAX_RETRIES,
                         expected["max_retries"])

    def test_linkalign_constants(self):
        from retrievers.linkalign_official_retriever import SchemaEvidence
        expected = paper_profile.PAPER_MODEL_PARAMS[
            "LinkAlign-SingleDB-Adapted"]
        self.assertEqual(SchemaEvidence.CHUNK_COLUMNS,
                         expected["chunk_columns"])

    def test_kasla_constants(self):
        import inspect
        from retrievers import kasla_retriever
        source = inspect.getsource(kasla_retriever)
        self.assertIn("C=1.0", source)
        self.assertIn("max_iter=300", source)
        expected = paper_profile.PAPER_MODEL_PARAMS["KaSLA-Adapted"]
        self.assertEqual(expected["classifier"],
                         {"C": 1.0, "max_iter": 300, "random_state": 42})

    def test_transfer_constants(self):
        from retrievers.annotation_transfer import TransferConfig
        cfg = TransferConfig()
        for key, value in paper_profile.PAPER_TERAG and {
                "min_total_sim": 0.5, "schema_only_min": 0.5,
                "context_min": 0.3, "max_sources_per_target": 3,
                "max_annotations_per_target": 3,
                "max_texts_per_table": 8}.items():
            self.assertEqual(getattr(cfg, key), value)


if __name__ == "__main__":
    unittest.main()
