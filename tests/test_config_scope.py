"""Round-2 P1 regressions: no configuration is accepted-but-ignored.

Every field is either (A) supported and wired, (B) a read-only mirror
validated against the actual runtime constant under EVERY profile, or
(C) rejected. The five counterexamples from the second independent
recheck must each fail loudly at resolution time.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

import terag_release.config as cfg_mod  # noqa: E402
from terag_release.config import ConfigError, resolve_run  # noqa: E402


def _mirror(tmp: Path) -> Path:
    import shutil
    dst = Path(tmp) / "configs"
    shutil.copytree(REPO_ROOT / "configs", dst)
    return dst


def _patch(path: Path, mutate):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True,
                                   sort_keys=False), encoding="utf-8")


class TestMirrorsRejectedUnderEveryProfile(unittest.TestCase):
    def test_transfer_defaults_deviation_rejected_even_under_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "base.yaml",
                   lambda d: d["transfer_defaults"].update(
                       {"min_total_sim": 0.1}))
            for profile in (None, "custom"):
                with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                    with self.assertRaises(ConfigError) as ctx:
                        resolve_run("demo", "demo_offline",
                                    method_names=["TERAG-Full"],
                                    profile=profile)
                self.assertIn("min_total_sim", str(ctx.exception))
                self.assertIn("read-only", str(ctx.exception))

    def test_transfer_defaults_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "base.yaml",
                   lambda d: d["transfer_defaults"].update({"gate": 1}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError):
                    resolve_run("demo", "demo_offline",
                                method_names=["TERAG-Full"])

    def test_model_name_change_rejected_even_under_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "methods" / "dense.yaml",
                   lambda d: d["model"].update(
                       {"name": "SYNTHETIC_AUDIT_MODEL"}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline", method_names=["Dense"],
                                profile="custom")
            self.assertIn("read-only", str(ctx.exception))

    def test_model_unknown_typo_rejected_under_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "methods" / "dense.yaml",
                   lambda d: d["model"].update({"unknown_typo": 123}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline", method_names=["Dense"])
            self.assertIn("unknown_typo", str(ctx.exception))

    def test_nested_model_typo_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "methods" / "kasla_adapted.yaml",
                   lambda d: d["model"]["classifier"].update(
                       {"max_iterr": 5}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline",
                                method_names=["KaSLA-Adapted"])
            self.assertIn("max_iterr", str(ctx.exception))

    def test_api_key_env_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "methods" / "dense_deepseek.yaml",
                   lambda d: d["api"].update({"key_env": "OTHER_KEY"}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError):
                    resolve_run("demo", "demo_offline",
                                method_names=["DenseDeepSeek"])

    def test_protocol_folds_change_rejected_even_under_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "protocols" / "question_level.yaml",
                   lambda d: d.update({"folds": 3}))
            for profile in (None, "custom"):
                with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                    with self.assertRaises(ConfigError) as ctx:
                        resolve_run("demo", "demo_offline",
                                    method_names=["BM25"], profile=profile)
                self.assertIn("folds", str(ctx.exception))

    def test_budget_change_rejected_even_under_custom(self):
        for mutate in (
            lambda tree: _patch(tree / "base.yaml",
                                lambda d: d["budget"].update(
                                    {"max_tables": 1})),
            lambda tree: _patch(tree / "protocols" / "question_level.yaml",
                                lambda d: d["budget"].update(
                                    {"max_tables": 1})),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                tree = _mirror(tmp)
                mutate(tree)
                with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                    with self.assertRaises(ConfigError) as ctx:
                        resolve_run("demo", "demo_offline",
                                    method_names=["BM25"], profile="custom")
                self.assertIn("budget", str(ctx.exception))

    def test_protocol_supervision_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "protocols" / "question_level.yaml",
                   lambda d: d.update({"supervision": "clean_projection"}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError):
                    resolve_run("demo", "demo_offline",
                                method_names=["BM25"], profile="custom")


class TestProtocolCanonicalNames(unittest.TestCase):
    def test_absolute_protocol_path_yields_canonical_execution_name(self):
        # The second-recheck probe: an absolute YAML path resolved but was
        # then used as the execution protocol name -> empty plan. The
        # canonical `name` inside the config must drive execution.
        resolved = resolve_run(
            "demo", "demo_offline",
            protocol_refs=[str(REPO_ROOT / "configs" / "protocols" /
                               "question_level.yaml")],
            method_names=["BM25"])
        self.assertEqual(resolved["experiment"]["protocols"],
                         ["question_v1"])
        self.assertEqual(resolved["experiment"]["protocol_refs"],
                         [str(REPO_ROOT / "configs" / "protocols" /
                              "question_level.yaml")])
        self.assertIn("question_v1", resolved["protocols"])

    def test_protocol_config_without_canonical_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            (tree / "protocols" / "weird.yaml").write_text(
                "name: protocol_x\nfolds: 5\nsupervision: full_sql\n",
                encoding="utf-8")
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                with self.assertRaises(ConfigError) as ctx:
                    resolve_run("demo", "demo_offline",
                                protocol_refs=["weird"],
                                method_names=["BM25"])
            self.assertIn("not a supported execution protocol",
                          str(ctx.exception))

    def test_duplicate_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                resolve_run("demo", "demo_offline",
                            protocol_refs=[
                                "question_level",
                                str(REPO_ROOT / "configs" / "protocols" /
                                    "question_level.yaml")],
                            method_names=["BM25"])


class TestCustomScopeStillWorks(unittest.TestCase):
    def test_custom_retriever_params_still_flow(self):
        # the genuinely wired custom surface must keep working
        with tempfile.TemporaryDirectory() as tmp:
            tree = _mirror(tmp)
            _patch(tree / "methods" / "bm25.yaml",
                   lambda d: d.update({"retriever": {"k1": 9.9}}))
            with mock.patch.object(cfg_mod, "CONFIG_DIR", tree):
                resolved = resolve_run("demo", "demo_offline",
                                       method_names=["BM25"],
                                       profile="custom")
            self.assertEqual(
                resolved["methods"]["BM25"]["effective"]
                ["retriever_params"]["k1"], 9.9)


if __name__ == "__main__":
    unittest.main()
