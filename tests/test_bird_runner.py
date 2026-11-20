"""BIRD runner wiring tests: method resolution, protocol availability,
network/budget guards, and the families sidecar verification."""

import unittest
from unittest import mock

from terag_release import bird_runner
from terag_release.bird_stats import verify_families


class TestMethodResolution(unittest.TestCase):
    def test_core_resolves_to_default_params(self):
        resolved = bird_runner.resolve_methods(["Core"])
        self.assertEqual(list(resolved), ["Core"])
        spec = resolved["Core"]
        self.assertEqual(spec["kind"], "engine")
        self.assertEqual(spec["engine"], "bird")
        self.assertEqual(spec["engine_variant"], "Full")
        from terag.engines.bird.study_defaults import DEFAULT
        self.assertEqual(spec["params"], dict(DEFAULT))

    def test_frozen_selected_methods_resolve(self):
        resolved = bird_runner.resolve_methods(
            ["qB_rerank_gate_v1", "full_href_1_8"])
        self.assertEqual(resolved["qB_rerank_gate_v1"]["engine"], "bird")
        self.assertEqual(resolved["full_href_1_8"]["engine"], "vnext")
        # mechanism namespaces present and must reach execution
        self.assertIn("field_rerank",
                      resolved["qB_rerank_gate_v1"]["params"])
        self.assertIn("hg", resolved["full_href_1_8"]["params"])

    def test_fixed_ablation_params_match_selected(self):
        """A fixed-ablation member carries the IDENTICAL params of its
        ablated selected config apart from the engine switch."""
        resolved = bird_runner.resolve_methods(
            ["qB_rerank_gate_v1", "qB_rerank_gate_v1-noPattern"])
        sel = dict(resolved["qB_rerank_gate_v1"]["params"])
        abl = dict(resolved["qB_rerank_gate_v1-noPattern"]["params"])
        self.assertEqual(resolved["qB_rerank_gate_v1-noPattern"]
                         ["engine_variant"], "noPattern")
        self.assertEqual(sel, abl, "fixed ablation must not re-tune params")

    def test_unknown_method_is_hard_error(self):
        with self.assertRaises(ValueError):
            bird_runner.resolve_methods(["NotAMethod"])

    def test_protocol_availability(self):
        for method, protocols in bird_runner.BASELINE_PROTOCOLS.items():
            if bird_runner.method_requires_network(method):
                self.assertEqual(protocols, ("question",),
                                 f"{method}: question-only in the paper")
        self.assertIn("heldout", bird_runner.BASELINE_PROTOCOLS["BM25"])

    def test_engine_kinds_valid(self):
        frozen = bird_runner.load_frozen_methods()
        self.assertGreater(len(frozen), 30)
        from terag.engines.bird.study_defaults import VARIANTS
        for name, entry in frozen.items():
            self.assertIn(entry["engine"], ("bird", "vnext"),
                          f"{name}: bad engine kind")
            self.assertIn(entry["engine_variant"], VARIANTS,
                          f"{name}: unknown predict switch")


class TestNetworkGuards(unittest.TestCase):
    def test_api_without_budget_cap_refuses(self):
        """run_baseline_cell must refuse API methods without an explicit
        user budget cap (the historical research budget never applies)."""
        from terag.engines.api.config import ApiRuntime, RUNTIME
        saved = (RUNTIME.run_dir, RUNTIME.budget_cap_cny)
        try:
            RUNTIME.run_dir = None
            RUNTIME.budget_cap_cny = None
            with mock.patch.dict("os.environ",
                                 {"DEEPSEEK_API_KEY": "dummy"},
                                 clear=True):
                with self.assertRaises(RuntimeError) as ctx:
                    bird_runner.run_baseline_cell(
                        "DenseDeepSeek", "question", 0, "db", ["q1"],
                        {"q1": "q"}, None, "f.csv", "t.csv", "/tmp/x",
                        allow_network=True, budget_cap_cny=None,
                        minilm_cache_dir=None)
                self.assertIn("budget cap", str(ctx.exception))
                with self.assertRaises(RuntimeError):
                    ApiRuntime().require_budget_cap()
        finally:
            RUNTIME.run_dir, RUNTIME.budget_cap_cny = saved

    def test_credentials_env_only(self):
        from terag.engines.api.config import load_credentials
        with mock.patch.dict("os.environ",
                             {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_BASE_URL": ""},
                             clear=True):
            key, base = load_credentials()
        self.assertEqual(key, "k")
        self.assertEqual(base, "https://api.deepseek.com")


class TestFamiliesVerification(unittest.TestCase):
    def test_sidecar_verifies_and_tamper_refuses(self):
        families, sha = verify_families()
        self.assertIn("protocols", families)
        # tampering must be refused: simulate a wrong sidecar
        from terag_release import bird_stats
        fake = mock.Mock()
        fake.read_text.return_value = "0" * 64 + "\n"
        with mock.patch.object(bird_stats, "FROZEN_FAMILIES_SHA", fake):
            with self.assertRaises(RuntimeError):
                verify_families()


if __name__ == "__main__":
    unittest.main()
