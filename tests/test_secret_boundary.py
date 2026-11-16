"""R8 regression: no repository-file credential fallback, ever.

Uses mock_open and fabricated placeholder values only — the real private
config.yaml is never read and no request is made.
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def _make_retriever(tmp):
    from retrievers.linkalign_retriever import LinkAlignRetriever
    tmp = Path(tmp)
    field_csv = tmp / "fields.csv"
    table_csv = tmp / "tables.csv"
    pd.DataFrame({"table": ["t"], "field_name": ["c"],
                  "field_name_desc": [""], "field_type": ["int"]}) \
        .to_csv(field_csv, index=False)
    pd.DataFrame({"table": ["t"], "table_desc": [""]}).to_csv(table_csv,
                                                              index=False)
    return LinkAlignRetriever(str(field_csv), str(table_csv))


class TestNoCredentialFallback(unittest.TestCase):
    def _openai_stub(self, ctor):
        """Inject a stub openai module so no real SDK is needed."""
        import sys, types
        module = types.ModuleType("openai")
        module.OpenAI = ctor
        return module

    def test_no_env_and_no_file_read_and_no_client(self):
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            retriever = _make_retriever(tmp)
            stub = self._openai_stub(
                lambda **kw: (_ for _ in ()).throw(AssertionError(
                    "client constructed without a key")))
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""},
                                 clear=False), \
                 mock.patch("builtins.open", mock.mock_open()), \
                 mock.patch.dict(sys.modules, {"openai": stub}):
                with self.assertRaises(ValueError) as ctx:
                    retriever._init_client()
            message = str(ctx.exception)
            self.assertIn("DEEPSEEK_API_KEY", message)
            self.assertNotIn("config.yaml", message,
                             "the error must point at the environment, not "
                             "at a repository file")

    def test_no_yaml_import_needed_for_client(self):
        # The removed fallback used to `import yaml`; constructing the
        # client must no longer touch yaml at all.
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            retriever = _make_retriever(tmp)
            stub = self._openai_stub(
                lambda **kw: (_ for _ in ()).throw(AssertionError(
                    "client constructed without a key")))
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                 mock.patch.dict(sys.modules,
                                 {"yaml": None, "openai": stub}):
                # sys.modules["yaml"] = None makes `import yaml` raise
                # ImportError if attempted
                with self.assertRaises(ValueError):
                    retriever._init_client()

    def test_env_key_creates_client_with_placeholder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            retriever = _make_retriever(tmp)
            import sys
            fake_client = object()
            calls = []

            def fake_openai(**kwargs):
                calls.append(kwargs)
                return fake_client

            stub = self._openai_stub(fake_openai)
            with mock.patch.dict(os.environ,
                                 {"DEEPSEEK_API_KEY":
                                  "sk-test-placeholder"}), \
                 mock.patch.dict(sys.modules, {"openai": stub}):
                retriever._init_client()
                self.assertIs(retriever._client, fake_client)
                self.assertEqual(calls[0]["api_key"],
                                 "sk-test-placeholder")

    def test_secret_values_rejected_by_config_loader(self):
        from terag_release.config import ConfigError, _check_secrets
        with self.assertRaises(ConfigError):
            _check_secrets({"api_key": "sk-real-looking-value"})
        with self.assertRaises(ConfigError):
            _check_secrets({"model": {"DEEPSEEK_API_KEY":
                                      "sk-real-looking-value"}})
        _check_secrets({"api_key": "${DEEPSEEK_API_KEY}"})
        _check_secrets({"api_key": ""})


if __name__ == "__main__":
    unittest.main()
