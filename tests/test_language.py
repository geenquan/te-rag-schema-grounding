"""Language check: comments and docstrings of shipped Python files contain no
Chinese text.

Chinese in RUNTIME STRING LITERALS is allowed where it is algorithm input
data (tokenization dictionaries, regexes, matching keywords, demo questions)
— those are enumerated in ALLOWED_DATA_HINTS below and are not comments. The
check inspects only COMMENT tokens and docstring positions, so translating
or rejecting them cannot change program behavior.
"""

import ast
import io
import tokenize
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that legitimately contain Chinese DATA inside string literals.
ALLOWED_DATA_FILES = {
    "terag/config.py",             # default keyword dictionaries (paper domain)
    "terag/terag_retriever_v2.py",  # (historical; kept free of data now)
    "terag/index_builder.py",       # demo test queries in module main
    "terag/code_mapper.py",         # demo test queries in module main
    "retrievers/annotation_miner.py",  # Chinese stopwords/date regex/CJK class
    "retrievers/annotation_transfer.py",
    "terag/pattern_miner.py",
    "terag/feature_extractor.py",
    "terag_release/protocols.py",   # CJK ranges in dup-group regexes (algorithm)
    "tests/test_language.py",       # this check's own CJK detection regex
    "tests/test_package_content.py",  # enterprise prefixes it guards against
    "tests/test_protocols.py",      # CJK test inputs (dup-group behavior)
    "tests/test_config.py",         # demo keyword-list values asserted as data
    "demo/generator.py",            # the synthetic dataset itself
}

# Pinned third-party snapshots under retrievers/upstream/ (LinkAlign prompts
# and KaSLA utilities) are kept BYTE-IDENTICAL to the audited upstream
# versions — including any Chinese prompt text — and are never rewritten.
UPSTREAM_DIR = "retrievers/upstream"

import re
CJK = re.compile(r"[一-鿿]")


def python_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        parts = path.relative_to(REPO_ROOT).parts
        if parts[0] in (".venv", "venv", "build", "dist", "runs"):
            continue
        yield path


def comment_text(source: str) -> str:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            out.append(tok.string)
    return "\n".join(out)


def docstring_lines(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            yield body[0].value.value


class TestEnglishComments(unittest.TestCase):
    def test_no_chinese_comments(self):
        offenders = []
        for path in python_files():
            source = path.read_text(encoding="utf-8")
            comments = comment_text(source)
            if CJK.search(comments):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [],
                         f"files with Chinese comments: {offenders}")

    def test_no_chinese_docstrings(self):
        offenders = []
        for path in python_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel.startswith(UPSTREAM_DIR):
                continue  # pinned third-party snapshot, never rewritten
            source = path.read_text(encoding="utf-8")
            for doc in docstring_lines(source):
                if CJK.search(doc):
                    offenders.append(rel)
                    break
        self.assertEqual(offenders, [],
                         f"files with Chinese docstrings: {offenders}")

    def test_chinese_data_limited_to_allowlist(self):
        """Chinese in string literals is only permitted where it is data."""
        offenders = []
        for path in python_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel in ALLOWED_DATA_FILES or rel.startswith(UPSTREAM_DIR):
                continue
            source = path.read_text(encoding="utf-8")
            # crude but effective: any CJK outside comments/docstrings
            for tok in tokenize.generate_tokens(
                    io.StringIO(source).readline):
                if tok.type == tokenize.STRING and CJK.search(tok.string):
                    offenders.append(f"{rel}:{tok.start[0]}")
                    break
        self.assertEqual(offenders, [],
                         f"unexpected Chinese string literals: {offenders}")


if __name__ == "__main__":
    unittest.main()
