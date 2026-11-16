"""Package-content safety checks: no credentials, private absolute paths,
enterprise identifiers, or forbidden files may enter the public tree."""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_FILES = {
    "config.yaml", ".env", "FREEZE_REPORT.md", "PAPER_RELEASE.md",
    "PAPER_RELEASE_REPORT.md", "SESSION_HANDOFF.md", "PROJECT_GUIDE.md",
    "AGENTS.md", "CLAUDE.md",
}

FORBIDDEN_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "API-key-shaped secret"),
    (re.compile(r"/Users/[A-Za-z]+/"), "absolute user path"),
    (re.compile(r"miniconda3|chatenv11|low-code-gpt"), "private environment"),
    (re.compile(r"DEEPSEEK_API_KEY:\s*\S+"), "key assignment in a file"),
    (re.compile(r"source_dataset/processed_"), "private enterprise export"),
    (re.compile(r"results/2026\d{4}_"), "private run directory"),
]

# The paper's enterprise question-id shape (pg + 3 digits) must not appear
# in shipped code/docs (synthetic demo uses the q-prefix).
QID_PATTERN = re.compile(r"\bpg\d{3}\b")

SCAN_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".csv",
                 ".json", ".jsonl"}


class TestPackageContent(unittest.TestCase):
    def test_no_forbidden_top_level_files(self):
        present = {p.name for p in REPO_ROOT.iterdir()}
        leaked = present & FORBIDDEN_FILES
        self.assertEqual(leaked, set(), f"forbidden files present: {leaked}")

    def test_no_private_patterns_in_text_files(self):
        offenders = []
        for path in sorted(REPO_ROOT.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            # tests/ legitimately contains the guard patterns themselves;
            # generated demo data is covered by dedicated assertions below.
            if rel.startswith((".venv", "venv", "runs", "demo/data",
                               "tests/")):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern, why in FORBIDDEN_PATTERNS:
                match = pattern.search(text)
                if match:
                    offenders.append(f"{rel}: {why} ({match.group(0)[:24]!r})")
                    break
        self.assertEqual(offenders, [], f"private content found: {offenders}")

    def test_no_enterprise_qids(self):
        offenders = []
        for path in sorted(REPO_ROOT.rglob("*.py")):
            rel = str(path.relative_to(REPO_ROOT))
            if rel.startswith((".venv", "venv")):
                continue
            if QID_PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)
        self.assertEqual(offenders, [],
                         f"enterprise qid shape found in: {offenders}")

    def test_third_party_licenses_present(self):
        upstream = REPO_ROOT / "retrievers" / "upstream"
        for name in ("LinkAlign_LICENSE", "KaSLA_LICENSE"):
            license_file = upstream / name
            self.assertTrue(license_file.exists(), f"missing {name}")
            text = license_file.read_text(encoding="utf-8")
            self.assertIn("MIT", text)

    def test_demo_data_present_and_fictional(self):
        qa = (REPO_ROOT / "demo" / "data" / "qa_data.csv").read_text(
            encoding="utf-8")
        tables = (REPO_ROOT / "demo" / "data" /
                  "table_schema.csv").read_text(encoding="utf-8")
        for fictional in ("wms_in_orders", "tms_shipment", "fin_receivable"):
            self.assertIn(fictional, tables)
        # None of the enterprise table prefixes may appear.
        for prefix in ("scyx_", "ads_", "dwd_", "ods_"):
            self.assertNotIn(prefix, tables)
            self.assertNotIn(prefix, qa)


if __name__ == "__main__":
    unittest.main()
