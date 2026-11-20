"""Final-paper profile identity tests: the frozen files the current paper
depends on must exist, hash-verify, and stay self-consistent."""

import hashlib
import json
import unittest
from pathlib import Path

from terag_release import paper_profile

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPaperFinalProfile(unittest.TestCase):
    def test_profile_constants(self):
        self.assertEqual(paper_profile.PAPER_FINAL_PROFILE,
                         "paper_final_20261120")
        identity = paper_profile.paper_final_identity()
        self.assertEqual(sorted(identity), sorted(
            paper_profile.PAPER_FINAL_FILES))
        for name, entry in identity.items():
            self.assertEqual(entry["path"],
                             paper_profile.PAPER_FINAL_FILES[name])
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

    def test_paper_pdf_byte_identity(self):
        pdf = REPO_ROOT / "docs" / "paper_with_appendix.pdf"
        h = hashlib.sha256(pdf.read_bytes()).hexdigest()
        self.assertEqual(h, paper_profile.PAPER_FINAL_PDF_SHA256)
        self.assertGreater(pdf.stat().st_size, 500_000)
        self.assertLess(pdf.stat().st_size, 2_000_000)  # 21 pages, no sources

    def test_frozen_bird_configs_shape(self):
        data = json.loads(
            (REPO_ROOT / "configs/paper/final_20261120"
             / "bird_final_configs.json").read_text(encoding="utf-8"))
        self.assertEqual(data["freeze_id"], "final_dev_v2")
        names = []
        for proto in ("question", "heldout"):
            entry = data["selected"][proto]
            self.assertIn(entry["engine"], ("bird", "vnext"))
            self.assertTrue(entry["params"])
            names.append(entry["name"])
        # the paper's named selections
        self.assertEqual(data["selected"]["question"]["name"],
                         "qB_rerank_gate_v1")
        self.assertEqual(data["selected"]["heldout"]["name"], "full_href_1_8")
        for group in ("fixed_ablation_members", "optimized_ablations",
                      "adaptive_variants"):
            self.assertTrue(data.get(group))
            names += [e["name"] for e in data[group]]
        self.assertEqual(len(names), len(set(names)),
                         "frozen method names must be unique")
        # no machine paths anywhere in the frozen file
        text = json.dumps(data)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("research_runs", text)

    def test_families_sidecar_verifies(self):
        fam = (REPO_ROOT / "configs/paper/final_20261120"
               / "bird_families.json").read_bytes()
        sha = hashlib.sha256(fam).hexdigest()
        side = (REPO_ROOT / "configs/paper/final_20261120"
                / "bird_families.sha256").read_text(encoding="utf-8").strip()
        self.assertEqual(sha, side)
        data = json.loads(fam.decode("utf-8"))
        # declared K values are frozen and never shrunk
        self.assertEqual(
            data["protocols"]["question"]["main"]["K"], 28)
        self.assertEqual(
            data["protocols"]["heldout"]["main"]["K"], 20)
        for proto in ("question", "heldout"):
            for fam_name in ("fixed_ablation", "optimized_ablation",
                             "fixed_vs_optimized_ablation"):
                self.assertEqual(
                    data["protocols"][proto][fam_name]["K"], 16)

    def test_grid_selected_configurations_present(self):
        csv_path = (REPO_ROOT / "configs/paper/final_20261120"
                    / "grid_selected_configurations.csv")
        text = csv_path.read_text(encoding="utf-8")
        self.assertIn("setting,fold,variant,chosen,params", text[:200])
        self.assertIn("question_v1", text)
        self.assertIn("table_v2", text)
        self.assertNotIn("/Users/", text)


if __name__ == "__main__":
    unittest.main()
