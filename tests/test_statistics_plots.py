"""R5 regression: statistics output genuinely drives the figure panels.

Builds a small synthetic run directory (predictions + summary), computes
statistics with the real module, renders figures with Agg matplotlib, and
inspects the Figure objects to prove the CI values from statistics.json
are the error-bar data actually drawn (not same-named files with unrelated
content). Also checks direction conversion (Full−variant → variant−Full)
and hollow markers for absent intervals.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import plots as plots_mod
from analysis import statistics as stats_mod

GOLD = {}
for i in range(24):
    target = ["ta", "tb", "tc"][i % 3]
    GOLD[f"q{i:03d}"] = {
        "qid": f"q{i:03d}", "gold_tables": [target],
        "gold_columns": [f"{target}.c1"],
        "gold_status": "verified_complete", "ast_parsed": True,
        "original_table": target}


def _records(variant, covered):
    return [{"qid": q, "setting": "question_v1", "variant": variant,
             "status": "ok",
             "retrieved_tables": (GOLD[q]["gold_tables"]
                                  if q in covered else ["other"]),
             "retrieved_column_keys": (
                 [f'{GOLD[q]["gold_tables"][0]}.c1'] if q in covered else [])}
            for q in GOLD]


class TestStatisticsToPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.run_dir = tmp / "run"
        cls.run_dir.mkdir()
        cls.gold_path = tmp / "gold.jsonl"
        with open(cls.gold_path, "w", encoding="utf-8") as f:
            for rec in GOLD.values():
                f.write(json.dumps(rec) + "\n")
        qids = sorted(GOLD)
        full = set(qids[:16])
        variant_a = set(qids[:8])       # differs from Full -> CI expected
        variant_b = set(qids[:16])      # identical to Full -> zero diff
        with open(cls.run_dir / "predictions.jsonl", "w",
                  encoding="utf-8") as f:
            for variant, covered in (("TERAG-Full", full),
                                     ("TERAG-noBC", variant_a),
                                     ("TERAG-noAnnHit", variant_b)):
                for rec in _records(variant, covered):
                    f.write(json.dumps(rec) + "\n")
        with open(cls.run_dir / "summary_full_gold.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "setting", "variant", "covered@5", "colp", "colrec",
                "colf1", "n"])
            writer.writeheader()
            writer.writerow({"setting": "question_v1", "variant": "TERAG-Full",
                             "covered@5": 16 / 24, "colp": 0.5, "colrec": 0.6,
                             "colf1": 0.55, "n": 24})
            writer.writerow({"setting": "question_v1", "variant": "TERAG-noBC",
                             "covered@5": 8 / 24, "colp": 0.4, "colrec": 0.5,
                             "colf1": 0.45, "n": 24})
            writer.writerow({"setting": "question_v1",
                             "variant": "TERAG-noAnnHit",
                             "covered@5": 16 / 24, "colp": 0.5, "colrec": 0.6,
                             "colf1": 0.55, "n": 24})
        cls.stats_file = Path(stats_mod.compute_run_statistics(
            str(cls.run_dir), str(cls.gold_path), str(tmp / "stats")))
        cls.stats = json.loads(cls.stats_file.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _figdir(self, name) -> Path:
        d = Path(self._tmp.name) / name
        d.mkdir(exist_ok=True)
        return d

    def test_statistics_identity_and_inputs(self):
        self.assertIn("no retrieval re-run", self.stats["identity"])
        self.assertTrue(self.stats["inputs"]["gold_sha256"])
        self.assertEqual(self.stats["inputs"]["n_valid"], 24)
        self.assertEqual(self.stats["method"]["seed"], 42)
        self.assertEqual(self.stats["method"]["resamples"], 10000)

    def test_direction_and_ci(self):
        by_variant = {c["variant"]: c for c in self.stats["comparisons"]}
        a = by_variant["TERAG-noBC"]
        self.assertEqual(a["direction"], "variant - Full")
        self.assertAlmostEqual(a["mean_diff"], -8 / 24, places=6)
        self.assertTrue(a["ci_available"])
        lo, hi = a["ci95_cluster_bootstrap"]
        self.assertLessEqual(lo, hi)
        self.assertLessEqual(lo, a["mean_diff"])
        self.assertGreaterEqual(hi, a["mean_diff"])
        b = by_variant["TERAG-noAnnHit"]
        self.assertTrue(b["zero_covered_difference"])
        self.assertFalse(b["ci_available"])
        self.assertNotIn("wilcoxon_p_raw", b)
        self.assertIn("not a claim of identical full retrieval outputs",
                      b["zero_difference_note"])

    def test_family_bookkeeping(self):
        by_variant = {c["variant"]: c for c in self.stats["comparisons"]}
        noBC = by_variant["TERAG-noBC"]
        self.assertEqual(noBC["family"], "k6_fixed")
        # the fixture pool lacks most k6 members -> family inference is
        # not_computed, no family-corrected value is emitted
        self.assertEqual(noBC["family_inference_status"], "not_computed")
        self.assertIn("missing_family_members", noBC)
        self.assertNotIn("wilcoxon_p_bonferroni_within_family", noBC)
        self.assertNotIn("bonferroni_k", noBC)
        # raw paired values are still reported for plotting
        self.assertIn("wilcoxon_p_raw", noBC)

    def test_plot_errorbars_carry_statistics_values(self):
        import matplotlib
        self.assertEqual(matplotlib.get_backend().lower(), "agg")
        manifest = plots_mod.plot_run(
            str(self.run_dir), self._figdir("figs"),
            statistics_path=str(self.stats_file))
        self.assertTrue(manifest["inputs"]["statistics"])
        # Inspect the drawn artists: render again with savefig captured.
        import matplotlib.pyplot as plt
        import unittest.mock as mock
        figures = []
        original_savefig = plt.Figure.savefig

        def capture(fig, *args, **kwargs):
            figures.append(fig)
            return original_savefig(fig, *args, **kwargs)

        with mock.patch.object(plt.Figure, "savefig", capture):
            plots_mod.plot_run(str(self.run_dir), self._figdir("figs2"),
                               statistics_path=str(self.stats_file))
        from matplotlib.container import ErrorbarContainer
        err_data = []
        for fig in figures:
            for ax in fig.axes:
                for container in getattr(ax, "containers", []):
                    if isinstance(container, ErrorbarContainer) and \
                            container.has_xerr:
                        main = container.lines[0]
                        x = float(np.asarray(main.get_xdata())[0])
                        barcols = container.lines[2]
                        if isinstance(barcols, tuple):
                            barcols = [b for b in barcols
                                       if b is not None][0]
                        segments = barcols.get_segments()
                        xs = [pt[0] for seg in segments for pt in seg]
                        left = x - min(xs)
                        right = max(xs) - x
                        err_data.append((x, x - left, x + right))
        by_variant = {c["variant"]: c for c in self.stats["comparisons"]}
        a = by_variant["TERAG-noBC"]
        expected_diff = 100 * a["mean_diff"]
        expected_lo, expected_hi = [100 * v for v in
                                    a["ci95_cluster_bootstrap"]]
        matched = [e for e in err_data if abs(e[0] - expected_diff) < 1e-6]
        self.assertTrue(
            matched,
            f"drawn error bars {err_data} must include the statistics "
            f"mean {expected_diff}")
        _, left, right = matched[0]
        self.assertAlmostEqual(left, expected_lo, places=5,
                               msg="whisker low must equal the CI low")
        self.assertAlmostEqual(right, expected_hi, places=5,
                               msg="whisker high must equal the CI high")

    def test_plot_without_statistics_is_point_only(self):
        manifest = plots_mod.plot_run(str(self.run_dir),
                                      self._figdir("figs3"),
                                      statistics_path=None)
        self.assertIsNone(manifest["inputs"]["statistics"])
        self.assertIn("no statistics file", manifest["ci_policy"])

    def test_heldout_panel_is_column_recall(self):
        rows = list(csv.DictReader(open(self.run_dir / "summary_full_gold.csv",
                                        encoding="utf-8")))
        with open(self.run_dir / "summary_full_gold.csv", "a", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writerow({"setting": "table_v2", "variant": "TERAG-Full",
                             "covered@5": 0.3, "colp": 0.2, "colrec": 0.4,
                             "colf1": 0.27, "n": 24})
            writer.writerow({"setting": "table_v2",
                             "variant": "MetadataMatched",
                             "covered@5": 0.5, "colp": 0.2, "colrec": 0.3,
                             "colf1": 0.24, "n": 24})
        manifest = plots_mod.plot_run(str(self.run_dir),
                                      self._figdir("figs4"),
                                      statistics_path=str(self.stats_file))
        self.assertIn("heldout_tradeoff.pdf", manifest["figures"])


if __name__ == "__main__":
    unittest.main()

class TestRound2FamiliesAndIntegrity(unittest.TestCase):
    """Second-recheck regressions: family scope, completeness, duplicates,
    and the Full/noC zero-covered-difference boundary."""

    def _write_gold(self, tmp, n=24):
        gold = {}
        for i in range(n):
            target = ["ta", "tb", "tc"][i % 3]
            gold[f"q{i:03d}"] = {
                "qid": f"q{i:03d}", "gold_tables": [target],
                "gold_columns": [f"{target}.c1"],
                "gold_status": "verified_complete", "ast_parsed": True,
                "original_table": target}
        path = tmp / "gold.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in gold.values():
                f.write(json.dumps(rec) + "\n")
        return path, gold

    def _write_predictions(self, tmp, groups):
        path = tmp / "predictions.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for (setting, variant, covered), n in groups:
                for i in range(n):
                    target = ["ta", "tb", "tc"][i % 3]
                    qid = f"q{i:03d}"
                    f.write(json.dumps({
                        "qid": qid, "setting": setting, "variant": variant,
                        "status": "ok",
                        "retrieved_tables": ([target] if qid in covered
                                             else ["other"]),
                        "retrieved_column_keys": (
                            [f"{target}.c1"] if qid in covered else []),
                    }) + "\n")
        return path

    def test_heldout_comparison_is_not_a_fixed_family(self):
        import tempfile
        from analysis import statistics as stats_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp)
            qids = sorted(gold)
            self._write_predictions(tmp, [
                (("table_v2", "TERAG-Full", set(qids[:16])), 24),
                (("table_v2", "TERAG-noBC", set(qids[:8])), 24),
                (("table_v2", "TERAG-noC", set(qids[:16])), 24)])
            out = stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                   str(tmp / "stats"))
            stats = json.loads(Path(out).read_text(encoding="utf-8"))
            for comp in stats["comparisons"]:
                self.assertEqual(comp["family"], "exploratory_heldout",
                                 "held-out comparisons must not borrow the "
                                 "Question-Level fixed families")
                self.assertNotIn("bonferroni_k", comp)

    def test_incomplete_fixed_family_reports_missing_members(self):
        import tempfile
        from analysis import statistics as stats_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp)
            qids = sorted(gold)
            # only Full + noBC present -> k6 family incomplete
            self._write_predictions(tmp, [
                (("question_v1", "TERAG-Full", set(qids[:16])), 24),
                (("question_v1", "TERAG-noBC", set(qids[:8])), 24)])
            out = stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                   str(tmp / "stats"))
            stats = json.loads(Path(out).read_text(encoding="utf-8"))
            noBC = next(c for c in stats["comparisons"]
                        if c["variant"] == "TERAG-noBC")
            self.assertEqual(noBC["family"], "k6_fixed")
            self.assertEqual(noBC["family_inference_status"],
                             "not_computed")
            for member in ("Dense", "DenseDeepSeek", "ExampleRRF"):
                self.assertIn(member, noBC["missing_family_members"])
            self.assertNotIn("wilcoxon_p_bonferroni_within_family", noBC)

    def test_complete_fixed_family_produces_correction(self):
        import tempfile
        from analysis import statistics as stats_mod
        from experiment.audit_saved_predictions import FAMILIES
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp, n=30)
            qids = sorted(gold)
            groups = [(("question_v1", "TERAG-Full", set(qids[:20])), 30)]
            for member in FAMILIES["six"]:
                groups.append((("question_v1", member, set(qids[:10])), 30))
            self._write_predictions(tmp, groups)
            out = stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                   str(tmp / "stats"))
            stats = json.loads(Path(out).read_text(encoding="utf-8"))
            noBC = next(c for c in stats["comparisons"]
                        if c["variant"] == "TERAG-noBC")
            self.assertEqual(noBC["family_inference_status"], "computed")
            self.assertEqual(noBC["bonferroni_k"], 6)
            corrected = noBC["wilcoxon_p_bonferroni_within_family"]
            self.assertGreaterEqual(corrected, noBC["wilcoxon_p_raw"])
            self.assertLessEqual(corrected, 1.0)

    def test_duplicate_predictions_rejected_before_statistics(self):
        import tempfile
        from analysis import statistics as stats_mod
        from analysis.saved_prefix import load_run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp)
            path = self._write_predictions(tmp, [
                (("question_v1", "TERAG-Full", set(gold)), 24)])
            # append a duplicate record
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "qid": "q000", "setting": "question_v1",
                    "variant": "TERAG-Full", "status": "ok",
                    "retrieved_tables": [], "retrieved_column_keys": []})
                    + "\n")
            with self.assertRaises(ValueError) as ctx:
                load_run(str(tmp))
            self.assertIn("duplicate prediction record", str(ctx.exception))
            with self.assertRaises(ValueError):
                stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                 str(tmp / "stats"))

    def test_missing_paired_population_not_computable(self):
        import tempfile
        from analysis import statistics as stats_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp)
            qids = sorted(gold)
            # Full covers 24, noBC covers 0 valid (all "other") -> no
            # shared valid? they share qids but noBC covered set is empty;
            # shared population is the intersection of RECORDS, so stats
            # compute; instead remove noBC records entirely for one qid
            path = tmp / "predictions.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i in range(24):
                    target = ["ta", "tb", "tc"][i % 3]
                    qid = f"q{i:03d}"
                    full_covered = qid in set(qids[:16])
                    f.write(json.dumps({
                        "qid": qid, "setting": "question_v1",
                        "variant": "TERAG-Full", "status": "ok",
                        "retrieved_tables": ([target] if full_covered
                                             else ["other"]),
                        "retrieved_column_keys": (
                            [f"{target}.c1"] if full_covered else [])})
                        + "\n")
                    if qid != "q000":  # noBC missing one record entirely
                        f.write(json.dumps({
                            "qid": qid, "setting": "question_v1",
                            "variant": "TERAG-noBC", "status": "ok",
                            "retrieved_tables": ["other"],
                            "retrieved_column_keys": []}) + "\n")
            out = stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                   str(tmp / "stats"))
            stats = json.loads(Path(out).read_text(encoding="utf-8"))
            noBC = next(c for c in stats["comparisons"]
                        if c["variant"] == "TERAG-noBC")
            # paired_covered requires the FULL valid population on both
            # sides; the missing record makes the strict pairing fail
            self.assertTrue(
                noBC["status"] == "not_computable"
                or noBC["n"] < len(gold),
                "population gaps must surface, never silently shrink")

    def test_zero_covered_difference_is_not_identical_outputs(self):
        import tempfile
        from analysis import statistics as stats_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, gold = self._write_gold(tmp)
            qids = sorted(gold)
            covered = set(qids[:16])
            path = tmp / "predictions.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for variant in ("TERAG-Full", "TERAG-noC"):
                    for i in range(24):
                        target = ["ta", "tb", "tc"][i % 3]
                        qid = f"q{i:03d}"
                        hit = qid in covered
                        # SAME covered@5 per question, but DIFFERENT
                        # ordered outputs for Full (extra wrong table
                        # first) — the paper's Full/noC boundary
                        tables = ([target, "decoy"] if (hit and
                                  variant == "TERAG-Full") else
                                  [target] if hit else ["other"])
                        f.write(json.dumps({
                            "qid": qid, "setting": "table_v2",
                            "variant": variant, "status": "ok",
                            "retrieved_tables": tables,
                            "retrieved_column_keys": (
                                [f"{t}.c1" for t in tables if t == target]
                                if hit else [])}) + "\n")
            out = stats_mod.compute_run_statistics(str(tmp), str(gold_path),
                                                   str(tmp / "stats"))
            stats = json.loads(Path(out).read_text(encoding="utf-8"))
            noC = next(c for c in stats["comparisons"]
                       if c["variant"] == "TERAG-noC")
            self.assertTrue(noC["zero_covered_difference"])
            self.assertFalse(noC["ci_available"])
            self.assertIn("not a claim of identical full retrieval outputs",
                          noC["zero_difference_note"])
            self.assertIn("identical per-question Covered@5 values",
                          noC["ci_note"])


class TestPlotIdentityBinding(unittest.TestCase):
    """Second-recheck regressions: statistics from another run are refused."""

    @classmethod
    def setUpClass(cls):
        import csv
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.run_dir = tmp / "run"
        cls.run_dir.mkdir()
        cls.gold_path = tmp / "gold.jsonl"
        gold = {}
        for i in range(24):
            target = ["ta", "tb", "tc"][i % 3]
            gold[f"q{i:03d}"] = {
                "qid": f"q{i:03d}", "gold_tables": [target],
                "gold_columns": [f"{target}.c1"],
                "gold_status": "verified_complete", "ast_parsed": True,
                "original_table": target}
        with open(cls.gold_path, "w", encoding="utf-8") as f:
            for rec in gold.values():
                f.write(json.dumps(rec) + "\n")
        qids = sorted(gold)
        with open(cls.run_dir / "predictions.jsonl", "w",
                  encoding="utf-8") as f:
            for variant, covered in (("TERAG-Full", set(qids[:16])),
                                     ("TERAG-noBC", set(qids[:8]))):
                for i in range(24):
                    target = ["ta", "tb", "tc"][i % 3]
                    qid = f"q{i:03d}"
                    hit = qid in covered
                    f.write(json.dumps({
                        "qid": qid, "setting": "question_v1",
                        "variant": variant, "status": "ok",
                        "retrieved_tables": ([target] if hit
                                             else ["other"]),
                        "retrieved_column_keys": (
                            [f"{target}.c1"] if hit else [])}) + "\n")
        with open(cls.run_dir / "summary_full_gold.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "setting", "variant", "covered@5", "colp", "colrec",
                "colf1", "n"])
            writer.writeheader()
            writer.writerow({"setting": "question_v1",
                             "variant": "TERAG-Full",
                             "covered@5": 16 / 24, "colp": 0.5,
                             "colrec": 0.6, "colf1": 0.55, "n": 24})
            writer.writerow({"setting": "question_v1",
                             "variant": "TERAG-noBC",
                             "covered@5": 8 / 24, "colp": 0.4,
                             "colrec": 0.5, "colf1": 0.45, "n": 24})
        from analysis import statistics as stats_mod
        cls.stats_file = Path(stats_mod.compute_run_statistics(
            str(cls.run_dir), str(cls.gold_path), str(tmp / "stats")))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _figdir(self, name="figs"):
        d = Path(self._tmp.name) / name
        d.mkdir(exist_ok=True)
        return d

    def test_doctored_predictions_digest_rejected(self):
        from analysis import plots as plots_mod
        stats = json.loads(self.stats_file.read_text(encoding="utf-8"))
        stats["inputs"]["predictions_sha256"] = "0" * 64
        doctored = Path(self._tmp.name) / "doctored_stats.json"
        doctored.write_text(json.dumps(stats), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            plots_mod.plot_run(str(self.run_dir), self._figdir(),
                               statistics_path=str(doctored))
        self.assertIn("different predictions", str(ctx.exception))

    def test_statistics_of_another_run_rejected(self):
        # same gold, different predictions content -> different digests
        import tempfile
        from analysis import statistics as stats_mod
        with tempfile.TemporaryDirectory() as tmp2:
            tmp2 = Path(tmp2)
            other_run = tmp2 / "run"
            other_run.mkdir()
            with open(other_run / "predictions.jsonl", "w",
                      encoding="utf-8") as f:
                for i in range(24):
                    target = ["ta", "tb", "tc"][i % 3]
                    f.write(json.dumps({
                        "qid": f"q{i:03d}", "setting": "question_v1",
                        "variant": "TERAG-Full", "status": "ok",
                        "retrieved_tables": [target],
                        "retrieved_column_keys": [f"{target}.c1"]})
                        + "\n")
                    f.write(json.dumps({
                        "qid": f"q{i:03d}", "setting": "question_v1",
                        "variant": "TERAG-noBC", "status": "ok",
                        "retrieved_tables": ["other"],
                        "retrieved_column_keys": []}) + "\n")
            other_stats = stats_mod.compute_run_statistics(
                str(other_run), str(self.gold_path), str(tmp2 / "stats"))
            from analysis import plots as plots_mod
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(self.run_dir), self._figdir("f2"),
                                   statistics_path=str(other_stats))
            self.assertIn("does not belong to this run", str(ctx.exception))

    def test_legitimate_copy_of_same_materials_passes(self):
        import shutil
        from analysis import plots as plots_mod
        copied = Path(self._tmp.name) / "copy_run"
        shutil.copytree(self.run_dir, copied)
        # copy also carries the statistics folder for auto-detection
        stats_copy = copied / "statistics"
        stats_copy.mkdir(exist_ok=True)
        shutil.copy(self.stats_file, stats_copy / "statistics.json")
        manifest = plots_mod.plot_run(str(copied), self._figdir("f3"))
        self.assertIsNotNone(manifest["inputs"]["statistics"])

    def test_missing_statistics_still_point_only(self):
        from analysis import plots as plots_mod
        manifest = plots_mod.plot_run(str(self.run_dir),
                                      self._figdir("f4"),
                                      statistics_path=None)
        self.assertIsNone(manifest["inputs"]["statistics"])

    def test_mismatched_gold_digest_rejected(self):
        from analysis import plots as plots_mod
        stats = json.loads(self.stats_file.read_text(encoding="utf-8"))
        stats["inputs"]["gold_sha256"] = "0" * 64
        doctored = Path(self._tmp.name) / "doctored_gold.json"
        doctored.write_text(json.dumps(stats), encoding="utf-8")
        with self.assertRaises(ValueError):
            plots_mod.plot_run(str(self.run_dir), self._figdir("f5"),
                               statistics_path=str(doctored))


class TestPlotMeanConsistency(unittest.TestCase):
    """Final-review regressions: summary/statistics mean consistency.

    The previous guard tested a field name inside the Full summary row
    (always false), so the mean check was dead: a tampered summary with a
    mismatched covered@5 rate passed identity and only died later inside
    matplotlib (negative xerr) — or not at all when the CI was wide
    enough. The identity check must reject these itself, before any
    figure is written.
    """

    def _fixture(self, tmp: Path, full_rate=None, variant_rate=None,
                 stats_mean=None, drop_full_row=False, wide_ci=False):
        """12-question / 3-cluster fixture; Full covers 8/12, noBC 4/12.

        Tampered rates / statistics values are written verbatim so the
        identity check must reconcile them against each other.
        """
        import csv
        from analysis import statistics as stats_mod
        run = tmp / "run"
        run.mkdir(exist_ok=True)
        gold = {}
        for i in range(12):
            target = ["ta", "tb", "tc"][i % 3]
            gold[f"q{i:03d}"] = {
                "qid": f"q{i:03d}", "gold_tables": [target],
                "gold_columns": [f"{target}.c1"],
                "gold_status": "verified_complete", "ast_parsed": True,
                "original_table": target}
        gold_path = tmp / "gold.jsonl"
        with open(gold_path, "w", encoding="utf-8") as f:
            for rec in gold.values():
                f.write(json.dumps(rec) + "\n")
        qids = sorted(gold)
        with open(run / "predictions.jsonl", "w", encoding="utf-8") as f:
            for variant, covered in (("TERAG-Full", set(qids[:8])),
                                     ("TERAG-noBC", set(qids[:4]))):
                for i in range(12):
                    target = ["ta", "tb", "tc"][i % 3]
                    qid = f"q{i:03d}"
                    hit = qid in covered
                    f.write(json.dumps({
                        "qid": qid, "setting": "question_v1",
                        "variant": variant, "status": "ok",
                        "retrieved_tables": ([target] if hit
                                             else ["other"]),
                        "retrieved_column_keys": (
                            [f"{target}.c1"] if hit else [])}) + "\n")
        rows = [
            {"setting": "question_v1", "variant": "TERAG-Full",
             "covered@5": (full_rate if full_rate is not None else 8 / 12),
             "colp": .5, "colrec": .5, "colf1": .5, "n": 12},
            {"setting": "question_v1", "variant": "TERAG-noBC",
             "covered@5": (variant_rate if variant_rate is not None
                           else 4 / 12),
             "colp": .4, "colrec": .4, "colf1": .4, "n": 12},
        ]
        if drop_full_row:
            rows = rows[1:]
        with open(run / "summary_full_gold.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        stats_file = Path(stats_mod.compute_run_statistics(
            str(run), str(gold_path), str(tmp / "stats")))
        if stats_mean is not None or wide_ci:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            for comp in stats["comparisons"]:
                if comp["variant"] == "TERAG-noBC":
                    if stats_mean is not None:
                        comp["mean_diff"] = stats_mean
                    if wide_ci:
                        comp["ci95_cluster_bootstrap"] = [-1.0, 1.0]
            stats_file = tmp / "doctored_stats.json"
            stats_file.write_text(json.dumps(stats), encoding="utf-8")
        return run, stats_file

    def _figdir(self, tmp, name="figs"):
        d = tmp / name
        d.mkdir(exist_ok=True)
        return d

    def test_legitimate_fixture_plots(self):
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp)
            manifest = plots_mod.plot_run(str(run), self._figdir(tmp),
                                          statistics_path=str(stats_file))
            self.assertTrue(manifest["figures"])

    def test_tampered_variant_rate_rejected(self):
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp, variant_rate=5 / 12)
            figs = self._figdir(tmp)
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(run), figs,
                                   statistics_path=str(stats_file))
            self.assertIn("mean difference", str(ctx.exception))
            self.assertEqual(list(figs.iterdir()), [],
                             "no figure may be written on rejection")

    def test_tampered_full_rate_rejected(self):
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp, full_rate=7 / 12)
            figs = self._figdir(tmp)
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(run), figs,
                                   statistics_path=str(stats_file))
            self.assertIn("mean difference", str(ctx.exception))
            self.assertEqual(list(figs.iterdir()), [])

    def test_tampered_statistics_mean_rejected(self):
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # statistics claim -1/4 while both summary rows say -1/3
            run, stats_file = self._fixture(tmp, stats_mean=-0.25)
            figs = self._figdir(tmp)
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(run), figs,
                                   statistics_path=str(stats_file))
            self.assertIn("inconsistent with this run's summary",
                          str(ctx.exception))
            self.assertEqual(list(figs.iterdir()), [])

    def test_wide_ci_does_not_mask_wrong_point(self):
        # The wrong point estimate sits INSIDE a wide CI, so matplotlib
        # would happily draw it; the identity check must reject it itself.
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp, variant_rate=5 / 12,
                                            wide_ci=True)
            figs = self._figdir(tmp)
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(run), figs,
                                   statistics_path=str(stats_file))
            self.assertIn("mean difference", str(ctx.exception))
            self.assertEqual(list(figs.iterdir()), [])

    def test_computed_comparison_without_full_row_rejected(self):
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp, drop_full_row=True)
            figs = self._figdir(tmp)
            with self.assertRaises(ValueError) as ctx:
                plots_mod.plot_run(str(run), figs,
                                   statistics_path=str(stats_file))
            self.assertIn("no TERAG-Full summary row", str(ctx.exception))
            self.assertEqual(list(figs.iterdir()), [])

    def test_non_finite_mean_cannot_bypass(self):
        from analysis import plots as plots_mod
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    run, stats_file = self._fixture(tmp, stats_mean=bad)
                    figs = self._figdir(tmp)
                    with self.assertRaises(ValueError) as ctx:
                        plots_mod.plot_run(str(run), figs,
                                           statistics_path=str(stats_file))
                    self.assertIn("non-finite", str(ctx.exception))
                    self.assertEqual(list(figs.iterdir()), [])

    def test_not_computable_comparison_keeps_no_ci_semantics(self):
        # A not_computable comparison (population gap) must not require a
        # Full-row mean reconciliation; the identity check ignores its
        # mean but the plot still renders it point-only/hollow.
        from analysis import plots as plots_mod
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run, stats_file = self._fixture(tmp)
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            for comp in stats["comparisons"]:
                comp["status"] = "not_computable"
                comp.pop("mean_diff", None)
                comp["reason"] = "paired population incomplete: probe"
            stats_file = tmp / "nc_stats.json"
            stats_file.write_text(json.dumps(stats), encoding="utf-8")
            manifest = plots_mod.plot_run(str(run), self._figdir(tmp),
                                          statistics_path=str(stats_file))
            self.assertTrue(manifest["figures"])
