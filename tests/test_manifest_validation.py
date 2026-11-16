"""R3 regression: external manifests are re-validated; qids stay aligned.

Every case uses synthetic data and never fits a real retriever — a guard
retriever raises if fit is ever reached on invalid input, proving
validation happens BEFORE execution.
"""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from terag_release import protocols
from terag_release import runner as runner_mod
from terag_release.protocols import ManifestValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent

FIELDS = [("ta", "c1", "a one", "int"), ("ta", "c2", "a two", "int"),
          ("tb", "c1", "b one", "int"), ("tb", "c2", "b two", "int")]
QUESTIONS = [f"question number {i} about tables" for i in range(10)]


def _rows():
    sqls, tables = [], []
    for i in range(10):
        if i < 5:
            sqls.append("SELECT c1 FROM ta WHERE c2 > 0")
            tables.append("ta")
        else:
            sqls.append("SELECT c1 FROM tb WHERE c2 > 0")
            tables.append("tb")
    return pd.DataFrame({"question": QUESTIONS, "sql": sqls,
                         "table": tables})


def _gold(rows):
    gold = {}
    for i, row in rows.iterrows():
        target = row["table"]
        gold[f"q{i:03d}"] = {
            "qid": f"q{i:03d}", "gold_tables": [target],
            "gold_columns": [f"{target}.c1", f"{target}.c2"],
            "gold_status": "verified_complete", "ast_parsed": True,
            "original_table": target, "sql": row["sql"],
        }
    return gold


def _manifest(rows, gold):
    q_folds = protocols.build_question_folds(rows)
    t_v1 = protocols.build_table_folds(rows)
    ast_folds, exclusions = protocols.apply_ast_exclusion(
        t_v1, gold, [f"q{i:03d}" for i in range(len(rows))])
    checks = protocols.verify_splits(rows, q_folds, t_v1)
    return protocols.build_manifest(rows, q_folds, t_v1, ast_folds,
                                    exclusions, checks)


class GuardFitted(Exception):
    pass


def _guard_run_experiments(**kwargs):
    """run_experiments with every retriever construction rigged to fail the
    test if validation did not already reject the input."""
    def boom(*a, **k):
        raise GuardFitted("fit/construction reached — validation missed it")
    with mock.patch.object(runner_mod, "create_retriever", boom):
        return runner_mod.run_experiments(**kwargs)


def _resolved(tmp, manifest_path=None, methods=("BM25",)):
    field_csv = os.path.join(tmp, "fields.csv")
    table_csv = os.path.join(tmp, "tables.csv")
    qa_csv = os.path.join(tmp, "qa.csv")
    gold_path = os.path.join(tmp, "gold.jsonl")
    pd.DataFrame(FIELDS, columns=["table", "field_name", "field_name_desc",
                                  "field_type"]).to_csv(field_csv, index=False)
    pd.DataFrame({"table": ["ta", "tb"],
                  "table_desc": ["table a", "table b"]}).to_csv(table_csv,
                                                                index=False)
    _rows().to_csv(qa_csv, index=False)
    with open(gold_path, "w", encoding="utf-8") as f:
        for rec in _gold(_rows()).values():
            f.write(json.dumps(rec) + "\n")
    data = {"field_csv": field_csv, "table_csv": table_csv,
            "qa_csv": qa_csv, "gold": gold_path}
    if manifest_path:
        data["splits_manifest"] = manifest_path
    return {
        "profile": "paper", "qid_prefix": "q",
        "dataset": {"data": data},
        "experiment": {"name": "test", "protocols": ["question_v1"],
                       "methods": list(methods)},
        "protocols": {}, "methods": {}, "budget": {"max_tables": 5,
                                                    "max_columns_per_table": 10},
    }


class TestManifestValidation(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()
        self.gold = _gold(self.rows)
        self.manifest = _manifest(self.rows, self.gold)
        self.qids = [f"q{i:03d}" for i in range(10)]

    def _rebuild_train(self, manifest, gold):
        """Rebuild each fold's train as the admissible complement of its
        test set (excluding unparseable SQL and target references) — the
        same repair an attacker/probe would apply to keep the OTHER checks
        green while corrupting target/test binding."""
        qids = self.qids
        for k in sorted(manifest["ast_refined_table_folds"], key=int):
            fd = manifest["ast_refined_table_folds"][k]
            targets = {str(x).split(".")[-1].strip().lower()
                       for x in fd["targets"]}
            test = set(fd["test"])
            train = []
            for q in qids:
                if q in test:
                    continue
                g = gold.get(q, {})
                refs = {str(x).split(".")[-1].strip().lower()
                        for x in (g.get("gold_tables") or [])}
                if not g.get("ast_parsed", True) or refs & targets:
                    continue
                train.append(q)
            fd["train"] = sorted(train)
        manifest["checks"] = {key: True for key in
                              manifest.get("checks", {})}
        return manifest

    def test_counterexample_a_emptied_targets_rejected(self):
        bad = copy.deepcopy(self.manifest)
        for fd in bad["ast_refined_table_folds"].values():
            fd["targets"] = []
        self._rebuild_train(bad, self.gold)
        with self.assertRaises(ManifestValidationError) as ctx:
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)
        self.assertIn("target tables do not match", str(ctx.exception))

    def test_counterexample_b_swapped_test_sets_rejected(self):
        bad = copy.deepcopy(self.manifest)
        folds = bad["ast_refined_table_folds"]
        keys = sorted(folds, key=int)
        # swap the TEST SETS of fold0 and fold1, keep targets, rebuild the
        # admissible train per fold
        folds[keys[0]]["test"], folds[keys[1]]["test"] = (
            folds[keys[1]]["test"], folds[keys[0]]["test"])
        self._rebuild_train(bad, self.gold)
        with self.assertRaises(ManifestValidationError) as ctx:
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)
        self.assertIn("test set does not match its declared target tables",
                      str(ctx.exception))

    def test_missing_and_unknown_targets_rejected(self):
        for mutate in (
            # drop one target table's fold declaration entirely
            lambda m: m["ast_refined_table_folds"].pop("0"),
            # invent an unknown target table
            lambda m: m["ast_refined_table_folds"]["0"]["targets"]
            .append("t_ghost"),
        ):
            bad = copy.deepcopy(self.manifest)
            mutate(bad)
            self._rebuild_train(bad, self.gold)
            with self.assertRaises(ManifestValidationError):
                protocols.validate_manifest(bad, self.rows, self.gold,
                                            self.qids)

    def test_same_table_questions_split_across_folds_rejected(self):
        bad = copy.deepcopy(self.manifest)
        # move one ta-targeted question into another fold's test set while
        # keeping all targets — the exact-membership rule must catch it
        folds = bad["ast_refined_table_folds"]
        keys = sorted(folds, key=int)
        movable = folds[keys[0]]["test"][0]
        folds[keys[0]]["test"].remove(movable)
        folds[keys[1]]["test"] = sorted(folds[keys[1]]["test"] + [movable])
        self._rebuild_train(bad, self.gold)
        with self.assertRaises(ManifestValidationError) as ctx:
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)
        self.assertIn("test set does not match", str(ctx.exception))

    def test_reordered_but_valid_manifest_passes(self):
        # shuffling qid order INSIDE a fold's test list must still validate
        ok = copy.deepcopy(self.manifest)
        for fd in ok["ast_refined_table_folds"].values():
            fd["test"] = list(reversed(fd["test"]))
            fd["train"] = list(reversed(fd["train"]))
        protocols.validate_manifest(ok, self.rows, self.gold, self.qids)

    def test_valid_manifest_passes(self):
        checks = protocols.validate_manifest(self.manifest, self.rows,
                                             self.gold, self.qids)
        self.assertTrue(all(checks.values()))

    def test_target_reference_leak_rejected(self):
        bad = copy.deepcopy(self.manifest)
        # q001 belongs to ta's fold as TEST elsewhere; put a *training* row
        # (q006, gold tb) into fold 0 whose target is ta... use a genuine
        # reference: fold with target tb gets a train qid whose SQL hits tb.
        fold_ids = sorted(bad["ast_refined_table_folds"],
                          key=lambda k: int(k))
        for k in fold_ids:
            fd = bad["ast_refined_table_folds"][k]
            if "tb" in fd["targets"] and fd["train"]:
                fd["train"] = fd["train"] + ["q005"]  # q005 gold=tb -> leak
                break
        with self.assertRaises(ManifestValidationError) as ctx:
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)
        self.assertIn("references held-out target", str(ctx.exception))

    def test_cached_checks_true_are_not_trusted(self):
        bad = copy.deepcopy(self.manifest)
        for k in bad["ast_refined_table_folds"]:
            fd = bad["ast_refined_table_folds"][k]
            if fd["train"]:
                fd["train"] = fd["train"] + ["q000"]
                break
        bad["checks"] = {k: True for k in bad.get("checks", {})}
        with self.assertRaises(ManifestValidationError):
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)

    def test_missing_and_unknown_and_duplicate_qids(self):
        for mutate in (
            lambda m: m["ast_refined_table_folds"]["0"]["train"].pop(),
            lambda m: m["ast_refined_table_folds"]["0"]["test"].append("q999"),
            lambda m: m["ast_refined_table_folds"]["0"]["test"].append(
                m["ast_refined_table_folds"]["0"]["test"][0]),
        ):
            bad = copy.deepcopy(self.manifest)
            mutate(bad)
            with self.assertRaises(ManifestValidationError):
                protocols.validate_manifest(bad, self.rows, self.gold,
                                            self.qids)

    def test_wrong_row_count_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["n_rows"] = 99
        with self.assertRaises(ManifestValidationError):
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)

    def test_missing_question_fold_coverage_rejected(self):
        bad = copy.deepcopy(self.manifest)
        key = next(iter(bad["question_folds"]))
        bad["question_folds"][key] = bad["question_folds"][key][:-1]
        with self.assertRaises(ManifestValidationError):
            protocols.validate_manifest(bad, self.rows, self.gold, self.qids)

    def test_unparseable_train_sql_rejected(self):
        bad_gold = dict(self.gold)
        first_train = self.manifest["ast_refined_table_folds"]["0"]["train"][0]
        bad_gold[first_train] = dict(bad_gold[first_train], ast_parsed=False)
        with self.assertRaises(ManifestValidationError) as ctx:
            protocols.validate_manifest(self.manifest, self.rows, bad_gold,
                                        self.qids)
        self.assertIn("unparseable SQL", str(ctx.exception))

    def test_runner_rejects_leaky_manifest_before_fit(self):
        bad = copy.deepcopy(self.manifest)
        for k in sorted(bad["ast_refined_table_folds"], key=int):
            fd = bad["ast_refined_table_folds"][k]
            if "tb" in fd["targets"] and fd["train"]:
                fd["train"] = fd["train"] + ["q005"]
                break
        bad["checks"] = {k: True for k in bad.get("checks", {})}
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, "splits_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(bad, f)
            resolved = _resolved(tmp, manifest_path=manifest_path)
            with self.assertRaises((ManifestValidationError, GuardFitted)) \
                    as ctx:
                _guard_run_experiments(resolved=resolved,
                                       run_dir=Path(tmp) / "run")
            self.assertIsInstance(
                ctx.exception, ManifestValidationError,
                "must fail in validation, not in execution")

    def test_missing_manifest_file_is_an_error_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = _resolved(
                tmp, manifest_path=os.path.join(tmp, "does_not_exist.json"))
            with self.assertRaises(FileNotFoundError):
                _guard_run_experiments(resolved=resolved,
                                       run_dir=Path(tmp) / "run")

    def test_reordered_test_qids_align_questions(self):
        # Reordering the manifest's test qids must NOT misalign questions:
        # run_unit orders the test frame BY qid and verifies each record.
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        rows = self.rows
        ordered = runner_mod._ordered_frame(
            rows.assign(qid=self.qids), ["q003", "q001", "q002"])
        self.assertEqual(list(ordered["qid"]), ["q003", "q001", "q002"])
        self.assertEqual(list(ordered["question"]),
                         [rows["question"][3], rows["question"][1],
                          rows["question"][2]])

    def test_run_unit_detects_misalignment(self):
        rows = self.rows.assign(qid=self.qids)
        test_qids = ["q003", "q001"]
        frame = runner_mod._ordered_frame(rows, ["q001", "q003"])
        method_entry = {"effective": {
            "retriever_params": {"k1": 1.5, "b": 0.75, "top_k": 5,
                                 "use_chinese_tokenizer": False},
            "extra_params": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(FIELDS, columns=["table", "field_name",
                                          "field_name_desc",
                                          "field_type"]).to_csv(
                os.path.join(tmp, "f.csv"), index=False)
            pd.DataFrame({"table": ["ta", "tb"],
                          "table_desc": ["a", "b"]}).to_csv(
                os.path.join(tmp, "t.csv"), index=False)
            dataset_cfg = {"data": {"field_csv": os.path.join(tmp, "f.csv"),
                                    "table_csv": os.path.join(tmp, "t.csv")}}
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=RuntimeError("constructed")):
                entry, records = runner_mod.run_unit(
                    "BM25", "question_v1", 0, frame, frame, test_qids,
                    self.gold, dataset_cfg, method_entry, tmp)
                # construction failure -> per-question error records that
                # keep qids aligned
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["stage"], "construct")
                self.assertEqual([r["qid"] for r in records], test_qids)
                self.assertEqual(
                    [r["question"] for r in records],
                    [rows["question"][3], rows["question"][1]])
                self.assertTrue(all(r["status"] == "error" for r in records))
                self.assertTrue(all(r["query_time"] is None
                                    for r in records))


if __name__ == "__main__":
    unittest.main()
