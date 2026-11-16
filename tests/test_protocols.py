"""Protocol tests: deterministic folds, duplicate groups, AST isolation.

Run from the repository root:
    python -m unittest tests.test_protocols -v
"""

import unittest

import pandas as pd

from terag_release import protocols


def make_rows(questions, sqls, tables):
    return pd.DataFrame({"question": questions, "sql": sqls, "table": tables})


class TestDuplicateGroups(unittest.TestCase):
    def test_exact_duplicates_group(self):
        groups = protocols.build_dup_groups(["月度营收", "月度营收", "别的"])
        self.assertEqual(sorted(groups), [[0, 1], [2]])

    def test_constant_masking_groups_date_variants(self):
        # Only the numeric constant differs -> same masked tokens -> grouped.
        groups = protocols.build_dup_groups(
            ["2024年1月的库存", "2024年2月的库存"])
        self.assertEqual(groups, [[0, 1]])

    def test_distinct_questions_do_not_group(self):
        groups = protocols.build_dup_groups(
            ["各仓库的入库件数", "客户应收总额", "拒收状态的运单号"])
        self.assertEqual(groups, [[0], [1], [2]])


class TestQuestionFolds(unittest.TestCase):
    def test_coverage_and_exclusivity_deterministic(self):
        rows = make_rows([f"问题编号{i}的内容查询" for i in range(23)],
                         [f"SELECT c{i} FROM t{i%4}" for i in range(23)],
                         [f"t{i%4}" for i in range(23)])
        folds = protocols.build_question_folds(rows)
        all_test = sorted(i for f in folds for i in f)
        self.assertEqual(all_test, list(range(23)))
        self.assertEqual(len(all_test), 23)
        again = protocols.build_question_folds(rows)
        self.assertEqual(folds, again, "folds must be deterministic")


class TestTableFolds(unittest.TestCase):
    def test_isolation_and_single_fold(self):
        rows = make_rows(
            [f"q{i}" for i in range(20)],
            ["FROM ta JOIN tb" if i % 2 else "FROM ta" for i in range(20)],
            ["ta" if i < 12 else "tb" for i in range(20)])
        folds = protocols.build_table_folds(rows)
        tables = {t for f in folds for t in f["test_tables"]}
        self.assertEqual(tables, {"ta", "tb"})
        for fd in folds:
            for i in fd["train"]:
                self.assertFalse(protocols.sql_ref_tables(rows["sql"][i])
                                  & set(fd["test_tables"]))
            self.assertFalse(set(fd["train"]) & set(fd["test"]))

    def test_ast_exclusion_rules(self):
        folds = [{"test_tables": ["tx"], "test": [0], "train": [1, 2, 3]}]
        gold = {
            "q1": {"gold_tables": ["tx"], "ast_parsed": True},   # leak
            "q2": {"gold_tables": ["other"], "ast_parsed": False},  # unparseable
            "q3": {"gold_tables": ["other"], "ast_parsed": True},  # fine
        }
        refined, excluded = protocols.apply_ast_exclusion(folds, gold,
                                                          ["q0", "q1", "q2",
                                                           "q3"])
        self.assertEqual(refined[0]["train"], ["q3"])
        reasons = {e["qid"]: r for e in excluded[0]["exclusion_reasons"]
                   for r in [e["reason"]]}
        self.assertEqual(reasons.get("q1"), "references_target_table")
        self.assertEqual(reasons.get("q2"), "unparseable_sql")


class TestVerifySplits(unittest.TestCase):
    def test_invariants(self):
        rows = make_rows(
            [f"distinct question {i}" for i in range(10)],
            [f"SELECT x{i%3} FROM t{i%3}" for i in range(10)],
            [f"t{i%3}" for i in range(10)])
        q_folds = protocols.build_question_folds(rows)
        t_folds = protocols.build_table_folds(rows)
        checks = protocols.verify_splits(rows, q_folds, t_folds)
        failed = [k for k, v in checks.items() if not v]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
