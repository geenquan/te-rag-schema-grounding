"""R4 regression: fold-construction helpers match the paper protocol.

The reference helpers below are verbatim ports of
experiment/run_p1b.py::_norm_raw/_norm_tokens/build_dup_groups at tag
paper-dasfaa2027-v1.0 (the private source is read-only; the reference is
embedded here so the parity test runs anywhere). The public
terag_release.protocols helpers must produce IDENTICAL groups on synthetic
inputs, including the two cases where v0.1.0 deviated (extra lowercase in
the token normalizer; masking the large-magnitude CJK numeral).
"""

import re
import unittest

from terag_release import protocols


# ---- reference implementation (paper tag, verbatim semantics) -----------

def paper_norm_raw(q):
    return re.sub(r"\s+", "", str(q).lower())


def paper_norm_tokens(q):
    q = re.sub(r"[0-9〇一二三四五六七八九十百千]+", "#", str(q))
    return set(t for t in re.split(r"[^\w#一-鿿]+", q) if t)


def paper_dup_groups(questions):
    n = len(questions)
    toks = [paper_norm_tokens(q) for q in questions]
    raws = [paper_norm_raw(q) for q in questions]
    assigned, groups = {}, []
    for i in range(n):
        if i in assigned:
            continue
        grp = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if j in assigned:
                continue
            near = (toks[i] and toks[j]
                    and len(toks[i] & toks[j]) / len(toks[i] | toks[j]) >= 0.8)
            if raws[i] == raws[j] or near:
                grp.append(j)
                assigned[j] = True
        groups.append(grp)
    return groups


# ---- synthetic boundary inputs (no enterprise content) -------------------

CASES = [
    # The two independent-review probes where v0.1.0 wrongly merged:
    ["A 1", "a 2"],                       # case difference must NOT merge
    ["库存一万件", "库存二件"],            # the large-magnitude numeral is NOT masked -> tokens differ
    # Stable basics:
    ["月度营收", "月度营收", "别的"],
    ["2024年1月的库存", "2024年2月的库存"],  # digit masking -> merge
    ["各仓库的入库件数", "客户应收总额", "拒收状态的运单号"],
    ["Q1 report", "q1 report"],           # raw normalization lowercases -> merge
    ["甲 A", "甲 a"],                      # token case keeps them apart
    ["一二三", "四五六"],                  # all-numeral masks -> {#} vs {#} merge
    ["库存100件", "库存99件"],             # digits masked -> merge
    ["库存一百件", "库存一百二十件"],       # both all-numeral strings reduce to the same masked token set
]


class TestPaperFoldParity(unittest.TestCase):
    def test_norm_raw_matches(self):
        for q in ["A b", "X Y", "库存 一万"]:
            self.assertEqual(protocols._norm_raw(q), paper_norm_raw(q))

    def test_norm_tokens_matches(self):
        for q in ["A 1", "a 2", "库存一万件", "库存二件", "q1 Report",
                  "2024年5月", "CASE"]:
            self.assertEqual(protocols._norm_tokens(q),
                             paper_norm_tokens(q),
                             f"token mismatch for {q!r}")

    def test_dup_groups_match_on_all_cases(self):
        for case in CASES:
            self.assertEqual(protocols.build_dup_groups(case),
                             paper_dup_groups(case),
                             f"group mismatch for {case}")

    def test_question_folds_keep_dup_groups_together(self):
        import pandas as pd
        rows = pd.DataFrame({
            "question": ["A 1", "a 2", "第三题", "第四题", "第五题",
                         "第六题", "第七题", "2024年1月报表", "2024年2月报表"],
            "sql": [f"SELECT x{i} FROM t{i % 3}" for i in range(9)],
            "table": [f"t{i % 3}" for i in range(9)]})
        folds = protocols.build_question_folds(rows)
        fold_of = {i: f for f, idxs in enumerate(folds) for i in idxs}
        # "A 1"/"a 2" form SEPARATE duplicate groups (parity case above);
        # the date-masked pair must share a fold (groups never split).
        self.assertNotEqual(protocols.build_dup_groups(
            ["A 1", "a 2"]), [[0, 1]])
        self.assertEqual(fold_of[7], fold_of[8])       # date-masked pair grouped
        # full coverage, exclusive
        all_test = sorted(i for f in folds for i in f)
        self.assertEqual(all_test, list(range(9)))


if __name__ == "__main__":
    unittest.main()
