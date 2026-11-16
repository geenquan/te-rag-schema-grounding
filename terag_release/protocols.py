"""Evaluation protocols: deterministic fold construction and isolation.

This module re-implements, faithfully, the two evaluation protocols of the
paper:

* ``question_v1`` — five question folds. Duplicate-question groups (exact
  normalized match OR token-set Jaccard >= 0.8 after constant masking) never
  cross folds; groups are assigned greedily to the currently lightest fold
  (ties broken by lower fold index). Every question is tested exactly once.
* ``table_v2`` — five target-table folds. Each gold target table is tested in
  exactly one fold; a question is *test* if its gold target table is in the
  fold's target set. Training questions are excluded when their SQL's full
  reference-table set (regex FROM/JOIN for the v1 split, AST physical tables
  from the full gold for the v2 refinement) intersects the fold's target
  tables. Unparseable training SQL is conservatively excluded as well.

Everything here is deterministic: no RNG anywhere. Given the same QA rows and
gold, the same folds result. A frozen manifest from a previous run can be
loaded instead of rebuilding (``load_manifest``).

Train supervision (``full_sql``) is applied by
``terag_release.supervision.inject_supervision`` to TRAINING frames only; test
SQL is used exclusively for gold and isolation audits.
"""

import json
import re
from typing import Dict, List, Sequence, Set

import pandas as pd

N_FOLDS = 5
DUP_JACCARD = 0.8


# --------------------------------------------------------------------------
# Duplicate-question grouping (question protocol)
# --------------------------------------------------------------------------

def _norm_raw(question: str) -> str:
    """Whitespace-stripped, lowercased raw question text."""
    return re.sub(r"\s+", "", str(question).lower())


def _norm_tokens(question: str) -> Set[str]:
    """Token set with numeric/CJK-numeral constants masked to '#'.

    Faithful re-implementation of the paper protocol helper
    (experiment/run_p1b.py::_norm_tokens at tag paper-dasfaa2027-v1.0):
    ASCII digits and the CJK numerals up to "thousand" (see the character
    class below) are masked; the larger-magnitude numeral is deliberately
    NOT masked, and the text is NOT lowercased here (case handling belongs
    to _norm_raw). tests/test_paper_fold_parity.py pins this behavior
    against the original helper on synthetic boundary inputs.
    """
    s = re.sub(r"[0-9〇一二三四五六七八九十百千]+", "#", str(question))
    return {t for t in re.split(r"[^\w#一-鿿]+", s) if t}


def build_dup_groups(questions: Sequence[str]) -> List[List[int]]:
    """Greedy duplicate grouping of question row indices.

    Row ``j`` joins the group of an earlier row ``i`` when their normalized
    raw texts are equal, or their masked token sets are both non-empty with
    Jaccard similarity >= 0.8. Groups never merge afterwards.
    """
    raws = [_norm_raw(q) for q in questions]
    tokens = [_norm_tokens(q) for q in questions]
    groups: List[List[int]] = []
    for j in range(len(questions)):
        placed = False
        for g in groups:
            i = g[0]
            if raws[i] == raws[j]:
                g.append(j)
                placed = True
                break
            ti, tj = tokens[i], tokens[j]
            if ti and tj:
                inter = len(ti & tj)
                union = len(ti | tj)
                if union and inter / union >= DUP_JACCARD:
                    g.append(j)
                    placed = True
                    break
        if not placed:
            groups.append([j])
    return groups


def _greedy_assign(units: List, weight_of, n_folds: int = N_FOLDS):
    """Deterministic greedy assignment: next unit -> lightest fold (min index
    on ties). Preserves the given unit order."""
    assign, load = {}, [0] * n_folds
    for unit in units:
        f = min(range(n_folds), key=lambda i: (load[i], i))
        assign[unit] = f
        load[f] += weight_of(unit)
    return assign, load


def build_question_folds(rows: pd.DataFrame) -> List[List[int]]:
    """Question-level fold test indices (0-based row indices, sorted)."""
    groups = build_dup_groups(rows["question"].tolist())
    units = [tuple(g) for g in sorted(groups, key=lambda g: min(g))]
    assign, _ = _greedy_assign(units, lambda g: len(g))
    return [sorted(i for g in units if assign[g] == f for i in g)
            for f in range(N_FOLDS)]


# --------------------------------------------------------------------------
# Table folds (held-out-table protocol)
# --------------------------------------------------------------------------

def sql_ref_tables(sql: str) -> Set[str]:
    """Regex FROM/JOIN reference tables (simple names, 'select' dropped).

    This is the v1 caliber used to build the initial table folds; the v2
    refinement below uses AST physical tables from the full gold instead.
    """
    ids = re.findall(r"(?:FROM|JOIN)\s+`?([a-zA-Z_][\w.]*)`?", str(sql), re.I)
    out = set()
    for x in ids:
        t = x.split(".")[-1]
        if t and t.lower() != "select":
            out.add(t)
    return out


def build_table_folds(rows: pd.DataFrame) -> List[Dict]:
    """Target-table folds from the gold table annotation.

    Units are gold target tables ordered by descending question count (then
    name); a row is *test* when its gold table is held out, and *v1-train*
    when its regex reference set is disjoint from the held-out targets.
    Test rows are explicitly excluded from training regardless of their
    reference set (on well-formed data — every SQL referencing its gold
    table — this changes nothing; it only guards malformed inputs where a
    test row's SQL would otherwise leak into its own fold's training set).
    Returns dicts with ``test_tables``/``test``/``train`` (row indices).
    """
    refs = [sql_ref_tables(s) for s in rows["sql"]]
    gold = [str(t).split(".")[-1] for t in rows["table"]]
    tables = sorted(set(gold))
    qcount = {t: sum(1 for g in gold if g == t) for t in tables}
    units = sorted(tables, key=lambda t: (-qcount[t], t))
    assign, _ = _greedy_assign(units, lambda t: qcount[t])
    folds = []
    for f in range(N_FOLDS):
        test_tables = {t for t in tables if assign[t] == f}
        test = sorted(i for i in range(len(rows)) if gold[i] in test_tables)
        test_set = set(test)
        folds.append({
            "test_tables": sorted(test_tables),
            "test": test,
            "train": sorted(i for i in range(len(rows))
                            if i not in test_set
                            and not (refs[i] & test_tables)),
        })
    return folds


def apply_ast_exclusion(table_folds: List[Dict], gold: Dict[str, Dict],
                        qids: List[str], exclude_unparseable: bool = True):
    """Refine table-fold training sets with AST full-gold isolation (v2).

    A training question is excluded when its AST physical-table set
    (``gold[qid]['gold_tables']``) intersects the fold's target tables, or —
    conservatively — when its SQL could not be AST-parsed at all. Returns a
    new list of fold dicts (``targets``/``train``/``test`` as qid lists) plus
    the per-fold exclusion bookkeeping.
    """
    out, excluded_log = [], []
    for f, fd in enumerate(table_folds):
        targets = set(fd["test_tables"])
        train_qids, excluded = [], []
        for i in fd["train"]:
            qid = qids[i]
            g = gold.get(qid)
            if g is None:
                excluded.append({"qid": qid, "reason": "missing_gold"})
                continue
            if exclude_unparseable and not g.get("ast_parsed"):
                excluded.append({"qid": qid, "reason": "unparseable_sql"})
                continue
            if set(g.get("gold_tables") or []) & targets:
                excluded.append({"qid": qid, "reason": "references_target_table"})
                continue
            train_qids.append(qid)
        test_qids = [qids[i] for i in fd["test"]]
        leak = sorted({q for q in train_qids
                       if set(gold.get(q, {}).get("gold_tables") or []) & targets})
        if leak:  # defensive: the loop above must already prevent this
            raise AssertionError(f"fold {f}: target-table leak {leak}")
        out.append({"fold": f, "targets": sorted(targets),
                    "train": train_qids, "test": test_qids})
        excluded_log.append({"fold": f, "excluded_qids": [e["qid"] for e in excluded],
                             "exclusion_reasons": excluded})
    return out, excluded_log


# --------------------------------------------------------------------------
# Split verification and manifest
# --------------------------------------------------------------------------

def verify_splits(rows: pd.DataFrame, q_folds, t_folds) -> Dict[str, bool]:
    """Protocol invariants; every check must hold before any fit."""
    n = len(rows)
    checks = {}
    all_test = sorted(i for f in q_folds for i in f)
    checks["question_coverage_complete"] = all_test == list(range(n))
    checks["question_exclusive"] = len(all_test) == n
    groups = build_dup_groups(rows["question"].tolist())
    fold_of = {i: f for f, idxs in enumerate(q_folds) for i in idxs}
    checks["dup_groups_no_cross"] = all(
        len({fold_of[i] for i in g}) == 1 for g in groups)

    gold_tables = [str(t).split(".")[-1] for t in rows["table"]]
    refs = [sql_ref_tables(s) for s in rows["sql"]]
    ok_exclusive = ok_iso = True
    seen_tables, all_t_test = {}, []
    for f, fd in enumerate(t_folds):
        for t in fd["test_tables"]:
            seen_tables[t] = f
        all_t_test.extend(fd["test"])
        if set(fd["train"]) & set(fd["test"]):
            ok_exclusive = False
        if any(refs[i] & set(fd["test_tables"]) for i in fd["train"]):
            ok_iso = False
    checks["table_single_fold"] = len(seen_tables) == len(set(gold_tables))
    checks["table_coverage_complete"] = sorted(all_t_test) == list(range(n))
    checks["table_train_test_exclusive"] = ok_exclusive
    checks["table_train_refs_isolated"] = ok_iso
    return checks


def build_manifest(rows: pd.DataFrame, q_folds, t_folds, ast_folds,
                   exclusions, checks) -> Dict:
    """Splits manifest (deterministic fold identity for run_meta)."""
    return {
        "n_rows": len(rows),
        "dup_groups": build_dup_groups(rows["question"].tolist()),
        "question_folds": {str(k): v for k, v in enumerate(q_folds)},
        "table_folds": {str(k): {"test_tables": fd["test_tables"],
                                 "test": fd["test"], "train": fd["train"]}
                        for k, fd in enumerate(t_folds)},
        "ast_refined_table_folds": {str(k): {"targets": fd["targets"],
                                             "train": fd["train"],
                                             "test": fd["test"]}
                                    for k, fd in enumerate(ast_folds)},
        "ast_exclusions": exclusions,
        "checks": checks,
    }


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def folds_from_manifest(manifest: Dict, qids: List[str], n_rows: int):
    """Rebuild qid-level folds from a saved manifest.

    Question folds are index lists; table folds use the AST-refined entries
    when present (falling back to the v1 folds without exclusions is NOT
    silent — it raises, because the v2 refinement is part of the protocol).
    """
    q_folds = [[qids[i] for i in manifest["question_folds"][str(k)]]
               for k in range(N_FOLDS)]
    if "ast_refined_table_folds" not in manifest:
        raise ValueError(
            "manifest lacks AST-refined table folds; rebuild folds with "
            "`terag_release make-folds` (the AST-based training exclusion is "
            "part of the table_v2 protocol)")
    t = manifest["ast_refined_table_folds"]
    t_folds = [{"fold": int(k), "targets": t[str(k)]["targets"],
                "train": t[str(k)]["train"], "test": t[str(k)]["test"]}
               for k in sorted(t, key=int)]
    if len(q_folds) != N_FOLDS or len(t_folds) != N_FOLDS:
        raise ValueError(f"expected {N_FOLDS} folds per protocol")
    return q_folds, t_folds


def assert_disjoint(train_qids: Sequence[str], test_qids: Sequence[str]):
    overlap = sorted(set(train_qids) & set(test_qids))
    if overlap:
        raise AssertionError(f"train/test overlap: {overlap}")


# --------------------------------------------------------------------------
# Manifest re-validation against the CURRENT data (R3)
# --------------------------------------------------------------------------

class ManifestValidationError(ValueError):
    """A loaded split manifest contradicts the current QA rows / gold.

    Raised before any fit/retrieve so a stale, edited, or malicious
    manifest can never reach execution. Cached ``checks`` booleans inside
    the manifest are never trusted; every invariant is recomputed.
    """


def validate_manifest(manifest: Dict, rows: pd.DataFrame,
                      gold: Dict[str, Dict], qids: List[str]) -> Dict:
    """Recompute every protocol invariant of a manifest on current data.

    Returns the recomputed checks dict (all True) or raises
    ManifestValidationError with a specific message. Validates:

    * row count and qid identity/order against the QA CSV,
    * question folds: valid indices, complete and exclusive coverage,
      duplicate-question groups never crossing folds,
    * table folds: valid indices, each target table in exactly one fold,
      complete test coverage, train/test disjointness,
    * AST isolation (recomputed from CURRENT gold): no training qid's
      gold tables intersect the fold's held-out targets, and training
      rows with unparseable SQL are conservatively excluded,
    * per-question test membership exactly once across each protocol.
    """
    errors: List[str] = []
    n = len(rows)
    if manifest.get("n_rows") != n:
        errors.append(f"n_rows {manifest.get('n_rows')} != {n} QA rows")

    q_folds = manifest.get("question_folds")
    if not q_folds or len(q_folds) != N_FOLDS:
        raise ManifestValidationError(
            f"question_folds must contain {N_FOLDS} folds")
    idx_folds = []
    for k in range(N_FOLDS):
        try:
            idx = [int(i) for i in q_folds[str(k)]]
        except (KeyError, TypeError, ValueError):
            errors.append(f"question fold {k}: invalid index list")
            continue
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            errors.append(f"question fold {k}: indices out of range {bad[:5]}")
            continue
        idx_folds.append(idx)
    if not errors and len(idx_folds) == N_FOLDS:
        covered = sorted(i for f in idx_folds for i in f)
        if covered != list(range(n)):
            errors.append("question folds do not cover every row exactly once")
        groups = build_dup_groups(rows["question"].tolist())
        fold_of = {i: f for f, idx in enumerate(idx_folds) for i in idx}
        crossing = [g for g in groups
                    if len({fold_of[i] for i in g if i in fold_of}) > 1]
        if crossing:
            errors.append(
                f"duplicate-question groups cross question folds: "
                f"{[[qids[i] for i in g] for g in crossing[:3]]}")

    t = manifest.get("ast_refined_table_folds")
    if not t or len(t) != N_FOLDS:
        raise ManifestValidationError(
            f"ast_refined_table_folds must contain {N_FOLDS} folds "
            f"(rebuild folds with `terag_release make-folds`)")
    # ---- target-table universe and exact fold/test binding -------------
    # The held-out target universe is RECOMPUTED from the current QA
    # annotation (`table` column, last dot-segment). The manifest's
    # declared targets must reproduce it exactly — an emptied, augmented,
    # or reshuffled target assignment is a protocol violation even when
    # train/test sets are internally consistent.
    def _norm_table(name):
        return str(name).split(".")[-1].strip().lower()

    orig_target = {}
    for i, raw in enumerate(rows["table"].tolist()):
        orig_target.setdefault(_norm_table(raw), []).append(qids[i])
    target_universe = set(orig_target)
    manifest_targets = set()
    for k in range(N_FOLDS):
        manifest_targets |= {_norm_table(x)
                             for x in t[str(k)].get("targets") or []}
    if manifest_targets != target_universe:
        missing = sorted(target_universe - manifest_targets)[:5]
        unknown = sorted(manifest_targets - target_universe)[:5]
        errors.append(
            "held-out target tables do not match the current QA "
            f"annotation (missing {missing}, unknown {unknown}; the "
            f"declared target universe cannot be emptied, extended, or "
            f"replaced)")
    seen_targets: Dict[str, int] = {}
    all_test: List[str] = []
    for k in range(N_FOLDS):
        fd = t[str(k)]
        raw_targets = list(fd["targets"])
        targets = {_norm_table(x) for x in raw_targets}
        for tab in targets:
            if tab in seen_targets and seen_targets[tab] != k:
                errors.append(f"target table {tab} in folds "
                              f"{seen_targets[tab]} and {k}")
            seen_targets[tab] = k
        train_qids, test_qids = list(fd["train"]), list(fd["test"])
        if len(set(train_qids)) != len(train_qids):
            errors.append(f"table fold {k}: duplicate train qids")
        if len(set(test_qids)) != len(test_qids):
            errors.append(f"table fold {k}: duplicate test qids")
        overlap = sorted(set(train_qids) & set(test_qids))
        if overlap:
            errors.append(f"table fold {k}: train/test overlap {overlap[:5]}")
        for qid in train_qids + test_qids:
            if qid not in gold:
                errors.append(f"table fold {k}: qid {qid} missing from gold")
        # Exact fold/test membership: a fold's test set is EXACTLY the
        # questions whose original target table is among the fold's
        # declared targets — swapping test sets between folds, stranding a
        # table's questions in another fold, or declaring targets without
        # their questions is rejected.
        expected_test = {q for tab in targets for q in orig_target.get(tab, [])}
        if set(test_qids) != expected_test:
            errors.append(
                f"table fold {k}: test set does not match its declared "
                f"target tables (missing "
                f"{sorted(expected_test - set(test_qids))[:5]}, unexpected "
                f"{sorted(set(test_qids) - expected_test)[:5]})")
        all_test.extend(test_qids)
        # Recomputed AST isolation with CURRENT gold — never trust cached
        # checks. Physical identity follows the data contract (last
        # dot-segment, case-insensitive); this cannot mask different
        # physical tables because a single database cannot hold two tables
        # differing only by case.
        for qid in train_qids:
            g = gold.get(qid)
            if g is None:
                continue  # reported above
            if not g.get("ast_parsed"):
                errors.append(
                    f"table fold {k}: train qid {qid} has unparseable SQL "
                    f"(conservative exclusion rule requires removal)")
                continue
            refs = {_norm_table(x) for x in (g.get("gold_tables") or [])}
            hit = sorted(refs & targets)
            if hit:
                errors.append(
                    f"table fold {k}: train qid {qid} references held-out "
                    f"target table(s) {hit}")
        # Training membership is fully determined by the protocol: every
        # qid that is neither this fold's test row nor conservatively
        # excluded (unparseable SQL, or gold tables intersecting the
        # targets — both recomputed from CURRENT gold) MUST be present. A
        # silently dropped training row would shrink supervision; an extra
        # one would add it.
        must_exclude = {
            q for q in qids
            if q in gold and (
                not gold[q].get("ast_parsed")
                or {_norm_table(x) for x in (gold[q].get("gold_tables")
                                             or [])} & targets)}
        expected_train = set(qids) - set(test_qids) - must_exclude
        actual_train = set(train_qids)
        if actual_train != expected_train:
            errors.append(
                f"table fold {k}: training set does not match the protocol "
                f"membership (missing "
                f"{sorted(expected_train - actual_train)[:5]}, unexpected "
                f"{sorted(actual_train - expected_train)[:5]})")
    if len(all_test) != len(set(all_test)):
        errors.append("a question is tested in more than one table fold")
    expected = set(qids)
    if set(all_test) != expected:
        errors.append("table folds do not cover every question exactly once "
                      f"(missing {sorted(expected - set(all_test))[:5]}, "
                      f"unexpected {sorted(set(all_test) - expected)[:5]})")

    if errors:
        raise ManifestValidationError(
            "split manifest failed revalidation on the current data:\n  - "
            + "\n  - ".join(errors))
    return {
        "question_coverage_complete": True,
        "question_exclusive": True,
        "dup_groups_no_cross": True,
        "table_single_fold": True,
        "table_coverage_complete": True,
        "table_train_test_exclusive": True,
        "table_ast_isolation_recomputed": True,
        "table_target_universe_matches_qa": True,
        "table_test_sets_match_declared_targets": True,
        "recomputed_at_load": True,
    }
