"""Analysis dispatch for `python -m terag_release analyze`."""

import json
from pathlib import Path

from terag_release import reporting
from terag_release.runner import gold_populations, load_gold


def _populations(gold_path: str, budget_path=None):
    gold = load_gold(gold_path)
    if budget_path:
        budget = json.loads(Path(budget_path).read_text(encoding="utf-8"))
        valid = {x["qid"] for x in budget.get("per_question", [])
                 if x.get("budget_feasible") is not None and
                 gold[x["qid"]]["gold_status"] == "verified_complete"}
        # valid population comes from gold, not from the budget file
        valid = {q for q, g in gold.items()
                 if g["gold_status"] == "verified_complete"}
        feasible = {x["qid"] for x in budget.get("per_question", [])
                    if x.get("budget_feasible")}
    else:
        valid, feasible, budget = gold_populations(gold)
    return gold, valid, feasible


def run_analysis(name: str, run_dir: str, gold: str, budget, output: str,
                 method_a: str, method_b: str):
    out = reporting.require_new_directory(output)
    run_path = Path(run_dir)

    # Gold defaults to the run's dataset when not supplied.
    if gold is None:
        meta = json.loads((run_path / "run_meta" / "run_meta.json")
                          .read_text(encoding="utf-8"))
        gold = meta["config"]["resolved"]["dataset"]["data"]["gold"]

    if name == "saved-prefix":
        from analysis import saved_prefix
        _, valid, feasible = _populations(gold, budget)
        feasible_by_budget = {}
        from analysis.saved_prefix import eligible
        g = load_gold(gold)
        for b in saved_prefix.BUDGETS:
            feasible_by_budget[b] = {q for q in valid if eligible(g[q], b)}
        methods = sorted({v for s, v in saved_prefix.load_run(run_dir)})
        path = saved_prefix.analyze(run_dir, gold, out, methods,
                                    valid, feasible_by_budget)
    elif name == "heldout":
        from analysis import heldout_failures
        _, valid, _ = _populations(gold, budget)
        # The paper's third method for this analysis is the noGraph variant;
        # include it only when the run actually contains that group.
        from analysis.saved_prefix import load_run
        method_c = ("TERAG-noGraph"
                    if ("table_v2", "TERAG-noGraph") in load_run(run_dir)
                    else None)
        path = heldout_failures.analyze(run_dir, gold, out,
                                        method_a=method_a, method_b=method_b,
                                        method_c=method_c, valid_qids=valid)
    elif name == "transfer":
        from analysis import transfer
        path = transfer.analyze(run_dir, out)
    elif name == "isolation":
        from terag_release import protocols
        manifest = json.loads((run_path / "run_meta" / "splits_manifest.json")
                              .read_text(encoding="utf-8"))
        gold_map = load_gold(gold)
        leaks = []
        for fold in manifest["ast_refined_table_folds"].values():
            targets = set(fold["targets"])
            for qid in fold["train"]:
                refs = set(gold_map.get(qid, {}).get("gold_tables") or [])
                hit = refs & targets
                if hit:
                    leaks.append({"qid": qid, "targets": sorted(targets),
                                  "referenced": sorted(hit)})
        report = {"n_folds": len(manifest["ast_refined_table_folds"]),
                  "train_leaks": leaks,
                  "checks": manifest.get("checks"),
                  "ast_exclusions": manifest.get("ast_exclusions")}
        reporting.write_json_exclusive(out / "isolation_report.json", report)
        path = out / "isolation_report.json"
    else:
        raise ValueError(f"unknown analysis: {name}")
    print(f"analysis '{name}' written under {out} (start: {path})")
