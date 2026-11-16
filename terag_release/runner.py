"""Experiment runner: fit, retrieve, save predictions, summarize.

This is the complete "user data -> saved results" pipeline (path A of the
release plan). Per (protocol, fold, variant) unit it:

1. builds the training frame (supervision injected for history-consuming
   methods, NEVER for test frames),
2. constructs a fresh retriever from the resolved configuration,
3. fits it (wall-clock timed) — with optional per-fold diagnostic export for
   TE-RAG variants (B/C evidence; disabled by default, never changes
   outputs),
4. evaluates every test question (per-question latency recorded; retrieval
   exceptions become ``status=error`` records that stay in the denominator),
5. stamps and appends prediction records to ``predictions.jsonl``,
6. summarizes with the full-gold metric engine and the paper's denominators
   (valid = ``verified_complete`` gold; feasible = gold-derived budget).

The prediction record schema matches the paper's saved artifacts so the
offline evaluation (``terag_release evaluate`` /
``experiment.audit_saved_predictions``) and the analysis modules consume both
identically.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from terag_release import protocols, reporting, supervision, variants
from terag_release.config import (build_terag_config, config_sha256,
                                  load_base, load_dataset)


# --------------------------------------------------------------------------
# Data / gold loading
# --------------------------------------------------------------------------

def load_rows(qa_csv: str, qid_prefix: str = "q") -> pd.DataFrame:
    """Load QA rows and assign positional qids ``<prefix>000...``."""
    rows = pd.read_csv(qa_csv)
    required = {"question", "sql", "table"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"QA CSV lacks required columns: {sorted(missing)}")
    rows["qid"] = [f"{qid_prefix}{i:03d}" for i in range(len(rows))]
    return rows


def load_gold(gold_path: str) -> Dict[str, Dict]:
    gold = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                gold[rec["qid"]] = rec
    return gold


def gold_populations(gold: Dict[str, Dict], max_tables: int = 5,
                     max_columns_per_table: int = 10):
    """Valid (complete gold) and budget-feasible qid sets, derived from gold.

    Budget feasibility depends only on gold: <= ``max_tables`` distinct gold
    tables and <= ``max_columns_per_table`` gold columns on every gold table.
    These populations are dataset properties; nothing is hardcoded.
    """
    valid = {q for q, g in gold.items()
             if g.get("gold_status") == "verified_complete"}
    feasible = set()
    detail = []
    for q in sorted(valid):
        g = gold[q]
        tables = sorted({t.split(".")[-1].lower() for t in g["gold_tables"]})
        per_table: Dict[str, int] = {}
        for c in g["gold_columns"]:
            t = c.rsplit(".", 1)[0]
            per_table[t] = per_table.get(t, 0) + 1
        ok = (len(tables) <= max_tables
              and max(per_table.values(), default=0) <= max_columns_per_table)
        if ok:
            feasible.add(q)
        detail.append({"qid": q, "n_gold_tables": len(tables),
                       "max_cols_per_table": max(per_table.values(), default=0),
                       "budget_feasible": ok})
    return valid, feasible, {
        "rule": ("gold tables <= max_tables and gold columns per table "
                 "<= max_columns_per_table (gold-only; predictions never "
                 "influence feasibility)"),
        "max_tables": max_tables,
        "max_columns_per_table": max_columns_per_table,
        "n_verified": len(valid), "n_feasible": len(feasible),
        "infeasible_qids": sorted(valid - feasible),
        "per_question": detail,
    }


# --------------------------------------------------------------------------
# Retriever construction
# --------------------------------------------------------------------------

def create_retriever(variant: str, dataset_cfg: Dict, base_sections: Dict,
                     artifacts_dir: str):
    """Create a fresh retriever instance for one (variant, fold) unit."""
    from retrievers.base_retriever import RetrieverConfig
    from retrievers.retriever_factory import RetrieverFactory

    spec = variants.VARIANTS[variant]
    factory = spec["factory"]
    if factory == "TE-RAG-V2" or variant == "MetadataMatched":
        sections = variants.variant_config(base_sections, variant)
        cfg = build_terag_config(sections, dataset_cfg["data"], artifacts_dir)
        extra = {"terag_config": cfg}
        if variant == "MetadataMatched":
            extra.update({"use_history": False, "use_transfer": False})
        rc = RetrieverConfig(name=factory, extra_params=extra)
    else:
        extra = {}
        if spec.get("fusion_mode"):
            extra["fusion_mode"] = spec["fusion_mode"]
        rc = RetrieverConfig(name=factory, extra_params=extra)
    return RetrieverFactory.create(factory, dataset_cfg["data"]["field_csv"],
                                   dataset_cfg["data"]["table_csv"], config=rc)


# --------------------------------------------------------------------------
# Unit execution
# --------------------------------------------------------------------------

def run_unit(variant: str, setting: str, fold: int, train_df, test_df,
             test_qids: List[str], gold: Dict[str, Dict], dataset_cfg: Dict,
             base_sections: Dict, artifacts_dir: str, diagnostics=None):
    """Fit + evaluate one (protocol, fold, variant) unit.

    Returns (run_status entry, prediction records).
    """
    spec = variants.VARIANTS[variant]
    entry = {"setting": setting, "fold": fold, "variant": variant,
             "factory": spec["factory"], "status": "pending",
             "fit_time_s": None, "avg_query_time_s": None,
             "n_test": len(test_qids), "error": None}
    records: List[Dict] = []
    retriever = None
    try:
        retriever = create_retriever(variant, dataset_cfg, base_sections,
                                     artifacts_dir)
        train_frame = train_df
        if spec["supervision"]:
            train_frame = supervision.inject_supervision(
                train_df, gold, spec["supervision"])
        t0 = time.time()
        if spec["factory"] == "TE-RAG-V2" and diagnostics is not None:
            retriever.fit(train_frame, diagnostics=diagnostics)
        else:
            retriever.fit(train_frame)
        entry["fit_time_s"] = round(time.time() - t0, 3)

        inner = getattr(retriever, "terag_retriever", retriever)
        miner = getattr(inner, "miner", None)
        if miner is not None:
            entry["n_b_terms"] = sum(
                1 for a in miner.annotations
                if a["type"] == "TERM" and a.get("status") == "confirmed")
        transfer = getattr(inner, "transfer_result", None)
        if transfer is not None:
            entry["n_c_accepted"] = transfer["provenance"]["n_accepted"]

        detailed = retriever.evaluate_detailed(test_df, k=5)
        for rec, qid in zip(detailed, test_qids):
            # Stamp identity fields FIRST so downstream consumers never see a
            # record without them.
            rec["qid"] = qid
            rec["setting"] = setting
            rec["fold"] = fold
            rec["variant"] = variant
            rec["method"] = spec["factory"]
            records.append(rec)
        if detailed:
            entry["avg_query_time_s"] = round(
                float(sum(r["query_time"] for r in detailed) / len(detailed)), 6)
        entry["status"] = "completed"
    except Exception as e:  # method-level failure is recorded, never silent
        entry["status"] = "failed"
        entry["error"] = f"{type(e).__name__}: {str(e)[:800]}"
    return entry, records


def run_experiments(dataset_cfg: Dict, resolved: Dict, base_sections: Dict,
                    run_dir: Path, variant_names: List[str],
                    protocol_names: List[str],
                    fold_override: Optional[int] = None,
                    diagnostics: bool = False) -> Dict:
    """Execute a method x protocol matrix; write predictions + summaries.

    ``resolved`` is the fully merged configuration (for run_meta).
    Returns the run_meta dict (also written to disk).
    """
    from eval.full_gold import summarize_predictions
    from terag.diagnostics import DiagnosticExport

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = reporting.require_new_directory(run_dir)
    meta_dir = run_dir / "run_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dataset_cfg["data"]
    rows = load_rows(data_cfg["qa_csv"],
                     qid_prefix=resolved.get("qid_prefix", "q"))
    qids = rows["qid"].tolist()
    gold = load_gold(data_cfg["gold"])

    max_tables = resolved.get("budget", {}).get("max_tables", 5)
    max_cols = resolved.get("budget", {}).get("max_columns_per_table", 10)
    valid, feasible, budget_info = gold_populations(
        gold, max_tables, max_cols)

    # Folds: frozen manifest when supplied, otherwise deterministic rebuild.
    splits_path = data_cfg.get("splits_manifest")
    if splits_path and Path(splits_path).exists():
        manifest = protocols.load_manifest(splits_path)
        q_folds, t_folds = protocols.folds_from_manifest(manifest, qids,
                                                         len(rows))
    else:
        q_folds_idx = protocols.build_question_folds(rows)
        t_folds_v1 = protocols.build_table_folds(rows)
        ast_folds, exclusions = protocols.apply_ast_exclusion(
            t_folds_v1, gold, qids)
        checks = protocols.verify_splits(rows, q_folds_idx, t_folds_v1)
        manifest = protocols.build_manifest(rows, q_folds_idx, t_folds_v1,
                                            ast_folds, exclusions, checks)
        q_folds = [[qids[i] for i in idx] for idx in q_folds_idx]
        t_folds = ast_folds
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise AssertionError(f"split invariants failed: {failed}")
    manifest_digest = reporting.json_bytes(manifest)
    import hashlib
    manifest_sha = hashlib.sha256(manifest_digest).hexdigest()

    run_meta = reporting.new_run_meta("run-experiments", resolved, repo_root)
    run_meta["data"] = {
        "inputs": reporting.input_checksums(
            [data_cfg["field_csv"], data_cfg["table_csv"],
             data_cfg["qa_csv"], data_cfg["gold"]]),
        "n_rows": len(rows), "qid_prefix": resolved.get("qid_prefix", "q"),
        "gold_populations": {
            "n_verified": len(valid), "n_feasible": len(feasible),
            "budget": budget_info["rule"],
        },
        "splits_manifest_sha256": manifest_sha,
    }
    reporting.write_json_exclusive(meta_dir / "splits_manifest.json",
                                   manifest)
    reporting.write_json_exclusive(meta_dir / "budget_feasibility.json",
                                   budget_info)

    pred_path = run_dir / "predictions.jsonl"
    statuses: List[Dict] = []
    summaries = []
    with open(pred_path, "x", encoding="utf-8") as pred_file:
        for setting, fold_sets in (("question_v1", q_folds),
                                   ("table_v2", t_folds)):
            if setting not in protocol_names:
                continue
            for fold, fold_qids in enumerate(fold_sets):
                if fold_override is not None and fold != fold_override:
                    continue
                if setting == "question_v1":
                    test_qids = list(fold_qids)
                    train_qids = [q for q in qids if q not in set(test_qids)]
                else:
                    test_qids = list(fold_qids["test"])
                    train_qids = list(fold_qids["train"])
                protocols.assert_disjoint(train_qids, test_qids)
                train_df = rows[rows["qid"].isin(train_qids)].reset_index(
                    drop=True)
                test_df = rows[rows["qid"].isin(test_qids)].reset_index(
                    drop=True)
                for variant in variant_names:
                    if setting not in variants.VARIANTS[variant]["protocols"]:
                        continue  # not run for this protocol in the paper
                    art = run_dir / "artifacts" / setting / f"fold{fold}" / variant
                    diag = None
                    if diagnostics and variants.VARIANTS[variant][
                            "factory"] == "TE-RAG-V2":
                        diag_dir = run_dir / "diagnostics" / setting / \
                            f"fold{fold}" / variant
                        # DiagnosticExport insists on creating a NEW leaf
                        # directory itself; create only the parents here.
                        diag_dir.parent.mkdir(parents=True, exist_ok=True)
                        diag = DiagnosticExport(str(diag_dir),
                                                run_id="public-release",
                                                protocol=setting, fold=fold)
                    entry, records = run_unit(
                        variant, setting, fold, train_df, test_df, test_qids,
                        gold, dataset_cfg, base_sections, str(art),
                        diagnostics=diag)
                    statuses.append(entry)
                    for rec in records:
                        pred_file.write(json.dumps(
                            rec, ensure_ascii=False, default=str) + "\n")
                    pred_file.flush()
                    print(f"[runner] {setting}/fold{fold}/{variant}: "
                          f"{entry['status']}"
                          + (f" ({entry['error']})" if entry["error"] else ""))

    # Offline summary with the full-gold engine (same code path as
    # `terag_release evaluate`). Full-population summaries require every
    # fold; a --fold override run is partial and gets an explicit note
    # instead of silently shrunken denominators.
    records_by_group: Dict[tuple, Dict[str, Dict]] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records_by_group.setdefault((rec["setting"], rec["variant"]),
                                        {})[rec["qid"]] = rec
    if fold_override is None:
        for (setting, variant), pool in sorted(records_by_group.items()):
            row = {"setting": setting, "variant": variant,
                   "source_file": str(pred_path)}
            row.update(summarize_predictions(list(pool.values()), gold,
                                             sorted(valid), sorted(feasible)))
            summaries.append(row)
        if summaries:
            import csv
            with open(run_dir / "summary_full_gold.csv", "x", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
    else:
        run_meta["execution"]["notes"].append(
            f"partial run (fold {fold_override} only): full-population "
            f"summaries intentionally NOT computed; predictions are saved "
            f"for diagnostics/benchmarking only")

    run_meta["execution"]["units"] = statuses
    run_meta["execution"]["summary"] = summaries
    run_meta["execution"]["notes"].append(
        "fit/query timings are NEW measurements of the public build; they are "
        "not the paper's original-run efficiency numbers")
    reporting.finalize_run_meta(run_meta, run_dir)
    return run_meta
