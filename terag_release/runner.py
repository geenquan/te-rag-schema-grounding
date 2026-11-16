"""Experiment runner: preflight, fit, retrieve, save predictions, summarize.

This is the complete "user data -> saved results" pipeline. Per
(protocol, fold, variant) unit it:

1. builds the training frame (``full_sql`` supervision injected for
   history-consuming methods — including KaSLA-Adapted, whose classifiers
   and capacity estimation consume the structured full gold exactly as in
   the paper runner — and NEVER for test frames),
2. constructs a fresh retriever FROM THE RESOLVED CONFIGURATION (method
   YAML parameters actually reach the constructors; the resolved config
   and its SHA256 cover every consumed value),
3. fits it (wall-clock timed) — with optional per-fold diagnostic export
   for TE-RAG variants (B/C evidence; disabled by default, never changes
   outputs),
4. evaluates every test question, aligned to test qids BY QID (never by
   positional zip); retrieval exceptions become ``status=error`` records
   that stay in the denominator,
5. on construction/fit failure, emits one ``status=error`` record per test
   question (empty outputs, no fabricated timings) so denominators and
   coverage stay intact, and marks the unit failed,
6. summarizes with the full-gold metric engine — only for complete runs —
   and writes run_meta whatever happened.

Run statuses: ``completed`` (all units ok), ``completed_with_failures``
(some units failed — CLI exits non-zero), ``partial`` (single-fold run),
``failed`` (nothing executed successfully). External split manifests are
re-validated against the CURRENT QA/gold before anything runs; cached
``checks`` inside a manifest are never trusted.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from terag_release import protocols, reporting, supervision, variants
from terag_release.config import build_terag_config, config_sha256

ADAPTER_FACTORIES = ("LinkAlign-SingleDB-Adapted", "KaSLA-Adapted")


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
                if rec["qid"] in gold:
                    raise ValueError(f"duplicate qid in gold: {rec['qid']}")
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
# Retriever construction — consumes the RESOLVED method configuration
# --------------------------------------------------------------------------

def create_retriever(variant: str, dataset_cfg: Dict, method_entry: Dict,
                     artifacts_dir: str, cache_dir: Optional[str] = None):
    """Create a retriever for one unit from its resolved effective config."""
    from retrievers.base_retriever import RetrieverConfig
    from retrievers.retriever_factory import RetrieverFactory

    spec = variants.VARIANTS[variant]
    factory = spec["factory"]
    effective = method_entry["effective"]
    if "terag_sections" in effective:
        cfg = build_terag_config(effective["terag_sections"],
                                 dataset_cfg["data"], artifacts_dir)
        extra = {"terag_config": cfg}
        if variant == "MetadataMatched":
            extra.update({"use_history": False, "use_transfer": False})
        rc = RetrieverConfig(name=factory, extra_params=extra)
    else:
        params = dict(effective["retriever_params"])
        extra = dict(effective.get("extra_params") or {})
        rc = RetrieverConfig(name=factory, k1=params.get("k1", 1.5),
                             b=params.get("b", 0.75),
                             top_k=params.get("top_k", 5),
                             use_chinese_tokenizer=params.get(
                                 "use_chinese_tokenizer", True),
                             extra_params=extra)
    kwargs = {}
    if factory in ADAPTER_FACTORIES:
        # The two adapted retrievers share a run-level API-response cache,
        # mirroring the paper runner's single shared cache directory. It
        # lives inside the run directory; nothing is written to the CWD.
        kwargs["cache_dir"] = str(cache_dir)
    return RetrieverFactory.create(factory, dataset_cfg["data"]["field_csv"],
                                   dataset_cfg["data"]["table_csv"],
                                   config=rc, **kwargs)


def effective_params_snapshot(variant: str, method_entry: Dict) -> Dict:
    """JSON-safe snapshot of what the constructor actually received."""
    effective = method_entry["effective"]
    if "terag_sections" in effective:
        return {"kind": "terag_config",
                "sections": effective["terag_sections"],
                "config_file": effective.get("config_file")}
    return {"kind": "retriever_config",
            "retriever": effective["retriever_params"],
            "extra_params": effective.get("extra_params") or {},
            "config_file": effective.get("config_file")}


# --------------------------------------------------------------------------
# Unit execution
# --------------------------------------------------------------------------

def _error_records(variant, setting, fold, test_qids, rows_by_qid,
                   stage: str, error: str) -> List[Dict]:
    """Per-question error records for a construct/fit failure.

    They keep the questions in every denominator with empty outputs and no
    fabricated timings; ``error_stage`` distinguishes them from per-query
    retrieval errors.
    """
    spec = variants.VARIANTS[variant]
    records = []
    for qid in test_qids:
        records.append({
            "qid": qid, "setting": setting, "fold": fold,
            "variant": variant, "method": spec["factory"],
            "question": rows_by_qid[qid]["question"],
            "status": "error", "error_type": "MethodExecutionError",
            "error_message": f"{stage}: {error}"[:500],
            "error_stage": stage,
            "retrieved_tables": [], "retrieved_column_keys": [],
            "retrieved_columns_detail": [],
            "query_time": None,
            "timing_note": "no query executed; timing not measured",
            "exec_accuracy": "not_run",
        })
    return records


def run_unit(variant: str, setting: str, fold: int, train_df, test_df,
             test_qids: List[str], gold: Dict[str, Dict], dataset_cfg: Dict,
             method_entry: Dict, artifacts_dir: str,
             cache_dir: Optional[str] = None, diagnostics=None):
    """Fit + evaluate one (protocol, fold, variant) unit.

    Returns (run_status entry, prediction records). ``test_df`` MUST be
    ordered exactly like ``test_qids`` (the runner guarantees this); the
    unit asserts the question/qid correspondence rather than trusting zip.
    """
    spec = variants.VARIANTS[variant]
    entry = {"setting": setting, "fold": fold, "variant": variant,
             "factory": spec["factory"], "status": "pending",
             "stage": None, "fit_time_s": None, "avg_query_time_s": None,
             "n_test": len(test_qids), "error": None,
             "n_attempted": 0, "n_retrieval_errors": 0,
             "n_legal_empty": 0, "n_succeeded": 0,
             "effective_params": effective_params_snapshot(
                 variant, method_entry)}
    records: List[Dict] = []
    rows_by_qid = dict(zip(test_df["qid"].tolist(),
                           test_df.to_dict("records")))
    try:
        if list(test_df["qid"]) != list(test_qids):
            raise RuntimeError("internal misalignment: test frame order "
                               "does not match test qids")
        retriever = create_retriever(variant, dataset_cfg, method_entry,
                                     artifacts_dir, cache_dir=cache_dir)
        entry["stage"] = "fit"
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

        entry["stage"] = "retrieve"
        detailed = retriever.evaluate_detailed(test_df, k=5)
        if len(detailed) != len(test_qids):
            raise RuntimeError(
                f"evaluator returned {len(detailed)} records for "
                f"{len(test_qids)} questions")
        for rec, qid in zip(detailed, test_qids):
            if rec.get("question") != rows_by_qid[qid]["question"]:
                raise RuntimeError(
                    f"question/qid misalignment at {qid}: evaluator order "
                    f"does not match the aligned test frame")
            rec["qid"] = qid
            rec["setting"] = setting
            rec["fold"] = fold
            rec["variant"] = variant
            rec["method"] = spec["factory"]
            records.append(rec)
        # Aggregate per-question retrieval outcomes. A retrieval exception
        # (status=error) is a FAILURE of the unit's execution, distinct
        # from a legal empty prediction (status=ok, no tables); neither
        # may silently count as full success.
        entry["n_attempted"] = len(detailed)
        entry["n_retrieval_errors"] = sum(
            1 for r in detailed if r.get("status") == "error")
        entry["n_legal_empty"] = sum(
            1 for r in detailed if r.get("status") == "ok"
            and not r.get("retrieved_tables"))
        entry["n_succeeded"] = (entry["n_attempted"]
                                - entry["n_retrieval_errors"])
        if detailed:
            entry["avg_query_time_s"] = round(
                float(sum(r["query_time"] for r in detailed)
                      / len(detailed)), 6)
        if entry["n_retrieval_errors"] == entry["n_attempted"]:
            entry["status"] = "failed"
            entry["error"] = (f"all {entry['n_attempted']} retrievals "
                              f"failed (per-question error records kept)")
        elif entry["n_retrieval_errors"]:
            entry["status"] = "completed_with_errors"
            entry["error"] = (f"{entry['n_retrieval_errors']}/"
                              f"{entry['n_attempted']} retrievals failed "
                              f"(per-question error records kept)")
        else:
            entry["status"] = "completed"
        entry["stage"] = None
    except Exception as e:  # method-level failure is recorded, never silent
        stage = entry["stage"] or "construct"
        entry["status"] = "failed"
        entry["stage"] = stage
        entry["error"] = f"{type(e).__name__}: {str(e)[:800]}"
        records = _error_records(variant, setting, fold, test_qids,
                                 rows_by_qid, stage, entry["error"])
    return entry, records


# --------------------------------------------------------------------------
# Run orchestration
# --------------------------------------------------------------------------

def _ordered_frame(rows: pd.DataFrame, qids: List[str]) -> pd.DataFrame:
    """Frame filtered to ``qids`` and ordered EXACTLY like ``qids``."""
    frame = rows.set_index("qid").loc[list(qids)].reset_index()
    return frame


def run_experiments(resolved: Dict, run_dir: Path,
                    fold_override: Optional[int] = None,
                    diagnostics: bool = False) -> Dict:
    """Execute a method x protocol matrix; write predictions + summaries.

    ``resolved`` is the output of ``terag_release.config.resolve_run`` (its
    canonical SHA256 is the run's configuration identity). Returns the
    run_meta dict (always written to disk, even on failure).
    """
    from eval.full_gold import summarize_predictions
    from terag.diagnostics import DiagnosticExport

    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = resolved["dataset"]
    data_cfg = dataset_cfg["data"]
    qid_prefix = resolved.get("qid_prefix", "q")
    protocol_names = resolved["experiment"]["protocols"]
    variant_names = resolved["experiment"]["methods"]

    # ---------------- preflight (no run dir, no fit on failure) ----------
    for key in ("field_csv", "table_csv", "qa_csv", "gold"):
        if not data_cfg.get(key) or not Path(data_cfg[key]).exists():
            raise FileNotFoundError(
                f"dataset input '{key}' missing or not found: "
                f"{data_cfg.get(key)}")
    rows = load_rows(data_cfg["qa_csv"], qid_prefix=qid_prefix)
    qids = rows["qid"].tolist()
    gold = load_gold(data_cfg["gold"])
    missing_gold = [q for q in qids if q not in gold]
    if missing_gold:
        raise ValueError(
            f"gold lacks records for {len(missing_gold)} QA rows (e.g. "
            f"{missing_gold[:3]}); rebuild gold with the same qid prefix "
            f"('{qid_prefix}') and row order")

    planned = []
    seen_units = set()
    if fold_override is not None and not 0 <= fold_override < protocols.N_FOLDS:
        raise ValueError(f"invalid --fold {fold_override} (0.."
                         f"{protocols.N_FOLDS - 1})")
    for setting in protocol_names:
        for variant in variant_names:
            if setting not in variants.VARIANTS[variant]["protocols"]:
                continue
            for fold in range(protocols.N_FOLDS):
                if fold_override is not None and fold != fold_override:
                    continue
                unit = (setting, fold, variant)
                if unit in seen_units:
                    raise ValueError(f"duplicate execution unit planned: {unit}")
                seen_units.add(unit)
                planned.append(unit)
    if not planned:
        raise ValueError(
            "empty execution plan: no (protocol, fold, variant) unit matches "
            "the requested methods/protocols/fold")

    max_tables = resolved["budget"]["max_tables"]
    max_cols = resolved["budget"]["max_columns_per_table"]
    valid, feasible, budget_info = gold_populations(gold, max_tables, max_cols)

    # Folds: external manifest (REVALIDATED against current data) or the
    # deterministic rebuild. A configured-but-missing manifest is an error,
    # never a silent rebuild.
    splits_path = data_cfg.get("splits_manifest")
    if splits_path:
        if not Path(splits_path).exists():
            raise FileNotFoundError(
                f"configured splits_manifest not found: {splits_path} "
                f"(remove the setting to rebuild folds, or point it at an "
                f"existing manifest)")
        manifest = protocols.load_manifest(splits_path)
        protocols.validate_manifest(manifest, rows, gold, qids)  # raises
        q_folds, t_folds = protocols.folds_from_manifest(manifest, qids,
                                                         len(rows))
    else:
        q_folds_idx = protocols.build_question_folds(rows)
        t_folds_v1 = protocols.build_table_folds(rows)
        ast_folds, exclusions = protocols.apply_ast_exclusion(
            t_folds_v1, gold, qids)
        checks = protocols.verify_splits(rows, q_folds_idx, t_folds_v1)
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise AssertionError(f"split invariants failed: {failed}")
        manifest = protocols.build_manifest(rows, q_folds_idx, t_folds_v1,
                                            ast_folds, exclusions, checks)
        q_folds = [[qids[i] for i in idx] for idx in q_folds_idx]
        t_folds = ast_folds

    # ---------------- run directory + provenance -------------------------
    run_dir = reporting.require_new_directory(run_dir)
    meta_dir = run_dir / "run_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    external_cache = run_dir / "external_cache"

    run_meta = reporting.new_run_meta("run-experiments", resolved, repo_root)
    run_meta["data"] = {
        "inputs": reporting.input_checksums(
            [data_cfg["field_csv"], data_cfg["table_csv"],
             data_cfg["qa_csv"], data_cfg["gold"]]),
        "n_rows": len(rows), "qid_prefix": qid_prefix,
        "gold_populations": {
            "n_verified": len(valid), "n_feasible": len(feasible),
            "budget": budget_info["rule"],
        },
        "splits_manifest_sha256": reporting.sha256_of_bytes(
            reporting.json_bytes(manifest)),
        "splits_source": ("external:" + str(splits_path) if splits_path
                          else "deterministic_rebuild"),
    }
    run_meta["execution"]["planned_units"] = [
        {"setting": s, "fold": f, "variant": v} for s, f, v in planned]
    run_meta["execution"]["skipped_units"] = [
        {"setting": setting, "variant": v,
         "reason": "not in the paper's protocol matrix for this method"}
        for setting in protocol_names for v in variant_names
        if setting not in variants.VARIANTS[v]["protocols"]]
    reporting.write_json_exclusive(meta_dir / "splits_manifest.json", manifest)
    reporting.write_json_exclusive(meta_dir / "budget_feasibility.json",
                                   budget_info)

    # ---------------- execution ------------------------------------------
    pred_path = run_dir / "predictions.jsonl"
    statuses: List[Dict] = []
    summaries = []
    summary_error = None
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
                train_df = _ordered_frame(rows, train_qids)
                test_df = _ordered_frame(rows, test_qids)
                for variant in variant_names:
                    if setting not in variants.VARIANTS[variant]["protocols"]:
                        continue
                    art = run_dir / "artifacts" / setting / f"fold{fold}" / variant
                    diag = None
                    if diagnostics and variants.VARIANTS[variant][
                            "factory"] == "TE-RAG-V2":
                        diag_dir = run_dir / "diagnostics" / setting / \
                            f"fold{fold}" / variant
                        diag_dir.parent.mkdir(parents=True, exist_ok=True)
                        diag = DiagnosticExport(str(diag_dir),
                                                run_id="public-release",
                                                protocol=setting, fold=fold)
                    entry, records = run_unit(
                        variant, setting, fold, train_df, test_df, test_qids,
                        gold, dataset_cfg, resolved["methods"][variant],
                        str(art), cache_dir=str(external_cache),
                        diagnostics=diag)
                    statuses.append(entry)
                    for rec in records:
                        pred_file.write(json.dumps(
                            rec, ensure_ascii=False, default=str) + "\n")
                    pred_file.flush()
                    print(f"[runner] {setting}/fold{fold}/{variant}: "
                          f"{entry['status']}"
                          + (f" ({entry['error']})" if entry["error"] else ""))

    # ---------------- coverage validation + summary ----------------------
    written: Dict[tuple, Dict[str, Dict]] = {}
    duplicate_qids = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["setting"], rec["variant"])
            bucket = written.setdefault(key, {})
            if rec["qid"] in bucket:
                duplicate_qids.append((key, rec["qid"]))
            bucket[rec["qid"]] = rec
    coverage_problems = []
    if duplicate_qids:
        coverage_problems.append(f"duplicate prediction qids: "
                                 f"{duplicate_qids[:3]}")
    for setting, fold, variant in planned:
        fold_qids = (q_folds if setting == "question_v1" else t_folds)[fold]
        expected = (list(fold_qids) if setting == "question_v1"
                    else list(fold_qids["test"]))
        unit_qids = set(written.get((setting, variant), {}))
        if set(expected) - unit_qids:
            coverage_problems.append(
                f"{setting}/fold{fold}/{variant}: missing predictions for "
                f"{sorted(set(expected) - unit_qids)[:3]}")

    failed_units = [u for u in statuses if u["status"] != "completed"]
    run_status = "completed"
    if failed_units:
        run_status = ("failed" if len(failed_units) == len(planned)
                      else "completed_with_failures")
    elif fold_override is not None:
        run_status = "partial"
    run_meta["execution"]["aggregate"] = {
        "units_planned": len(planned),
        "units_completed": sum(u["status"] == "completed" for u in statuses),
        "units_with_retrieval_errors":
            sum(u["status"] == "completed_with_errors" for u in statuses),
        "units_failed": sum(u["status"] == "failed" for u in statuses),
        "retrieval_errors_total":
            sum(u.get("n_retrieval_errors") or 0 for u in statuses),
    }

    if fold_override is None and not coverage_problems and not failed_units:
        try:
            for (setting, variant), pool in sorted(written.items()):
                row = {"setting": setting, "variant": variant,
                       "source_file": str(pred_path)}
                row.update(summarize_predictions(list(pool.values()), gold,
                                                 sorted(valid),
                                                 sorted(feasible)))
                summaries.append(row)
            if summaries:
                import csv
                with open(run_dir / "summary_full_gold.csv", "x", newline="",
                          encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
        except Exception as e:  # summary failure must not lose the run
            summary_error = f"{type(e).__name__}: {e}"
            run_status = "completed_with_failures"
    elif fold_override is not None:
        run_meta["execution"]["notes"].append(
            f"partial run (fold {fold_override} only): full-population "
            f"summaries intentionally NOT computed; predictions are saved "
            f"for diagnostics/benchmarking only")
    elif coverage_problems or failed_units:
        run_meta["execution"]["notes"].append(
            "full-population summaries NOT computed: "
            + ("unit failures present; " if failed_units else "")
            + ("coverage problems present; " if coverage_problems else "")
            + "error records keep the affected questions in the "
              "denominators of any offline evaluation; a summary over "
              "failed retrievals would misrepresent the run as successful")

    run_meta["execution"]["units"] = statuses
    run_meta["execution"]["summary"] = summaries
    run_meta["execution"]["coverage_problems"] = coverage_problems
    if summary_error:
        run_meta["execution"]["summary_error"] = summary_error
    run_meta["execution"]["run_status"] = run_status
    run_meta["execution"]["notes"].append(
        "fit/query timings are NEW measurements of the public build; they "
        "are not the paper's original-run efficiency numbers")
    run_meta["config"]["sha256"] = config_sha256(resolved)
    reporting.finalize_run_meta(run_meta, run_dir)
    return run_meta
