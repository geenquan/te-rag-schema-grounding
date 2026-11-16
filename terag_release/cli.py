"""Unified public CLI.

    python -m terag_release <command> ...

Commands
--------
validate-data    Check schema/QA/gold/split files against the input contract.
generate-demo    Generate the synthetic demo dataset into a NEW directory.
build-gold       Derive full gold (AST) from user QA + schema CSVs.
make-folds       Build the deterministic question/table fold manifest.
run-demo         Run the offline demo pipeline end to end (no network).
run-experiments  Run a method x protocol matrix on user data.
evaluate         Offline full-gold evaluation of saved predictions.
statistics       Offline paired statistics (Wilcoxon + cluster bootstrap).
analyze          Saved-result analyses (saved-prefix budgets, transfer
                 evidence, held-out disagreements, isolation).
plot             Render result figures from a run directory (+ statistics).
benchmark        Fresh timing of the current build (NEW measurement).

Execution commands resolve the layered configuration (base <- dataset <-
protocol <- method <- experiment <- CLI) into one effective structure whose
canonical SHA256 is the run's configuration identity; run_meta records the
resolved configuration, per-unit effective parameters, input checksums,
environment versions, and per-unit status/timing. Failed units make the
command exit non-zero; error records keep affected questions in every
denominator. API keys are read from environment variables only.
"""

import argparse
import json
import sys
from pathlib import Path

from terag_release import reporting
from terag_release.config import (CONFIG_DIR, REPO_ROOT, config_sha256,
                                  resolve_run)

OFFLINE_METHODS = ["BM25", "TFIDF", "Hybrid", "MetadataMatched",
                   "ExampleRRF", "CooccurRRF",
                   "TERAG-Full", "TERAG-noBC", "TERAG-noAnnIdx",
                   "TERAG-noAnnHit"]


def _fail(message: str, code: int = 2):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------- validate

def cmd_validate_data(args):
    from terag_release.validation import validate_dataset
    from terag_release import protocols
    from terag_release.runner import load_gold

    splits_report = {}
    if args.splits_manifest:
        import pandas as pd
        rows = pd.read_csv(args.qa_csv)
        qids = [f"{args.qid_prefix}{i:03d}" for i in range(len(rows))]
        try:
            manifest = protocols.load_manifest(args.splits_manifest)
            checks = protocols.validate_manifest(manifest, rows,
                                                 load_gold(args.gold), qids)
            splits_report = {"revalidated_checks": checks}
        except protocols.ManifestValidationError as e:
            splits_report = {"revalidated_checks": None, "errors":
                             [str(e)]}

    report = validate_dataset(
        field_csv=args.field_csv, table_csv=args.table_csv,
        qa_csv=args.qa_csv, gold=args.gold,
        splits_manifest=None)  # manifest handled above with revalidation
    report["splits_manifest"] = splits_report or {
        "note": "no manifest supplied; folds will be rebuilt at run time"}
    ok = report["ok"] and splits_report.get("revalidated_checks") is not None \
        if args.splits_manifest else report["ok"]
    report["ok"] = ok
    out = reporting.require_new_directory(args.output)
    reporting.write_json_exclusive(out / "validation_report.json", report)
    print(("PASS" if ok else "FAIL") + f" — {report['summary']}")
    for err in report["errors"][:20]:
        print(f"  error: {err}")
    for err in splits_report.get("errors", [])[:20]:
        print(f"  error: {err}")
    for warn in report.get("warnings", [])[:20]:
        print(f"  warning: {warn}")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------- demo

def cmd_generate_demo(args):
    from demo.generator import generate

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        _fail(f"output directory exists and is not empty: {out} — choose a "
              f"NEW directory (the bundled demo/data is tracked and is "
              f"never overwritten; regeneration is always into a fresh "
              f"directory)")
    generate(out, scale=args.scale)
    print(f"demo dataset generated under {out} (scale={args.scale}); "
          f"build its gold with scripts/build_full_gold.py or point a "
          f"dataset config at the new directory")


def cmd_run_demo(args):
    dataset_ref = args.dataset
    resolved = resolve_run(dataset_ref, "demo_offline",
                           method_names=OFFLINE_METHODS)
    if resolved["profile"] != "paper":
        _fail("run-demo always uses the paper profile")
    from terag_release.runner import run_experiments
    resolved["execution_note"] = (
        "synthetic demo execution; verifies the public code path, NOT a "
        "reproduction of paper numbers")
    meta = run_experiments(resolved=resolved, run_dir=Path(args.output),
                           diagnostics=args.diagnostics)
    status = meta["execution"]["run_status"]
    print(f"demo run '{status}': {args.output}")
    if status != "completed":
        raise SystemExit(1)


# ---------------------------------------------------------------- gold/folds

def cmd_build_gold(args):
    from scripts.build_full_gold import build

    build(field_csv=args.field_csv, table_csv=args.table_csv,
          qa_csv=args.qa_csv, output_dir=args.output_dir,
          qid_prefix=args.qid_prefix, dialect=args.dialect)


def cmd_make_folds(args):
    from terag_release import protocols
    from terag_release.runner import load_gold, load_rows

    rows = load_rows(args.qa_csv, qid_prefix=args.qid_prefix)
    gold = load_gold(args.gold)
    q_folds = protocols.build_question_folds(rows)
    t_v1 = protocols.build_table_folds(rows)
    ast_folds, exclusions = protocols.apply_ast_exclusion(
        t_v1, gold, rows["qid"].tolist())
    checks = protocols.verify_splits(rows, q_folds, t_v1)
    manifest = protocols.build_manifest(rows, q_folds, t_v1, ast_folds,
                                        exclusions, checks)
    failed = [k for k, v in checks.items() if not v]
    out = reporting.require_new_directory(args.output)
    reporting.write_json_exclusive(out / "splits_manifest.json", manifest)
    print("fold checks:", "all passed" if not failed else f"FAILED {failed}")
    for entry in exclusions:
        if entry["excluded_qids"]:
            print(f"  fold {entry['fold']}: AST exclusions "
                  f"{entry['excluded_qids']}")
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------- experiments

def cmd_run_experiments(args):
    from terag_release import variants as variants_mod
    from terag_release.config import load_experiment
    from terag_release.runner import run_experiments

    method_names = args.methods
    if args.offline_only:
        if method_names is None:
            experiment, _ = load_experiment(args.experiment)
            method_names = experiment.get("methods",
                                          variants_mod.FULL_MATRIX)
        network = {m for m in method_names
                   if variants_mod.method_requires_network(m)}
        if network:
            print(f"offline-only: skipping network methods {sorted(network)}")
            method_names = [m for m in method_names if m not in network]
            if not method_names:
                _fail("offline-only selection is empty after removing "
                      "network methods")

    try:
        resolved = resolve_run(args.dataset, args.experiment,
                               method_names=method_names,
                               profile=args.profile)
    except Exception as e:
        _fail(str(e))
    network = {m for m in resolved["experiment"]["methods"]
               if variants_mod.method_requires_network(m)}
    if network and not args.allow_network:
        _fail(f"methods {sorted(network)} require a model download or the "
              f"DeepSeek API; pass --allow-network to enable them (never in "
              f"the demo)")

    meta = run_experiments(resolved=resolved, run_dir=Path(args.output),
                           fold_override=args.fold,
                           diagnostics=args.diagnostics)
    status = meta["execution"]["run_status"]
    print(f"run '{status}': {args.output}")
    if status in ("completed_with_failures", "failed"):
        raise SystemExit(1)


# ---------------------------------------------------------------- evaluate

def cmd_evaluate(args):
    from experiment.audit_saved_predictions import audit, write_audit_result

    result = audit(gold_path=args.gold, budget_path=args.budget,
                   prediction_paths=args.predictions, families=args.families)
    write_audit_result(result, args.output)


# ---------------------------------------------------------------- statistics

def cmd_statistics(args):
    from analysis.statistics import compute_run_statistics

    output = args.output or str(Path(args.run) / "statistics")
    compute_run_statistics(run_dir=args.run, gold=args.gold, output=output)


# ---------------------------------------------------------------- analyze

def cmd_analyze(args):
    from analysis.runner import run_analysis

    run_analysis(name=args.analysis, run_dir=args.run, gold=args.gold,
                 budget=args.budget, output=args.output,
                 method_a=args.method_a, method_b=args.method_b)


# ---------------------------------------------------------------- plot

def cmd_plot(args):
    from analysis.plots import plot_run

    out = reporting.require_new_directory(args.output)
    # plot_run auto-detects <run>/statistics/statistics.json; an explicit
    # --statistics path is content-verified against the run before use
    plot_run(args.run, out, statistics_path=args.statistics)


# ---------------------------------------------------------------- benchmark

def cmd_benchmark(args):
    from terag_release.benchmark import benchmark

    resolved = resolve_run(args.dataset, "question_only_offline",
                           method_names=args.variants)
    report = benchmark(resolved=resolved, output=args.output)
    print(json.dumps(report.get("summary", report), indent=2,
                     ensure_ascii=False, default=str))
    if report.get("run_status") in ("failed", "completed_with_failures"):
        raise SystemExit(1)


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terag_release",
        description="TE-RAG public release CLI (see docs/*.md)")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-data", help="check input files against the "
                       "documented contract (splits manifests are "
                       "re-validated, cached checks are never trusted)")
    v.add_argument("--field-csv", required=True)
    v.add_argument("--table-csv", required=True)
    v.add_argument("--qa-csv", required=True)
    v.add_argument("--gold", required=True)
    v.add_argument("--splits-manifest")
    v.add_argument("--qid-prefix", default="q")
    v.add_argument("--output", required=True)
    v.set_defaults(func=cmd_validate_data)

    g = sub.add_parser("generate-demo", help="generate the synthetic demo "
                       "dataset into a NEW directory (deterministic)")
    g.add_argument("--output", required=True,
                   help="NEW output directory (the bundled demo/data is "
                        "never overwritten)")
    g.add_argument("--scale", default="small", choices=["small", "full"])
    g.set_defaults(func=cmd_generate_demo)

    d = sub.add_parser("run-demo", help="run the offline synthetic demo "
                       "pipeline (no network, no API)")
    d.add_argument("--dataset", default="demo",
                   help="demo dataset config name or path (default: demo)")
    d.add_argument("--output", required=True,
                   help="NEW output directory for the demo run")
    d.add_argument("--diagnostics", action="store_true",
                   help="export per-fold B/C evidence (does not change "
                        "outputs)")
    d.set_defaults(func=cmd_run_demo)

    b = sub.add_parser("build-gold", help="derive full gold from QA SQL via "
                       "AST (no SQL execution)")
    b.add_argument("--field-csv", required=True)
    b.add_argument("--table-csv", required=True)
    b.add_argument("--qa-csv", required=True)
    b.add_argument("--output-dir", required=True)
    b.add_argument("--qid-prefix", default="q")
    b.add_argument("--dialect", default="mysql")
    b.set_defaults(func=cmd_build_gold)

    f = sub.add_parser("make-folds", help="build the deterministic "
                       "question/table fold manifest")
    f.add_argument("--qa-csv", required=True)
    f.add_argument("--gold", required=True)
    f.add_argument("--qid-prefix", default="q")
    f.add_argument("--output", required=True)
    f.set_defaults(func=cmd_make_folds)

    r = sub.add_parser("run-experiments", help="run a method x protocol "
                       "matrix on user data")
    r.add_argument("--dataset", required=True,
                   help="dataset config NAME under configs/datasets/ or an "
                        "explicit YAML file path")
    r.add_argument("--experiment", default="full_matrix",
                   help="experiment config name or path")
    r.add_argument("--methods", nargs="*", help="override method list")
    r.add_argument("--profile", choices=["paper", "custom"], default=None,
                   help="paper (default): frozen parameters are enforced; "
                        "custom: deviations are allowed and recorded")
    r.add_argument("--offline-only", action="store_true",
                   help="skip methods needing model downloads or the API")
    r.add_argument("--allow-network", action="store_true",
                   help="permit model/API methods (never default)")
    r.add_argument("--fold", type=int, help="run a single fold only "
                   "(produces a labeled partial run)")
    r.add_argument("--diagnostics", action="store_true")
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_run_experiments)

    e = sub.add_parser("evaluate", help="offline full-gold evaluation of "
                       "saved predictions")
    e.add_argument("--gold", required=True)
    e.add_argument("--budget", required=True)
    e.add_argument("--predictions", nargs="+", required=True,
                   help="prediction JSONL files; a later file's COMPLETE "
                        "method/protocol group replaces an earlier one")
    e.add_argument("--families", choices=["none", "six", "two", "both"],
                   default="none")
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("statistics", help="offline paired statistics "
                       "(Wilcoxon + cluster bootstrap) for a run directory")
    s.add_argument("--run", required=True, help="run directory")
    s.add_argument("--gold", required=True)
    s.add_argument("--output", default=None,
                   help="output directory (default: <run>/statistics, which "
                        "`plot` auto-detects)")
    s.set_defaults(func=cmd_statistics)

    a = sub.add_parser("analyze", help="saved-result analyses")
    a.add_argument("--analysis", required=True,
                   choices=["saved-prefix", "transfer", "heldout",
                            "isolation"])
    a.add_argument("--run", required=True, help="run directory")
    a.add_argument("--gold")
    a.add_argument("--budget")
    a.add_argument("--method-a", default="TERAG-Full")
    a.add_argument("--method-b", default="MetadataMatched")
    a.add_argument("--output", required=True)
    a.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plot", help="render figures from a run directory "
                       "(CI whiskers need the statistics step's output)")
    pl.add_argument("--run", required=True)
    pl.add_argument("--statistics", default=None,
                    help="statistics.json from `terag_release statistics` "
                         "(auto-detected under <run>/statistics/)")
    pl.add_argument("--output", required=True)
    pl.set_defaults(func=cmd_plot)

    bm = sub.add_parser("benchmark", help="fresh timing of the current "
                       "build (NEW measurement, not the paper's table)")
    bm.add_argument("--dataset", default="demo")
    bm.add_argument("--variants", nargs="*")
    bm.add_argument("--output", required=True)
    bm.set_defaults(func=cmd_benchmark)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
