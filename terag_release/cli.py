"""Unified public CLI.

    python -m terag_release <command> ...

Commands
--------
validate-data    Check schema/QA/gold/split files against the input contract.
generate-demo    Generate the bundled synthetic demo dataset (deterministic).
build-gold       Derive full gold (AST) from user QA + schema CSVs.
make-folds       Build the deterministic question/table fold manifest.
run-demo         Run the offline demo pipeline end to end (no network).
run-experiments  Run a method x protocol matrix on user data.
evaluate         Offline full-gold evaluation of saved predictions.
analyze          Saved-result analyses (saved-prefix budgets, transfer
                 evidence, held-out disagreements, isolation).
plot             Render result figures from a run directory.
benchmark        Fresh timing of the current build (NEW measurement).

Every command writes only to NEW output directories and records a run_meta
snapshot (resolved config + SHA256, input checksums, environment versions,
per-unit status/timing). API keys are read from environment variables only.
"""

import argparse
import json
import sys
from pathlib import Path

from terag_release import reporting
from terag_release.config import (CONFIG_DIR, REPO_ROOT, config_sha256,
                                  load_base, load_dataset, load_experiment,
                                  load_protocol, terag_sections)

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

    report = validate_dataset(
        field_csv=args.field_csv, table_csv=args.table_csv,
        qa_csv=args.qa_csv, gold=args.gold,
        splits_manifest=args.splits_manifest)
    out = reporting.require_new_directory(args.output)
    reporting.write_json_exclusive(out / "validation_report.json", report)
    ok = report["ok"]
    print(("PASS" if ok else "FAIL") + f" — {report['summary']}")
    for err in report["errors"][:20]:
        print(f"  error: {err}")
    for warn in report.get("warnings", [])[:20]:
        print(f"  warning: {warn}")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------- demo

def cmd_generate_demo(args):
    from demo.generator import generate

    out = Path(args.output)
    if out.exists() and not args.force:
        _fail(f"output exists: {out} (use --force to regenerate demo data)")
    generate(out, scale=args.scale)
    print(f"demo dataset generated under {out} (scale={args.scale})")


def cmd_run_demo(args):
    from terag_release.runner import run_experiments

    dataset = load_dataset("demo")
    demo_dir = REPO_ROOT / "demo" / "data"
    if not (demo_dir / "qa_data.csv").exists():
        _fail("demo data missing; run `python -m terag_release generate-demo` "
              "first")
    base = load_base()
    resolved = {
        "command": "run-demo",
        "dataset": dataset,
        "methods": OFFLINE_METHODS,
        "protocols": ["question_v1", "table_v2"],
        "budget": base.get("budget"),
        "offline_only": True,
        "note": ("synthetic demo execution; verifies the public code path, "
                 "NOT a reproduction of paper numbers"),
    }
    run_experiments(dataset_cfg=dataset, resolved=resolved,
                    base_sections=terag_sections(base, dataset),
                    run_dir=Path(args.output),
                    variant_names=OFFLINE_METHODS,
                    protocol_names=["question_v1", "table_v2"],
                    diagnostics=args.diagnostics)
    print(f"demo run complete: {args.output}")


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
    ast_folds, exclusions = protocols.apply_ast_exclusion(t_v1, gold,
                                                          rows["qid"].tolist())
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
    from terag_release.runner import run_experiments

    dataset = load_dataset(args.dataset)
    experiment = load_experiment(args.experiment)
    base = load_base()
    protocol_names = experiment.get("protocols",
                                    ["question_v1", "table_v2"])
    method_names = args.methods or experiment.get("methods",
                                                  variants_mod.FULL_MATRIX)
    unknown = [m for m in method_names if m not in variants_mod.VARIANTS]
    if unknown:
        _fail(f"unknown methods: {unknown}; available: "
              f"{sorted(variants_mod.VARIANTS)}")

    network_free = [m for m in method_names
                    if not variants_mod.method_requires_network(m)]
    skipped = sorted(set(method_names) - set(network_free))
    if args.offline_only:
        method_names = network_free
        if skipped:
            print(f"offline-only: skipping network methods {skipped}")
    elif skipped and not args.allow_network:
        _fail(f"methods {skipped} require a model download or the DeepSeek "
              f"API; pass --allow-network to enable them (never in the demo)")

    resolved = {
        "command": "run-experiments",
        "dataset": dataset, "experiment": experiment,
        "methods": method_names, "protocols": protocol_names,
        "budget": base.get("budget"), "seed": base.get("terag", {}).get("seed", 42),
    }
    run_experiments(dataset_cfg=dataset, resolved=resolved,
                    base_sections=terag_sections(base, dataset),
                    run_dir=Path(args.output), variant_names=method_names,
                    protocol_names=protocol_names,
                    fold_override=args.fold, diagnostics=args.diagnostics)
    print(f"run complete: {args.output}")


# ---------------------------------------------------------------- evaluate

def cmd_evaluate(args):
    from experiment.audit_saved_predictions import audit, write_audit_result

    result = audit(gold_path=args.gold, budget_path=args.budget,
                   prediction_paths=args.predictions, families=args.families)
    write_audit_result(result, args.output)


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
    plot_run(args.run, out)


# ---------------------------------------------------------------- benchmark

def cmd_benchmark(args):
    from terag_release.benchmark import benchmark

    dataset = load_dataset(args.dataset)
    base = load_base()
    report = benchmark(dataset_cfg=dataset,
                       base_sections=terag_sections(base, dataset),
                       variants=args.variants or ["TERAG-Full"],
                       output=args.output)
    print(json.dumps(report.get("summary", report), indent=2,
                     ensure_ascii=False, default=str))


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terag_release",
        description="TE-RAG public release CLI (see docs/*.md)")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-data", help="check input files against the "
                       "documented contract")
    v.add_argument("--field-csv", required=True)
    v.add_argument("--table-csv", required=True)
    v.add_argument("--qa-csv", required=True)
    v.add_argument("--gold")
    v.add_argument("--splits-manifest")
    v.add_argument("--output", required=True)
    v.set_defaults(func=cmd_validate_data)

    g = sub.add_parser("generate-demo", help="generate the synthetic demo "
                       "dataset (deterministic)")
    g.add_argument("--output", default=str(REPO_ROOT / "demo" / "data"))
    g.add_argument("--scale", default="small", choices=["small", "full"])
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=cmd_generate_demo)

    d = sub.add_parser("run-demo", help="run the offline synthetic demo "
                       "pipeline (no network, no API)")
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
                   help="dataset config name under configs/datasets/")
    r.add_argument("--experiment", default="full_matrix",
                   help="experiment config name under configs/experiments/")
    r.add_argument("--methods", nargs="*", help="override method list")
    r.add_argument("--offline-only", action="store_true",
                   help="skip methods needing model downloads or the API")
    r.add_argument("--allow-network", action="store_true",
                   help="permit model/API methods (never default)")
    r.add_argument("--fold", type=int, help="run a single fold only")
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

    pl = sub.add_parser("plot", help="render figures from a run directory")
    pl.add_argument("--run", required=True)
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
