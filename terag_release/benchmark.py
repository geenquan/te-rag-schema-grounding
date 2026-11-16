"""Fresh timing of the current build (paper Table `tab:efficiency` analogue).

Measurement boundaries mirror the paper's protocol: Fit = wall-clock seconds
per fold (mean over the executed folds); Query = mean of per-fold mean
per-question latency. All run records of each fold are included.

Whatever this produces is a NEW measurement of the public build on the
current machine. It is not, and must not be presented as, a reproduction of
the paper's original-run efficiency numbers. The benchmark honors the same
network boundary as every other command: offline methods only unless the
resolved configuration was explicitly created with --allow-network.
"""

from pathlib import Path

from terag_release import reporting
from terag_release.runner import run_experiments


def benchmark(resolved: dict, output: str) -> dict:
    from terag_release import variants as variants_mod

    network = {m for m in resolved["experiment"]["methods"]
               if variants_mod.method_requires_network(m)}
    if network:
        raise ValueError(
            f"benchmark refuses network/API methods {sorted(network)}; time "
            f"them inside a normal run-experiments execution instead")
    resolved = dict(resolved)
    resolved["benchmark_note"] = (
        "NEW measurement of the public build; not comparable to the paper's "
        "original-run efficiency table")
    resolved["experiment"] = dict(resolved["experiment"])
    resolved["experiment"]["protocols"] = ["question_v1"]
    run_dir = Path(output)
    meta = run_experiments(resolved=resolved, run_dir=run_dir)

    summary = {}
    for unit in meta["execution"]["units"]:
        variant = unit["variant"]
        entry = summary.setdefault(variant, {"fits": [], "query_means": [],
                                             "status": unit["status"]})
        if unit.get("fit_time_s") is not None:
            entry["fits"].append(unit["fit_time_s"])
        if unit.get("avg_query_time_s") is not None:
            entry["query_means"].append(unit["avg_query_time_s"])
    for variant, entry in summary.items():
        fits = entry["fits"]
        queries = entry["query_means"]
        entry["fit_mean_s"] = round(sum(fits) / len(fits), 3) if fits else None
        entry["query_mean_of_fold_means_s"] = round(
            sum(queries) / len(queries), 6) if queries else None
        entry.pop("fits"), entry.pop("query_means")

    report = {
        "identity": "new measurement; not the paper's original-run table",
        "run_status": meta["execution"]["run_status"],
        "variants": summary,
        "environment": meta["environment"],
    }
    reporting.write_json_exclusive(run_dir / "benchmark.json", report)
    if meta["execution"]["run_status"] in ("failed",
                                           "completed_with_failures"):
        report["exit_nonzero_reason"] = (
            "the benchmark run itself had failing units or retrieval "
            "errors; timings above are incomplete")
    return report
