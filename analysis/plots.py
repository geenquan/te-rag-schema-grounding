"""Figure regeneration from a run directory (matplotlib).

Renders the three result-figure families of the paper from ANY conforming
run directory (synthetic or user data):

1. overall trade-off — five question-level controls: Covered@5 bars + macro
   column precision-recall scatter with F1 annotations;
2. variant boundary — variant−Full Covered@5 differences per protocol with
   unadjusted target-table cluster 95% CIs as error bars when a statistics
   file is supplied (``terag_release statistics`` writes it; the CLI
   auto-detects ``<run>/statistics/statistics.json``). Comparisons without
   an interval (identical outputs / zero difference / not computable) are
   drawn as HOLLOW markers with an explicit legend entry;
3. held-out panel — Covered@5 versus macro COLUMN RECALL (not table
   recall) per method on the table_v2 protocol.

Point estimates come from the run's summary CSV; intervals come ONLY from
the statistics file — plotting never re-runs retrieval, never calls models,
and records its inputs in a manifest next to the figures. Numbers differ
from the paper unless the inputs are the authorized private data.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError as _e:  # pragma: no cover - optional dependency group
    raise SystemExit(
        "plotting requires the optional 'plots' dependency group: "
        "pip install matplotlib  (or pip install -e '.[plots]')") from _e

COLORS = {"TERAG-Full": "#4c78a8", "ExampleRRF": "#f58518",
          "MetadataMatched": "#54a24b", "CooccurRRF": "#e45756",
          "BM25": "#72b7b2"}
NEUTRAL = "#9d755d"


def _read_summary(run_dir: Path):
    path = run_dir / "summary_full_gold.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for row in rows:
        for key in ("covered@5", "colp", "colrec", "colf1", "fgcs",
                    "tr@1", "tr@3", "tr@5"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
        if row.get("n") not in (None, ""):
            row["n"] = int(row["n"])
    return rows


def _read_statistics(statistics_path):
    if statistics_path is None:
        return None
    path = Path(statistics_path)
    if not path.exists():
        raise FileNotFoundError(
            f"statistics file not found: {path} — run "
            f"`python -m terag_release statistics` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_statistics_identity(run_dir: Path, statistics: Dict,
                                summary_rows: List[Dict]):
    """Content-identity check before combining point estimates with CIs.

    The statistics file's recorded predictions/gold digests must match the
    CURRENT run's predictions content and its gold input, and every
    comparison must line up with this run's summary: metric, direction,
    paired n versus BOTH method rows' evaluation population, and the mean
    difference versus the summary's covered@5 difference for the same
    protocol. Identity is by CONTENT HASH, so a legitimate copy of the
    same materials in another directory passes, while another run's
    statistics — a doctored digest — or a tampered SUMMARY is rejected
    before any figure is written, never silently supplying whiskers for
    points they do not describe. Validation raises explicitly (ValueError)
    and never relies on ``assert`` (which ``python -O`` would strip).
    """
    problems = []
    inputs = statistics.get("inputs") or {}
    predictions_path = run_dir / "predictions.jsonl"
    actual_predictions = _sha256(predictions_path)
    recorded_predictions = inputs.get("predictions_sha256")
    if recorded_predictions != actual_predictions:
        problems.append(
            f"statistics were computed for different predictions "
            f"(recorded {recorded_predictions}, this run "
            f"{actual_predictions})")
    gold_path = inputs.get("gold")
    if gold_path and Path(gold_path).exists():
        if inputs.get("gold_sha256") != _sha256(Path(gold_path)):
            problems.append("statistics gold digest does not match the "
                            "recorded gold file content")
    elif inputs.get("gold_sha256"):
        problems.append(f"statistics gold input not found: {gold_path}")

    summary_by_key = {(r["setting"], r["variant"]): r for r in summary_rows}
    mean_tolerance = 1e-9

    def _finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    for comp in statistics.get("comparisons", []):
        key = (comp.get("setting"), comp.get("variant"))
        row = summary_by_key.get(key)
        if row is None:
            problems.append(f"statistics comparison {key} has no matching "
                            f"summary row in this run")
            continue
        if comp.get("metric") != "covered@5" or \
                comp.get("direction") != "variant - Full":
            problems.append(f"statistics comparison {key}: unexpected "
                            f"metric/direction "
                            f"({comp.get('metric')}/{comp.get('direction')})")
            continue

        # Required summary fields for any comparison that will be drawn.
        variant_rate = row.get("covered@5")
        if variant_rate is None or not _finite(variant_rate):
            problems.append(f"statistics comparison {key}: summary row "
                            f"lacks a finite covered@5 value")
            continue

        # Population consistency: the paired n must equal BOTH rows'
        # evaluation populations (Full and variant share the denominator).
        comp_n = comp.get("n")
        for label, source in (("variant", row),):
            if comp_n is not None and source.get("n") is not None \
                    and int(comp_n) != int(source["n"]):
                problems.append(
                    f"statistics comparison {key}: paired n {comp_n} != "
                    f"{label} summary n {source['n']}")

        full_key = (key[0], "TERAG-Full")
        full_row = summary_by_key.get(full_key)
        computed = comp.get("status") == "computed"
        if full_row is None:
            if computed:
                # A computed comparison without the anchor Full row can
                # never be reconciled — reject rather than skip.
                problems.append(
                    f"statistics comparison {key}: no TERAG-Full summary "
                    f"row for setting {key[0]!r}; cannot verify the mean "
                    f"difference")
            continue
        full_rate = full_row.get("covered@5")
        if full_rate is None or not _finite(full_rate):
            problems.append(f"statistics comparison {key}: TERAG-Full "
                            f"summary row lacks a finite covered@5 value")
            continue
        if comp_n is not None and full_row.get("n") is not None \
                and int(comp_n) != int(full_row["n"]):
            problems.append(
                f"statistics comparison {key}: paired n {comp_n} != Full "
                f"summary n {full_row['n']} (evaluation populations must "
                f"agree)")

        # Mean-difference consistency: statistics.mean_diff must equal the
        # difference of THIS summary's covered@5 rates. NaN/Infinity can
        # never satisfy the comparison and are reported, not skipped.
        mean = comp.get("mean_diff")
        if computed:
            if mean is None or not _finite(mean):
                problems.append(
                    f"statistics comparison {key}: mean_diff is missing or "
                    f"non-finite ({mean!r}); it must equal "
                    f"variant covered@5 - Full covered@5 from this run's "
                    f"summary")
                continue
            expected = float(variant_rate) - float(full_rate)
            if not _finite(expected) or \
                    abs(float(mean) - expected) > mean_tolerance:
                problems.append(
                    f"statistics comparison {key}: mean difference "
                    f"{mean:.6f} inconsistent with this run's summary "
                    f"({variant_rate:.6f} - {full_rate:.6f} = "
                    f"{expected:.6f})")
        # not_computable comparisons carry no interval semantics; they are
        # kept as explicit no-CI markers and need no mean reconciliation.

    if problems:
        raise ValueError(
            "statistics file does not belong to this run's content — "
            "refusing to draw its intervals over these point estimates "
            "(recompute with `terag_release statistics`):\n  - "
            + "\n  - ".join(problems))


def plot_run(run_dir: str, out_dir: Path, statistics_path=None) -> dict:
    run_dir = Path(run_dir)
    rows = _read_summary(run_dir)
    if statistics_path is None:
        # auto-detect the statistics step's default output location
        candidate = run_dir / "statistics" / "statistics.json"
        statistics_path = str(candidate) if candidate.exists() else None
    statistics = _read_statistics(statistics_path)
    if statistics is not None:
        _verify_statistics_identity(run_dir, statistics, rows)
    q = [r for r in rows if r["setting"] == "question_v1"]
    h = [r for r in rows if r["setting"] == "table_v2"]
    written = []

    # ---- Figure 1: question-level trade-off -----------------------------
    five = [r for r in q if r["variant"] in
            ("TERAG-Full", "ExampleRRF", "MetadataMatched", "CooccurRRF",
             "BM25")]
    if five:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        names = [r["variant"] for r in five]
        covered = [100 * r["covered@5"] for r in five]
        axes[0].barh(names, covered,
                     color=[COLORS.get(n, NEUTRAL) for n in names])
        axes[0].set_xlabel("Covered@5 (%)")
        axes[0].set_title(f"All-gold-table coverage (n={five[0]['n']})")
        for r, x in zip(five, covered):
            axes[0].text(x, r["variant"], f" {x:.1f}", va="center")
        for r in five:
            axes[1].scatter(100 * r["colrec"], 100 * r["colp"], s=60,
                            color=COLORS.get(r["variant"], NEUTRAL))
            axes[1].annotate(
                f"{r['variant']}\nF1 {100 * r['colf1']:.1f}",
                (100 * r["colrec"], 100 * r["colp"]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel("Macro column recall (%)")
        axes[1].set_ylabel("Macro column precision (%)")
        axes[1].set_title("Macro column P–R")
        fig.tight_layout()
        fig.savefig(out_dir / "overall_tradeoff.png", dpi=200)
        fig.savefig(out_dir / "overall_tradeoff.pdf")
        plt.close(fig)
        written.append("overall_tradeoff.pdf")

    # ---- Figure 2: variant−Full differences with CIs ---------------------
    full_q = next((r for r in q if r["variant"] == "TERAG-Full"), None)
    full_h = next((r for r in h if r["variant"] == "TERAG-Full"), None)
    stats_by_key = {}
    if statistics:
        for comp in statistics.get("comparisons", []):
            stats_by_key.setdefault(comp["setting"], {})[comp["variant"]] = comp
    if full_q or full_h:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        drew_ci = drew_hollow = False
        for panel, setting in enumerate(("question_v1", "table_v2")):
            variants = [r for r in (q if setting == "question_v1" else h)
                        if r["variant"] != "TERAG-Full"
                        and r["variant"].startswith("TERAG-")]
            full = full_q if setting == "question_v1" else full_h
            ax = axes[panel]
            if not variants or full is None:
                ax.text(0.5, 0.5, f"no run data for {setting}",
                        ha="center", va="center")
                ax.set_axis_off()
                continue
            labels, diffs, lo, hi, have_ci = [], [], [], [], []
            for r in variants:
                labels.append(r["variant"].replace("TERAG-", ""))
                diffs.append(100 * (r["covered@5"] - full["covered@5"]))
                comp = stats_by_key.get(setting, {}).get(r["variant"])
                if comp and comp.get("ci_available"):
                    lo.append(100 * comp["ci95_cluster_bootstrap"][0])
                    hi.append(100 * comp["ci95_cluster_bootstrap"][1])
                    have_ci.append(True)
                else:
                    lo.append(diffs[-1])  # zero-height bar -> hollow marker
                    hi.append(diffs[-1])
                    have_ci.append(False)
            y = range(len(labels))
            for i in y:
                color = "#4c78a8" if diffs[i] >= 0 else "#e45756"
                if have_ci[i]:
                    ax.errorbar(diffs[i], i,
                                xerr=[[diffs[i] - lo[i]],
                                      [hi[i] - diffs[i]]],
                                fmt="o", color=color, capsize=3)
                    drew_ci = True
                else:
                    ax.scatter(diffs[i], i, facecolors="none",
                               edgecolors=color)
                    drew_hollow = True
            ax.set_yticks(list(y), labels)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"Covered@5 diff vs Full (pp) — {setting}")
        note = []
        if statistics is None:
            note.append("no statistics file: point estimates only "
                        "(run `terag_release statistics` for CIs)")
        if drew_ci:
            note.append("filled + whiskers: unadjusted 95% cluster "
                        "bootstrap CI")
        if drew_hollow:
            note.append("hollow: no interval (identical per-question "
                        "Covered@5 values / not computable)")
        fig.suptitle("Variant minus Full; " + "; ".join(note), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_boundary.png", dpi=200)
        fig.savefig(out_dir / "ablation_boundary.pdf")
        plt.close(fig)
        written.append("ablation_boundary.pdf")

    # ---- Figure 3: held-out Covered@5 vs COLUMN RECALL -------------------
    if h:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for r in h:
            label = r["variant"]
            ax.scatter(100 * r["colrec"], 100 * r["covered@5"], s=60,
                       color=COLORS.get(label, NEUTRAL))
            ax.annotate(label, (100 * r["colrec"], 100 * r["covered@5"]),
                        fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
        ax.set_xlabel("Held-out macro COLUMN recall (%)")
        ax.set_ylabel("Held-out Covered@5 (%)")
        ax.set_title(f"Held-out-table coverage vs column recall "
                     f"(n={h[0]['n']})")
        fig.tight_layout()
        fig.savefig(out_dir / "heldout_tradeoff.png", dpi=200)
        fig.savefig(out_dir / "heldout_tradeoff.pdf")
        plt.close(fig)
        written.append("heldout_tradeoff.pdf")

    if not written:
        raise ValueError("run directory has nothing plottable "
                         "(need question_v1 summaries)")

    manifest = {
        "figures": written,
        "inputs": {
            "summary_full_gold": str(run_dir / "summary_full_gold.csv"),
            "statistics": (str(statistics_path) if statistics else None),
            "statistics_inputs": (statistics.get("inputs")
                                  if statistics else None),
        },
        "ci_policy": ("; ".join(note) if note else
                      "error bars come only from the supplied statistics "
                      "file; hollow markers mark absent intervals"),
    }
    (out_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"figures written: {written}")
    return manifest
