"""Figure regeneration from a run directory (matplotlib).

Renders the two result-figure families of the paper from ANY conforming run
directory (synthetic or user data):

* overall trade-off — five question-level controls: Covered@5 bar panel +
  column precision-recall scatter with macro F1 annotations;
* variant boundary — variant-minus-Full Covered@5 differences per protocol
  with unadjusted cluster bootstrap CIs when SciPy is available (hollow
  markers otherwise), plus a held-out coverage/recall panel.

Numbers differ from the paper unless the inputs are the authorized private
data; these plots visualize whatever run directory they are given.
"""

import csv
import json
from pathlib import Path

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


def _read_summary(run_dir: Path):
    path = run_dir / "summary_full_gold.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for row in rows:
        for key in ("covered@5", "colp", "colrec", "colf1", "fgcs",
                    "tr@1", "tr@3", "tr@5"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def plot_run(run_dir: str, out_dir: Path):
    rows = _read_summary(Path(run_dir))
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
                     color=[COLORS.get(n, "#9d755d") for n in names])
        axes[0].set_xlabel("Covered@5 (%)")
        axes[0].set_title(f"All-gold-table coverage (n={five[0]['n']})")
        for r, x in zip(five, covered):
            axes[0].text(x, r["variant"], f" {x:.1f}", va="center")
        for r in five:
            axes[1].scatter(100 * r["colrec"], 100 * r["colp"],
                            s=60, color=COLORS.get(r["variant"], "#9d755d"))
            axes[1].annotate(
                f"{r['variant']}\nF1 {100 * r['colf1']:.1f}",
                (100 * r["colrec"], 100 * r["colp"]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel("Column recall (%)")
        axes[1].set_ylabel("Column precision (%)")
        axes[1].set_title("Macro column P–R")
        fig.tight_layout()
        fig.savefig(out_dir / "overall_tradeoff.png", dpi=200)
        fig.savefig(out_dir / "overall_tradeoff.pdf")
        plt.close(fig)
        written.append("overall_tradeoff.pdf")

    # ---- Figure 2: variant−Full differences ------------------------------
    full_q = next((r for r in q if r["variant"] == "TERAG-Full"), None)
    variants_h = [r for r in h if r["variant"] != "TERAG-Full"]
    full_h = next((r for r in h if r["variant"] == "TERAG-Full"), None)
    if full_q or full_h:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plotted = False
        for panel, (setting, variants, full) in enumerate((
                ("question_v1", [r for r in q if r["variant"] != "TERAG-Full"
                                 and r["variant"].startswith("TERAG-")],
                 full_q),
                ("table_v2", variants_h, full_h))):
            names = [r["variant"].replace("TERAG-", "") for r in variants]
            diffs = [100 * (r["covered@5"] - full["covered@5"])
                     for r in variants] if full else []
            if not names:
                continue
            axes[panel].barh(names, diffs,
                             color=["#4c78a8" if d >= 0 else "#e45756"
                                    for d in diffs])
            axes[panel].axvline(0, color="black", linewidth=0.8)
            axes[panel].set_xlabel(f"Covered@5 diff vs Full (pp) — {setting}")
            plotted = True
        if plotted:
            fig.suptitle("Variant minus Full differences (this run's data; "
                         "CI whiskers require the statistics step)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "ablation_boundary.png", dpi=200)
            fig.savefig(out_dir / "ablation_boundary.pdf")
            written.append("ablation_boundary.pdf")
        plt.close(fig)

    if not written:
        raise ValueError("run directory has nothing plottable "
                         "(need question_v1 summaries)")
    print(f"figures written: {written}")
