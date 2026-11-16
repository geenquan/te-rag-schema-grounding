"""Offline paired statistics for a run directory (R5).

Computes, per protocol, every present TE-RAG variant's Covered@5 difference
against TERAG-Full using the paper's machinery — question-paired Wilcoxon
signed-rank (two-sided, zero_method='wilcox', no correction, 'auto') and
target-table cluster bootstrap (clusters = gold ``original_table``, 10,000
resamples, RNG ``default_rng(42)``, percentile 2.5/97.5, unadjusted).

Direction: eval.full_gold.paired_covered computes Full − variant; the paper
figures plot variant − Full, so both the mean difference and the CI
endpoints are NEGATED here (and checked symmetric). Zero-difference
comparisons carry no p-value and are marked ``ci_available: false`` — the
plot draws them as hollow markers. Family membership (k=6 / k=2) is
recorded per comparison but NO global correction is implied across
families or exploratory comparisons.

Purely offline: reads saved predictions + gold; never re-runs retrieval,
never calls models. Inputs are fingerprinted into the output for
traceability.
"""

import json
from pathlib import Path

from analysis.saved_prefix import load_gold, load_run
from eval.full_gold import full_gold_metrics, paired_covered
from experiment.audit_saved_predictions import FAMILIES

# The fixed comparison families are QUESTION-LEVEL comparisons in the
# paper (both are anchored at question_v1/TERAG-Full in `evaluate`).
# Family membership and completeness policy are imported from the
# evaluate module so the two entry points cannot drift apart.
SIX_FAMILY = set(FAMILIES["six"])
TWO_FAMILY = set(FAMILIES["two"])
FIXED_FAMILY_SETTING = "question_v1"


def _covered(record, gold) -> float:
    return float(full_gold_metrics(record, gold)[1])


def _variant_diffs(records, full_records, gold, valid, clusters):
    """variant − Full per-qid Covered@5 differences over sorted(valid)."""
    diffs = {}
    for qid in sorted(valid):
        if qid not in records or qid not in full_records:
            continue
        diffs[qid] = _covered(records[qid], gold) - \
            _covered(full_records[qid], gold)
    return diffs


def compute_run_statistics(run_dir: str, gold: str, output: str) -> Path:
    from terag_release import reporting

    run_path = Path(run_dir)
    gold_map = load_gold(gold)
    pool = load_run(run_dir)
    valid = {q for q, g in gold_map.items()
             if g.get("gold_status") == "verified_complete"}
    clusters = {q: g.get("original_table", "") for q, g in gold_map.items()}

    comparisons = []
    for setting in ("question_v1", "table_v2"):
        full_key = (setting, "TERAG-Full")
        if full_key not in pool:
            continue
        full_records = pool[full_key]
        setting_variants = sorted(v for (s, v) in pool
                                  if s == setting and v != "TERAG-Full")
        for variant in setting_variants:
            records = pool[(setting, variant)]
            # Fixed-family inference status is decided by (setting,
            # comparison): the paper's k=6 / k=2 families are Question-
            # Level only, and a family corrected value requires the WHOLE
            # family to be present — mirroring `evaluate`.
            fixed_family = None
            fixed_members = None
            if setting == FIXED_FAMILY_SETTING:
                if variant in SIX_FAMILY:
                    fixed_family, fixed_members = "k6_fixed", SIX_FAMILY
                elif variant in TWO_FAMILY:
                    fixed_family, fixed_members = "k2_fixed", TWO_FAMILY
            family_status = "not_applicable"
            missing_members = None
            if fixed_family:
                missing = sorted(m for m in fixed_members
                                 if (setting, m) not in pool)
                if missing:
                    family_status = "not_computed"
                    missing_members = missing
                else:
                    family_status = "computed"
            shared = sorted(set(records) & set(full_records) & valid)
            if not shared:
                comparisons.append({
                    "setting": setting, "variant": variant,
                    "status": "not_computable",
                    "reason": "no shared valid questions between the two "
                              "methods",
                    "family": fixed_family or (
                        "exploratory_heldout"
                        if setting == "table_v2" else "exploratory"),
                    "family_inference_status": family_status,
                    **({"missing_family_members": missing_members}
                       if missing_members else {})})
                continue
            # paired_covered computes Full − opponent; run it with the
            # argument order (Full, variant) so the machinery (Wilcoxon +
            # seeded cluster bootstrap) is the paper's, then present
            # variant − Full by negation. Population gaps (a qid missing
            # on either side) are reported as not_computable — never
            # silently shrunk.
            try:
                res = paired_covered(list(full_records.values()),
                                     list(records.values()), gold_map,
                                     sorted(valid), clusters)
            except ValueError as e:
                comparisons.append({
                    "setting": setting, "variant": variant,
                    "status": "not_computable",
                    "reason": f"paired population incomplete: {e}",
                    "family": fixed_family or (
                        "exploratory_heldout"
                        if setting == "table_v2" else "exploratory"),
                    "family_inference_status": family_status,
                    **({"missing_family_members": missing_members}
                       if missing_members else {})})
                continue
            n_clusters = res["statistical_method"]["n_clusters"]
            entry = {
                "setting": setting, "variant": variant,
                "status": "computed",
                "metric": "covered@5",
                "direction": "variant - Full",
                "n": res["n"],
                "n_clusters": n_clusters,
                "mean_diff": (-res["mean_diff"]
                              if res["mean_diff"] is not None else None),
                # zero_difference means the per-question COVERED@5 values
                # are identical — NOT that the full retrieval outputs are
                # identical (the paper's Full/noC boundary: metrics equal,
                # some ordered outputs differ).
                "zero_covered_difference": res["zero_difference"],
                "zero_difference_note": (
                    "identical per-question Covered@5 values; this is not "
                    "a claim of identical full retrieval outputs"),
                "statistical_method": res["statistical_method"],
                "family": fixed_family or (
                    "exploratory_heldout" if setting == "table_v2"
                    else "exploratory"),
                "family_inference_status": family_status,
            }
            if missing_members:
                entry["missing_family_members"] = missing_members
                entry["family_note"] = (
                    "fixed family incomplete in this run; no family-"
                    "corrected value is reported (raw paired values only)")
            if fixed_family and family_status == "computed":
                entry["bonferroni_k"] = len(fixed_members)
            if "wilcoxon_p_raw" in res:
                entry["wilcoxon_p_raw"] = res["wilcoxon_p_raw"]
                if fixed_family and family_status == "computed":
                    entry["wilcoxon_p_bonferroni_within_family"] = min(
                        1.0, res["wilcoxon_p_raw"] * len(fixed_members))
            if "ci95_cluster_bootstrap" in res:
                lo, hi = res["ci95_cluster_bootstrap"]
                entry["ci95_cluster_bootstrap"] = [-hi, -lo]  # flip direction
                entry["ci_available"] = True
            else:
                entry["ci_available"] = False
                entry["ci_note"] = (
                    "no interval estimated: identical per-question "
                    "Covered@5 values (zero difference) or insufficient "
                    "sample/clusters")
            comparisons.append(entry)

    if not comparisons:
        raise ValueError(f"no comparable method pairs in {run_dir}")

    payload = {
        "identity": ("offline paired statistics over SAVED predictions; "
                     "no retrieval re-run, no model/API calls"),
        "inputs": {
            "predictions": str(run_path / "predictions.jsonl"),
            "predictions_sha256": reporting.sha256_file(
                run_path / "predictions.jsonl"),
            "gold": str(gold),
            "gold_sha256": reporting.sha256_file(gold),
            "n_valid": len(valid),
        },
        "method": {
            "test": "scipy.stats.wilcoxon (two-sided, zero_method='wilcox', "
                    "correction=False, method='auto')",
            "cluster_key": "full_gold.original_table",
            "resamples": 10000, "seed": 42,
            "ci": "percentile 2.5/97.5; unadjusted",
            "correction_note": ("Bonferroni is applied WITHIN the declared "
                                "families only (k=6, k=2); exploratory "
                                "comparisons carry no family correction"),
        },
        "comparisons": comparisons,
    }
    out_dir = reporting.require_new_directory(output)
    reporting.write_json_exclusive(out_dir / "statistics.json", payload)
    print(f"statistics written: {out_dir / 'statistics.json'} "
          f"({len(comparisons)} comparisons)")
    return out_dir / "statistics.json"
