"""Audit saved full-gold predictions without importing a runner or model.

Example (explicit sources, later complete groups override earlier ones):
python -m experiment.audit_saved_predictions --gold GOLD --budget BUDGET \
    --predictions ORIGINAL EXPANSION DDS KASLA LINKALIGN --output NEW_DIRECTORY \
    --families both
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from eval.full_gold import (index_predictions, paired_covered, prediction_groups,
                            summarize_predictions, validate_population)
from eval.retrieval_metrics import normalize_column_key, normalize_table

FAMILIES = {
    'six': ['TERAG-noBC', 'TERAG-noAnnIdx', 'TERAG-noAnnHit', 'Dense',
            'DenseDeepSeek', 'ExampleRRF'],
    'two': ['LinkAlign-SingleDB-Adapted', 'KaSLA-Adapted'],
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_sources(paths):
    pool, provenance = {}, {}
    for path in paths:
        for key, records in prediction_groups(read_jsonl(path)).items():
            pool[key] = records  # whole group replaces; never fill missing qids from another run
            provenance[key] = str(path)
    return pool, provenance


def audit(gold_path, budget_path, prediction_paths, families='none'):
    gold = index_predictions(read_jsonl(gold_path))
    valid = {q for q, g in gold.items() if g['gold_status'] == 'verified_complete'}
    budget = json.loads(Path(budget_path).read_text())
    budget_rows = index_predictions(budget['per_question'], valid)
    feasible = {q for q, r in budget_rows.items() if r['budget_feasible']}
    recomputed = set()
    for q in valid:
        g = gold[q]
        counts = Counter(c.rsplit('.', 1)[0] for c in {normalize_column_key(c) for c in g['gold_columns']})
        if len({normalize_table(t) for t in g['gold_tables']}) <= 5 and max(counts.values(), default=0) <= 10:
            recomputed.add(q)
    if (feasible != recomputed or budget['n_verified'] != len(valid)
            or budget['n_feasible'] != len(feasible)
            or set(budget['infeasible_qids']) != valid-feasible):
        raise ValueError('budget file disagrees with supplied full gold')
    validate_population(gold, valid, feasible)
    pool, sources = load_sources(prediction_paths)
    if not pool:
        raise ValueError('no prediction groups supplied')
    summaries = []
    for (setting, method), records in sorted(pool.items()):
        summaries.append({'setting': setting, 'method': method,
                          'source_file': sources[(setting, method)],
                          **summarize_predictions(records.values(), gold, valid, feasible)})
    comparisons = {}
    chosen = list(FAMILIES) if families == 'both' else ([] if families == 'none' else [families])
    clusters = {q: gold[q]['original_table'] for q in valid}
    for family in chosen:
        members = FAMILIES[family]
        anchor = ('question_v1', 'TERAG-Full')
        missing = [m for m in members + ['TERAG-Full']
                   if ('question_v1', m) not in pool]
        if missing:
            # A fixed family is either complete or not computed — it is
            # never silently shrunk or imputed. The paper's runs supply the
            # complete families; an offline demo run records this instead.
            comparisons[family] = {
                'status': 'not_computed',
                'missing_members': missing,
                'reason': ('fixed comparison family incomplete in the '
                           'supplied predictions; family size k is fixed, so '
                           'no partial correction is reported'),
            }
            continue
        comparisons[family] = {}
        for opponent in members:
            a, b = anchor, ('question_v1', opponent)
            r = paired_covered(pool[a].values(), pool[b].values(), gold, valid, clusters)
            r['bonferroni_k'] = len(members)
            r['family_members'] = members
            if 'wilcoxon_p_raw' in r:
                r['wilcoxon_p_bonferroni'] = min(1., r['wilcoxon_p_raw'] * len(members))
            comparisons[family]['Full_vs_' + opponent] = r
    paths = [gold_path, budget_path, *prediction_paths]
    root = Path(__file__).resolve().parents[1]
    code_paths = ['eval/full_gold.py', 'eval/retrieval_metrics.py',
                  'experiment/audit_saved_predictions.py']
    return {'summary': summaries, 'comparisons': comparisons,
            'evaluation_code_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                       for p in code_paths},
            'valid_qids': sorted(valid), 'feasible_qids': sorted(feasible),
            'source_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            'note': 'offline re-evaluation of saved predictions; no retriever fit/retrieve or API'}


def write_audit_result(result, output):
    """Write audit.json + summary.csv into a NEW directory (never overwrite)."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (out/'summary.csv').open('x') as f:
        w = csv.DictWriter(f, fieldnames=list(result['summary'][0]))
        w.writeheader()
        w.writerows(result['summary'])
    print(f"Audited {len(result['summary'])} method/protocol groups; "
          f"n={len(result['valid_qids'])}, feasible={len(result['feasible_qids'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--budget', required=True)
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--families', choices=['none', 'six', 'two', 'both'], default='none')
    args = parser.parse_args()
    result = audit(args.gold, args.budget, args.predictions, args.families)
    write_audit_result(result, args.output)


if __name__ == '__main__':
    main()
