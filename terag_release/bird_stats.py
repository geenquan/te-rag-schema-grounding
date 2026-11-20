"""BIRD paired statistics over a run's per-question metrics, using the
frozen declared families.

Reimplements the final evaluation's statistical protocol directly on the
run's per-question metric CSVs (`<run>/metrics/<method>/<protocol>
_per_question.csv`, written by `run-bird`):

* pairing: per-question differences on the frozen Dev OOF population; the
  qid sets of the two methods must be IDENTICAL (population identity);
* cluster = database (from the qid prefix before ``::``);
* test: db-cluster EXACT sign-flip (the accepted prior-round
  implementation, ported verbatim in ``terag.engines.evalkit.stats_core``);
  Monte Carlo fallback only when m > 20 clusters;
* taxonomy: zero_difference / cancels_within_cluster / computed;
* CI: db-cluster bootstrap (10,000 resamples, seed 42, percentile
  2.5/97.5) — descriptive, unadjusted;
* families: the frozen declaration
  (``configs/paper/final_20261120/bird_families.json`` + sha sidecar — a
  mismatch refuses everything); Bonferroni multiplies by the DECLARED K,
  never shrunk; zero/cancel members keep their slot;
* labels: confirmatory; post_hoc_exploratory for the ``V1:*`` opponents;
  extra_exploratory for the fixed-vs-reoptimized track;
* direction convention: Delta = Full − Ablation (A − B where A is the
  declared left side).
"""
import csv
import json
from pathlib import Path

from terag_release import reporting
from terag_release.bird_runner import (FROZEN_FAMILIES_PATH,
                                       FROZEN_FAMILIES_SHA)

METRICS = ('colf1', 'covered@5')
LABEL_CONFIRMATORY = 'confirmatory'
LABEL_POST_HOC = 'post_hoc_exploratory'
LABEL_EXTRA = 'extra_exploratory'


def verify_families():
    import hashlib
    data = FROZEN_FAMILIES_PATH.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    side = FROZEN_FAMILIES_SHA.read_text(encoding='utf-8').strip()
    if sha != side:
        raise RuntimeError('bird_families.json hash does not match its '
                           'declaration sidecar (edited after declaration?); '
                           'refusing to compute anything')
    return json.loads(data.decode('utf-8')), sha


def _load_per_question(run_dir, method, protocol):
    path = (Path(run_dir) / 'metrics' / method
            / f'{protocol}_per_question.csv')
    if not path.is_file():
        return None
    out = {}
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['qid']] = row
    return out


def _db_of(qid):
    return qid.split('::', 1)[0]


def _comparison(a_rows, b_rows, metric, label):
    """Paired A − B differences, db clusters, sign-flip + bootstrap."""
    from terag.engines.evalkit.stats_core import (classify_difference,
                                                  cluster_bootstrap_ci,
                                                  sign_flip)
    import numpy as np
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        return {'status': 'missing_population', 'label': label,
                'metric': metric, 'n': 0}
    if set(a_rows) != set(b_rows):
        return {'status': 'population_mismatch', 'label': label,
                'metric': metric, 'n': len(common)}
    diffs = np.array([float(a_rows[q][metric]) - float(b_rows[q][metric])
                      for q in common])
    clusters = [_db_of(q) for q in common]
    by = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(float(d))
    names = sorted(by)
    sums = [sum(by[c]) for c in names]
    taxonomy = classify_difference(diffs, np.array(sums))
    p, test = (None, None)
    if taxonomy == 'computed':
        p, test = sign_flip(sums)
    ci, _ = cluster_bootstrap_ci(diffs, clusters)
    return {'status': taxonomy, 'label': label, 'metric': metric,
            'n': len(common), 'n_clusters': len(names),
            'mean_diff_pp': float(diffs.mean() * 100),
            'positive_questions': int((diffs > 0).sum()),
            'negative_questions': int((diffs < 0).sum()),
            'unchanged_questions': int((diffs == 0).sum()),
            'nonzero_cluster_sums': int(sum(1 for s in sums
                                            if abs(s) > 1e-12)),
            'p_sign_flip_raw': p, 'test': test,
            'ci95_bootstrap_pp': [v * 100 for v in ci]}


def run_statistics(run_dir, output):
    families, sha = verify_families()
    run_dir = Path(run_dir)
    out = reporting.require_new_directory(Path(output))
    frozen = json.loads((Path(__file__).parent
                         .joinpath('..', 'configs', 'paper', 'final_20261120',
                                   'bird_final_configs.json')
                         .resolve()).read_text(encoding='utf-8'))

    rows = []
    for protocol in ('question', 'heldout'):
        spec = families['protocols'][protocol]
        selected_name = frozen['selected'][protocol]['name']
        selected = _load_per_question(run_dir, selected_name, protocol)

        def ensure(method, label):
            rows_local = _load_per_question(run_dir, method, protocol)
            if rows_local is None:
                rows.append({'protocol': protocol, 'family': 'missing',
                             'comparison': f'{selected_name} - {method}',
                             'metric': None, 'n': 0, 'status': 'missing',
                             'label': label,
                             'note': 'per-question metrics not in this run'})
                return None
            return rows_local

        # ---- main family ------------------------------------------------
        fam = spec['main']
        for opp in fam['opponents']:
            method = ('ORIGINAL_FULL' if opp == 'ORIGINAL_FULL'
                      else opp.split(':', 1)[1] if opp.startswith('V1:')
                      else opp)
            label = (LABEL_POST_HOC if opp in fam.get('post_hoc_opponents',
                                                      [])
                     else LABEL_CONFIRMATORY)
            opp_rows = ensure(method, label)
            if opp_rows is None or selected is None:
                continue
            for metric in METRICS:
                r = _comparison(selected, opp_rows, metric, label)
                r.update(protocol=protocol, family='main',
                         comparison=f'{selected_name} - {method}',
                         K_family=fam['K'])
                r['p_bonferroni'] = (min(1.0, r['p_sign_flip_raw']
                                         * fam['K'])
                                     if r.get('p_sign_flip_raw') is not None
                                     else None)
                rows.append(r)

        # ---- ablation tracks --------------------------------------------
        # T1 members = the 8 declared stock switches PLUS every mechanism
        # off-switch appended at freeze time (entries of the frozen configs'
        # fixed_ablation_members named <selected>-X with X not a stock
        # switch, e.g. field_filter_gate_off / hg_gate_off). Per the
        # declaration's vnext_extension_slots rule each appended member
        # adds 2 to the family's Bonferroni K (16 -> 18 with one member).
        stock = spec['fixed_ablation']['members']
        frozen_fixed = [e['name'] for e in frozen.get(
            'fixed_ablation_members') or []]
        mechanism_members = [n[len(selected_name) + 1:]
                             for n in frozen_fixed
                             if n.startswith(selected_name + '-')
                             and n[len(selected_name) + 1:] not in stock]
        t1_members = list(stock) + mechanism_members
        t1_k = spec['fixed_ablation']['K'] + 2 * len(mechanism_members)

        for family, left_name, right_of, label, members, fam_k in (
                ('fixed_ablation', selected_name,
                 lambda m: f'{selected_name}-{m}', LABEL_CONFIRMATORY,
                 t1_members, t1_k),
                ('optimized_ablation',
                 f'optimized_full_{"Q" if protocol == "question" else "H"}',
                 lambda m: (f'optimized_{"Q" if protocol == "question" else "H"}_'
                            f'{m}'), LABEL_CONFIRMATORY,
                 spec['optimized_ablation']['members'],
                 spec['optimized_ablation']['K']),
                ('fixed_vs_optimized_ablation',
                 f'{selected_name}-{{}}',
                 lambda m: (f'optimized_{"Q" if protocol == "question" else "H"}_'
                            f'{m}'), LABEL_EXTRA,
                 spec['fixed_vs_optimized_ablation']['members'],
                 spec['fixed_vs_optimized_ablation']['K'])):
            fam = spec[family]
            for member in members:
                left = (left_name.format(member) if '{' in left_name
                        else left_name)
                right = right_of(member)
                a = _load_per_question(run_dir, left, protocol)
                b = _load_per_question(run_dir, right, protocol)
                if a is None or b is None:
                    rows.append({'protocol': protocol, 'family': family,
                                 'comparison': f'{left} - {right}',
                                 'metric': None, 'n': 0, 'status': 'missing',
                                 'label': label, 'K_family': fam_k,
                                 'note': 'per-question metrics not in this '
                                         'run'})
                    continue
                for metric in METRICS:
                    r = _comparison(a, b, metric, label)
                    r.update(protocol=protocol, family=family,
                             comparison=f'{left} - {right}',
                             K_family=fam_k)
                    r['p_bonferroni'] = (min(1.0, r['p_sign_flip_raw']
                                             * fam_k)
                                         if r.get('p_sign_flip_raw')
                                         is not None else None)
                    rows.append(r)

    # CSV output
    cols = ['protocol', 'family', 'comparison', 'metric', 'n', 'n_clusters',
            'mean_diff_pp', 'positive_questions', 'negative_questions',
            'unchanged_questions', 'nonzero_cluster_sums', 'status',
            'label', 'p_sign_flip_raw', 'p_bonferroni', 'K_family', 'test',
            'ci95_bootstrap_pp']
    lines = [','.join(cols)]
    for r in rows:
        lines.append(','.join(
            json.dumps(r[c], ensure_ascii=False)
            if isinstance(r.get(c), (list, dict)) else str(r.get(c, ''))
            for c in cols))
    (out / 'bird_statistics.csv').write_text('\n'.join(lines) + '\n',
                                             encoding='utf-8')
    meta = {'families_sha256': sha,
            'direction_convention': 'Delta = left - right (Full - Ablation '
                                    'for ablation tracks)',
            'n_comparisons': len(rows),
            'missing': sum(1 for r in rows if r['status'] == 'missing')}
    reporting.write_json_exclusive(out / 'statistics_meta.json', meta)
    print(f"bird statistics: {len(rows)} comparison rows "
          f"({meta['missing']} missing members) -> {out}")
    return out
