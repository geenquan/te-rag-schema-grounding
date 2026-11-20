"""Final V2 statistics over the declared families
(eval/FAMILIES_DECLARED_V2.json, hash-frozen sidecar).

THREE ablation tracks as SEPARATE declared families (V2 protocol):
- T1 fixed_ablation          selected Full - fixed ablation (same frozen
                             params, switch off). Confirmatory.
- T2 optimized_ablation      optimized_full - optimized_no<member>, the
                             REQUIRED direct comparisons absent from v1
                             (v1 defect b). Confirmatory; a partial
                             optimized track is a REFUSAL, never a silent
                             drop.
- T3 fixed_vs_optimized_ablation  fixed ablation - re-optimized
                             counterpart (same switch). Retained from v1
                             but DEMOTED: labeled extra_exploratory (a
                             fixed-vs-reoptimized difference mixes module
                             removal with re-optimization).
Plus the main families per protocol (selected_for[protocol] vs 10/6 paper
baselines + ORIGINAL_FULL + the 3 v1 config rows). Every comparison whose
opponent is a v1_post_hoc row carries label post_hoc_exploratory (v1
defect d: v1 configs were selected after v1's Dev inspection; V2-vs-v1
comparisons are exploratory, never confirmatory).

Reads ONLY saved outputs: the freeze JSON (method roles, selected_for,
family_extensions) and per-question metric CSVs under
<final_root>/metrics/<method>/. Imports the accepted statistics core
VERBATIM (evalkit/stats_core.py: exact db-cluster sign-flip, zero/cancel
taxonomy, db-cluster bootstrap CI seed 42 x 10000).

PUBLIC-PORT: the research copy resolved the families file from the run's
eval/ directory and the metrics root from the run's eval_output/ tree
(runpaths constants). This port takes both as explicit arguments: the
families JSON path plus the recorded sha256 it must match (sidecar file or
literal), and the final-run root holding metrics/ and stats/.

Rules
- The families file is immutable: its sha256 must match the recorded value
  passed alongside, else refuse.
- Per-family Bonferroni multiplies by the DECLARED K (frozen file) plus 2
  per freeze-declared extension member (vnext_extension_slots rule); never
  shrunk after seeing results; zero/cancel members keep their slot and K
  contribution.
- Computed comparisons beyond declared members + extensions => refuse
  (undeclared multiplicity would silently weaken the family).
- Paired population identity is re-verified per comparison (identical qid
  sets; cluster = db).
- No Q/H splicing: each protocol's family uses selected_for[protocol].
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from terag.engines.evalkit import stats_core  # verbatim accepted core

EXIT_OK, EXIT_REFUSED = 0, 3

# Fixed stock ablation set (engine switches) of any engine config.
# PUBLIC-PORT: inlined from the research runpaths module.
STOCK_ABLATIONS = ['noBIndex', 'noBHit', 'noB', 'noTableGraph', 'noFieldGraph',
                   'noGraph', 'noPattern', 'noC']

LABEL_CONFIRMATORY = 'confirmatory'
LABEL_POST_HOC = 'post_hoc_exploratory'
LABEL_EXTRA = 'extra_exploratory'

STAT_COLS = ['protocol', 'family', 'track', 'comparison', 'metric', 'n',
             'n_clusters', 'mean_diff_pp', 'positive_questions',
             'negative_questions', 'unchanged_questions',
             'nonzero_cluster_sums', 'status', 'label',
             'p_sign_flip_raw', 'p_bonferroni', 'K_family', 'test',
             'ci95_bootstrap_pp']


def verify_families(families_file, expected_sha256=None):
    """Load the families JSON after verifying its immutability.

    PUBLIC-PORT: the recorded sha256 is passed alongside (a literal hex
    string or a sidecar FILE whose stripped text is the hash) instead of
    being read from a run-relative sidecar path. Refuses (exit 3) on any
    mismatch.
    """
    families_file = Path(families_file)
    if not families_file.is_file():
        print(f'stats_v2: REFUSED - families file missing: {families_file}',
              file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    if expected_sha256 is None:
        sidecar = families_file.with_suffix('.sha256')
        if not sidecar.is_file():
            print('stats_v2: REFUSED - no families sha256 recorded '
                  '(pass expected_sha256 or provide a .sha256 sidecar)',
                  file=sys.stderr)
            sys.exit(EXIT_REFUSED)
        expected_sha256 = sidecar.read_text(encoding='utf-8').strip()
    elif Path(expected_sha256).is_file():
        expected_sha256 = Path(expected_sha256).read_text(encoding='utf-8').strip()
    sha = hashlib.sha256(families_file.read_bytes()).hexdigest()
    if sha != expected_sha256:
        print('stats_v2: REFUSED - families json hash does not '
              'match its declaration (edited after declaration)',
              file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    return json.loads(families_file.read_text(encoding='utf-8'))


def load_per_question(metrics_root, method, protocols):
    """{(protocol, qid): row} for one method; refuses on duplicate/missing."""
    path = Path(metrics_root) / method / 'per_question.csv'
    if not path.is_file():
        raise FileNotFoundError(f'metrics missing: {path}')
    out = {}
    for row in csv.DictReader(path.open(encoding='utf-8')):
        if protocols and row['protocol'] not in protocols:
            continue
        key = (row['protocol'], row['qid'])
        if key in out:
            raise RuntimeError(f'{method}: duplicate metric row {key}')
        for k in row:
            if k not in ('protocol', 'qid', 'status', 'db'):
                try:
                    row[k] = float(row[k])
                except (TypeError, ValueError):
                    pass
        out[key] = row
    return out


def method_by(freeze, role=None, name=None, member=None, ablation_of=None,
              also_tradeoff=None):
    out = []
    for m in freeze['methods']:
        if role is not None and m.get('role') != role:
            continue
        if name is not None and m['name'] != name:
            continue
        if member is not None and m.get('ablation_member') != member:
            continue
        if ablation_of is not None and m.get('ablation_of') != ablation_of:
            continue
        if also_tradeoff is not None and \
                bool(m.get('also_tradeoff_branch')) != also_tradeoff:
            continue
        out.append(m)
    return out


def comparison_row(protocol, family, track, label, metric, a_rows, b_rows):
    """a - b paired per question; exact sign-flip + Bonferroni + CI."""
    common = sorted(k for k in a_rows if k in b_rows)
    if not common or len(common) != len(a_rows) or len(common) != len(b_rows):
        raise RuntimeError(
            f'{protocol}/{family}: population mismatch '
            f'({len(a_rows)} vs {len(b_rows)} vs common {len(common)})')
    diffs = [a_rows[k][metric] - b_rows[k][metric] for k in common]
    clusters = [a_rows[k]['db'] for k in common]
    by = {}
    for d, c in zip(diffs, clusters):
        by[c] = by.get(c, 0.0) + float(d)
    cluster_sums = [by[c] for c in sorted(by)]
    status = stats_core.classify_difference(diffs, cluster_sums)
    row = {'protocol': protocol, 'family': family, 'track': track,
           'comparison': '', 'metric': metric, 'n': len(common),
           'n_clusters': len(by),
           'mean_diff_pp': float(sum(diffs) / len(diffs)) * 100.0,
           'positive_questions': sum(1 for d in diffs if d > 1e-12),
           'negative_questions': sum(1 for d in diffs if d < -1e-12),
           'unchanged_questions': sum(1 for d in diffs
                                      if abs(d) <= 1e-12),
           'nonzero_cluster_sums': sum(1 for s in cluster_sums
                                       if abs(s) > 1e-12),
           'status': status, 'label': label, 'p_sign_flip_raw': '',
           'p_bonferroni': '', 'K_family': '', 'test': '',
           'ci95_bootstrap_pp': ''}
    if status == 'computed':
        p, test = stats_core.sign_flip(cluster_sums)
        row['p_sign_flip_raw'] = float(p)
        row['test'] = test
        ci, _ = stats_core.cluster_bootstrap_ci(diffs, clusters)
        row['ci95_bootstrap_pp'] = [ci[0] * 100.0, ci[1] * 100.0]
    return row


def attach_bonferroni(rows, K):
    for r in rows:
        r['K_family'] = K
        if r['status'] == 'computed' and r['p_sign_flip_raw'] != '':
            r['p_bonferroni'] = min(1.0, float(r['p_sign_flip_raw']) * K)


def effective_members(freeze, protocol, family):
    """Declared stock members + freeze-time extension members of one
    ablation family/protocol, and the effective K (declared + 2/ext)."""
    fam_decl = freeze['_families']['protocols'][protocol][family]
    members = list(fam_decl['members'])
    ext = (freeze.get('family_extensions') or {}).get(
        protocol, {}).get(family, [])
    return members + list(ext), fam_decl['K'] + 2 * len(ext)


def compute_main(protocol, spec, freeze, metrics_root):
    sel_name = freeze['selected_for'][protocol]
    sel = method_by(freeze, role='selected', name=sel_name)
    if len(sel) != 1:
        sys.exit(f'stats_v2: selected_for[{protocol}]={sel_name!r} not '
                 'found exactly once in the freeze')
    a = load_per_question(metrics_root, sel_name, [protocol])
    a_p = {k: v for k, v in a.items() if k[0] == protocol}
    post_hoc_set = set(spec.get('post_hoc_opponents') or [])
    rows, missing = [], []
    for opp in spec['opponents']:
        if opp == 'ORIGINAL_FULL':
            m = method_by(freeze, role='original_full')
            if len(m) != 1:
                sys.exit('stats_v2: exactly one original_full required')
            name, label = m[0]['name'], f'{sel_name}-ORIGINAL_FULL'
        elif opp.startswith('V1:'):
            v1 = opp[3:]
            m = method_by(freeze, role='v1_post_hoc', name=v1)
            if len(m) != 1:
                missing.append(f'V1:{v1}@{protocol}:not_in_freeze')
                continue
            name, label = v1, f'{sel_name}-V1:{v1}'
        else:
            name, label = opp, f'{sel_name}-{opp}'
        try:
            b = load_per_question(metrics_root, name, [protocol])
        except FileNotFoundError:
            missing.append(f'{name}@{protocol}:not_run')
            continue
        b_p = {k: v for k, v in b.items() if k[0] == protocol}
        if not b_p:
            missing.append(f'{name}@{protocol}:not_run')
            continue
        row_label = LABEL_POST_HOC if opp in post_hoc_set else \
            LABEL_CONFIRMATORY
        for metric in ['colf1', 'covered@5']:
            r = comparison_row(protocol, 'main', '-', row_label, metric,
                               a_p, b_p)
            r['comparison'] = label
            rows.append(r)
    return rows, missing


def _ablation_anchor(freeze, protocol):
    """selected config of the protocol + its fixed members by switch."""
    sel_name = freeze['selected_for'][protocol]
    fixed = {m['ablation_member']: m for m in
             method_by(freeze, role='fixed_ablation', ablation_of=sel_name)}
    fulls = method_by(freeze, role='optimized_ablation', member='Full',
                      ablation_of=sel_name)
    return sel_name, fixed, fulls


def compute_t1(protocol, spec, freeze, metrics_root):
    """T1 fixed-vs-fixed: selected Full - fixed ablation members."""
    sel_name, fixed, _fulls = _ablation_anchor(freeze, protocol)
    members, K = effective_members(freeze, protocol, 'fixed_ablation')
    a = load_per_question(metrics_root, sel_name, [protocol])
    a_p = {k: v for k, v in a.items() if k[0] == protocol}
    rows, missing = [], []
    for member in members:
        m = fixed.get(member)
        if m is None:
            missing.append(member)
            continue
        b = load_per_question(metrics_root, m['name'], [protocol])
        b_p = {k: v for k, v in b.items() if k[0] == protocol}
        for metric in ['colf1', 'covered@5']:
            r = comparison_row(protocol, 'fixed_ablation', 'T1',
                               LABEL_CONFIRMATORY, metric, a_p, b_p)
            r['comparison'] = f'{sel_name}-{m["name"]}'
            rows.append(r)
    return rows, missing, K


def compute_t2(protocol, spec, freeze, metrics_root):
    """T2 optimized-vs-optimized: optimized_full - optimized_no<member>.

    REQUIRED (v1 defect b): when the optimized track exists for the
    protocol, EVERY declared member must be present and computed; anything
    missing is a REFUSAL (exit 3), never a silent drop. Track absent =>
    family not computed, loudly recorded in the audit.
    """
    sel_name, _fixed, fulls = _ablation_anchor(freeze, protocol)
    if not fulls:
        return None, {'state': 'track_absent'}
    if len(fulls) > 1:
        sys.exit(f'stats_v2: {protocol}: {len(fulls)} optimized Fulls bound '
                 f'to {sel_name!r} - T2 pairing is ambiguous, refusing')
    full = fulls[0]
    members, K = effective_members(freeze, protocol, 'optimized_ablation')
    kids = {m['ablation_member']: m for m in
            method_by(freeze, role='optimized_ablation',
                      ablation_of=full['name'])}
    absent = [mm for mm in members if mm not in kids]
    if absent:
        print(f'stats_v2: REFUSED - T2 family is REQUIRED but {protocol} '
              f'optimized track lacks members {absent} under '
              f'{full["name"]!r}', file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    a = load_per_question(metrics_root, full['name'], [protocol])
    a_p = {k: v for k, v in a.items() if k[0] == protocol}
    rows = []
    for member in members:
        m = kids[member]
        b = load_per_question(metrics_root, m['name'], [protocol])
        b_p = {k: v for k, v in b.items() if k[0] == protocol}
        for metric in ['colf1', 'covered@5']:
            r = comparison_row(protocol, 'optimized_ablation', 'T2',
                               LABEL_CONFIRMATORY, metric, a_p, b_p)
            r['comparison'] = f'{full["name"]}-{m["name"]}'
            rows.append(r)
    return rows, {'state': 'computed', 'optimized_full': full['name'],
                  'members': members}


def compute_t3(protocol, spec, freeze, metrics_root):
    """T3 fixed-vs-reoptimized, labeled extra_exploratory (EXTRA track)."""
    sel_name, fixed, fulls = _ablation_anchor(freeze, protocol)
    if not fulls:
        return None, {'state': 'track_absent'}
    full = fulls[0]
    members, K = effective_members(freeze, protocol,
                                   'fixed_vs_optimized_ablation')
    kids = {m['ablation_member']: m for m in
            method_by(freeze, role='optimized_ablation',
                      ablation_of=full['name'])}
    rows, missing = [], []
    for member in members:
        f = fixed.get(member)
        o = kids.get(member)
        if f is None or o is None:
            missing.append(member)
            continue
        aa = load_per_question(metrics_root, f['name'], [protocol])
        bb = load_per_question(metrics_root, o['name'], [protocol])
        aa_p = {k: v for k, v in aa.items() if k[0] == protocol}
        bb_p = {k: v for k, v in bb.items() if k[0] == protocol}
        for metric in ['colf1', 'covered@5']:
            r = comparison_row(protocol, 'fixed_vs_optimized_ablation',
                               'T3', LABEL_EXTRA, metric, aa_p, bb_p)
            r['comparison'] = f'{f["name"]}-vs-opt-{o["name"]}'
            rows.append(r)
    return rows, {'state': 'computed', 'missing': missing}


def compute_tradeoff(protocol, fam_spec, freeze, metrics_root):
    tb = method_by(freeze, role='tradeoff_branch') or \
        method_by(freeze, also_tradeoff=True)
    of = method_by(freeze, role='original_full')
    if len(tb) != 1 or len(of) != 1:
        return None, {'state': 'absent'}
    a = load_per_question(metrics_root, tb[0]['name'], [protocol])
    b = load_per_question(metrics_root, of[0]['name'], [protocol])
    a_p = {k: v for k, v in a.items() if k[0] == protocol}
    b_p = {k: v for k, v in b.items() if k[0] == protocol}
    rows = []
    for metric in ['colf1', 'covered@5']:
        r = comparison_row(protocol, 'tradeoff_branch', '-',
                           LABEL_CONFIRMATORY, metric, a_p, b_p)
        r['comparison'] = f'{tb[0]["name"]}-ORIGINAL_FULL'
        rows.append(r)
    return rows, {'state': 'computed'}


def write_rows(path, rows):
    lines = [','.join(STAT_COLS)]
    for r in rows:
        cells = []
        for c in STAT_COLS:
            v = r[c]
            if isinstance(v, list):
                v = f'"[{v[0]:.4f},{v[1]:.4f}]"'  # quote: CI contains a comma
            elif isinstance(v, float):
                v = f'{v:.6f}'
            cells.append(str(v))
        lines.append(','.join(cells))
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def check_extension_consistency(freeze):
    """Freeze-recorded extensions must equal the extensions derivable from
    the freeze's own methods (declare/build agreement). All-empty protocol
    entries are normalized away on both sides before comparing."""
    def drop_empty(d):
        return {p: {k: v for k, v in fam.items() if v}
                for p, fam in d.items() if any(fam.values())}
    derived = {}
    for proto in ('question', 'heldout'):
        sel_name = freeze['selected_for'][proto]
        have = {m['ablation_member'] for m in
                method_by(freeze, role='fixed_ablation', ablation_of=sel_name)}
        ext_f = sorted(have - set(STOCK_ABLATIONS))
        ext_o = set()
        for f in method_by(freeze, role='optimized_ablation', member='Full',
                           ablation_of=sel_name):
            kids = {m['ablation_member'] for m in
                    method_by(freeze, role='optimized_ablation',
                              ablation_of=f['name'])}
            ext_o |= kids - set(STOCK_ABLATIONS)
        if ext_f or ext_o:
            derived[proto] = {'fixed_ablation': ext_f,
                              'optimized_ablation': sorted(ext_o)}
    recorded = drop_empty(freeze.get('family_extensions') or {})
    if drop_empty(derived) != recorded:
        sys.exit('stats_v2: freeze family_extensions disagree with the '
                 'freeze methods - refusing')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--final-root', required=True,
                    help='final-run root holding metrics/ and stats/')
    ap.add_argument('--families', required=True,
                    help='declared families JSON (immutable)')
    ap.add_argument('--families-sha256', default=None,
                    help='recorded sha256 of the families JSON (literal hex '
                         'or sidecar file; default: <families>.sha256)')
    args = ap.parse_args(argv)

    families = verify_families(args.families, args.families_sha256)
    freeze = json.loads(Path(args.freeze).read_text(encoding='utf-8'))
    freeze['_families'] = families  # declaration in reach of helpers
    fid = freeze['freeze_id']
    final_root = Path(args.final_root)
    metrics_root = final_root / 'metrics'
    stats_root = final_root / 'stats'
    stats_root.mkdir(parents=True, exist_ok=True)
    protocols = sorted(families['protocols'])
    check_extension_consistency(freeze)

    all_rows, audit = [], {}
    for protocol in protocols:
        proto_decl = families['protocols'][protocol]

        rows, missing = compute_main(protocol, proto_decl['main'], freeze,
                                     metrics_root)
        K = proto_decl['main']['K']
        if len(rows) > K:
            sys.exit(f'stats_v2: main/{protocol} computed {len(rows)} '
                     f'comparisons > declared K {K} - undeclared '
                     'multiplicity, refusing')
        attach_bonferroni(rows, K)
        all_rows += rows
        audit[f'main/{protocol}'] = {'declared_K': K,
                                     'computed': len(rows),
                                     'missing': missing}
        write_rows(stats_root / f'main__{protocol}.csv', rows)

        rows, missing, K = compute_t1(protocol, proto_decl['fixed_ablation'],
                                      freeze, metrics_root)
        if len(rows) > K:
            sys.exit(f'stats_v2: fixed_ablation/{protocol} computed '
                     f'{len(rows)} > effective K {K} - refusing')
        attach_bonferroni(rows, K)
        all_rows += rows
        audit[f'fixed_ablation/{protocol}'] = {
            'track': 'T1 fixed-vs-fixed', 'declared_K': K,
            'computed': len(rows), 'missing': missing}
        write_rows(stats_root / f'fixed_ablation__{protocol}.csv', rows)

        rows, t2_audit = compute_t2(protocol,
                                    proto_decl['optimized_ablation'],
                                    freeze, metrics_root)
        if rows is None:
            audit[f'optimized_ablation/{protocol}'] = {
                'track': 'T2 optimized-vs-optimized',
                'state': t2_audit['state'],
                'note': 'optimized track absent from the freeze - the '
                        'REQUIRED direct comparisons were not produced; '
                        'recorded, not silently dropped'}
            write_rows(stats_root / f'optimized_ablation__{protocol}.csv',
                       [])
        else:
            members, K = effective_members(freeze, protocol,
                                           'optimized_ablation')
            if len(rows) != 2 * len(members):
                sys.exit(f'stats_v2: optimized_ablation/{protocol} computed '
                         f'{len(rows)} != 2*{len(members)} required pairs')
            attach_bonferroni(rows, K)
            all_rows += rows
            audit[f'optimized_ablation/{protocol}'] = {
                'track': 'T2 optimized-vs-optimized', 'declared_K': K,
                'computed': len(rows), **t2_audit}
            write_rows(stats_root / f'optimized_ablation__{protocol}.csv',
                       rows)

        rows, t3_audit = compute_t3(
            protocol, proto_decl['fixed_vs_optimized_ablation'], freeze,
            metrics_root)
        if rows is None:
            audit[f'fixed_vs_optimized_ablation/{protocol}'] = {
                'track': 'T3 fixed-vs-reoptimized', 'label': LABEL_EXTRA,
                'state': t3_audit['state']}
            write_rows(
                stats_root / f'fixed_vs_optimized_ablation__{protocol}.csv',
                [])
        else:
            members, K = effective_members(
                freeze, protocol, 'fixed_vs_optimized_ablation')
            attach_bonferroni(rows, K)
            all_rows += rows
            audit[f'fixed_vs_optimized_ablation/{protocol}'] = {
                'track': 'T3 fixed-vs-reoptimized', 'label': LABEL_EXTRA,
                'declared_K': K, 'computed': len(rows),
                'missing': t3_audit.get('missing', [])}
            write_rows(
                stats_root / f'fixed_vs_optimized_ablation__{protocol}.csv',
                rows)

        if freeze.get('tradeoff_branch_present'):
            rows, tb_audit = compute_tradeoff(
                protocol, families['tradeoff_branch'], freeze, metrics_root)
            if rows:
                attach_bonferroni(rows, families['tradeoff_branch']['K'])
                all_rows += rows
                audit[f'tradeoff_branch/{protocol}'] = tb_audit
                write_rows(stats_root / f'tradeoff_branch__{protocol}.csv',
                           rows)

    payload = {'freeze_id': fid, 'freeze_sha256':
               hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest(),
               'families_sha256':
                   hashlib.sha256(Path(args.families).read_bytes()).hexdigest(),
               'n_comparisons': len(all_rows), 'audit': audit,
               'labels': {'confirmatory': LABEL_CONFIRMATORY,
                          'post_hoc': LABEL_POST_HOC,
                          'extra_exploratory': LABEL_EXTRA},
               'note': 'Bonferroni per declared family K (no shrinkage; '
                       'freeze-time extension slots add 2 K each); CI is '
                       'descriptive db-cluster bootstrap (seed 42, 10000 '
                       'resamples), not the formal test. T3 and every '
                       'v1-opponent comparison are exploratory by label.',
               'rows': all_rows}
    (stats_root / 'statistics.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8')
    n_sig = sum(1 for r in all_rows
                if r['p_bonferroni'] != '' and r['p_bonferroni'] < 0.05)
    print(json.dumps({'stats': str(stats_root / 'statistics.json'),
                      'n_comparisons': len(all_rows),
                      'n_bonferroni_significant': n_sig,
                      'audit': audit}))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
