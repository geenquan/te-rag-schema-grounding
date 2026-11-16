"""
SQL context construction (established during revision; corrected before
the formal experiments).

Turns one retrieval result into the structured context required by
downstream SQL generation:
- the question, table columns, and **the real schema descriptions** (only
  the columns actually returned within budget; never silently expanded);
- hit annotations (channel B direct / channel C transferred, with source
  qids) — bound only to actually returned columns;
- predicate slots (operator + value class; **training date/institution
  instances never become default conditions for a new question**, they
  are always given as slots; CODE gets no value);
- table-relation evidence at two levels:
  * join_conditions: only from **AST join predicates of training SQL**
    (a.x=b.y) — same-named id columns are not join evidence;
  * cooccurrence: two tables co-occurring in a training SQL (reference
    evidence only, not a join condition).

No LLM calls, no gold-based column completion, no budget expansion; the
output is saved for later stages.
"""

from typing import Dict, List


def _train_join_evidence(train_data, schema_cols) -> Dict:
    """AST join predicates of training SQL: {(t1,c1,t2,c2) -> number of
    supporting questions}."""
    from terag.sql_role_parser import SQLRoleParser
    import sqlglot
    from sqlglot import exp as _exp
    evidence = {}
    if train_data is None:
        return evidence
    parser = SQLRoleParser(dialect='mysql')
    for _, row in train_data.iterrows():
        pr = parser.parse(str(row.get('sql', '')), schema_columns=schema_cols)
        if not pr.ast_parsed:
            continue
        sql = str(row.get('sql', ''))
        tree = None
        for d in ('mysql', 'hive'):
            try:
                tree = sqlglot.parse_one(sql, dialect=d)
                break
            except Exception:  # noqa: BLE001
                continue
        if tree is None:
            continue
        # alias -> physical table within each SELECT scope
        for select in tree.find_all(_exp.Select):
            tables = {}
            frm = select.args.get('from_') or select.args.get('from')
            sources = [frm.this] if frm is not None else []
            sources += [j.this for j in (select.args.get('joins') or [])]
            for src in sources:
                if isinstance(src, _exp.Table) and src.name:
                    tables[(src.alias or src.name).lower()] = src.name
            n_phys = len({t for t in tables.values()})
            for join in (select.args.get('joins') or []):
                on = join.args.get('on')
                if on is None:
                    continue
                for eq in on.find_all(_exp.EQ):
                    cols = [c for c in eq.find_all(_exp.Column)]
                    if len(cols) != 2:
                        continue
                    resolved = []
                    for c in cols:
                        if c.table:
                            resolved.append(tables.get(c.table.lower()))
                        elif n_phys == 1:
                            resolved.append(next(iter(
                                {t for t in tables.values()})))
                        else:
                            resolved.append(None)
                    if all(resolved) and resolved[0] != resolved[1]:
                        key = (resolved[0], cols[0].name,
                               resolved[1], cols[1].name)
                        evidence[key] = evidence.get(key, 0) + 1
    return evidence


def build_sql_context(query: str, results: List, retriever=None) -> Dict:
    """Assemble retrieval results into a SQL generation context (see
    module docstring)."""
    miner = getattr(retriever, 'miner', None)
    transfer = getattr(retriever, 'transfer_result', None)
    train_data = getattr(retriever, 'train_data', None)
    field_df = getattr(retriever, 'field_df', None)

    # Column descriptions (real schema; only for actually returned columns)
    desc_map = {}
    if field_df is not None:
        for _, r in field_df.iterrows():
            desc_map[(str(r['table']).lower(),
                      str(r['field_name']).lower())] = str(
                r.get('field_name_desc', '') or '')

    schema_cols = miner.schema_columns if miner is not None else {}
    join_ev = _train_join_evidence(train_data, schema_cols)

    # Training co-occurrence (reference evidence, not a join condition)
    co_occur = set()
    if miner is not None and train_data is not None:
        from terag.sql_role_parser import SQLRoleParser
        parser = SQLRoleParser(dialect='mysql')
        for _, row in train_data.iterrows():
            pr = parser.parse(str(row.get('sql', '')),
                              schema_columns=schema_cols)
            if not pr.ast_parsed:
                continue
            tabs = pr.get_gold_tables()
            for i, t1 in enumerate(tabs):
                for t2 in tabs[i + 1:]:
                    co_occur.add((min(t1, t2), max(t1, t2)))

    c_terms_by_target = {}
    if transfer is not None:
        for r in transfer.get('accepted', []):
            if r.get('status') == 'accepted' and r.get('source_type') == 'TERM' \
                    and isinstance(r.get('adapted_content'), str):
                c_terms_by_target.setdefault(r['target_element'], []).append(
                    {'term': r['adapted_content'],
                     'source': r['source_element'],
                     'sim': r['similarity']['total']})

    returned_tables = [r.table for r in results]
    tables_out = []
    for r in results:
        t = r.table
        returned_cols = {c.split('.')[-1] for c, _ in r.columns}
        cols = []
        for c, s in r.columns:
            fname = c.split('.')[-1]
            cols.append({'name': fname,
                         'desc': desc_map.get((t.lower(), fname.lower()), ''),
                         'score': round(float(s), 4)})

        # Annotations bind only to actually returned columns
        anns = []
        if miner is not None:
            for a in miner.annotations:
                if a['type'] == 'TERM' and a.get('status') == 'confirmed':
                    tgt = a.get('target', '')
                    tt, tc = tgt.split('.', 1) if '.' in tgt else (tgt, '')
                    if tt == t.lower() and tc in {c.lower()
                                                  for c in returned_cols}:
                        anns.append({'term': str(a['content']),
                                     'column': tc,
                                     'channel': 'B',
                                     'rule': a.get('rule'),
                                     'source_qids': a.get('source_qids', [])})
        for col_key, items in c_terms_by_target.items():
            tt, tc = col_key.split('.', 1)
            if tt == t.lower() and tc in {c.lower() for c in returned_cols}:
                for it in items:
                    anns.append({'term': it['term'], 'column': tc,
                                 'channel': 'C',
                                 'source_element': it['source'],
                                 'sim': it['sim']})

        # Predicate slots: bound only to actually returned columns;
        # training date/institution instances become slots (no value given)
        slots = []
        if miner is not None:
            for a in miner.annotations:
                if a['type'] == 'PREDICATE' and a.get('status') == 'confirmed':
                    tgt = a.get('target', '')
                    tt, tc = tgt.split('.', 1) if '.' in tgt else (tgt, '')
                    if tt == t.lower() and tc in {c.lower()
                                                  for c in returned_cols}:
                        c_ = a['content']
                        slots.append({'column': tc,
                                      'operator': c_['operator'],
                                      'value_class': c_['value_class'],
                                      'value': None,
                                      'note': ('training instance values are '
                                               'not default conditions for a new '
                                               'question; CODE meaning awaits '
                                               'target evidence')})

        # Join conditions (AST join predicates only) + co-occurrence reference
        join_conditions = []
        for (t1, c1, t2, c2), sup in join_ev.items():
            if t1 == t.lower() and t2 in [x.lower() for x in returned_tables]:
                join_conditions.append({'left': f"{t1}.{c1}",
                                        'right': f"{t2}.{c2}",
                                        'support': sup})
            elif t2 == t.lower() and t1 in [x.lower() for x in returned_tables]:
                join_conditions.append({'left': f"{t2}.{c2}",
                                        'right': f"{t1}.{c1}",
                                        'support': sup})
        cooccurrence = []
        for other in returned_tables:
            if other == t:
                continue
            pair = (min(t.lower(), other.lower()),
                    max(t.lower(), other.lower()))
            if pair in co_occur:
                cooccurrence.append({'table': other,
                                     'evidence': 'cooccurrence'})

        tables_out.append({'table': t,
                           'score': round(float(r.table_score), 4),
                           'columns': cols,
                           'annotations': anns,
                           'predicate_slots': slots,
                           'join_conditions': join_conditions,
                           'cooccurrence': cooccurrence})

    return {
        'query': query,
        'tables': tables_out,
        'budget_note': ('tables<=5, columns<=10 per table (the budget the '
                        'retriever actually enforces); no LLM was called for '
                        'this context, ExecAcc=not_run'),
        'annotation_channels': {'B': miner is not None,
                                'C': transfer is not None},
    }
