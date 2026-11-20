"""Direct domain-annotation mining (P2-D1; the "direct evidence" subset of
the paper's Sec. 3.3 annotation generation).

Extracts traceable annotations from the CURRENT training fold's NLQ-SQL
pairs (B class: direct evidence from historical logs; distinct from the
A-class raw schema metadata and the C-class transfer annotations of stage
2).

Annotation types and rules (all explicit and interpretable, no external
models):
- TERM (business phrase -> table column):
  R1 select_alias - a term in a training SQL's SELECT output alias is
  aligned with that output column (the alias is the training evidence; the
  term must not already occur in the column name/description or the table
  name/description - metadata-covered terms are not duplicated, only the
  lexicon_overlap flag is kept). Status confirmed.
  R2 co-occurrence - a term t of the question outside the schema
  vocabulary co-occurs with a referenced column e:
  support(t,e)>=2 and concentration support(t,e)/sum_e'
  support(t,e') >= 0.8 -> confirmed; several elements tied within 20% ->
  ambiguous (recorded, not injected); support=1 -> rejected_single.
  No "every word of the sentence <-> every field" binding.
- TEMPLATE (table-level query template): the question is slot-normalized
  (date -> #DATE, number -> #NUM, ENTITY-class predicate value -> #ORG),
  aggregated per table, top K=5 by support.
- PREDICATE (predicate annotation): the training WHERE's (column,
  operator, value class[, value]); value classes DATE/ENTITY/CODE/
  THRESHOLD/UNKNOWN (rules in _classify_value). Code values without NL
  evidence only record the value co-occurrence; no business meaning is
  invented.
- ROLE (column role prior): the column role set from AST parsing (reuses
  sql_role_parser).

Every annotation: type/target/content/source_qids/rule/support/
confidence/status.
Index channel: table_annotation_texts() (confirmed TERM + TEMPLATE, fixed
weight field_weights.pattern); column-ranking channel: column_bonus()
(w_ann fixed at 0.5).
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from terag.engines.bird_runtime.sql_role_parser import SQLRoleParser, SQLGLOT_AVAILABLE

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None

W_ANN = 0.5            # column-ranking annotation hit bonus (fixed, declared before the run)
TEMPLATE_TOPK = 5      # templates injected per table (fixed)
TERM_MIN_SUPPORT = 2   # R2 minimum support
TERM_CONCENTRATION = 0.8
AMBIGUITY_WINDOW = 0.2

# Term-extraction stop terms (pure function words/verbs; organizations and
# numbers are handled by the slot rules). Runtime DATA: frozen Chinese
# stopword list of the accepted identity - kept byte-for-byte.
STOP_TERMS = {'的', '是', '有哪些', '多少', '哪些', '什么', '请问', '查询',
              '统计', '显示', '列出', '分析', '怎么', '如何'}

_DATE_RE = re.compile(r'\d{4}年\d{1,2}月|\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|\d{6}|\d{8}')
_NUM_RE = re.compile(r'\d+')
_CODE_RE = re.compile(r'^[A-Za-z0-9]{1,4}$')
_ORG_COL_RE = re.compile(r'org|name|dept|unit|station|company', re.I)


def _norm_key(table: str, col: str) -> str:
    return f"{table.lower()}.{col.lower()}"


class DirectAnnotationMiner:
    """Extracts direct annotations from the current training fold (rules in
    the module docstring)."""

    def __init__(self, field_df: pd.DataFrame, table_df: pd.DataFrame,
                 dialect: str = 'mysql'):
        self.field_df = field_df
        self.table_df = table_df
        self.dialect = dialect
        self.schema_columns = {t: set(g['field_name'].astype(str))
                               for t, g in field_df.groupby('table')}
        self.annotations: List[Dict] = []
        self._term_by_key: Dict[str, List[Tuple[str, float]]] = {}
        self._templates_by_table: Dict[str, List[Tuple[str, int]]] = {}
        self._entity_values: List[str] = []

    # ---------------------------------------------------------------- schema lexicon

    def _local_lexicon(self, table: str, col: Optional[str] = None) -> set:
        from terag.engines.bird_runtime.tokenization import tokenize_text
        words = set()
        trow = self.table_df[self.table_df['table'] == table]
        for _, r in trow.iterrows():
            words |= {w for w in tokenize_text(str(r.get('table_desc', '') or ''), 'chinese')
                      if len(w) > 1}
        cols = self.field_df[self.field_df['table'] == table]
        if col is not None:
            cols = cols[cols['field_name'] == col]
        for _, r in cols.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), 'chinese') if len(w) > 1}
        return words

    def _global_lexicon(self) -> set:
        from terag.engines.bird_runtime.tokenization import tokenize_text
        words = set()
        for _, r in self.field_df.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), 'chinese') if len(w) > 1}
        for _, r in self.table_df.iterrows():
            words |= {w for w in tokenize_text(
                str(r.get('table_desc', '') or ''), 'chinese') if len(w) > 1}
        return words

    # ---------------------------------------------------------------- mining entry

    def _append(self, ann: Dict) -> None:
        """Assemble the annotation and assign a stable annotation_id (for
        transfer provenance)."""
        ann['annotation_id'] = f"a{len(self.annotations):04d}"
        self.annotations.append(ann)

    def mine(self, train_df: pd.DataFrame) -> 'DirectAnnotationMiner':
        from terag.engines.bird_runtime.tokenization import tokenize_text
        assert SQLGLOT_AVAILABLE and sqlglot is not None, 'annotation mining requires sqlglot (AST)'
        parser = SQLRoleParser(dialect=self.dialect)
        glob_lex = self._global_lexicon()

        alias_pairs = defaultdict(lambda: defaultdict(set))  # (t,c) -> term -> {qids actually containing the alias}
        alias_meta = defaultdict(lambda: defaultdict(set))   # (t,c) -> term -> {full alias texts}
        cooc = defaultdict(lambda: defaultdict(set))  # term -> element_key -> {qid}
        templates = defaultdict(lambda: defaultdict(set))  # table -> norm_q -> {qid}
        tmpl_bindings = defaultdict(lambda: defaultdict(list))  # table -> norm_q -> [(element, role)]
        pred_counts = defaultdict(set)        # (t,c,op,vclass[,v]) -> {qid}
        role_sets = defaultdict(set)          # element -> roles
        role_qids = defaultdict(set)
        entity_vals = defaultdict(set)        # value -> {qid}

        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            sql = str(row.get('sql', ''))
            question = str(row.get('question', ''))
            pr = parser.parse(sql, schema_columns=self.schema_columns)
            if not pr.ast_parsed:
                continue
            tables = pr.get_gold_tables()

            # --- R1 alias alignment (output columns resolved in local scope) ---
            for t, c in pr.get_all_fields():
                for r_ in pr.field_roles[t][c].roles:
                    role_sets[(t, c)].add(r_)
                role_qids[(t, c)].add(qid)
            for term, target, alias_text in self._alias_terms(sql):
                alias_pairs[target][term].add(qid)
                alias_meta[target][term].add(alias_text)

            # --- R2 co-occurrence terms ---
            q_tokens = [w for w in tokenize_text(question, 'chinese') if len(w) > 1]
            unknown_terms = [w for w in q_tokens
                             if w not in glob_lex and w not in STOP_TERMS
                             and not _NUM_RE.search(w)]
            ref_cols = [k for k in pr.get_all_fields()]
            for term in set(unknown_terms):
                for key in ref_cols:
                    cooc[term][_norm_key(*key)].add(qid)

            # --- predicates ---
            for (t, c), op, val, vclass in self._predicates(sql):
                pred_counts[(t, c, op, vclass, val)].add(qid)
                if vclass == 'ENTITY':
                    entity_vals[val].add(qid)

            # --- templates (with the minimal binding structure: the
            #     (element, role) list referenced by this question) ---
            norm_q = self._normalize_template(question)
            for t in tables:
                templates[t][norm_q].add(qid)
                binds = sorted({( _norm_key(tt, cc), rr)
                                for (tt, cc) in pr.get_all_fields()
                                for rr in pr.field_roles[tt][cc].roles})
                tmpl_bindings[t][norm_q].append({'qid': qid, 'bindings': binds})

        self._entity_values = sorted(entity_vals.keys())
        # Second template pass (ENTITY value normalization needs the entity
        # values collected first)
        templates2 = defaultdict(lambda: defaultdict(set))
        parser2 = SQLRoleParser(dialect=self.dialect)
        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            pr2 = parser2.parse(str(row.get('sql', '')),
                                schema_columns=self.schema_columns)
            if not pr2.ast_parsed:
                continue
            for t in pr2.get_gold_tables():
                templates2[t][self._normalize_template(str(row.get('question', '')))].add(qid)

        # --- assemble the annotations ---
        self.annotations = []
        self._annotation_ids = {}
        for (t, c), per_term in alias_pairs.items():
            for term, qids in per_term.items():
                local = self._local_lexicon(t, c)
                lex_overlap = term in local
                # P2-D2 fix: source/support = the qid set actually containing
                # the alias (role_qids only feeds the ROLE annotations);
                # confidence is a heuristic support score (= number of
                # questions containing the alias, not a calibrated
                # probability)
                self._append({
                    'type': 'TERM', 'rule': 'select_alias',
                    'target': _norm_key(t, c), 'content': term,
                    'alias_texts': sorted(alias_meta[(t, c)][term]),
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed',
                    'lexicon_overlap': lex_overlap})
        for term, per_elem in cooc.items():
            counts = {k: len(v) for k, v in per_elem.items()}
            total = sum(counts.values())
            best = max(counts.values())
            top = [k for k, n in counts.items() if n >= best * (1 - AMBIGUITY_WINDOW)]
            if best < TERM_MIN_SUPPORT:
                status = 'rejected_single'
            elif best / total >= TERM_CONCENTRATION and len(top) == 1:
                status = 'confirmed'
            else:
                status = 'ambiguous'
            qids = sorted(set().union(*per_elem.values()))
            self._append({
                'type': 'TERM', 'rule': 'co_occurrence',
                'target': top[0] if len(top) == 1 and status == 'confirmed' else None,
                'candidates': sorted(counts.items(), key=lambda x: -x[1])[:5],
                'content': term, 'source_qids': qids, 'support': total,
                'confidence': f'heuristic:concentration={round(best / total, 3)}',
                'status': status})
        for (t, c, op, vclass, val), qids in pred_counts.items():
            self._append({
                'type': 'PREDICATE', 'rule': 'where_ast',
                'target': _norm_key(t, c), 'content': {'operator': op,
                                                       'value_class': vclass,
                                                       'value': val},
                'source_qids': sorted(qids), 'support': len(qids),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for (t, c), roles in role_sets.items():
            self._append({
                'type': 'ROLE', 'rule': 'ast_roles',
                'target': _norm_key(t, c), 'content': sorted(roles),
                'source_qids': sorted(role_qids[(t, c)]),
                'support': len(role_qids[(t, c)]),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for t, per_tmpl in templates2.items():
            ranked = sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))
            for text, qids in ranked[:TEMPLATE_TOPK]:
                binds = []
                for rec in tmpl_bindings[t].get(text, []):
                    binds.append(rec['bindings'])
                self._append({
                    'type': 'TEMPLATE', 'rule': 'slot_normalization',
                    'target': t, 'content': text,
                    'bindings': binds,
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed'})

        # Injection views
        self._term_by_key = defaultdict(list)
        for a in self.annotations:
            if a['type'] == 'TERM' and a['status'] == 'confirmed' and a.get('target'):
                self._term_by_key[a['target']].append((a['content'], a['confidence']))
        self._templates_by_table = {
            t: [(text, len(qids)) for text, qids in
                sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))[:TEMPLATE_TOPK]]
            for t, per_tmpl in templates2.items()}
        return self

    # ---------------------------------------------------------------- local SQL parsing

    def _parse_sql(self, sql):
        for d in (self.dialect, 'hive', 'mysql'):
            try:
                return sqlglot.parse_one(sql, dialect=d), d
            except Exception:  # noqa: BLE001
                continue
        return None, None

    def _scope_sources(self, select):
        """Physical tables and aliases of one SELECT's FROM/JOIN (simplified
        local parsing)."""
        tables = {}
        frm = select.args.get('from_') or select.args.get('from')
        sources = []
        if frm is not None:
            sources.append(frm.this)
        sources += [j.this for j in (select.args.get('joins') or [])]
        for src in sources:
            if isinstance(src, exp.Table) and src.name:
                name = src.name
                alias = (src.alias or name).lower()
                tables[alias] = name
                tables[name.lower()] = name
        return tables

    def _resolve(self, col, tables, n_physical):
        if col.table:
            return tables.get(col.table.lower())
        if n_physical == 1:
            return next(iter({t for t in tables.values()}))
        return None

    def _alias_terms(self, sql):
        """R1: new terms in SELECT output aliases <-> output columns (only
        uniquely attributable columns).

        Returns [(term, (table, column), alias_text), ...] - alias_text
        keeps the full alias for provenance auditing (P2-D2: support must
        come from the questions actually containing the alias).
        """
        from terag.engines.bird_runtime.tokenization import tokenize_text
        out = []
        tree, _ = self._parse_sql(sql)
        if tree is None:
            return out
        for select in tree.find_all(exp.Select):
            tables = self._scope_sources(select)
            n_phys = len({t for t in tables.values()})
            for e in select.expressions:
                alias = e.alias
                if not alias:
                    continue
                cols = [c for c in e.find_all(exp.Column)
                        if not isinstance(c.this, exp.Star)]
                if len(cols) != 1:
                    continue
                t = self._resolve(cols[0], tables, n_phys)
                if t is None:
                    continue
                local = self._local_lexicon(t, cols[0].name)
                for w in tokenize_text(alias, 'chinese'):
                    if len(w) > 1 and w not in STOP_TERMS \
                            and not _NUM_RE.search(w) and w not in local:
                        out.append((w, (t, cols[0].name), alias))
        return out

    def _predicates(self, sql):
        """WHERE predicates: (column, operator, value, value class). Only
        comparison conditions over uniquely attributable columns."""
        out = []
        tree, _ = self._parse_sql(sql)
        if tree is None:
            return out
        op_map = {exp.EQ: '=', exp.GT: '>', exp.LT: '<', exp.GTE: '>=',
                  exp.LTE: '<=', exp.Like: 'LIKE', exp.In: 'IN',
                  exp.Between: 'BETWEEN'}
        for select in tree.find_all(exp.Select):
            w = select.args.get('where')
            if w is None:
                continue
            tables = self._scope_sources(select)
            n_phys = len({t for t in tables.values()})
            for node in w.find_all(exp.Binary):
                op = op_map.get(type(node))
                if op is None:
                    continue
                cols = [c for c in node.find_all(exp.Column)]
                lits = [l for l in node.find_all(exp.Literal)]
                if len(cols) != 1 or not lits:
                    continue
                t = self._resolve(cols[0], tables, n_phys)
                if t is None:
                    continue
                val = lits[0].this if hasattr(lits[0], 'this') else str(lits[0])
                out.append(((t, cols[0].name), op, str(val),
                            self._classify_value(str(val), cols[0].name, op)))
            for node in w.find_all(exp.Is):
                cols = [c for c in node.find_all(exp.Column)]
                if len(cols) == 1:
                    t = self._resolve(cols[0], tables, n_phys)
                    if t is not None:
                        out.append(((t, cols[0].name), 'IS', 'NULL', 'NULL'))
        return out

    def _classify_value(self, val: str, col: str, op: str) -> str:
        """Value classification (declared rules): DATE/ENTITY/CODE/
        THRESHOLD/UNKNOWN/NULL."""
        if _DATE_RE.fullmatch(val):
            return 'DATE'
        if re.search(r'[一-鿿]', val) and len(val) >= 3 \
                and _ORG_COL_RE.search(col):
            return 'ENTITY'
        if _CODE_RE.match(val) and op == '=' and re.search(
                r'cls|type|stat|flag|code|level|kind', col, re.I):
            return 'CODE'
        if op in ('>', '<', '>=', '<=') and _NUM_RE.match(val):
            return 'THRESHOLD'
        return 'UNKNOWN'

    def _normalize_template(self, question: str) -> str:
        """Slot normalization: date -> #DATE; ENTITY value -> #ORG; the
        remaining numbers -> #NUM."""
        q = question
        for v in self._entity_values:
            if v in q:
                q = q.replace(v, '#ORG')
        q = _DATE_RE.sub('#DATE', q)
        q = _NUM_RE.sub('#NUM', q)
        return re.sub(r'\s+', '', q)

    # ---------------------------------------------------------------- retrieval channels

    def table_annotation_texts(self, pattern_weight: float) -> Dict[str, List[Tuple[str, float]]]:
        """Table-level index injection content: confirmed TERM (of columns) +
        template texts."""
        out = defaultdict(list)
        for key, terms in self._term_by_key.items():
            t = key.split('.', 1)[0]
            for term, _conf in terms:
                out[t].append((term, pattern_weight))
        for t, tmpls in self._templates_by_table.items():
            for text, _sup in tmpls:
                out[t].append((text, pattern_weight))
        return dict(out)

    def column_bonus(self, query_tokens, table: str, column: str) -> float:
        """Column-ranking annotation hit bonus: a query term hits one of this
        column's confirmed TERM annotations."""
        key = _norm_key(table, column)
        bonus = 0.0
        for term, _conf in self._term_by_key.get(key, []):
            if term in query_tokens:
                bonus += W_ANN
        return bonus

    def summary(self) -> Dict:
        by = defaultdict(lambda: defaultdict(int))
        for a in self.annotations:
            by[a['type']][a['status']] += 1
        return {t: dict(v) for t, v in by.items()}

    def source_qids(self) -> set:
        return {q for a in self.annotations for q in a['source_qids']}
