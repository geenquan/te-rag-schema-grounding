"""English-language subclass of the frozen DirectAnnotationMiner (B class mining).

Copied verbatim from workspace/retrievers/annotation_miner.py. The ONLY deviation
is the tokenization language at the fixed 'chinese' call sites inside
_local_lexicon / _global_lexicon / mine (q_tokens) / _alias_terms (marked
# BIRD-ADAPT:); the SQL dialect is already a constructor parameter (sqlite for
BIRD). Every mining rule (R1 alias, R2 co-occurrence support>=2 concentration>=0.8,
TEMPLATE top-5, PREDICATE, ROLE) is untouched.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from terag.engines.bird_runtime.retrievers.annotation_miner import (
    DirectAnnotationMiner, STOP_TERMS, _NUM_RE, _norm_key,
    TEMPLATE_TOPK, TERM_MIN_SUPPORT, TERM_CONCENTRATION, AMBIGUITY_WINDOW)
from terag.engines.bird_runtime.sql_role_parser import SQLRoleParser, SQLGLOT_AVAILABLE
from terag.engines.bird_runtime.tokenization import tokenize_text

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None


class AnnotationMinerBird(DirectAnnotationMiner):
    """DirectAnnotationMiner with English tokenization (BIRD); dialect='sqlite'."""

    def __init__(self, field_df, table_df, dialect: str = 'sqlite',
                 language: str = 'english'):
        super().__init__(field_df, table_df, dialect=dialect)
        self.language = language  # BIRD-ADAPT: tokenization language for lexicons/terms

    def _local_lexicon(self, table: str, col: Optional[str] = None) -> set:
        from terag.engines.bird_runtime.tokenization import tokenize_text
        words = set()
        trow = self.table_df[self.table_df['table'] == table]
        for _, r in trow.iterrows():
            words |= {w for w in tokenize_text(str(r.get('table_desc', '') or ''), self.language)  # BIRD-ADAPT

                      if len(w) > 1}
        cols = self.field_df[self.field_df['table'] == table]
        if col is not None:
            cols = cols[cols['field_name'] == col]
        for _, r in cols.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), self.language) if len(w) > 1}  # BIRD-ADAPT

        return words
    def _global_lexicon(self) -> set:
        from terag.engines.bird_runtime.tokenization import tokenize_text
        words = set()
        for _, r in self.field_df.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), self.language) if len(w) > 1}  # BIRD-ADAPT

        for _, r in self.table_df.iterrows():
            words |= {w for w in tokenize_text(
                str(r.get('table_desc', '') or ''), self.language) if len(w) > 1}  # BIRD-ADAPT

        return words
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
            q_tokens = [w for w in tokenize_text(question, self.language) if len(w) > 1]  # BIRD-ADAPT

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
                for w in tokenize_text(alias, self.language):  # BIRD-ADAPT

                    if len(w) > 1 and w not in STOP_TERMS \
                            and not _NUM_RE.search(w) and w not in local:
                        out.append((w, (t, cols[0].name), alias))
        return out
