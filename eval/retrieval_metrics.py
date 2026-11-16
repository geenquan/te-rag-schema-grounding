"""
Retrieval evaluation metrics (official semantics; the historical
semantics of the earlier revision are documented alongside each legacy
alias below)

This module is the single authoritative definition of the metrics used
in the official reports. Semantics:

Single-table task (PowerGrid):
- Hit@K      : the gold table appears among the top-K returned tables
               ranked by score (boolean, K in {1,3,5}).
- Annotated-Field Precision/Recall/F1 : set-level metrics over
               normalized "table.column" keys, averaged per query
               (macro). The predicted column set = the union of the
               columns carried by all top-K returned tables (the
               per-table column budget is whatever the retriever
               actually enforces, see the protocol); the gold column
               set = "gold-table-short-name.field", **consisting solely
               of the existing manually annotated fields (SELECT
               targets)** and excluding unannotated fields such as
               WHERE/JOIN. A table miss or a retrieval failure naturally
               yields 0.
- AFCS (Annotated-Field Coverage Success): the gold table hits top-K
               and the gold annotated column set is a subset of the
               predicted column set. The name explicitly limits the
               metric to coverage of existing annotated fields — it is
               NOT full SQL schema coverage (gold semantics are defined
               in the evaluation protocol).

Multi-table task (BIRD):
- Table Recall@K = |gold ∩ topK| / |gold|.
- All-Gold-Tables-Covered@K = gold ⊆ topK (subset coverage;
  predictions may contain extra tables — this must not be called set
  Exact Match).
- Column / coverage metrics as above.

Common rules (failure handling):
- Gold field status: present / missing (raw annotation absent) /
  empty (empty after parsing) / parse_error (parsing raised).
- Queries with valid gold annotation (present): an empty prediction or
  a retrieval error (status=error) always counts column P/R/F1 and
  AFCS as 0 **inside the valid denominator** — such queries must never
  disappear from the denominator.
- Missing annotation / parse failure: tallied separately from model
  failures; not counted as a prediction error, not counted as a
  coverage success, and excluded from the column-level and AFCS
  denominators.
- Budget bookkeeping: each query records the actual number of tables
  returned, the number of columns per table, and budget_ok (tables<=5,
  columns per table<=10).
- Retrieval exceptions: status=error; counted in the denominators of
  the main metrics such as Hit@K (as 0) and tallied separately.
- SQL generation and execution: out of scope for this module;
  exec_accuracy is always "not_run" — never filled with 0 or an
  estimate.
- Legacy aliases (old field names, internal use only): table_accuracy
  / table_recall == single-table Hit@5; sql_accuracy == the old SCS
  variant (empty gold counted as correct); schema_coverage_success ==
  AFCS (same name and value; AFCS is the official name).

Normalization rules: keys are lowercased, stripped of leading/trailing
whitespace and of "T:"/"C:" prefixes; a table name keeps only its last
"." segment (dropping any db prefix). This rule applies only to the
current single-database PowerGrid setting (short names are unique); it
does not hold for future cross-database retrieval (the same table name
in different dbs), which would require full identifiers that retain the
database name — recorded as a known issue; the BIRD pipeline is left
unchanged at this stage.
"""

from typing import Dict, List, Optional, Sequence, Tuple

K_VALUES = (1, 3, 5)


def normalize_table(name: str) -> str:
    """Normalize a table name: strip prefix/whitespace/case; keep the last
    segment (drops any db prefix)."""
    if not name:
        return ''
    s = str(name).strip().lower()
    if s.startswith('t:'):
        s = s[2:]
    return s.split('.')[-1]


def normalize_column_key(key: str) -> str:
    """Normalize a "table.column" key: strip prefix/whitespace/case; the
    table name keeps only its last segment."""
    if not key:
        return ''
    s = str(key).strip().lower()
    if s.startswith('c:'):
        s = s[2:]
    parts = [p for p in s.split('.') if p]
    if len(parts) >= 2:
        table, col = parts[-2], parts[-1]
        return f"{table.split('.')[-1]}.{col}"
    return s


def classify_gold_fields(raw_value) -> Tuple[str, List[str]]:
    """Classify the status of a gold field annotation.

    Returns:
        (status, fields): status is one of present/missing/empty/
        parse_error; fields is the raw field-name list (non-empty only
        when present).
    """
    if raw_value is None:
        return 'missing', []
    try:
        import pandas as pd
        if pd.isna(raw_value):
            return 'missing', []
        if not isinstance(raw_value, str):
            return 'missing', []
        fields = [f.strip() for f in raw_value.split('|') if f.strip()]
        if not fields:
            return 'empty', []
        return 'present', fields
    except Exception:
        return 'parse_error', []


def hits_at_k(gold_table: str, retrieved_tables: Sequence[str],
              k_values: Sequence[int] = K_VALUES) -> Dict[int, bool]:
    """Single-table Hit@K."""
    gold = normalize_table(gold_table)
    ordered = [normalize_table(t) for t in retrieved_tables]
    return {k: bool(gold) and gold in ordered[:k] for k in k_values}


def table_recall_at_k(gold_tables: Sequence[str], retrieved_tables: Sequence[str],
                      k_values: Sequence[int] = K_VALUES) -> Dict[int, Optional[float]]:
    """Multi-table Table Recall@K = |gold ∩ topK| / |gold|; returns None
    when gold is empty."""
    gold = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold:
        return {k: None for k in k_values}
    ordered = [normalize_table(t) for t in retrieved_tables]
    out = {}
    for k in k_values:
        topk = set(ordered[:k])
        out[k] = len(gold & topk) / len(gold)
    return out


def all_gold_covered(gold_tables: Sequence[str], retrieved_tables: Sequence[str],
                     k: int = 5) -> Optional[bool]:
    """All-Gold-Tables-Covered@K: gold ⊆ topK (subset coverage, not Exact
    Match).

    Returns None when gold is empty (excluded from the denominator).
    """
    gold = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold:
        return None
    topk = {normalize_table(t) for t in retrieved_tables[:k]}
    return gold.issubset(topk)


def column_prf(gold_keys: Sequence[str], pred_keys: Sequence[str]) -> Tuple[
        Optional[float], Optional[float], Optional[float]]:
    """Set-level Column Precision/Recall/F1 (official semantics).

    With non-empty gold (valid annotation), an empty prediction or a
    retrieval failure always counts as 0 — failures must never disappear
    from the valid-annotation denominator; with empty gold (missing or
    unparseable annotation) returns (None, None, None), excluded from
    every column-level denominator and not counted as a model error.
    """
    gold = {normalize_column_key(k) for k in gold_keys if normalize_column_key(k)}
    pred = {normalize_column_key(k) for k in pred_keys if normalize_column_key(k)}
    if not gold:
        return None, None, None
    inter = gold & pred
    recall = len(inter) / len(gold)
    precision = len(inter) / len(pred) if pred else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return precision, recall, f1


def schema_coverage_success(gold_tables: Sequence[str], gold_column_keys: Sequence[str],
                            retrieved_tables: Sequence[str],
                            retrieved_column_keys: Sequence[str], k: int = 5) -> Optional[bool]:
    """SCS: all gold tables make the top-K and the gold column keys are
    fully covered by the predicted column keys.

    Returns None when the gold column set is empty (empty/missing/
    parse_error) — an empty or missing annotation is neither a coverage
    success nor part of the SCS denominator.
    """
    gold_t = {normalize_table(t) for t in gold_tables if normalize_table(t)}
    if not gold_t:
        return None
    gold_c = {normalize_column_key(c) for c in gold_column_keys if normalize_column_key(c)}
    if not gold_c:
        return None
    topk = [normalize_table(t) for t in retrieved_tables[:k]]
    if not gold_t.issubset(set(topk)):
        return False
    pred_c = {normalize_column_key(c) for c in retrieved_column_keys}
    return gold_c.issubset(pred_c)


def evaluate_single_table_query(gold_table: str, gold_fields_raw,
                                retrieved_tables: Sequence[str],
                                retrieved_column_keys: Sequence[str],
                                k_values: Sequence[int] = K_VALUES) -> Dict:
    """Per-query metrics for the single-table task (PowerGrid).

    Args:
        gold_fields_raw: raw gold field annotation ('|'-separated string,
            or None/NaN).
        retrieved_tables: returned tables ranked by score (as-is; prefixes
            allowed).
        retrieved_column_keys: union of the "table.column" keys carried by
            all returned tables.
    """
    status, fields = classify_gold_fields(gold_fields_raw)
    gold_keys = [f"{normalize_table(gold_table)}.{f.lower().strip()}" for f in fields]

    out = {
        'gold_field_status': status,
        'gold_column_keys': gold_keys,
        'hit': hits_at_k(gold_table, retrieved_tables, k_values),
        'column_precision': None, 'column_recall': None, 'column_f1': None,
        'annotated_field_coverage_success': None,
        'schema_coverage_success': None,  # legacy alias (== AFCS)
    }
    if status == 'present':
        p, r, f1 = column_prf(gold_keys, retrieved_column_keys)
        out['column_precision'], out['column_recall'], out['column_f1'] = p, r, f1
        scs = schema_coverage_success(
            [gold_table], gold_keys, retrieved_tables, retrieved_column_keys,
            k=max(k_values))
        # Empty predictions count as 0 (not None): failures must not vanish
        # from the valid denominator
        out['annotated_field_coverage_success'] = bool(scs) if scs is not None else False
        out['schema_coverage_success'] = out['annotated_field_coverage_success']
    return out


def evaluate_multi_table_query(gold_tables: Sequence[str], gold_column_keys: Sequence[str],
                               retrieved_tables: Sequence[str],
                               retrieved_column_keys: Sequence[str],
                               k_values: Sequence[int] = K_VALUES) -> Dict:
    """Per-query metrics for the multi-table task (BIRD). Gold column keys
    use the "table.column" form."""
    gold_keys = list(gold_column_keys)
    p, r, f1 = column_prf(gold_keys, retrieved_column_keys) if gold_keys else (None, None, None)
    scs = schema_coverage_success(
        gold_tables, gold_keys, retrieved_tables, retrieved_column_keys,
        k=max(k_values)) if gold_keys else None
    afcs = (bool(scs) if scs is not None else False) if gold_keys else None
    return {
        'gold_field_status': 'present' if gold_keys else 'missing',
        'gold_column_keys': gold_keys,
        'table_recall_at_k': table_recall_at_k(gold_tables, retrieved_tables, k_values),
        'all_gold_covered': {k: all_gold_covered(gold_tables, retrieved_tables, k)
                             for k in k_values},
        'column_precision': p, 'column_recall': r, 'column_f1': f1,
        'annotated_field_coverage_success': afcs,
        'schema_coverage_success': afcs,  # legacy alias (== AFCS)
    }


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate_records(records: List[Dict]) -> Dict:
    """Aggregate method-level metrics from per-query records (macro).

    Main denominator = all test queries in the group (status=error
    queries count as 0 in Hit-type metrics); the valid denominator for
    column-level metrics and SCS = the number of queries with
    gold_field_status == 'present'. All counts are output explicitly.
    Every input must be an already-persisted predictions record.
    """
    n_total = len(records)
    if n_total == 0:
        return {'n_total': 0}

    n_error = sum(1 for r in records if r.get('status') == 'error')
    status_counts = {}
    for r in records:
        s = r.get('gold_field_status', 'missing')
        status_counts[s] = status_counts.get(s, 0) + 1
    n_present = status_counts.get('present', 0)

    single = 'hit_at_1' in records[0]
    out = {
        'n_total': n_total,
        'n_ok': n_total - n_error,
        'n_error': n_error,
        'n_gold_present': n_present,
        'n_gold_missing': status_counts.get('missing', 0),
        'n_gold_empty': status_counts.get('empty', 0),
        'n_gold_parse_error': status_counts.get('parse_error', 0),
    }

    if single:
        for k in K_VALUES:
            hits = [1.0 if (r.get('status') == 'ok' and r.get(f'hit_at_{k}')) else 0.0
                    for r in records]
            hits_valid = [1.0 if r.get(f'hit_at_{k}') else 0.0
                          for r in records if r.get('status') == 'ok']
            out[f'hit@{k}'] = _mean(hits)
            out[f'hit@{k}_valid'] = _mean(hits_valid)
    else:
        for k in K_VALUES:
            vals, vals_valid = [], []
            for r in records:
                tr = r.get('table_recall_at_k', {})
                v = tr.get(str(k), tr.get(k))
                if v is not None:
                    vals.append(v if r.get('status') == 'ok' else 0.0)
                    vals_valid.append(v)
            out[f'table_recall@{k}'] = _mean(vals)
            out[f'table_recall@{k}_valid'] = _mean(vals_valid)
        def _cov5(r):
            agc = r.get('all_gold_covered') or {}
            return agc.get('5', agc.get(5))

        cov = [1.0 if (r.get('status') == 'ok' and _cov5(r)) else 0.0 for r in records]
        cov_valid = [1.0 if _cov5(r) else 0.0 for r in records if r.get('status') == 'ok']
        out['all_gold_covered@5'] = _mean(cov)
        out['all_gold_covered@5_valid'] = _mean(cov_valid)

    present = [r for r in records if r.get('gold_field_status') == 'present']
    # All queries with valid gold enter the column-level denominator
    # (empty predictions/errors count as 0, never dropped)
    out['afcs_denominator'] = len(present)
    for name in ('column_precision', 'column_recall', 'column_f1'):
        vals = [r.get(name) if r.get(name) is not None else 0.0 for r in present]
        out[name] = _mean(vals)
        out[f'{name}_denominator'] = len(present)
    afcs_vals = [1.0 if r.get('annotated_field_coverage_success',
                              r.get('schema_coverage_success')) else 0.0
                 for r in present]
    out['afcs'] = _mean(afcs_vals)
    # Budget bookkeeping: actual tables/columns returned and the budget
    # compliance rate (tables<=5, columns per table<=10)
    n_cols = [sum(len(t.get('columns', [])) for t in (r.get('retrieved_columns_detail') or []))
              for r in records]
    budget_ok = []
    for r in records:
        tables = r.get('retrieved_columns_detail') or []
        ok = (len(r.get('retrieved_tables') or []) <= 5
              and all(len(t.get('columns', [])) <= 10 for t in tables))
        budget_ok.append(1.0 if ok else 0.0)
    out['avg_columns_returned'] = _mean(n_cols)
    out['avg_tables_returned'] = _mean(
        [len(r.get('retrieved_tables') or []) for r in records])
    out['budget_ok_rate'] = _mean(budget_ok)
    out['exec_accuracy'] = 'not_run'
    return out
