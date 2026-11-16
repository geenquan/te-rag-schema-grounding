"""
Dense+DeepSeek Schema Selector

Method identity: a Dense retrieval + LLM table/column selector of our own
design. **This is NOT a LinkAlign reproduction or adaptation**.

Candidate restrictions (must be registered in the protocol):
- Table candidates: the top CTX_TABLE_LIMIT(20) tables from Dense
  embedding retrieval.
- Column display: CTX_COL_LIMIT(10) columns per table, ranked by Dense
  cosine similarity descending (protocol expansion_v2; falls back to CSV
  row order when q_emb is missing or the table has no column embeddings).
- **A column that exists in a table but was not displayed can never be
  chosen by the LLM** — this is a known limitation of the current
  method, unlike the Dense baseline (which ranks all fields as columns).

Mechanism:
1. Dense embedding retrieval fetches candidate tables.
2. The candidate tables' schema (table descriptions + column
   descriptions) is formatted into the prompt context.
3. The DeepSeek LLM selects the relevant tables and columns.
4. The JSON reply is parsed and validated to contain only the
   tables/columns that were shown to the LLM.
"""


import json
import hashlib
import os
import re
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig
from retrievers.dense_retriever import _embed, MODEL_NAME as EMBED_MODEL

COLUMN_BUDGET = 10
TABLE_BUDGET = 5
CANDIDATE_K = 20
CTX_COL_LIMIT = 10        # max columns displayed per table in the prompt context
CTX_TABLE_LIMIT = 20      # max candidate tables displayed in the prompt context
REQUEST_TIMEOUT = 60
MAX_RETRIES = 1

SCHEMA_SELECT_PROMPT = """You are a schema linking expert. Given a natural language question and candidate database schema elements, select the tables and columns needed to answer the question.

## Candidate Schema Elements
{schema_text}

## Question
{question}

## Task
Select the tables and columns from the above candidates that are relevant to answering this question. Only select from what is shown above.

Output your answer in this exact JSON format:
{{
  "tables": [
    {{
      "table": "table_name",
      "columns": ["col1", "col2"]
    }}
  ]
}}

Rules:
- At most {table_budget} tables.
- At most {col_budget} columns per table.
- Only select tables and columns explicitly shown above.
- Output only valid JSON, no other text."""


def _is_fatal_api_error(e: Exception) -> bool:
    """Auth/balance/config errors: not retryable, and per protocol §5 the
    method must be stopped.

    Covers the exception types the openai SDK maps to 400/401/403/404,
    plus status codes it does not map (e.g. 402 for exhausted DeepSeek
    balance). 429/5xx/connection errors are not fatal.
    """
    try:
        import openai as _openai
        fatal_types = (_openai.AuthenticationError,
                       _openai.PermissionDeniedError,
                       _openai.BadRequestError,
                       _openai.NotFoundError)
    except ImportError:
        fatal_types = ()
    if fatal_types and isinstance(e, fatal_types):
        return True
    status = getattr(e, 'status_code', None)
    if status is None:
        status = getattr(getattr(e, 'response', None), 'status_code', None)
    return status in (400, 401, 402, 403)


class LinkAlignRetriever(BaseRetriever):
    """Dense+DeepSeek schema selector (not a LinkAlign reproduction)."""

    # Global latch for unrecoverable auth/balance/config errors
    # (protocol §5: stop the method). Class-level attribute: subsequent
    # instances in the same process (including instances created in later
    # folds) no longer make API calls.
    _FATAL_API_ERROR: str = ''

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='DenseDeepSeek',
                description='Dense+DeepSeek Schema Selector (custom, NOT LinkAlign)')
        super().__init__(field_csv, table_csv, config)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._col_keys: Dict[str, List[str]] = {}
        self._col_descs: Dict[str, Dict[str, str]] = {}
        self._table_descs: Dict[str, str] = {}
        self._col_embs: Dict[str, np.ndarray] = {}
        self._client = None
        self._api_calls = 0
        self._api_retries = 0
        self._api_errors = 0
        self._parse_errors = 0
        self._illegal_items: List[dict] = []  # per-query record of illegal LLM output items
        self._last_resp_id = ''  # id of the most recent LLM response (written to per-query metadata)
        self._last_usage = {}    # token usage of the most recent LLM call (written to per-query metadata)
        self._last_resp_model = 'not_provided'  # model name actually returned in the response
        self._last_attempts = 0        # actual HTTP attempts for the current query
        self._last_http_times: List[float] = []  # wall time of each attempt for the current query
        self._last_query_evidence: Dict = {}     # standalone evidence for the current query (independent of the returned result list)
        # Set by the runner before each per-query call (the base call chain
        # does not thread qid through)
        self._current_qid = ''
        self._current_fold = None

    def _init_client(self):
        """Initialize the OpenAI-compatible client at runtime. The API key
        is never printed or logged."""
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get('DEEPSEEK_API_KEY', '')
            base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
            if not api_key:
                cfg_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'config.yaml')
                try:
                    import yaml
                    with open(cfg_path, encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                    api_key = cfg.get('DEEPSEEK_API_KEY', '')
                    base_url = cfg.get('DEEPSEEK_BASE_URL', base_url)
                except Exception:
                    pass
            if not api_key:
                raise ValueError(
                    'No DeepSeek API key. Set DEEPSEEK_API_KEY env var '
                    'or add DEEPSEEK_API_KEY to config.yaml')
            self._client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=REQUEST_TIMEOUT,
                                  max_retries=0)  # disable SDK-internal retries; retrying is controlled by the outer layer

    def fit(self, train_data: pd.DataFrame = None):
        """Build the Dense embedding index (same document construction as
        DenseRetriever)."""
        self._table_names = []
        self._table_docs = []
        self._table_embs = None
        self._col_keys = {}
        self._col_descs = {}
        self._table_descs = {}

        from retrievers.dense_retriever import TABLE_COL_LIMIT
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            col_names = ' '.join(
                cols['field_name'].astype(str).tolist()[:TABLE_COL_LIMIT])
            doc = f"{tname} {tdesc} {col_names}"
            self._table_names.append(tname)
            self._table_docs.append(doc)
            self._table_descs[tname] = tdesc
            keys, descs = [], {}
            for _, c in cols.iterrows():
                cname = str(c['field_name'])
                cdesc = str(c.get('field_name_desc', '') or '')
                keys.append(cname)
                descs[cname] = cdesc
            self._col_keys[tname] = keys
            self._col_descs[tname] = descs

        self._table_embs = _embed(self._table_docs)
        # Per-table column embeddings (used to rank candidate columns,
        # replacing CSV order)
        from retrievers.dense_retriever import DenseRetriever as _DR
        for tname in self._table_names:
            cols = self.field_df[self.field_df['table'] == tname]
            docs = [f"{str(c['field_name'])} {str(c.get('field_name_desc', '') or '')}" for _, c in cols.iterrows()]
            if docs:
                self._col_embs[tname] = _embed(docs)
        self._is_fitted = True

    def _build_schema_text(self, candidate_indices: List[int],
                           q_emb: np.ndarray = None) -> Tuple[str, List[str], Dict[str, List[str]]]:
        """Build the prompt context (including table and column descriptions).

        When q_emb is not None, each table shows its top CTX_COL_LIMIT
        columns ranked by Dense cosine similarity descending (protocol
        expansion_v2); when q_emb is None or the table has no column
        embeddings, fall back to CSV row order.

        Returns:
            (schema_text, shown_tables, shown_columns_by_table) —
            shown_tables in candidate order and shown_columns_by_table in
            display order (ordered lists so the prompt can be reconstructed
            exactly); the parser may accept only these columns.
        """
        parts = []
        shown_tables: List[str] = []
        shown_cols: Dict[str, List[str]] = {}
        for idx in candidate_indices[:CTX_TABLE_LIMIT]:
            tname = self._table_names[idx]
            tdesc = self._table_descs.get(tname, '')
            keys = self._col_keys.get(tname, [])
            descs = self._col_descs.get(tname, {})
            if q_emb is not None and keys and tname in self._col_embs:
                col_scores = self._col_embs[tname] @ q_emb
                # Stable ordering matching the Dense baseline: score
                # descending, then column name ascending
                col_order = sorted(
                    range(len(keys)),
                    key=lambda i: (-col_scores[i], keys[i]))[:CTX_COL_LIMIT]
                display_keys = [keys[i] for i in col_order]
            else:
                display_keys = keys[:CTX_COL_LIMIT]
            cols_str = '; '.join(
                f"{k}({descs.get(k, '')})" for k in display_keys)
            parts.append(f"Table: {tname}\nDescription: {tdesc}\nColumns: {cols_str}")
            shown_tables.append(tname)
            shown_cols[tname] = list(display_keys)
        return '\n\n'.join(parts), shown_tables, shown_cols

    def _call_llm(self, prompt: str) -> Tuple[str, str]:
        """Call the DeepSeek API. SDK-internal retries are disabled
        (max_retries=0).

        Every actual HTTP attempt is counted and timed (written to
        per-query evidence); fields the SDK response does not provide
        (model/id/usage) are recorded as 'not_provided', never fabricated.
        """
        self._init_client()
        self._api_calls += 1
        self._last_attempts = 0
        self._last_http_times = []
        for attempt in range(MAX_RETRIES + 1):
            self._last_attempts += 1
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    max_tokens=2000)
                self._last_http_times.append(round(time.time() - t0, 3))
                self._last_resp_id = getattr(resp, 'id', '') or 'not_provided'
                self._last_resp_model = getattr(resp, 'model', None) or 'not_provided'
                self._last_usage = {
                    'prompt_tokens': getattr(resp.usage, 'prompt_tokens', None),
                    'completion_tokens': getattr(resp.usage, 'completion_tokens', None),
                    'total_tokens': getattr(resp.usage, 'total_tokens', None)} if resp.usage else 'not_provided'
                return resp.choices[0].message.content, 'ok'
            except Exception as e:
                self._last_http_times.append(round(time.time() - t0, 3))
                if _is_fatal_api_error(e):
                    # Protocol §5: auth/balance/config errors are not
                    # retryable and must stop the method
                    self._api_errors += 1
                    type(self)._FATAL_API_ERROR = (
                        f'{type(e).__name__}: {str(e)[:200]}')
                    raise RuntimeError(
                        '[DenseDeepSeek] Non-retryable API error '
                        '(auth/balance/config, method stopped per '
                        f'protocol): {type(e).__name__}') from e
                if attempt < MAX_RETRIES:
                    self._api_retries += 1
                    time.sleep(2)
                    continue
                self._api_errors += 1
                raise RuntimeError(
                    f'LLM API failed after {MAX_RETRIES + 1} attempts: '
                    f'{type(e).__name__}') from e

    def _parse_response(self, response: str, shown_tables: Set[str],
                        shown_cols: Dict[str, Set[str]]) -> Tuple[List[Tuple[str, List[str]]], str]:
        """Parse the LLM JSON response, accepting only the tables and columns
        that were actually shown to the LLM.

        Returns:
            (selected, status)
        """
        self._illegal_items = []  # reset before each parse
        text = response.strip()
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
        if m:
            text = m.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self._parse_errors += 1
            return [], 'json_error'

        # Type checks: the root must be a dict and 'tables' must be a list
        if not isinstance(data, dict):
            self._parse_errors += 1
            return [], 'json_error'
        tables = data.get('tables', [])
        if not isinstance(tables, list):
            self._parse_errors += 1
            return [], 'json_error'

        result = []
        seen_tables_in_result = set()
        for t in tables:
            if not isinstance(t, dict):
                self._illegal_items.append({'type': 'bad_table_entry', 'value': str(t)[:50]})
                continue
            tname_raw = t.get('table')
            tname = str(tname_raw) if tname_raw is not None else ''
            columns_raw = t.get('columns', [])
            if not isinstance(columns_raw, list):
                self._illegal_items.append(
                    {'type': 'bad_columns_type', 'table': tname,
                     'value_type': type(columns_raw).__name__})
                continue

            # Table-level check: must have been shown to the LLM
            if tname not in shown_tables:
                self._illegal_items.append(
                    {'type': 'unshown_table', 'table': tname})
                continue

            # Deduplicate tables
            if tname in seen_tables_in_result:
                self._illegal_items.append(
                    {'type': 'duplicate_table', 'table': tname})
                continue

            # Column-level check: accept only columns shown to the LLM
            shown_for_this = shown_cols.get(tname, set())
            valid = []
            seen_cols = set()
            for c in columns_raw:
                cname = str(c)
                if cname not in shown_for_this:
                    self._illegal_items.append(
                        {'type': 'unshown_column', 'table': tname,
                         'column': cname})
                    continue
                if cname in seen_cols:
                    self._illegal_items.append(
                        {'type': 'duplicate_column', 'table': tname,
                         'column': cname})
                    continue
                seen_cols.add(cname)
                valid.append(cname)

            valid = valid[:COLUMN_BUDGET]
            if valid:
                result.append((tname, valid))
                seen_tables_in_result.add(tname)
            if len(result) >= TABLE_BUDGET:
                break

        if not result:
            if self._illegal_items:
                return [], 'illegal_elements'
            return [], 'valid_empty'
        # Valid results, possibly alongside illegal items (already recorded
        # in _illegal_items)
        return result, 'ok' if not self._illegal_items else 'partial_illegal'

    def _retrieve(self, query: str, k: int = 5, qid: str = '') -> List[RetrievalResult]:
        t_start = time.time()
        qid = qid or self._current_qid
        # Per-query standalone evidence: independent of the returned result
        # list (empty-result and exception paths are also fully recorded)
        ev: Dict = {
            'qid': qid, 'fold': self._current_fold, 'question': query,
            'request_model': 'deepseek-chat',
            'request': {'model': 'deepseek-chat', 'temperature': 0.1,
                        'max_tokens': 2000, 'timeout_s': REQUEST_TIMEOUT,
                        'sdk_max_retries': 0, 'outer_max_retries': MAX_RETRIES},
            'candidate_tables': [], 'shown_columns': {}, 'prompt': '',
            'prompt_sha256': '',
            'api': {'attempts': 0, 'retries': 0, 'http_time_s': [],
                    'resp_model': 'not_provided', 'resp_id': 'not_provided',
                    'usage': 'not_provided', 'raw_response': None,
                    'error': None},
            'parse': {'status': None, 'illegal_items': []},
            'final_selection': [], 'status': None,
        }
        self._last_query_evidence = ev

        if type(self)._FATAL_API_ERROR:
            # Protocol §5: stop the method after an auth/balance/config
            # error (fail fast, no further API calls)
            ev['status'] = 'not_attempted_fatal_stop'
            ev['api']['error'] = type(self)._FATAL_API_ERROR
            ev['wall_time_s'] = round(time.time() - t_start, 3)
            raise RuntimeError(
                '[DenseDeepSeek] Method stopped after non-retryable API '
                f'error: {type(self)._FATAL_API_ERROR}')

        self._illegal_items = []
        self._last_resp_id = 'not_provided'
        self._last_usage = {}
        self._last_resp_model = 'not_provided'
        q_emb = _embed([query])[0]
        scores = self._table_embs @ q_emb
        order = sorted(range(len(scores)),
                      key=lambda i: (-scores[i], self._table_names[i]))
        top_idx = order[:CANDIDATE_K]
        ev['candidate_tables'] = [[self._table_names[i], float(scores[i])]
                                  for i in top_idx]

        schema_text, shown_tables, shown_cols = self._build_schema_text(top_idx, q_emb=q_emb)
        ev['shown_columns'] = shown_cols
        prompt = SCHEMA_SELECT_PROMPT.format(
            schema_text=schema_text, question=query,
            table_budget=TABLE_BUDGET, col_budget=COLUMN_BUDGET)
        ev['prompt'] = prompt
        ev['prompt_sha256'] = hashlib.sha256(prompt.encode('utf-8')).hexdigest()

        try:
            response, api_status = self._call_llm(prompt)
        except RuntimeError as e:
            ev['api'].update({
                'attempts': self._last_attempts,
                'retries': max(0, self._last_attempts - 1),
                'http_time_s': list(self._last_http_times),
                'error': str(e)[:300]})
            ev['status'] = 'api_error'
            ev['wall_time_s'] = round(time.time() - t_start, 3)
            raise RuntimeError(
                f'[DenseDeepSeek] LLM API failure for query '
                f'"{query[:50]}...": {e}') from e

        ev['api'].update({
            'attempts': self._last_attempts,
            'retries': max(0, self._last_attempts - 1),
            'http_time_s': list(self._last_http_times),
            'resp_model': self._last_resp_model,
            'resp_id': self._last_resp_id,
            'usage': self._last_usage,
            'raw_response': response})

        selected, parse_status = self._parse_response(
            response, shown_tables, shown_cols)
        ev['parse'] = {'status': parse_status,
                       'illegal_items': self._illegal_items[:10]}
        ev['status'] = parse_status

        score_map = {self._table_names[i]: float(scores[i]) for i in top_idx}
        ev['final_selection'] = [
            {'table': tname, 'columns': cols, 'table_score': score_map.get(tname, 0.0)}
            for tname, cols in selected[:k]]
        ev['wall_time_s'] = round(time.time() - t_start, 3)

        results = []
        for tname, cols in selected[:k]:
            ts = score_map.get(tname, 0.0)
            columns = [(f"C:{tname}.{c}", ts) for c in cols]
            results.append(RetrievalResult(
                table=tname, table_score=ts, columns=columns,
                metadata={
                    'method': 'DenseDeepSeek',
                    'qid': qid,
                    'prompt_version': 'expansion_v2',
                    'embedding': EMBED_MODEL,
                    'llm': 'deepseek-chat',
                    'llm_resp_model': self._last_resp_model,
                    'api_status': api_status,
                    'resp_id': self._last_resp_id,
                    'usage': self._last_usage,
                    'parse_status': parse_status,
                    'illegal_items': self._illegal_items[:10],
                    'candidate_tables': len(shown_tables),
                    'shown_cols_per_table': {t: len(c) for t, c in shown_cols.items()},
                    'api_calls': self._api_calls,
                    'api_retries': self._api_retries,
                    'api_errors': self._api_errors,
                    'parse_errors': self._parse_errors,
                }))
        return results
