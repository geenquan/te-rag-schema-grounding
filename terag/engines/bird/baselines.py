"""Offline baseline adapter for the BIRD schema-grounding runs.

Wires the frozen bird_runtime retrievers with the exact method identities
used by the paper baselines:

- BM25            : bird_runtime retrievers/bm25_retriever.BM25Retriever over
                    table name+desc documents only, English tokenizer, fit
                    ignores train data.
- MetadataMatched : retrievers/matched_schema_retriever.MatchedSchemaRetriever.
                    Builds the SAME enhanced metadata index documents as TE-RAG
                    Full's metadata index (bird_runtime
                    terag.index_builder.IndexBuilder: table name/desc + field
                    name/desc, weighted, English) with use_history=False and
                    use_transfer=False. That index (and no history/transfer
                    channel) IS this baseline's definition.
- ExampleRRF      : retrievers/example_rag_retriever.ExampleRAGRetriever with
                    fusion_mode='rank_equal' (the frozen implementation applies
                    the P2-C corrected RRF, recorded as 'rank_equal_v2' in the
                    result evidence metadata: only positive-evidence tables get
                    an evidence rank; the BM25 fallback list uses fixed depth
                    BM25_RANK_DEPTH=20). Example evidence comes from the current
                    train fold's full-SQL gold tables/fields (all roles, parsed
                    via datasets.gt_utils.parse_ground_truth_fields), never from
                    SELECT-only projections.
- TFIDF           : retrievers/vector_retriever.VectorRetriever - log-TF x idf
                    field-level vectors, weighted table aggregation, cosine
                    scoring - with ENGLISH tokenization; the index is
                    schema-only (fit ignores train data).
- Hybrid          : retrievers/hybrid_retriever.HybridRetriever - BM25 + TFIDF
                    linear score fusion, default alpha=0.5 (equal weight;
                    min-max normalization per retriever before fusion), English
                    tokenization propagated to both sub-retrievers.
- CooccurRRF      : retrievers/cooccurrence_retriever.CooccurrenceRetriever
                    with fusion_mode='rank_equal' (the preregistered
                    rank_equal_v2 RRF). Supervision = the current train rows,
                    full-SQL gold tables/fields via gt_utils.
- Dense           : DenseBirdRetriever below - real MiniLM
                    paraphrase-multilingual-MiniLM-L12-v2 (revision
                    e8f8c211226b894fcb81acc59f3b34ba3efd5f42) table
                    embeddings, per-db cache keyed by the schema CSV hashes,
                    BM25-baseline overlap column ranking (top 10 per table).
                    Requires the optional torch/transformers dependencies and
                    raises a clear error when they are absent.

Boundary fixes (identical to the research adapters; the frozen retriever
files are never edited): ExampleRRF and CooccurRRF flip the internal
fallback BM25Retriever to the English tokenizer BEFORE fit() so the
fallback index is built with the same tokenizer as the rest of the method.

Record shape: {qid, db_id, variant, status: 'ok'|'error', retrieved_tables,
retrieved_column_keys, retrieved_columns_detail, query_time, error}.
Errors keep the question in the denominator (record kept, outputs empty).
Budgets: <=5 tables per question, <=10 columns per table; violations are
recorded as status='error'.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from terag.engines.bird_runtime.retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult)

METHODS = ('BM25', 'MetadataMatched', 'ExampleRRF', 'TFIDF', 'Hybrid',
           'CooccurRRF', 'Dense')

K_TABLES = 5              # table budget per question
COLUMN_BUDGET = 10        # column budget per table (enforced, not truncated)


# ---------------------------------------------------------------------------
# Retriever construction (exact identities; deviations not allowed)
# ---------------------------------------------------------------------------

def build_fitted_retriever(method: str, cfg, field_csv: str, table_csv: str,
                           train_rows: pd.DataFrame, cache_root=None,
                           db_id: str = None):
    """Construct and fit the frozen retriever for `method` (exact identity).

    Exposed for tests; run_baseline() is the contract entry point.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown baseline method: {method!r}; expected one of {METHODS}")

    if method == 'BM25':
        from terag.engines.bird_runtime.retrievers.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(
            field_csv, table_csv,
            RetrieverConfig(name='BM25', use_chinese_tokenizer=False))
        retriever.fit(train_rows)  # fit() ignores train data (identity note)
        return retriever

    if method == 'MetadataMatched':
        from terag.engines.bird_runtime.retrievers.matched_schema_retriever import MatchedSchemaRetriever
        retriever = MatchedSchemaRetriever(
            field_csv, table_csv,
            RetrieverConfig(
                name='MetadataMatched',
                use_chinese_tokenizer=False,
                extra_params={'terag_config': cfg,
                              'use_history': False,
                              'use_transfer': False}))
        retriever.fit(train_rows)  # use_history=False: builds pure metadata index
        return retriever

    if method == 'ExampleRRF':
        from terag.engines.bird_runtime.retrievers.example_rag_retriever import ExampleRAGRetriever
        retriever = ExampleRAGRetriever(
            field_csv, table_csv,
            RetrieverConfig(
                name='ExampleRRF',
                use_chinese_tokenizer=False,
                extra_params={'fusion_mode': 'rank_equal'}))
        # BOUNDARY FIX (do not edit the frozen retriever files): the internal
        # fallback BM25Retriever was constructed with the default config
        # (Chinese jieba tokenizer). Flip it to English BEFORE fit() so the
        # fallback index and its query tokenization use the same English
        # tokenizer as everything else.
        retriever._fallback.config.use_chinese_tokenizer = False
        retriever.fit(train_rows)  # example evidence = full-SQL gold via gt_utils
        return retriever

    if method == 'TFIDF':
        from terag.engines.bird_runtime.retrievers.vector_retriever import VectorRetriever
        retriever = VectorRetriever(
            field_csv, table_csv,
            RetrieverConfig(name='TFIDF', use_chinese_tokenizer=False))
        retriever.fit(train_rows)  # schema-only index (identity note)
        return retriever

    if method == 'Hybrid':
        from terag.engines.bird_runtime.retrievers.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            field_csv, table_csv,
            config=RetrieverConfig(name='Hybrid', use_chinese_tokenizer=False),
            alpha=0.5)  # CONTRACTS6: default alpha (equal BM25/TFIDF weight); NO tuning
        retriever.fit(train_rows)
        return retriever

    if method == 'CooccurRRF':
        from terag.engines.bird_runtime.retrievers.cooccurrence_retriever import CooccurrenceRetriever
        retriever = CooccurrenceRetriever(
            field_csv, table_csv,
            RetrieverConfig(name='CooccurRRF', use_chinese_tokenizer=False,
                            extra_params={'fusion_mode': 'rank_equal'}))
        # BOUNDARY FIX (same kind as the ExampleRRF fix; see the docstring)
        retriever._fallback.config.use_chinese_tokenizer = False
        retriever.fit(train_rows)  # evidence = full-SQL gold via gt_utils
        return retriever

    # method == 'Dense'
    if cache_root is None:
        raise ValueError("Dense requires cache_root (per-db embedding cache directory)")
    retriever = DenseBirdRetriever(
        field_csv, table_csv, db_id if db_id is not None else 'db', cache_root)
    retriever.fit(train_rows)
    return retriever


# ---------------------------------------------------------------------------
# Record normalization (shared by all methods)
# ---------------------------------------------------------------------------

def _strip_c_prefix(col_key: str) -> str:
    """'C:{table}.{field}' -> '{table}.{field}' (keep table.field)."""
    return col_key[2:] if col_key.startswith('C:') else col_key


def _result_to_detail(results) -> List[Dict[str, Any]]:
    """RetrievalResult list -> retrieved_columns_detail (C: prefix stripped)."""
    detail = []
    for r in results:
        detail.append({
            'table': r.table,
            'table_score': float(r.table_score),
            'columns': [{'key': _strip_c_prefix(c), 'score': float(s)}
                        for c, s in r.columns],
        })
    return detail


def _column_keys(results) -> List[str]:
    """Ordered, de-duplicated simple column keys ('table.field')."""
    keys: List[str] = []
    for r in results:
        for col, _ in r.columns:
            key = _strip_c_prefix(col)
            if key not in keys:
                keys.append(key)
    return keys


def _check_budgets(results) -> Optional[str]:
    """Return a violation message, or None if within budgets."""
    if len(results) > K_TABLES:
        return f"budget violation: {len(results)} tables > {K_TABLES}"
    for r in results:
        if len(r.columns) > COLUMN_BUDGET:
            return (f"budget violation: table {r.table!r} has "
                    f"{len(r.columns)} columns > {COLUMN_BUDGET}")
    return None


def _question_map(questions) -> Dict[str, str]:
    """Accept {qid: question} mapping or DataFrame with qid/question columns."""
    if questions is None:
        return {}
    if isinstance(questions, pd.DataFrame):
        qmap = {}
        for _, row in questions.iterrows():
            qmap[str(row['qid'])] = str(row['question'])
        return qmap
    if isinstance(questions, pd.Series):
        return {str(k): str(v) for k, v in questions.items()}
    return {str(k): str(v) for k, v in dict(questions).items()}


def run_retriever_records(retriever, method: str, db_id: str,
                          test_qids: List[str], questions
                          ) -> List[Dict[str, Any]]:
    """Run a fitted retriever over all test questions with per-question isolation.

    One record per test qid (same order). Any per-question exception becomes
    status='error' with empty outputs (fixed denominator). Budget violations
    (tables > 5 or columns/table > 10) also become error records.
    """
    qmap = _question_map(questions)
    records: List[Dict[str, Any]] = []
    for qid in test_qids:
        record = {
            'qid': qid,
            'db_id': db_id,
            'variant': method,
            'status': 'ok',
            'retrieved_tables': [],
            'retrieved_column_keys': [],
            'retrieved_columns_detail': [],
            'query_time': 0.0,
            'error': None,
        }
        t0 = time.perf_counter()
        try:
            question = qmap.get(str(qid))
            if question is None:
                raise KeyError(f"missing question text for qid {qid!r}")
            results = retriever.retrieve(question, k=K_TABLES)
            violation = _check_budgets(results)
            if violation is not None:
                raise RuntimeError(violation)
            record['retrieved_tables'] = [r.table for r in results]
            record['retrieved_column_keys'] = _column_keys(results)
            record['retrieved_columns_detail'] = _result_to_detail(results)
        except Exception as exc:  # noqa: BLE001 - per-question error isolation
            record['status'] = 'error'
            record['retrieved_tables'] = []
            record['retrieved_column_keys'] = []
            record['retrieved_columns_detail'] = []
            record['error'] = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            record['query_time'] = time.perf_counter() - t0
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Contract entry point
# ---------------------------------------------------------------------------

def run_baseline(method: str, cfg, db_id: str, train_rows: pd.DataFrame,
                 test_qids: List[str], questions, field_csv: str,
                 table_csv: str, cache_root=None) -> List[Dict[str, Any]]:
    """Run one baseline over all test questions of one database.

    Args:
        method: one of METHODS ('BM25', 'MetadataMatched', 'ExampleRRF',
            'TFIDF', 'Hybrid', 'CooccurRRF', 'Dense').
        cfg: TERAGConfig from terag.engines.bird.config_factory.make_bird_config
             (only consumed by MetadataMatched's IndexBuilder wiring).
        db_id: database id for the records (also the Dense per-db cache key).
        train_rows: DataFrame with qid, question, sql, db_id, table, field,
                    gold_tables (JSON list), gold_fields (JSON list of
                    [simple_table, col]) - full SQL physical needs.
        test_qids: qids to evaluate, in order; every qid yields a record.
        questions: {qid: question_text} mapping (or DataFrame qid/question).
        field_csv, table_csv: per-db schema CSV paths.
        cache_root: directory under which dense/<db_id>/ caches live
                    (Dense only; required for method='Dense').

    Returns:
        One record per test qid (same order). Errors produce status='error'
        with empty outputs and stay in the denominator.
    """
    retriever = build_fitted_retriever(method, cfg, field_csv, table_csv,
                                       train_rows, cache_root=cache_root,
                                       db_id=db_id)
    return run_retriever_records(retriever, method, db_id, test_qids, questions)


# ---------------------------------------------------------------------------
# Dense identity (per-db encode, cache-aware)
# ---------------------------------------------------------------------------

DENSE_BATCH_SIZE = 64       # dense_retriever.py fixed batch size
DENSE_MAX_TOKENS = 256      # dense_retriever.py fixed truncation
DENSE_TABLE_COL_LIMIT = 30  # dense_retriever.py: column names per table doc
DENSE_EMB_DIM = 384         # MiniLM-L12 output dimension
DENSE_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DENSE_MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

DENSE_IDENTITY = {
    'method': 'Dense',
    'model': DENSE_MODEL_NAME,
    'revision': DENSE_MODEL_REVISION,
    # PUBLIC-PORT: the research run opened a pinned local snapshot directory;
    # the public port pins model id + revision and resolves the weights
    # through the standard Hugging Face cache (TERAG_DENSE_CACHE_DIR /
    # HF_HOME override), loading with transformers.AutoModel/AutoTokenizer
    # (math identical to SentenceTransformer mean pooling + L2 normalize).
    'loading': 'transformers.AutoModel/AutoTokenizer, model id + pinned '
               'revision, standard HF cache (offline after first download)',
    'pooling': 'masked mean over attention_mask, L2 normalized',
    'max_tokens': DENSE_MAX_TOKENS,
    'batch_size': DENSE_BATCH_SIZE,
    'table_doc': 'table name + table_desc + first '
                 f'{DENSE_TABLE_COL_LIMIT} column names (schema CSV order)',
    'column_ranking': 'BM25-baseline overlap family (English tokens), '
                      'per-table top-10 by (-score, field name)',
    'torch': 'torch.set_num_threads(1) at model load',
}


def dense_embed(texts: List[str], batch_size: int = DENSE_BATCH_SIZE) -> np.ndarray:
    """Encode texts: masked mean pooling + L2 normalize (dense_retriever math).

    Order guarantee: input order = output order. Empty input -> (0, 384).
    Deterministic: eval mode, no_grad, fixed batch order, single thread.

    PUBLIC-PORT: raises a clear English error when torch/transformers are
    not installed (they are optional dependencies).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            'The Dense baseline requires the optional dependencies `torch` '
            'and `transformers` (model '
            f'{DENSE_MODEL_NAME} @ {DENSE_MODEL_REVISION}). Install them to '
            'run Dense; all other offline baselines and engines work '
            'without them.') from exc
    if not texts:
        return np.zeros((0, DENSE_EMB_DIM), dtype=np.float32)
    import os
    cache_dir = os.environ.get('TERAG_DENSE_CACHE_DIR') or os.environ.get('HF_HOME') or None
    tok = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME,
                                        revision=DENSE_MODEL_REVISION,
                                        cache_dir=cache_dir)
    model = AutoModel.from_pretrained(DENSE_MODEL_NAME,
                                      revision=DENSE_MODEL_REVISION,
                                      cache_dir=cache_dir)
    torch.set_num_threads(1)  # deterministic, single thread per the run env
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=DENSE_MAX_TOKENS)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.numpy())
    result = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / (norms + 1e-10)


def table_doc_text(table_name: str, table_desc: str,
                   column_names: List[str]) -> str:
    """dense_retriever.py identity: '{name} {desc} {first-30 column names}'."""
    col_names = ' '.join(str(c) for c in list(column_names)[:DENSE_TABLE_COL_LIMIT])
    return f"{table_name} {table_desc} {col_names}"


def _schema_sha256(field_csv: str, table_csv: str) -> str:
    """Cache key: SHA-256 over both schema CSV file bytes."""
    h = hashlib.sha256()
    for path in (field_csv, table_csv):
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                h.update(block)
        h.update(b'\n\x00schema-file-boundary\x00\n')
    return h.hexdigest()


class DenseBirdRetriever(BaseRetriever):
    """Dense schema retrieval: real MiniLM table embeddings, per-db cache.

    Subclasses the frozen BaseRetriever for schema loading + English
    tokenization (column overlap scoring); no frozen file is modified.
    """

    def __init__(self, field_csv: str, table_csv: str, db_id: str,
                 cache_root):
        super().__init__(field_csv, table_csv,
                         RetrieverConfig(name='Dense',
                                         use_chinese_tokenizer=False))
        self.db_id = str(db_id)
        self.cache_root = Path(cache_root)
        self.schema_sha256 = _schema_sha256(field_csv, table_csv)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._cache_path = None
        self._cache_loaded = False  # test hook: True when cache file reused

    # ---------------- fit (per-db, cache keyed by schema SHA) ----------------

    def fit(self, train_data: pd.DataFrame = None):
        """Build the per-db table embedding index (schema-only, no history).

        Encodes table documents ONCE per database; the result is cached under
        <cache_root>/dense/<db_id>/<schema_sha256>.npz and reloaded verbatim on
        any later fit with the same schema (train_data is accepted for
        interface parity and ignored - the Dense identity is unsupervised).
        """
        cache_dir = self.cache_root / 'dense' / self.db_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path = cache_dir / f'{self.schema_sha256}.npz'
        sidecar_path = cache_dir / f'{self.schema_sha256}.json'
        self._cache_path = npz_path

        if npz_path.is_file() and sidecar_path.is_file():
            # Unicode-dtype string arrays load without pickle.
            saved = np.load(npz_path, allow_pickle=False)
            self._table_names = [str(x) for x in saved['table_names'].tolist()]
            self._table_docs = [str(x) for x in saved['table_docs'].tolist()]
            self._table_embs = saved['table_embs'].astype(np.float32)
            self._cache_loaded = True
            if len(self._table_names) != len(self._table_embs):
                raise RuntimeError('dense cache corruption: name/vector count mismatch')
            self._is_fitted = True
            return

        docs, names = [], []
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            docs.append(table_doc_text(tname, tdesc,
                                       cols['field_name'].astype(str).tolist()))
            names.append(tname)
        self._table_names = names
        self._table_docs = docs
        self._table_embs = dense_embed(docs).astype(np.float32)

        # Write via an open file object (np.savez would otherwise append
        # '.npz' to a tmp name), then atomically move into place.
        tmp = npz_path.with_name(npz_path.name + '.tmp')
        with tmp.open('wb') as f:
            np.savez(f, table_names=np.array(names, dtype=np.str_),
                     table_docs=np.array(docs, dtype=np.str_),
                     table_embs=self._table_embs)
        tmp.replace(npz_path)
        sidecar_path.write_text(json.dumps(
            {'db_id': self.db_id, 'schema_sha256': self.schema_sha256,
             'n_tables': len(names), 'identity': DENSE_IDENTITY}, ensure_ascii=False,
            indent=1, default=str) + '\n', encoding='utf-8')
        self._cache_loaded = False
        self._is_fitted = True

    def cache_info(self) -> Dict[str, Any]:
        """Cache status for tests/auditing."""
        return {'cache_path': str(self._cache_path) if self._cache_path else None,
                'schema_sha256': self.schema_sha256,
                'loaded_from_cache': self._cache_loaded,
                'n_tables': len(self._table_names)}

    # ---------------- retrieve ----------------

    def _columns_for(self, query_terms: List[str], table: str) -> List[tuple]:
        """BM25-baseline overlap scoring family; dense_retriever selection
        mechanics (per-table sort by (-score, field name), top 10)."""
        query_term_set = set(query_terms)
        scored = []
        for _, row in self.field_df[self.field_df['table'] == table].iterrows():
            field_name = str(row['field_name'])
            field_desc = str(row.get('field_name_desc', '') or '')
            field_terms = set(t for t in self.tokenize(
                f"{field_name} {field_desc}") if len(t) > 1)
            overlap = len(query_term_set & field_terms)
            col_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in field_name for t in query_term_set):
                col_score += 0.5
            scored.append((field_name, col_score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(f"C:{table}.{f}", s) for f, s in scored[:COLUMN_BUDGET]]

    def _retrieve(self, query: str, k: int = K_TABLES) -> List[RetrievalResult]:
        q_emb = dense_embed([query])[0]
        scores = self._table_embs @ q_emb
        # Stable order exactly as dense_retriever.py: (-score, table name).
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self._table_names[i]))[:k]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        results = []
        for idx in order:
            tname = self._table_names[idx]
            results.append(RetrievalResult(
                table=tname, table_score=float(scores[idx]),
                columns=self._columns_for(query_terms, tname),
                metadata={'method': 'Dense', 'model': DENSE_MODEL_NAME,
                          'revision': DENSE_MODEL_REVISION}))
        return results
