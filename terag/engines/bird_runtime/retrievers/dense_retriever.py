"""
Dense Schema Retrieval Baseline

Loads a real pretrained embedding model (invoked directly through the
transformers API) and ranks table-level composite documents against the
query by cosine similarity. No supervision from historical queries.

Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, supports Chinese),
pinned to revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42
(PUBLIC-PORT: model id + revision instead of a local snapshot path; the
weights resolve through the standard Hugging Face cache, configurable via
TERAG_DENSE_CACHE_DIR / HF_HOME).
Loading: transformers.AutoModel + mean pooling + L2 normalization.
Batched encoding: fixed BATCH_SIZE=64, order-preserving.
Document construction: table doc = table name + table description + the
first TABLE_COL_LIMIT(30) column names; column doc = column name +
column description. Truncation is registered in the protocol.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from terag.engines.bird_runtime.retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig

import os

COLUMN_BUDGET = 10
BATCH_SIZE = 64
MAX_TOKENS = 256
TABLE_COL_LIMIT = 30  # max column names included in a table doc (CSV order, not relevance-ranked)
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# PUBLIC-PORT: the research run loaded a locally cached snapshot by absolute
# path. The public port pins the model id + revision and resolves weights
# through the standard Hugging Face cache (override the cache directory with
# the TERAG_DENSE_CACHE_DIR / HF_HOME environment variables).
MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
EMB_DIM = 384

_MODEL = None
_TOKENIZER = None


def _dense_cache_dir():
    """PUBLIC-PORT: configurable model cache directory (no absolute paths)."""
    env = os.environ.get('TERAG_DENSE_CACHE_DIR') or os.environ.get('HF_HOME')
    return env or None


def _get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                'The Dense baseline requires the optional dependencies '
                '`torch` and `transformers` (model '
                f'{MODEL_NAME} @ {MODEL_REVISION}). Install them to run '
                'Dense; all other offline baselines and engines work '
                'without them.') from exc
        cache_dir = _dense_cache_dir()
        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, cache_dir=cache_dir)
        _MODEL = AutoModel.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, cache_dir=cache_dir)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def _embed(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode texts in batches with mean pooling; return L2-normalized vectors.

    Order guarantee: output order = input order. Empty input returns an
    empty array.
    """
    import torch
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tok, model = _get_model()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.numpy())
    result = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / (norms + 1e-10)


class DenseRetriever(BaseRetriever):
    """Dense schema retrieval (real pretrained embeddings, not TF-IDF)."""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='Dense',
                description='Dense retrieval with pretrained multilingual embedding')
        super().__init__(field_csv, table_csv, config)
        self._table_names: List[str] = []
        self._table_docs: List[str] = []
        self._table_embs: np.ndarray = None
        self._col_keys: Dict[str, List[str]] = {}
        self._col_embs: Dict[str, np.ndarray] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """Build the table-level and column-level embedding indexes (schema
        only, no query history).

        Safe to call repeatedly: all state is cleared before each fit, so
        nothing accumulates.
        """
        # Clear all state (safe under repeated fit)
        self._table_names = []
        self._table_docs = []
        self._table_embs = None
        self._col_keys = {}
        self._col_embs = {}

        # Table-level doc: table name + table description + the first
        # TABLE_COL_LIMIT column names (truncated in CSV row order, not
        # relevance-ranked; registered in the protocol)
        for _, row in self.table_df.iterrows():
            tname = str(row['table'])
            tdesc = str(row.get('table_desc', '') or '')
            cols = self.field_df[self.field_df['table'] == tname]
            col_names = ' '.join(
                cols['field_name'].astype(str).tolist()[:TABLE_COL_LIMIT])
            doc = f"{tname} {tdesc} {col_names}"
            self._table_names.append(tname)
            self._table_docs.append(doc)

        self._table_embs = _embed(self._table_docs)

        # Column-level docs: column name + column description (all columns
        # of the table)
        for tname in self._table_names:
            cols = self.field_df[self.field_df['table'] == tname]
            docs, keys = [], []
            for _, c in cols.iterrows():
                cname = str(c['field_name'])
                cdesc = str(c.get('field_name_desc', '') or '')
                docs.append(f"{cname} {cdesc}")
                keys.append(cname)
            if docs:
                self._col_keys[tname] = keys
                self._col_embs[tname] = _embed(docs)

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        q_emb = _embed([query])[0]
        scores = self._table_embs @ q_emb
        # Stable ordering: score descending, then table name ascending
        # (independent of gold)
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self._table_names[i]))[:k]

        results = []
        for idx in order:
            tname = self._table_names[idx]
            score = float(scores[idx])
            if tname in self._col_embs:
                col_scores = self._col_embs[tname] @ q_emb
                col_order = sorted(
                    range(len(col_scores)),
                    key=lambda i: (-col_scores[i], self._col_keys[tname][i])
                )[:COLUMN_BUDGET]
                columns = [
                    (f"C:{tname}.{self._col_keys[tname][i]}", float(col_scores[i]))
                    for i in col_order
                ]
            else:
                columns = []
            results.append(RetrievalResult(
                table=tname, table_score=score, columns=columns,
                metadata={'method': 'Dense', 'model': MODEL_NAME}))
        return results
