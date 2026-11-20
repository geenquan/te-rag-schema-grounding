"""Minimal retrievers package for the BIRD runs (no heavy deps)."""
from terag.engines.bird_runtime.retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult, EvaluationMetrics)
from terag.engines.bird_runtime.retrievers.bm25_retriever import (
    BM25Retriever as BM25SchemaRetriever)
