"""Retriever package for the public TE-RAG release.

Exposes the base interfaces and the retrievers reported in the paper (see
docs/experiment_catalog.md for the method matrix). Legacy research
implementations that were never part of the paper's reported experiments
(simulated LLM retriever, TE-RAG v1, graph-only variant, RAT-SQL/RESDSQL/
Seq2SQL/IRNet adaptations) are not shipped in this repository.
"""

from retrievers.base_retriever import (
    BaseRetriever,
    RetrieverConfig,
    RetrievalResult,
    EvaluationMetrics,
)
from retrievers.bm25_retriever import BM25Retriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.matched_schema_retriever import MatchedSchemaRetriever
from retrievers.example_rag_retriever import ExampleRAGRetriever
from retrievers.cooccurrence_retriever import CooccurrenceRetriever
from retrievers.retriever_factory import RetrieverFactory, RetrieverManager

__all__ = [
    'BaseRetriever',
    'RetrieverConfig',
    'RetrievalResult',
    'EvaluationMetrics',
    'BM25Retriever',
    'VectorRetriever',
    'HybridRetriever',
    'MatchedSchemaRetriever',
    'ExampleRAGRetriever',
    'CooccurrenceRetriever',
    'RetrieverFactory',
    'RetrieverManager',
]
