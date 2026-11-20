"""TE-RAG core modules (frozen paper version, BIRD-run identity).

This package is the final research workspace ``terag/`` package re-rooted
under ``terag.engines.bird_runtime`` with import rewrites only; scoring
behavior is identical to the frozen research runs.

Modules:
- config: configuration management (TERAGConfig and friends)
- graph_builder: query-table-column bipartite graph construction
- sql_role_parser: SQL role parsing (sqlglot AST with scoped resolution)
- pattern_miner: query pattern mining
- index_builder: BM25 inverted index construction
- feature_extractor: table/field feature extraction
- ranker: table/field rankers
- terag_retriever_v2: the integrated TE-RAG retriever
- weight_learner: supervised ranking-weight learning
- tokenization: language-dispatched tokenization (jieba / English regex)
"""

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.sql_role_parser import SQLRoleParser
from terag.engines.bird_runtime.graph_builder import BipartiteGraphBuilder
from terag.engines.bird_runtime.pattern_miner import PatternMiner
from terag.engines.bird_runtime.index_builder import IndexBuilder
from terag.engines.bird_runtime.feature_extractor import FeatureExtractor
from terag.engines.bird_runtime.ranker import TableRanker, FieldRanker, CombinedRanker
from terag.engines.bird_runtime.weight_learner import WeightLearner
from terag.engines.bird_runtime.terag_retriever_v2 import TERAGRetrieverV2, RetrievalResultV2

__all__ = [
    'TERAGConfig',
    'SQLRoleParser',
    'BipartiteGraphBuilder',
    'PatternMiner',
    'IndexBuilder',
    'FeatureExtractor',
    'TableRanker',
    'FieldRanker',
    'CombinedRanker',
    'WeightLearner',
    'TERAGRetrieverV2',
    'RetrievalResultV2',
]
