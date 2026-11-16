"""
TE-RAG core modules (paper release).

Contains:
- config: configuration management
- graph_builder: bipartite graph construction
- sql_role_parser: SQL role parsing
- pattern_miner: pattern mining
- index_builder: index building
- feature_extractor: feature extraction
- ranker: rankers
- terag_retriever_v2: current-generation retriever

Public-release note: the private research repository also exposed a weight
learning module here; ranking weights are fixed in the paper's experiments
and that module is not shipped.
"""

from terag.config import TERAGConfig
from terag.sql_role_parser import SQLRoleParser
from terag.graph_builder import BipartiteGraphBuilder
from terag.pattern_miner import PatternMiner
from terag.index_builder import IndexBuilder
from terag.feature_extractor import FeatureExtractor
from terag.ranker import TableRanker, FieldRanker, CombinedRanker
from terag.terag_retriever_v2 import TERAGRetrieverV2, RetrievalResultV2

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
    'TERAGRetrieverV2',
    'RetrievalResultV2',
]
