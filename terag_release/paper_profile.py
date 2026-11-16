"""The paper profile: the single code source of truth for fixed parameters.

Everything the paper froze before evaluation lives HERE as code constants.
The YAML files under configs/ mirror these values for human review; under
the default ``paper`` profile the resolver validates YAML against these
constants and REFUSES to run on any mismatch ("changed the config but it
silently did not take effect" becomes impossible). Runs that intentionally
deviate must request ``profile: custom`` — the deviation then flows into
the retrievers for real and is stamped into run_meta.

tests/test_config_binding.py separately asserts that the constants imported
from the retriever modules (model IDs, API limits, transfer caps) equal the
values declared here, so code, YAML, and this profile cannot drift apart.

Dataset-dependent values (paths, qid prefix, tokenizer language, domain
keyword lists) are data, not algorithm parameters; they are freely
configurable in every profile.
"""

PAPER_PROFILE = "paper"
CUSTOM_PROFILE = "custom"

PAPER_TERAG = {
    "version": "1.0.0",
    "seed": 42,
    "index": {
        "bm25": {"k1": 1.5, "b": 0.75},
        "field_weights": {
            "table_name": 2.0, "table_desc": 1.5, "field_name": 1.0,
            "field_desc": 0.8, "pattern": 0.5,
        },
        "include_field_desc": True,
    },
    "graph": {
        "role_weights": {
            "SELECT": 1.2, "WHERE": 2.0, "JOIN": 1.8, "GROUP_BY": 1.5,
            "ORDER_BY": 1.3, "HAVING": 1.4, "FROM": 1.0,
        },
        "parse_mode": "ast",
    },
    "pattern_mining": {
        "mode": "keywords",
        "similarity_threshold": 0.3,
        "min_pattern_count": 1,
    },
    "feature": {
        "table_weights": {"bm25_score": 0.35, "graph_score": 0.35,
                          "pattern_score": 0.30},
        "field_weights": {"direct_match": 0.40, "graph_propagation": 0.20,
                          "role_prior": 0.15, "train_recommend": 0.25,
                          "annotation_hit": 1.0},
        "graph_propagation_threshold": 0.15,
        "train_similarity_threshold": 0.30,
        "graph_propagation_weight": 3.0,
        "template_match_weight": 0.6,
        "direct_match_multiplier": 1.5,
        "field_name_match_bonus": 0.4,
        "field_graph_score_cap": 5.0,
        "train_recommend_multiplier": 1.5,
        "train_recommend_cap": 3.0,
    },
    "annotations": {"use_history": False, "use_transfer": False,
                    "use_annotation_index": True},
    "ablation": {
        "use_graph_weight": True, "use_template_mining": True,
        "use_pattern_generalization": True, "use_enhanced_index": True,
        "use_role_parser": True, "use_role_weight": True,
    },
    "evaluation": {"table_k": [1, 3, 5, 10], "column_k": [5, 10, 20]},
}

# Base-constructor parameters of the lexical baselines (RetrieverConfig
# fields). The BM25 baseline shares the protocol's fixed BM25 parameters.
PAPER_BASELINE_PARAMS = {
    "BM25": {"k1": 1.5, "b": 0.75},
    "TFIDF": {}, "Hybrid": {}, "MetadataMatched": {},
    "ExampleRRF": {}, "CooccurRRF": {}, "Dense": {},
    "DenseDeepSeek": {}, "LinkAlign-SingleDB-Adapted": {},
    "KaSLA-Adapted": {},
}

# Model / API constants frozen in the retriever modules. The resolver
# validates configs/methods/*.yaml `model:` sections against these;
# test_config_binding asserts equality with the imported module constants.
PAPER_MODEL_PARAMS = {
    "Dense": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "max_tokens": 256,
        "table_column_limit": 30,
    },
    "DenseDeepSeek": {
        "request_model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 2000,
        "timeout_s": 60,
        "max_retries": 1,
        "candidate_tables": 20,
        "displayed_columns_per_table": 10,
    },
    "LinkAlign-SingleDB-Adapted": {
        "request_model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens_isolation": 32768,
        "max_tokens_other": 8192,
        "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "chunk_columns": 24,
        "chunk_max_tokens": 256,
        "retrieve_chunks": 20,
    },
    "KaSLA-Adapted": {
        "request_model": "deepseek-chat",
        "temperature": 0.1,
        "classifier": {"C": 1.0, "max_iter": 300, "random_state": 42},
        "neighbors_for_capacity": 30,
        "shown_tables": 20,
        "shown_columns_per_table": 20,
    },
}

PAPER_BUDGET = {"max_tables": 5, "max_columns_per_table": 10}

PAPER_PROTOCOL = {
    "folds": 5,
    "supervision": "full_sql",
    "isolation": {"mode": "question"},
    "isolation_table": {"mode": "target_table_ast",
                        "exclude_unparseable_train_sql": True},
}

# Keys of the terag sections that carry DATASET-provided content (freely
# configurable in every profile) rather than frozen algorithm values.
DATASET_FREE_TERAG_PATHS = {
    ("tokenizer",),                 # language choice follows the questions
    ("pattern_mining", "entity_keywords"),
    ("pattern_mining", "attribute_keywords"),
    ("pattern_mining", "aggregation_keywords"),
    ("pattern_mining", "english_aggregation_keywords"),
    ("pattern_mining", "time_patterns"),
    ("pattern_mining", "english_time_patterns"),
}


def _dig(node, path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None, False
        node = node[key]
    return node, True


def _diff_fixed(expected, actual, path="", out=None):
    """Collect (path, expected, actual) triples where actual deviates from
    the frozen paper values. ``actual`` may contain extra keys (reported);
    ``expected`` keys must match exactly."""
    out = [] if out is None else out
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append((path, expected, actual))
            return out
        for key, value in expected.items():
            _diff_fixed(value, actual.get(key),
                        f"{path}.{key}" if path else key, out)
        for key in actual:
            if key not in expected:
                out.append((f"{path}.{key}" if path else key,
                            "<not a recognized parameter>", actual[key]))
    elif expected != actual:
        out.append((path, expected, actual))
    return out


def diff_against_paper(configured_terag: dict) -> list:
    """Deviations of a configured terag mapping from the paper constants.

    Dataset-free paths (tokenizer language, keyword lists) are exempt.
    """
    diffs = _diff_fixed(PAPER_TERAG, configured_terag or {})
    return [d for d in diffs
            if not any(len(d[0].split(".")) <= len(p) and
                       tuple(d[0].split(".")[:len(p)]) == p
                       for p in DATASET_FREE_TERAG_PATHS)]


# --------------------------------------------------------------------------
# Read-only mirrors and structural constants
# --------------------------------------------------------------------------
# These configuration fields are MIRRORS of values fixed in code. They are
# validated against the actual runtime sources under EVERY profile (paper
# and custom alike): a YAML value that contradicts the real constant is a
# startup error, because execution would silently ignore it. They are not
# customizable in this release — the supported custom surface is the
# retriever parameters (k1/b/top_k/tokenizer) and the terag sections
# (weights/thresholds/toggles), which are genuinely wired through to the
# constructors under profile=custom.

# Mirror of retrievers/annotation_transfer.TransferConfig defaults.
TRANSFER_MIRROR_KEYS = ("min_total_sim", "schema_only_min", "context_min",
                        "max_sources_per_target",
                        "max_annotations_per_target", "max_texts_per_table")

# Mirror of the model/API constants declared above; the method YAML
# `model:`/`api:` sections may only restate these values.
MODEL_MIRROR = PAPER_MODEL_PARAMS

# Allowed api: section keys and their fixed values.
API_MIRROR = {
    "key_env": "DEEPSEEK_API_KEY",
    "base_url_env": "DEEPSEEK_BASE_URL",
    "base_url_default": "https://api.deepseek.com",
}

# Structural protocol constants (fold count, supervision arm, retrieval
# budget). The public runner is not parameterized for these; deviations are
# rejected under every profile.
STRUCTURAL = {
    "folds": 5,
    "supervision": "full_sql",
    "budget": PAPER_BUDGET,
}

SUPPORTED_PROTOCOL_NAMES = ("question_v1", "table_v2")

# The custom-profile surface: parameters that genuinely reach execution.
CUSTOM_SUPPORTED = ("retriever parameters (k1, b, top_k, use_chinese_tokenizer"
                    " via the dataset tokenizer language)",
                    "terag sections (weights, thresholds, annotation and "
                    "ablation toggles)")


def actual_transfer_defaults():
    """The real TransferConfig defaults, read from the implementation."""
    from retrievers.annotation_transfer import TransferConfig
    cfg = TransferConfig()
    return {key: getattr(cfg, key) for key in TRANSFER_MIRROR_KEYS}
