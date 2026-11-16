"""Method/variant matrix: the single code source of truth for how every
reported method and ablation is constructed.

Each entry maps a paper variant name to:
* ``factory``     — registry name in retrievers/retriever_factory.py
* ``supervision`` — training supervision arm (None = raw training rows)
* ``terag``       — TERAGConfig mutations for TE-RAG-V2 / MatchedSchema
  (the annotation toggles reproduce the accepted `transfer_v3_target_cap`
  semantics exactly; see docs/experiment_catalog.md §2)
* ``protocols``   — protocols the paper formally ran (absent groups must not
  be zero-filled or imputed)
* ``config``      — the method configuration file (stem under
  configs/methods/) that documents the method and carries its parameters;
  the resolver validates the file against this entry (single truth: the
  code), so YAML and code cannot silently diverge.

Supervision notes (verified against experiment/run_expansion.py at tag
paper-dasfaa2027-v1.0): the paper runner injects ``full_sql`` supervision
into the training frame BEFORE the reviewer-method loop, and
KaSLA-Adapted's classifiers/capacity estimation consume that structured
full gold (all tables/columns the SQL needs) — NOT the original SELECT
annotation. LinkAlign-SingleDB-Adapted builds only the schema/dense index
(``fit`` ignores train data), so it correctly takes no supervision.
ExampleRRF/CooccurRRF and all TE-RAG variants also use ``full_sql``;
the raw-metadata baselines (BM25/TF-IDF/Hybrid/MetadataMatched) get the
raw frame, exactly as in the paper runner.
"""

from copy import deepcopy

# Baselines: (variant, factory, fusion_mode, supervision, config stem)
BASELINES = [
    ("BM25", "BM25", None, None, "bm25"),
    ("TFIDF", "Vector", None, None, "tfidf"),
    ("Hybrid", "Hybrid", None, None, "hybrid"),
    ("MetadataMatched", "MatchedSchema", None, None, "metadata_matched"),
    ("ExampleRRF", "ExampleRAG", "rank_equal", "full_sql", "example_rrf"),
    ("CooccurRRF", "Cooccurrence", "rank_equal", "full_sql", "cooccur_rrf"),
    ("Dense", "Dense", None, None, "dense"),
    ("DenseDeepSeek", "DenseDeepSeek", None, None, "dense_deepseek"),
    ("LinkAlign-SingleDB-Adapted", "LinkAlign-SingleDB-Adapted", None,
     None, "linkalign_adapted"),
    ("KaSLA-Adapted", "KaSLA-Adapted", None, "full_sql", "kasla_adapted"),
]

FULL_ANNOTATIONS = {"use_history": True, "use_transfer": True,
                    "use_annotation_index": True}


def _terag(mutations=None):
    cfg = {"annotations": dict(FULL_ANNOTATIONS), "graph": {"parse_mode": "ast"}}
    for section, values in (mutations or {}).items():
        cfg.setdefault(section, {}).update(values)
    return cfg


VARIANTS = {
    "TERAG-Full": {
        "factory": "TE-RAG-V2", "supervision": "full_sql", "terag": _terag(),
        "protocols": ["question_v1", "table_v2"], "config": "terag_full",
    },
    "TERAG-noBC": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_history": False,
                                         "use_transfer": False}}),
        "protocols": ["question_v1", "table_v2"], "config": "terag_noBC",
    },
    "TERAG-noC": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_transfer": False}}),
        "protocols": ["table_v2"], "config": "terag_noC",
    },
    "TERAG-noAnnIdx": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_annotation_index": False}}),
        "protocols": ["question_v1"], "config": "terag_noAnnIdx",
    },
    "TERAG-noAnnHit": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"feature": {"field_weights": {"annotation_hit": 0.0}}}),
        "protocols": ["question_v1"], "config": "terag_noAnnHit",
    },
    "TERAG-noGraph": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"ablation": {"use_graph_weight": False}}),
        "protocols": ["table_v2"], "config": "terag_noGraph",
    },
    "TERAG-noPattern": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"ablation": {"use_template_mining": False}}),
        "protocols": ["table_v2"], "config": "terag_noPattern",
    },
}

for _name, _factory, _fusion, _sup, _cfg in BASELINES:
    VARIANTS[_name] = {
        "factory": _factory, "supervision": _sup, "fusion_mode": _fusion,
        "terag": None, "config": _cfg,
        # Dense, DenseDeepSeek and the two adapted methods were run on
        # Question-Level only in the paper (appendix Table 3 shows "--" for
        # their held-out columns); absent groups are never imputed.
        "protocols": (["question_v1"] if _name in
                      ("Dense", "DenseDeepSeek",
                       "LinkAlign-SingleDB-Adapted", "KaSLA-Adapted")
                      else ["question_v1", "table_v2"]),
    }

# MatchedSchema builds TE-RAG's enriched index; the paper's control keeps all
# history channels explicitly OFF.
VARIANTS["MetadataMatched"]["terag"] = {
    "annotations": {"use_history": False, "use_transfer": False},
    "graph": {"parse_mode": "ast"},
}

QUESTION_MATRIX = [v for v, d in VARIANTS.items()
                   if "question_v1" in d["protocols"]]      # 14 groups
TABLE_MATRIX = [v for v, d in VARIANTS.items()
                if "table_v2" in d["protocols"]]            # 11 groups
FULL_MATRIX = QUESTION_MATRIX + TABLE_MATRIX                # 25 groups


def variant_config(base_terag: dict, variant: str) -> dict:
    """Deep-merged TERAGConfig sections for a variant.

    ``base_terag`` is the resolved ``terag`` mapping from the dataset/base
    configuration; the variant's mutations are applied on top.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    merged = deepcopy(base_terag or {})
    for section, values in (VARIANTS[variant]["terag"] or {}).items():
        if section == "feature" and "field_weights" in values:
            merged.setdefault("feature", {}).setdefault("field_weights", {})
            merged["feature"]["field_weights"].update(values["field_weights"])
        else:
            merged.setdefault(section, {}).update(values)
    return merged


def method_requires_network(variant: str) -> bool:
    return VARIANTS[variant]["factory"] in (
        "Dense", "DenseDeepSeek",
        "LinkAlign-SingleDB-Adapted", "KaSLA-Adapted")
