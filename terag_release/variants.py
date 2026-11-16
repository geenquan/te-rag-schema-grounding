"""Method/variant matrix: the single source of truth for how every reported
method and ablation is constructed.

Each entry maps a paper variant name to:
* ``factory``     — registry name in retrievers/retriever_factory.py
* ``supervision`` — training supervision arm (None = raw training rows)
* ``terag``       — TERAGConfig mutations for TE-RAG-V2 / MatchedSchema
  (the annotation toggles reproduce the accepted `transfer_v3_target_cap`
  semantics exactly; see docs/experiment_catalog.md §2)
* ``protocols``   — protocols the paper formally ran (absent groups must not
  be zero-filled or imputed)

The toggle semantics:
* Full             use_history=True, use_transfer=True, use_annotation_index=True
* noBC             use_history=False (no B, hence no C)
* noC              use_transfer=False (B only)
* noAnnIdx         use_annotation_index=False (B/C kept out of the index)
* noAnnHit         feature weight annotation_hit = 0.0 (column TERM hit off)
* noGraph          ablation.use_graph_weight=False (graph scores off; the
                   graph itself is still built for pattern mining)
* noPattern        ablation.use_template_mining=False (keyword-pattern channel)
"""

from copy import deepcopy

# Baselines: (variant, factory, fusion_mode, supervision)
BASELINES = [
    ("BM25", "BM25", None, None),
    ("TFIDF", "Vector", None, None),
    ("Hybrid", "Hybrid", None, None),
    ("MetadataMatched", "MatchedSchema", None, None),
    ("ExampleRRF", "ExampleRAG", "rank_equal", "full_sql"),
    ("CooccurRRF", "Cooccurrence", "rank_equal", "full_sql"),
    ("Dense", "Dense", None, None),
    ("DenseDeepSeek", "DenseDeepSeek", None, None),
    ("LinkAlign-SingleDB-Adapted", "LinkAlign-SingleDB-Adapted", None, None),
    ("KaSLA-Adapted", "KaSLA-Adapted", None, None),
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
        "protocols": ["question_v1", "table_v2"],
    },
    "TERAG-noBC": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_history": False,
                                         "use_transfer": False}}),
        "protocols": ["question_v1", "table_v2"],
    },
    "TERAG-noC": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_transfer": False}}),
        "protocols": ["table_v2"],
    },
    "TERAG-noAnnIdx": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"annotations": {"use_annotation_index": False}}),
        "protocols": ["question_v1"],
    },
    "TERAG-noAnnHit": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"feature": {"field_weights": {"annotation_hit": 0.0}}}),
        "protocols": ["question_v1"],
    },
    "TERAG-noGraph": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"ablation": {"use_graph_weight": False}}),
        "protocols": ["table_v2"],
    },
    "TERAG-noPattern": {
        "factory": "TE-RAG-V2", "supervision": "full_sql",
        "terag": _terag({"ablation": {"use_template_mining": False}}),
        "protocols": ["table_v2"],
    },
}

for _name, _factory, _fusion, _sup in BASELINES:
    VARIANTS[_name] = {
        "factory": _factory, "supervision": _sup, "fusion_mode": _fusion,
        "terag": None,
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
