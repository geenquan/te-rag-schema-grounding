"""Configuration loading for the public release CLI.

Layered YAML configuration:

* ``configs/base.yaml``           — fixed algorithm parameters (paper values)
* ``configs/datasets/*.yaml``     — data paths, tokenizer language, domain
                                     keyword lists
* ``configs/protocols/*.yaml``    — protocol definitions
* ``configs/methods/*.yaml``      — per-method documentation and overrides
* ``configs/experiments/*.yaml``  — which dataset/protocols/methods to run

Files may declare ``extends:`` (relative path) to inherit and override. The
loader resolves the full tree into a single mapping that is:

* exported verbatim next to every run (``run_meta/resolved_config.json``), and
* fingerprinted (SHA256 of the canonical JSON) into run_meta, so any run can
  be traced to its exact configuration.

Secret hygiene: values whose keys look like credentials (api_key, token,
secret, password) are refused at load time unless they are placeholders
(``${ENV_VAR}`` / ``<...>`` / empty). Real keys belong in environment
variables — never in these files.
"""

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

_SECRET_KEY = re.compile(r"(api_?key|token|secret|password|credential)", re.I)
_PLACEHOLDER = re.compile(r"^(\$\{[^}]+\}|<[^>]+>|)$")


class ConfigError(ValueError):
    """Raised for malformed or unsafe configuration."""


def _check_secrets(node: Any, path: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            key = str(k)
            if _SECRET_KEY.search(key) and isinstance(v, str) \
                    and not _PLACEHOLDER.match(v.strip()):
                raise ConfigError(
                    f"config key '{path}{key}' looks like a real credential; "
                    f"use environment variables instead (got "
                    f"{v.strip()[:4]}...)")
            _check_secrets(v, f"{path}{key}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_secrets(v, f"{path}[{i}].")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    return data


def load_config(path: os.PathLike) -> Dict:
    """Load one YAML file, resolving its ``extends:`` chain (deepest first)."""
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    data = load_yaml(path)
    extends = data.pop("extends", None)
    merged: Dict = {}
    if extends:
        parent = (path.parent / str(extends)).resolve()
        merged = load_config(parent)
    merged = _deep_merge(merged, data)
    _check_secrets(merged)
    return merged


def canonical_json(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def config_sha256(node: Any) -> str:
    return hashlib.sha256(canonical_json(node).encode("utf-8")).hexdigest()


def load_base() -> Dict:
    return load_config(CONFIG_DIR / "base.yaml")


def load_dataset(name: str) -> Dict:
    """Load a dataset config and anchor its relative data paths.

    Relative paths in dataset configs are resolved against the repository
    root (they are written relative to the repo so the bundled demo works
    from any working directory); absolute paths pass through unchanged.
    """
    cfg = load_config(CONFIG_DIR / "datasets" / f"{name}.yaml")
    for key, value in list(cfg.get("data", {}).items()):
        if isinstance(value, str) and value and not os.path.isabs(value):
            cfg["data"][key] = str((REPO_ROOT / value).resolve())
    return cfg


def load_protocol(name: str) -> Dict:
    return load_config(CONFIG_DIR / "protocols" / f"{name}.yaml")


def load_experiment(name: str) -> Dict:
    return load_config(CONFIG_DIR / "experiments" / f"{name}.yaml")


def terag_sections(base: Dict, dataset: Dict) -> Dict:
    """Merge base ``terag`` sections with dataset-level overrides."""
    return _deep_merge(base.get("terag", {}), dataset.get("terag", {}))


# --------------------------------------------------------------------------
# TERAGConfig construction (programmatic; no implicit config.yaml discovery)
# --------------------------------------------------------------------------

def build_terag_config(sections: Dict, data_cfg: Dict, artifacts_dir: str):
    """Construct a ``terag.config.TERAGConfig`` from resolved sections.

    ``sections`` is the merged ``terag`` mapping; ``data_cfg`` supplies the
    data paths; ``artifacts_dir`` isolates this run's intermediate artifacts.
    """
    from terag.config import (
        TERAGConfig, DataConfig, TokenizerConfig, BM25Config,
        FieldWeightsConfig, IndexConfig, GraphConfig, PatternMiningConfig,
        FeatureConfig, AnnotationsConfig, AblationConfig, EvaluationConfig,
        OutputConfig,
    )

    def sec(name):
        return sections.get(name, {}) or {}

    data = DataConfig(
        dataset=Path(data_cfg.get("name", "dataset")).stem,
        field_csv=data_cfg.get("field_csv", ""),
        table_csv=data_cfg.get("table_csv", ""),
        qa_csv=data_cfg.get("qa_csv", ""),
    )
    tok = sec("tokenizer")
    language = tok.get("default", "chinese")
    tokenizer = TokenizerConfig(default=language,
                                datasets={}, language=language)

    idx = sec("index")
    bm25 = idx.get("bm25", {})
    fw = idx.get("field_weights", {})
    index = IndexConfig(
        bm25=BM25Config(k1=bm25.get("k1", 1.5), b=bm25.get("b", 0.75)),
        field_weights=FieldWeightsConfig(
            table_name=fw.get("table_name", 2.0),
            table_desc=fw.get("table_desc", 1.5),
            field_name=fw.get("field_name", 1.0),
            field_desc=fw.get("field_desc", 0.8),
            pattern=fw.get("pattern", 0.5),
        ),
        include_field_desc=idx.get("include_field_desc", True),
    )

    graph = GraphConfig(
        role_weights=sec("graph").get("role_weights", None) or None,
        parse_mode=sec("graph").get("parse_mode", "ast"),
    )

    pm = sec("pattern_mining")
    pattern_mining = PatternMiningConfig(
        mode=pm.get("mode", "keywords"),
        entity_keywords=pm.get("entity_keywords", []),
        attribute_keywords=pm.get("attribute_keywords", []),
        aggregation_keywords=pm.get("aggregation_keywords", {}),
        time_patterns=pm.get("time_patterns", []),
        similarity_threshold=pm.get("similarity_threshold", 0.3),
        min_pattern_count=pm.get("min_pattern_count", 1),
    )

    feat = sec("feature")
    feature = FeatureConfig(
        table_weights=feat.get("table_weights", None) or None,
        field_weights=feat.get("field_weights", None) or None,
        graph_propagation_threshold=feat.get("graph_propagation_threshold", 0.15),
        train_similarity_threshold=feat.get("train_similarity_threshold", 0.30),
        graph_propagation_weight=feat.get("graph_propagation_weight", 3.0),
        template_match_weight=feat.get("template_match_weight", 0.6),
        direct_match_multiplier=feat.get("direct_match_multiplier", 1.5),
        field_name_match_bonus=feat.get("field_name_match_bonus", 0.4),
        field_graph_score_cap=feat.get("field_graph_score_cap", 5.0),
        train_recommend_multiplier=feat.get("train_recommend_multiplier", 1.5),
        train_recommend_cap=feat.get("train_recommend_cap", 3.0),
    )

    ann = sec("annotations")
    annotations = AnnotationsConfig(
        use_history=ann.get("use_history", False),
        use_transfer=ann.get("use_transfer", False),
        use_annotation_index=ann.get("use_annotation_index", True),
    )

    abl = sec("ablation")
    ablation = AblationConfig(
        use_graph_weight=abl.get("use_graph_weight", True),
        use_template_mining=abl.get("use_template_mining", True),
        use_pattern_generalization=abl.get("use_pattern_generalization", True),
        use_enhanced_index=abl.get("use_enhanced_index", True),
        use_role_parser=abl.get("use_role_parser", True),
        use_role_weight=abl.get("use_role_weight", True),
    )

    evaluation = EvaluationConfig(
        table_k=sec("evaluation").get("table_k", [1, 3, 5, 10]),
        column_k=sec("evaluation").get("column_k", [5, 10, 20]),
    )

    output = OutputConfig(artifacts_dir=artifacts_dir,
                          results_dir=artifacts_dir,
                          save_graph=False, save_index=False,
                          save_patterns=False, save_role_stats=False)

    return TERAGConfig(
        version=sections.get("version", "1.0.0"),
        seed=sections.get("seed", 42),
        data=data, index=index, graph=graph, pattern_mining=pattern_mining,
        feature=feature, ablation=ablation, annotations=annotations,
        evaluation=evaluation, output=output, tokenizer=tokenizer,
    )
