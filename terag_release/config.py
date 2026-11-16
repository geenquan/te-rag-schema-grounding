"""Configuration loading, resolution, and enforcement for the public CLI.

Layered YAML configuration:

* ``configs/base.yaml``           — fixed algorithm parameters (paper values)
* ``configs/datasets/*.yaml``     — data paths, tokenizer language, domain
                                     keyword lists
* ``configs/protocols/*.yaml``    — protocol definitions
* ``configs/methods/*.yaml``      — per-method parameters (consumed by the
                                     runner, not documentation-only)
* ``configs/experiments/*.yaml``  — which dataset/protocols/methods to run

Resolution rules (later overrides earlier):

    paper-profile constants (code)
      <- base.yaml
      <- dataset yaml
      <- protocol yaml (per protocol)
      <- method yaml (per method)
      <- experiment yaml
      <- CLI overrides

Under the default ``paper`` profile, every frozen paper parameter is
validated against ``terag_release.paper_profile`` after merging: a changed
value is a hard startup error (never a silent no-op). ``profile: custom``
(experiment yaml, dataset yaml, or ``--profile custom``) lets deviations
through — they then genuinely reach the retrievers and are stamped into
run_meta.

Every reference accepts either a NAME (resolved under ``configs/<kind>/``,
with or without the ``.yaml`` suffix) or an explicit FILE PATH; paths are
never re-suffixed or re-joined onto the config directory.

Secret hygiene: values whose keys look like credentials (api_key, token,
secret, password, credential) are refused at load time unless they are
placeholders (``${ENV_VAR}`` / ``<...>`` / empty). Real keys belong in
environment variables — never in these files.
"""

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from terag_release import paper_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

_SECRET_KEY = re.compile(r"(api_?key|token|secret|password|credential)", re.I)
_PLACEHOLDER = re.compile(r"^(\$\{[^}]+\}|<[^>]+>|)$")


class ConfigError(ValueError):
    """Raised for malformed, unsafe, or conflicting configuration."""


# --------------------------------------------------------------------------
# Primitive loading
# --------------------------------------------------------------------------

def _check_secrets(node: Any, path: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            key = str(k)
            if _SECRET_KEY.search(key) and isinstance(v, str) \
                    and not _PLACEHOLDER.match(v.strip()):
                raise ConfigError(
                    f"config key '{path}{key}' looks like a real credential; "
                    f"use environment variables instead")
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


def load_config(path) -> Dict:
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


def load_named(reference: str, kind: str) -> Tuple[Dict, Path]:
    """Load a config by NAME or by explicit FILE PATH.

    ``kind`` is the directory under configs/ (``datasets``, ``protocols``,
    ``methods``, ``experiments``). A reference containing a path separator
    or ending in .yaml/.yml is treated as a path and used verbatim;
    otherwise it is looked up as ``configs/<kind>/<reference>.yaml``.
    Returns (config, resolved_path).
    """
    if not isinstance(reference, str) or not reference.strip():
        raise ConfigError(f"empty {kind} configuration reference")
    ref = reference.strip()
    looks_like_path = ("/" in ref or ref.startswith(".")
                       or Path(ref).is_absolute())
    if looks_like_path:
        path = Path(ref)
        if not path.exists():
            raise ConfigError(f"{kind} config path not found: {ref}")
    else:
        stem = ref[:-5] if ref.endswith(".yaml") else \
            ref[:-4] if ref.endswith(".yml") else ref
        path = CONFIG_DIR / kind / f"{stem}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in (CONFIG_DIR / kind).glob("*.yaml"))
            raise ConfigError(f"unknown {kind} config '{reference}'; "
                              f"available: {available}")
    return load_config(path), path.resolve()


# --------------------------------------------------------------------------
# Strict key schemas
# --------------------------------------------------------------------------

_METHOD_KEYS = {"method", "family", "variant", "retriever", "terag", "model",
                "api", "protocols", "requires_network", "note", "notes"}
_RETRIEVER_KEYS = {"k1", "b", "top_k", "use_chinese_tokenizer"}
_EXPERIMENT_KEYS = {"dataset", "protocols", "methods", "profile", "note",
                    "notes"}
_DATASET_KEYS = {"name", "data", "terag", "sql", "qid_prefix", "notes",
                 "note", "extends", "profile"}
_PROTOCOL_KEYS = {"name", "folds", "seed", "supervision", "budget",
                  "isolation", "note", "notes", "extends", "profile"}

_TERAG_SECTION_PARAMS = {
    "version": set(), "seed": set(),
    "index": {"bm25", "field_weights", "include_field_desc"},
    "graph": {"role_weights", "parse_mode"},
    "pattern_mining": {"mode", "entity_keywords", "attribute_keywords",
                       "aggregation_keywords", "english_aggregation_keywords",
                       "time_patterns", "english_time_patterns",
                       "similarity_threshold", "min_pattern_count"},
    "feature": {"table_weights", "field_weights",
                "graph_propagation_threshold", "train_similarity_threshold",
                "graph_propagation_weight", "template_match_weight",
                "direct_match_multiplier", "field_name_match_bonus",
                "field_graph_score_cap", "train_recommend_multiplier",
                "train_recommend_cap"},
    "annotations": {"use_history", "use_transfer", "use_annotation_index"},
    "ablation": {"use_graph_weight", "use_template_mining",
                 "use_pattern_generalization", "use_enhanced_index",
                 "use_role_parser", "use_role_weight"},
    "evaluation": {"table_k", "column_k"},
    "tokenizer": {"default", "datasets", "language"},
}


def _validate_keys(cfg: Dict, allowed: set, kind: str, path: Path):
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ConfigError(f"{path} ({kind}): unknown keys {unknown}; "
                          f"allowed: {sorted(allowed)}")


def _validate_terag(terag: Dict, where: str):
    for section, params in terag.items():
        if section not in _TERAG_SECTION_PARAMS:
            raise ConfigError(
                f"{where}: unknown terag section '{section}'; recognized: "
                f"{sorted(_TERAG_SECTION_PARAMS)}")
        if section in ("version", "seed"):
            continue  # scalar values
        if params is None:
            continue
        if not isinstance(params, dict):
            raise ConfigError(f"{where}: terag.{section} must be a mapping")
        unknown = set(params) - _TERAG_SECTION_PARAMS[section]
        if unknown:
            raise ConfigError(
                f"{where}: unknown parameter(s) under terag.{section}: "
                f"{sorted(unknown)}; recognized: "
                f"{sorted(_TERAG_SECTION_PARAMS[section])}")


# --------------------------------------------------------------------------
# Run-level resolution
# --------------------------------------------------------------------------

def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_base() -> Dict:
    """Load configs/base.yaml (the shared fixed-parameter mirror)."""
    return load_config(CONFIG_DIR / "base.yaml")


def load_dataset(reference: str) -> Tuple[Dict, Path]:
    """Load a dataset config by name or path and anchor relative data paths
    to the repository root."""
    cfg, path = load_named(reference, "datasets")
    _validate_keys(cfg, _DATASET_KEYS, "dataset", path)
    _validate_terag(cfg.get("terag") or {}, str(path))
    for key, value in list(cfg.get("data", {}).items()):
        if isinstance(value, str) and value and not os.path.isabs(value):
            cfg["data"][key] = str((REPO_ROOT / value).resolve())
    return cfg, path


def load_protocol(reference: str) -> Tuple[Dict, Path]:
    # Protocol settings are referenced by their execution names; map them to
    # the descriptive config file names under configs/protocols/.
    setting_to_file = {"question_v1": "question_level",
                       "table_v2": "heldout_table"}
    reference = setting_to_file.get(reference, reference)
    cfg, path = load_named(reference, "protocols")
    _validate_keys(cfg, _PROTOCOL_KEYS, "protocol", path)
    return cfg, path


def load_experiment(reference: str) -> Tuple[Dict, Path]:
    cfg, path = load_named(reference, "experiments")
    _validate_keys(cfg, _EXPERIMENT_KEYS, "experiment", path)
    return cfg, path


def _validate_model_mirror(variant: str, cfg: Dict, path: Path):
    """Model/API sections are READ-ONLY MIRRORS of code constants.

    Unknown keys (typos) and any value deviation are hard errors under
    EVERY profile: the constructors read the constants from the modules,
    so a changed YAML value would be silently ignored. Supported custom
    surface: retriever parameters and terag sections only (see
    paper_profile.CUSTOM_SUPPORTED).
    """
    expected = paper_profile.MODEL_MIRROR.get(variant, {})
    model = cfg.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise ConfigError(f"{path}: 'model' must be a mapping")
        unknown = sorted(set(model) - set(expected))
        if unknown:
            raise ConfigError(
                f"{path}: unknown model key(s) {unknown} for {variant}; "
                f"recognized: {sorted(expected)} (model parameters are "
                f"fixed in code — typos and unsupported overrides are "
                f"rejected)")
        for key, value in expected.items():
            if key in model and model[key] != value:
                raise ConfigError(
                    f"{path}: model.{key}={model[key]!r} contradicts the "
                    f"fixed constant {value!r}; model/API parameters are "
                    f"read-only in this release and cannot be customized "
                    f"even under profile=custom")
        _validate_nested_model_keys(variant, model, path)
    api = cfg.get("api")
    if api is not None:
        if not isinstance(api, dict):
            raise ConfigError(f"{path}: 'api' must be a mapping")
        unknown = sorted(set(api) - set(paper_profile.API_MIRROR))
        if unknown:
            raise ConfigError(
                f"{path}: unknown api key(s) {unknown}; recognized: "
                f"{sorted(paper_profile.API_MIRROR)}")
        for key, value in paper_profile.API_MIRROR.items():
            if key in api and api[key] != value:
                raise ConfigError(
                    f"{path}: api.{key}={api[key]!r} contradicts the fixed "
                    f"boundary {value!r}; credentials come from the "
                    f"environment only and API endpoints are fixed")


def _validate_nested_model_keys(variant: str, model: Dict, path: Path):
    """Deep key validation for nested model sections (classifier, ...)."""
    expected = paper_profile.MODEL_MIRROR.get(variant, {})
    for key, value in model.items():
        expected_value = expected.get(key)
        if isinstance(expected_value, dict) and isinstance(value, dict):
            unknown = sorted(set(value) - set(expected_value))
            if unknown:
                raise ConfigError(
                    f"{path}: unknown model.{key} sub-key(s) {unknown}; "
                    f"recognized: {sorted(expected_value)}")


def load_method(variant: str) -> Tuple[Dict, Path]:
    """Load the method config bound to a paper variant name."""
    from terag_release import variants
    spec = variants.VARIANTS[variant]
    cfg, path = load_named(spec["config"], "methods")
    _validate_keys(cfg, _METHOD_KEYS, "method", path)
    if "retriever" in cfg:
        _validate_keys(cfg["retriever"], _RETRIEVER_KEYS,
                       "method retriever", path)
    if "terag" in cfg:
        _validate_terag(cfg["terag"] or {}, str(path))
    declared = cfg.get("method")
    if declared not in (spec["factory"], variant):
        raise ConfigError(
            f"{path}: 'method: {declared}' does not match variant "
            f"{variant} (factory {spec['factory']})")
    if "variant" in cfg and cfg["variant"] != variant:
        raise ConfigError(f"{path}: 'variant: {cfg['variant']}' != {variant}")
    if "protocols" in cfg and sorted(cfg["protocols"]) != \
            sorted(spec["protocols"]):
        raise ConfigError(
            f"{path}: 'protocols: {cfg['protocols']}' contradicts the "
            f"variant matrix {spec['protocols']} (code is the source of "
            f"truth; fix the YAML)")
    _validate_model_mirror(variant, cfg, path)
    if cfg.get("terag") and variant != "MetadataMatched":
        expected = variants.variant_config(
            paper_profile.PAPER_TERAG, variant)
        declared_terag = cfg["terag"]
        for section, values in declared_terag.items():
            if section in ("graph",) and values.get("parse_mode") == "ast":
                continue  # documented mirror of the fixed parse mode
            for key, value in values.items():
                if key == "field_weights" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if expected.get(section, {}).get(
                                "field_weights", {}).get(sub_key) != sub_value:
                            raise ConfigError(
                                f"{path}: terag.{section}.field_weights."
                                f"{sub_key}={sub_value} contradicts the "
                                f"variant definition (expected "
                                f"{expected[section]['field_weights'][sub_key]})")
                elif expected.get(section, {}).get(key) != value:
                    raise ConfigError(
                        f"{path}: terag.{section}.{key}={value} contradicts "
                        f"the variant definition (expected "
                        f"{expected.get(section, {}).get(key)}); the code "
                        f"matrix is the source of truth — fix the YAML")
    return cfg, path


def resolve_run(dataset_ref: str, experiment_ref: str,
                protocol_refs: Optional[List[str]] = None,
                method_names: Optional[List[str]] = None,
                profile: Optional[str] = None) -> Dict:
    """Resolve the full effective configuration for one execution.

    Returns a mapping consumed by the runner and stamped verbatim into
    run_meta (whose canonical SHA256 is the run's configuration identity).
    Any change to a value that execution actually consumes changes that
    hash.
    """
    from terag_release import variants

    experiment, exp_path = load_experiment(experiment_ref)
    dataset, ds_path = load_dataset(dataset_ref)

    profile = (profile or experiment.get("profile")
               or dataset.get("profile") or paper_profile.PAPER_PROFILE)
    if profile not in (paper_profile.PAPER_PROFILE,
                       paper_profile.CUSTOM_PROFILE):
        raise ConfigError(f"unknown profile '{profile}' (use "
                          f"'{paper_profile.PAPER_PROFILE}' or "
                          f"'{paper_profile.CUSTOM_PROFILE}')")

    protocol_refs = (protocol_refs if protocol_refs is not None
                     else experiment.get("protocols",
                                         ["question_v1", "table_v2"]))
    protocols_cfg = {}
    protocol_paths = []
    canonical_protocols = []
    for ref in protocol_refs:
        cfg, path = load_protocol(ref)
        # A file path only LOCATES the config; the canonical `name` inside
        # it is the execution protocol identity (so absolute paths never
        # leak into unit planning).
        name = cfg.get("name")
        if name not in paper_profile.SUPPORTED_PROTOCOL_NAMES:
            raise ConfigError(
                f"{path}: protocol 'name: {name!r}' is not a supported "
                f"execution protocol "
                f"({list(paper_profile.SUPPORTED_PROTOCOL_NAMES)}); the "
                f"name field, not the file reference, drives execution")
        if name in protocols_cfg:
            raise ConfigError(f"protocol {name} supplied twice")
        protocols_cfg[name] = cfg
        protocol_paths.append(path)
        canonical_protocols.append(name)

    method_names = (method_names if method_names is not None
                    else experiment.get("methods", variants.FULL_MATRIX))
    unknown = [m for m in method_names if m not in variants.VARIANTS]
    if unknown:
        raise ConfigError(f"unknown methods: {unknown}; available: "
                          f"{sorted(variants.VARIANTS)}")

    methods_cfg = {}
    for variant in method_names:
        spec = variants.VARIANTS[variant]
        yaml_cfg, path = load_method(variant)
        effective: Dict = {"config_file": str(path)}
        if spec["terag"] is not None or spec["factory"] == "TE-RAG-V2":
            base_terag = _deep_merge(
                _deep_merge(paper_profile.PAPER_TERAG,
                            load_base().get("terag", {})),
                dataset.get("terag", {}))
            sections = variants.variant_config(base_terag, variant)
            _validate_terag(sections, f"effective/{variant}")
            effective["terag_sections"] = sections
        else:
            retriever_params = {"k1": 1.5, "b": 0.75, "top_k": 5,
                                "use_chinese_tokenizer": True}
            retriever_params.update(yaml_cfg.get("retriever") or {})
            language = (dataset.get("terag", {}).get("tokenizer", {})
                        .get("default"))
            if language == "english":
                retriever_params["use_chinese_tokenizer"] = False
            effective["retriever_params"] = retriever_params
            effective["extra_params"] = (
                {"fusion_mode": spec["fusion_mode"]}
                if spec.get("fusion_mode") else {})
        methods_cfg[variant] = {
            "yaml": yaml_cfg,
            "factory": spec["factory"],
            "supervision": spec["supervision"],
            "protocols": spec["protocols"],
            "effective": effective,
        }

    base_cfg = load_base()
    budget = _deep_merge(paper_profile.PAPER_BUDGET,
                         base_cfg.get("budget", {}))
    resolved = {
        "profile": profile,
        "qid_prefix": dataset.get("qid_prefix", "q"),
        "dataset": dataset,
        "experiment": {"name": str(exp_path),
                       "protocol_refs": list(protocol_refs),
                       "protocols": canonical_protocols,
                       "methods": method_names},
        "protocols": protocols_cfg,
        "methods": methods_cfg,
        "budget": budget,
        "paper_terag_reference": paper_profile.PAPER_TERAG,
        "config_sources": {
            "base": {"path": str(CONFIG_DIR / "base.yaml"),
                     "sha256": _file_digest(CONFIG_DIR / "base.yaml")},
            "dataset": {"path": str(ds_path), "sha256": _file_digest(ds_path)},
            "experiment": {"path": str(exp_path),
                           "sha256": _file_digest(exp_path)},
            "protocols": [{"path": str(p), "sha256": _file_digest(p)}
                          for p in protocol_paths],
            "methods": {v: {"path": entry["effective"]["config_file"],
                            "sha256": _file_digest(
                                Path(entry["effective"]["config_file"]))}
                        for v, entry in methods_cfg.items()},
        },
    }
    # Read-only mirrors and structural constants are validated under EVERY
    # profile: execution would silently ignore a contradicting YAML value.
    enforce_read_only_mirrors(resolved, base_cfg)
    if profile == paper_profile.PAPER_PROFILE:
        enforce_paper_profile(resolved)
    return resolved


def enforce_read_only_mirrors(resolved: Dict, base_cfg: Dict):
    """Reject configuration fields that execution would silently ignore.

    Applies to BOTH profiles: transfer defaults (mirror of the actual
    TransferConfig), protocol structure (fold count, supervision arm), and
    the retrieval budget are fixed in code for this release. Model/API
    mirrors were already validated per method in load_method.
    """
    problems: List[str] = []
    declared_transfer = base_cfg.get("transfer_defaults") or {}
    unknown = sorted(set(declared_transfer) -
                     set(paper_profile.TRANSFER_MIRROR_KEYS))
    if unknown:
        problems.append(f"transfer_defaults: unknown key(s) {unknown}; "
                        f"recognized: "
                        f"{sorted(paper_profile.TRANSFER_MIRROR_KEYS)}")
    actual_transfer = paper_profile.actual_transfer_defaults()
    for key in paper_profile.TRANSFER_MIRROR_KEYS:
        got = declared_transfer.get(key, actual_transfer[key])
        if got != actual_transfer[key]:
            problems.append(
                f"transfer_defaults.{key}={got!r} contradicts the actual "
                f"TransferConfig default {actual_transfer[key]!r}; "
                f"transfer gates are read-only in this release (not "
                f"customizable even under profile=custom)")
    for name, cfg in resolved["protocols"].items():
        if cfg.get("folds") != paper_profile.STRUCTURAL["folds"]:
            problems.append(
                f"protocol {name}: folds={cfg.get('folds')} — the runner "
                f"executes exactly "
                f"{paper_profile.STRUCTURAL['folds']} folds; fold count is "
                f"structural and cannot be changed in this release")
        if cfg.get("supervision") != paper_profile.STRUCTURAL["supervision"]:
            problems.append(
                f"protocol {name}: supervision={cfg.get('supervision')!r} "
                f"— only 'full_sql' is implemented; deviations are not "
                f"supported")
        budget = cfg.get("budget") or {}
        for key, value in paper_profile.STRUCTURAL["budget"].items():
            if key in budget and budget[key] != value:
                problems.append(
                    f"protocol {name}: budget.{key}={budget[key]} — the "
                    f"retrieval budget is fixed in code ({value}); "
                    f"feasible-population budgets cannot be changed either")
    for key, value in paper_profile.STRUCTURAL["budget"].items():
        if resolved["budget"].get(key) != value:
            problems.append(
                f"budget.{key}={resolved['budget'][key]} — the retrieval "
                f"budget is fixed in code ({value}) and not parameterized")
    if problems:
        raise ConfigError(
            "read-only mirror violations (execution would silently ignore "
            "these values, so they are rejected under every profile; "
            "supported custom surface: "
            + "; ".join(paper_profile.CUSTOM_SUPPORTED)
            + "):\n  - " + "\n  - ".join(problems))


def enforce_paper_profile(resolved: Dict):
    """Refuse deviations from frozen paper values under the paper profile.

    Each violation names the parameter, the paper value, and the configured
    value, and tells the user how to run a custom-profile experiment.
    """
    problems: List[str] = []
    dataset_terag = _deep_merge(
        _deep_merge(paper_profile.PAPER_TERAG,
                    load_base().get("terag", {})),
        resolved["dataset"].get("terag", {}))
    for path, expected, actual in paper_profile.diff_against_paper(
            dataset_terag):
        problems.append(f"terag.{path}: paper={expected!r}, "
                        f"configured={actual!r}")
    for variant, entry in resolved["methods"].items():
        yaml_cfg = entry["yaml"]
        fixed = paper_profile.PAPER_BASELINE_PARAMS.get(variant)
        if fixed is not None:
            for key, value in fixed.items():
                got = (yaml_cfg.get("retriever") or {}).get(key, value)
                if got != value:
                    problems.append(
                        f"method {variant}: retriever.{key}={got} "
                        f"(paper: {value})")
        # Variant toggles are code-defined; a YAML terag section that
        # contradicts them already failed in load_method.
        if variant == "MetadataMatched":
            annotations = (entry.get("effective", {})
                           .get("terag_sections", {}).get("annotations", {}))
            if annotations.get("use_history") or \
                    annotations.get("use_transfer"):
                problems.append(
                    "method MetadataMatched: history channels must stay off")
    if problems:
        raise ConfigError(
            "paper-profile violations (these parameters are frozen for "
            "paper reproduction; run with profile=custom to deviate "
            "deliberately):\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------

def canonical_json(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def config_sha256(node: Any) -> str:
    return hashlib.sha256(canonical_json(node).encode("utf-8")).hexdigest()


def config_sources_digests(paths: List[Path]) -> Dict[str, str]:
    return {str(p): _file_digest(p) for p in paths}


# --------------------------------------------------------------------------
# TERAGConfig construction (programmatic; no implicit config.yaml discovery)
# --------------------------------------------------------------------------

def build_terag_config(sections: Dict, data_cfg: Dict, artifacts_dir: str):
    """Construct a ``terag.config.TERAGConfig`` from resolved sections."""
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
    tokenizer = TokenizerConfig(default=language, datasets={}, language=language)

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
