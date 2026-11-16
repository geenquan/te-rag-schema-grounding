"""Run provenance: run_meta snapshots for every executed command.

Every run records:
* code identity  — public release version + SHA256 of every executed module
* configuration — the fully resolved config plus its SHA256 (secrets-checked
  at load time; nothing credential-shaped is ever snapshotted)
* data identity  — SHA256 of every input file, qid count, gold/budget
  populations and split-manifest digest
* environment    — python + library versions actually importable at run time
* execution      — per (protocol, fold, variant) status, timing boundaries
  (fit wall-clock, per-query latency), errors, and output paths

Outputs always go to a NEW directory; existing directories are refused.
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

PUBLIC_RELEASE_VERSION = "0.1.0"

# Modules whose hashes identify the executed algorithm/runner code.
CODE_MODULES = [
    "terag/config.py", "terag/tokenization.py", "terag/graph_builder.py",
    "terag/sql_role_parser.py", "terag/pattern_miner.py",
    "terag/index_builder.py", "terag/feature_extractor.py",
    "terag/ranker.py", "terag/code_mapper.py", "terag/diagnostics.py",
    "terag/terag_retriever_v2.py",
    "retrievers/base_retriever.py", "retrievers/retriever_factory.py",
    "retrievers/annotation_miner.py", "retrievers/annotation_transfer.py",
    "retrievers/bm25_retriever.py", "retrievers/vector_retriever.py",
    "retrievers/hybrid_retriever.py", "retrievers/matched_schema_retriever.py",
    "retrievers/example_rag_retriever.py",
    "retrievers/cooccurrence_retriever.py", "retrievers/dense_retriever.py",
    "eval/retrieval_metrics.py", "eval/full_gold.py",
    "datasets/gt_utils.py",
    "terag_release/config.py", "terag_release/protocols.py",
    "terag_release/supervision.py", "terag_release/variants.py",
    "terag_release/runner.py", "terag_release/reporting.py",
]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2) + "\n").encode("utf-8")


def require_new_directory(path) -> Path:
    """Refuse to overwrite: the path must not exist."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"output directory already exists: {path} — choose a new one "
            f"(runs are never overwritten)")
    path.mkdir(parents=True)
    return path


def write_json_exclusive(path, value):
    with open(path, "xb") as f:
        f.write(json_bytes(value))


def code_state(repo_root: Path) -> Dict:
    files = {}
    for rel in CODE_MODULES:
        p = repo_root / rel
        files[rel] = sha256_file(p) if p.exists() else "missing"
    return {
        "public_release_version": PUBLIC_RELEASE_VERSION,
        "paper_tag": "paper-dasfaa2027-v1.0",
        "paper_commit": "25aa5e807d5cd6f8d5a3911e253ef89424d743e4",
        "algorithm_identity": "transfer_v3_target_cap",
        "files_sha256": files,
    }


def environment_snapshot() -> Dict:
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {name: version(name) for name in (
            "pandas", "numpy", "networkx", "jieba", "scikit-learn", "PyYAML",
            "sqlglot", "scipy", "matplotlib", "torch", "transformers",
            "openai")},
    }


def new_run_meta(command: str, resolved_config, repo_root: Path) -> Dict:
    from terag_release.config import config_sha256
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "code": code_state(repo_root),
        "config": {
            "sha256": config_sha256(resolved_config),
            "resolved": resolved_config,
        },
        "environment": environment_snapshot(),
        "execution": {"units": [], "notes": []},
    }


def finalize_run_meta(run_meta: Dict, run_dir: Path):
    write_json_exclusive(run_dir / "run_meta" / "run_meta.json", run_meta)


def input_checksums(paths) -> Dict:
    return {str(p): sha256_file(p) for p in paths if Path(p).exists()}
