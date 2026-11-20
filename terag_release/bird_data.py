"""Read-only access to the shipped BIRD experiment inputs (data/bird).

The ``data/bird`` tree carries the complete retrieval-side inputs for the
BIRD experiments: the raw official benchmark files (hash-pinned local
copies; see ``data/bird/SOURCES.md``), the frozen Dev five-fold
construction (gold, question/heldout manifests, per-db schema CSVs,
per-cell train rows), and the frozen Train development construction
(opt gold/manifest, per-db schema CSVs and train/dev rows, plus the
held-out-table inner two-way split).

Everything in this module is strictly read-only and side-effect free:

* files are opened and parsed, never written or rebuilt;
* a missing file is a hard ``FileNotFoundError`` naming the path —
  manifests are never silently regenerated or substituted;
* structural deviations from the frozen expectations (e.g. the 55/50
  applicable-cell counts) are hard errors, because the population
  identity is part of the reuse contract.

``identity_report()`` exists so runners can stamp the SHA256 of every
shipped file into ``run_meta`` for provenance.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from terag_release.config import REPO_ROOT

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

BIRD_DATA_ROOT = REPO_ROOT / "data" / "bird"

MANIFEST_FILES = {
    "question": "q5_manifest.json",
    "heldout": "h5_manifest.json",
}

# Frozen SHA256 of the raw official files (data/bird/raw/**). These are the
# authoritative identity of the local copies — the upstream distribution
# publishes no per-file release manifest for this snapshot.
RAW_SHA256 = {
    "raw/train/train.json":
        "abf17d3def22f56f41b3e139b94bba3fa4a986280abcbf6df2d7f30150695520",
    "raw/train/train_tables.json":
        "85300c745b263db88a5e0cdf083e8b2427d90272e954bc8c5f92aa1dc66c836b",
    "raw/dev/dev.json":
        "630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06",
    "raw/dev/dev_tables.json":
        "35e7f73f2bc3a3bc4def9466215f7617ff8b37e9b2837f03539a42c9e9c69d97",
}

# Frozen applicable-cell counts of the Dev five-fold manifests.
EXPECTED_CELLS = {"question": 55, "heldout": 50}


def bird_data_root() -> Path:
    """Absolute path of ``data/bird`` in this repository."""
    return BIRD_DATA_ROOT


# ---------------------------------------------------------------------------
# Hashing / verification
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    """SHA256 hex digest of a file (streamed; safe for large files)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw() -> Dict:
    """Check the four raw BIRD files against the pinned SHA256 values.

    Returns a report dict ``{"ok": bool, "files": {...}}`` where each file
    maps to ``{"expected", "actual", "match"}`` and, for missing files,
    an ``"error"`` entry instead of a digest. This never repairs or
    re-downloads anything — a mismatch means the shipped inputs are not
    the frozen ones and every downstream result is suspect.
    """
    files = {}
    ok = True
    for rel, expected in RAW_SHA256.items():
        path = BIRD_DATA_ROOT / rel
        if not path.is_file():
            files[rel] = {"expected": expected, "error": "file missing"}
            ok = False
            continue
        actual = sha256_file(path)
        match = actual == expected
        ok = ok and match
        files[rel] = {"expected": expected, "actual": actual,
                      "match": match}
    return {"ok": ok, "files": files}


# ---------------------------------------------------------------------------
# Dev (processed) inputs
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"required BIRD data file missing: {path} "
            f"(data/bird is part of this repository; do not rebuild "
            f"manifests — restore the shipped file)")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(protocol: str) -> Dict:
    """Load a frozen Dev five-fold manifest by protocol.

    ``protocol`` is ``"question"`` (q5_manifest.json, question_v5) or
    ``"heldout"`` (h5_manifest.json, heldout_table_v5). The returned
    mapping is the verbatim manifest content; callers must treat it as
    read-only.
    """
    if protocol not in MANIFEST_FILES:
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of "
            f"{sorted(MANIFEST_FILES)}")
    return _load_json(BIRD_DATA_ROOT / "processed" / MANIFEST_FILES[protocol])


def load_dev_gold() -> Dict:
    """Load the frozen Dev gold (all 1,534 questions, keyed by qid).

    Entries carry ``db_id``, ``parse_status``, ``gold_tables`` (simple
    names) and ``gold_columns`` as ``[table, column]`` pairs, plus the
    canonical ``db.table.column`` forms; see data/bird/SOURCES.md.
    """
    return _load_json(BIRD_DATA_ROOT / "processed" / "gold" / "gold.json")


def schema_dir(db: str, split: str) -> Path:
    """Path of a per-database schema directory.

    ``split`` is ``"processed"`` (the 11 Dev databases) or ``"train_dev"``
    (the 69 Train databases). Raises if the directory is not shipped.
    """
    if split not in ("processed", "train_dev"):
        raise ValueError(
            f"unknown split {split!r}; expected 'processed' or 'train_dev'")
    path = BIRD_DATA_ROOT / split / "schema" / db
    if not path.is_dir():
        raise FileNotFoundError(
            f"schema directory missing for db {db!r} (split {split!r}): "
            f"{path}")
    return path


def train_rows(protocol: str, fold: int, db: str) -> List[Dict]:
    """Training rows of one applicable Dev cell, as a list of dicts.

    Reads ``processed/train/{protocol}_f{fold}_{db}.jsonl`` (one JSON
    object per line; row keys ``qid, question, sql, db_id, table, field,
    gold_tables, gold_fields``). Files exist only for applicable cells —
    exactly the cells returned by :func:`applicable_cells`.
    """
    if protocol not in MANIFEST_FILES:
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of "
            f"{sorted(MANIFEST_FILES)}")
    path = (BIRD_DATA_ROOT / "processed" / "train" /
            f"{protocol}_f{fold}_{db}.jsonl")
    if not path.is_file():
        raise FileNotFoundError(
            f"train rows missing for ({protocol}, fold {fold}, {db}): "
            f"{path} — cells without a file are not applicable in the "
            f"frozen manifest")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def applicable_cells(protocol: str) -> List[Dict]:
    """Applicable (db, fold) cells of a frozen Dev five-fold manifest.

    Mirrors the frozen Dev population semantics exactly: databases in
    sorted order, folds ascending 0..4, and only cells whose manifest
    ``per_fold`` entry has ``applicable: true``. Each entry is
    ``{"db", "fold", "test_qids", "train_qids", "heldout_tables"}``
    (``heldout_tables`` is non-empty only for the heldout protocol).

    The cell count must equal the frozen expectation (question: 55,
    heldout: 50); any deviation is a hard error because it means the
    shipped manifest is not the frozen one.
    """
    manifest = load_manifest(protocol)
    cells = []
    per_db = manifest["per_db"]
    for db in sorted(per_db):
        for fold in range(manifest["folds"]):
            spec = per_db[db]["per_fold"].get(str(fold))
            if not spec or not spec.get("applicable", False):
                continue
            cells.append({
                "db": db,
                "fold": fold,
                "test_qids": list(spec["test_qids"]),
                "train_qids": list(spec["train_qids"]),
                "heldout_tables": list(spec.get("heldout_tables") or []),
            })
    expected = EXPECTED_CELLS[protocol]
    if len(cells) != expected:
        raise RuntimeError(
            f"{protocol}: {len(cells)} applicable cells in the shipped "
            f"manifest, expected {expected} — population identity broken; "
            f"restore the frozen manifest instead of re-splitting")
    return cells


# ---------------------------------------------------------------------------
# Train development inputs
# ---------------------------------------------------------------------------

def train_dev_rows(db: str) -> Dict[str, List[Dict]]:
    """Opt-train/opt-dev development rows of one Train database.

    Returns ``{"opt_train": [...], "opt_dev": [...]}`` parsed from
    ``train_dev/train_rows/<db>/{opt_train,opt_dev}.jsonl`` (same row
    shape as the Dev train rows). All 69 Train databases are shipped.
    """
    base = BIRD_DATA_ROOT / "train_dev" / "train_rows" / db
    out = {}
    for part in ("opt_train", "opt_dev"):
        path = base / f"{part}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"train_dev rows missing for db {db!r}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            out[part] = [json.loads(line) for line in f if line.strip()]
    return out


def h_manifest() -> Dict:
    """Load the frozen Train-side held-out-table inner two-way manifest.

    ``train_dev/h_splits/h_manifest.json`` (heldout_table_inner_2way):
    69 databases of which 55 pass the inner gate, declaring the 4,387
    dev-population and 2,154 inner-train questions. Verbatim content,
    read-only.
    """
    return _load_json(BIRD_DATA_ROOT / "train_dev" / "h_splits"
                      / "h_manifest.json")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def identity_report() -> Dict[str, str]:
    """SHA256 of every file shipped under ``data/bird``.

    Keys are repository-relative POSIX paths (``data/bird/...``), values
    are hex digests — suitable for stamping verbatim into ``run_meta``
    so a run's data inputs are reproducibly identifiable. Sorting makes
    the report stable across platforms. macOS ``.DS_Store`` metadata is
    never part of the shipped inputs and is skipped.
    """
    root = BIRD_DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"data/bird not found: {root} — the BIRD inputs are part of "
            f"this repository and must not be missing")
    report = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            report[path.relative_to(REPO_ROOT).as_posix()] = \
                sha256_file(path)
    return report
