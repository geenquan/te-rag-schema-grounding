"""Prediction record IO: loading, hashing and generic JSONL writing.

PUBLIC-PORT: the research module centered on hash-verified REUSE of
external baseline predictions (REUSE_SOURCES.json machinery, multi-root
resolution, refusal exits). The public release ships reviewed reference
predictions instead, so this port keeps only the generic pieces:
- sha256_file / content_digest (the accepted group-digest recipe);
- load_cell_records / cell_prediction_path for a results-like tree
  (<root>/<protocol>/fold<k>/<db>/<method>/predictions.jsonl);
- write helpers for prediction/evidence JSONL files.

Population/data/budget identity is enforced by evaluate.py's gates (exact
test-set equality vs frozen manifests, budgets, db scope).
"""
import hashlib
import json
from pathlib import Path


def sha256_file(path) -> str:
    """SHA-256 of file bytes."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def content_digest(entries):
    """Group content digest, recipe copied from the accepted reused_manifest:
    sha256 of ''.join(sha256_hex + '  ' + relpath + '\\n') over sorted relpaths.
    entries: iterable of (relpath_str, sha256_hex)."""
    joined = ''.join(f"{sha}  {rel}\n" for rel, sha in sorted(entries))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()


def cell_prediction_path(source_run, protocol, fold, db, method):
    """Path of one method's predictions.jsonl in a results-like tree.

    Layout: <source_run>/<protocol>/fold<fold>/<db>/<method>/predictions.jsonl
    """
    return Path(source_run) / protocol / f'fold{fold}' / db / method / 'predictions.jsonl'


def load_cell_records(source_run, protocol, fold, db, method):
    """Load one (protocol, fold, db, method) cell's prediction records.

    Returns (records, path); raises FileNotFoundError when the cell file is
    missing (population-exact evaluation treats that as a gate violation).
    """
    path = cell_prediction_path(source_run, protocol, fold, db, method)
    if not path.is_file():
        raise FileNotFoundError(f'missing predictions file: {path}')
    records = [json.loads(line) for line in
               path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return records, path


def iter_jsonl(path):
    """Iterate records of a JSONL file (blank lines skipped)."""
    path = Path(path)
    return (json.loads(line) for line in
            path.read_text(encoding='utf-8').splitlines() if line.strip())


def write_jsonl(path, records):
    """Generic write helper: records (dicts) -> JSONL (one per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    return path


def write_json(path, value, *, allow_nan=False):
    """Generic write helper: pretty JSON (allow_nan off by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1,
                               allow_nan=allow_nan) + '\n', encoding='utf-8')
    return path
