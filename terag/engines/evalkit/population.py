"""Frozen Dev five-fold population (READ-ONLY ground truth access).

The Dev run (research bird_full_v1/run_20261119_0115, released under
data/bird/processed) defines, for each protocol:
- the applicable (db, fold) cells (question: 55, heldout: 50),
- each cell's exact test_qids and train_qids,
- the heldout-protocol per-fold heldout_tables,
- gold (db-local simple-name view),
- per-db schema table sets (table_schema.csv).

Everything here is deterministic and side-effect free. Any structural
deviation from the frozen expectation is a hard error (population identity
is part of the reuse contract).

PUBLIC-PORT: the research DevPopulation resolved the run root from a
research-absolute path (runpaths.dev_run, env override for fixtures). The
public BirdPopulation takes the processed-data root as a constructor
argument, defaulting to data/bird/processed in the repository; the layout
under that root is gold/gold.json, q5_manifest.json / h5_manifest.json,
schema/<db>/table_schema.csv, train/<protocol>_f<fold>_<db>.jsonl.
"""
import hashlib
import json
from pathlib import Path

MANIFEST_NAME = {'question': 'q5_manifest.json', 'heldout': 'h5_manifest.json'}


def _default_root():
    from terag_release.config import REPO_ROOT
    return REPO_ROOT / 'data' / 'bird' / 'processed'


def sha256_file(path) -> str:
    """Verbatim runpaths helper."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class BirdPopulation:
    FROZEN_EXPECTED_CELLS = {'question': 55, 'heldout': 50}

    def __init__(self, root=None, expected_cells=None):
        self.root = Path(root) if root else _default_root()
        self._manifests = {}
        self._cells = {}
        self._schema_tables = {}
        # fixture tests may declare smaller expectations; production always
        # uses the frozen counts (no override in real runs)
        self.expected_cells = expected_cells or self.FROZEN_EXPECTED_CELLS

    # ---------------------------------------------------------- manifests
    def manifest(self, protocol):
        if protocol not in self._manifests:
            path = self.root / MANIFEST_NAME[protocol]
            if not path.is_file():
                raise FileNotFoundError(f'manifest missing: {path}')
            self._manifests[protocol] = json.loads(path.read_text(encoding='utf-8'))
        return self._manifests[protocol]

    def cells(self, protocol):
        """Applicable (db, fold) cells in canonical order: db sorted, fold
        ascending. Each entry: {'db', 'fold', 'test_qids', 'train_qids',
        'heldout_tables'}."""
        if protocol in self._cells:
            return self._cells[protocol]
        out = []
        per_db = self.manifest(protocol)['per_db']
        for db in sorted(per_db):
            for fold in range(5):
                spec = per_db[db]['per_fold'].get(str(fold))
                if not spec or not spec.get('applicable', False):
                    continue
                out.append({'db': db, 'fold': fold,
                            'test_qids': list(spec['test_qids']),
                            'train_qids': list(spec['train_qids']),
                            'heldout_tables': list(spec.get('heldout_tables') or [])})
        expected = self.expected_cells
        if len(out) != expected[protocol]:
            raise RuntimeError(
                f'{protocol}: {len(out)} applicable cells, expected '
                f'{expected[protocol]} - population identity broken')
        self._cells[protocol] = out
        return out

    def test_qids(self, protocol):
        """All OOF test qids across cells (should equal the manifest's
        oof_population; verified)."""
        qids = []
        for cell in self.cells(protocol):
            qids.extend(cell['test_qids'])
        oof = set(self.manifest(protocol)['oof_population'][protocol])
        if set(qids) != oof or len(qids) != len(set(qids)):
            raise RuntimeError(f'{protocol}: cell test qids != oof_population')
        return qids

    # ---------------------------------------------------------- gold
    def gold(self):
        path = self.root / 'gold' / 'gold.json'
        raw = json.loads(path.read_text(encoding='utf-8'))
        gold = {}
        for qid, g in raw.items():
            gold[qid] = {
                'db_id': g['db_id'],
                'gold_tables': list(g['gold_tables']),
                'gold_columns': [f'{t}.{c}' for t, c in g['gold_columns']],
                'question': g['question'],
            }
        return gold

    def gold_entry(self, raw_gold, qid):
        """Db-local simple-name view for metric_values (reference shape)."""
        g = raw_gold[qid]
        return {'gold_tables': g['gold_tables'],
                'gold_columns': g['gold_columns']}

    # ---------------------------------------------------------- schema
    def schema_tables(self, db):
        """RAW table names of the db's table_schema.csv (first CSV column)."""
        if db not in self._schema_tables:
            path = self.root / 'schema' / db / 'table_schema.csv'
            if not path.is_file():
                raise FileNotFoundError(f'schema missing: {path}')
            tables = set()
            for i, row in enumerate(path.read_text(encoding='utf-8').splitlines()):
                if i == 0 or not row.strip():
                    continue
                tables.add(row.split(',')[0])
            self._schema_tables[db] = tables
        return self._schema_tables[db]

    # ---------------------------------------------------------- train rows
    def train_rows(self, protocol, fold, db):
        path = self.root / 'train' / f'{protocol}_f{fold}_{db}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'train rows missing: {path}')
        return [json.loads(line) for line in
                path.read_text(encoding='utf-8').splitlines() if line.strip()]

    # ---------------------------------------------------------- identity
    def identity(self):
        """Digest of every population-defining input (manifests, gold,
        schema, train rows) - used by the final-run pre-guard."""
        parts = {}
        for protocol, name in MANIFEST_NAME.items():
            parts[name] = sha256_file(self.root / name)
        parts['gold/gold.json'] = sha256_file(self.root / 'gold' / 'gold.json')
        entries = []
        for db in sorted(self.manifest('question')['per_db']):
            for f in sorted((self.root / 'schema' / db).glob('*.csv')):
                entries.append((str(f.relative_to(self.root)), sha256_file(f)))
        for protocol in MANIFEST_NAME:
            for f in sorted((self.root / 'train').glob(f'{protocol}_f*.jsonl')):
                entries.append((str(f.relative_to(self.root)), sha256_file(f)))
        joined = ''.join(f'{sha}  {rel}\n' for rel, sha in sorted(entries))
        parts['schema_and_train_digest'] = hashlib.sha256(
            joined.encode('utf-8')).hexdigest()
        return parts
