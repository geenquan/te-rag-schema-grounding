"""Regression tests for the shipped BIRD inputs (data/bird).

These checks pin the identity of the public experiment data exactly as
documented in data/bird/SOURCES.md: the hash-pinned raw files, the frozen
Dev five-fold population (55/50 applicable cells, 1,530-question OOF
population with the 4 unverified questions excluded), the Train
development manifest (69 databases, 9,068 verified questions), the
Train-side held-out-table split, and the exclusion of regenerable
engine caches. Everything is read-only — no retriever is imported and
nothing is fitted.
"""

import json
import unittest
from pathlib import Path

from terag_release import bird_data

REPO_ROOT = Path(__file__).resolve().parent.parent
BIRD_ROOT = REPO_ROOT / "data" / "bird"

PROTOCOLS = ("question", "heldout")


class RawFilesTest(unittest.TestCase):
    """The raw BIRD copies must be the hash-pinned local snapshots."""

    def test_verify_raw_passes(self):
        report = bird_data.verify_raw()
        self.assertTrue(
            report["ok"],
            msg=f"raw file verification failed: {report['files']}")
        self.assertEqual(len(report["files"]), 4)

    def test_identity_report_covers_raw_files(self):
        report = bird_data.identity_report()
        for rel in bird_data.RAW_SHA256:
            self.assertIn(f"data/bird/{rel}", report)
            self.assertEqual(report[f"data/bird/{rel}"],
                             bird_data.RAW_SHA256[rel])


class DevManifestTest(unittest.TestCase):
    """The frozen Dev five-fold manifests define the population."""

    def test_applicable_cell_counts_are_frozen(self):
        for protocol, expected in (("question", 55), ("heldout", 50)):
            cells = bird_data.applicable_cells(protocol)
            self.assertEqual(len(cells), expected,
                             msg=f"{protocol}: expected {expected} cells")

    def test_cells_are_in_canonical_order(self):
        for protocol in PROTOCOLS:
            cells = bird_data.applicable_cells(protocol)
            keys = [(c["db"], c["fold"]) for c in cells]
            self.assertEqual(keys, sorted(keys),
                             msg=f"{protocol}: cells must be db-sorted, "
                                 f"fold-ascending")

    def test_q5_oof_population_equals_cell_test_qids(self):
        cells = bird_data.applicable_cells("question")
        union = [qid for cell in cells for qid in cell["test_qids"]]
        manifest = bird_data.load_manifest("question")
        oof = manifest["oof_population"]["question"]
        # No duplicates and exact set identity with the manifest record.
        self.assertEqual(len(union), len(set(union)))
        self.assertEqual(set(union), set(oof))
        self.assertEqual(len(oof), 1530)

    def test_heldout_cells_carry_heldout_tables(self):
        cells = bird_data.applicable_cells("heldout")
        with self.assertRaises(ValueError):
            bird_data.applicable_cells("no_such_protocol")
        for cell in cells:
            self.assertIsInstance(cell["heldout_tables"], list)

    def test_manifests_reject_unknown_protocol(self):
        with self.assertRaises(ValueError):
            bird_data.load_manifest("no_such_protocol")


class DevGoldTest(unittest.TestCase):
    """The frozen Dev gold covers all 1,534 raw questions."""

    def test_gold_entry_count(self):
        gold = bird_data.load_dev_gold()
        self.assertEqual(len(gold), 1534)

    def test_gold_columns_are_table_qualified(self):
        gold = bird_data.load_dev_gold()
        for qid, entry in gold.items():
            columns = entry["gold_columns"]
            self.assertTrue(columns, msg=f"{qid}: empty gold_columns")
            for column in columns:
                # Shipped form: [table, column] pairs, i.e. the
                # "table.column" physical identity of the column.
                self.assertIsInstance(column, list)
                self.assertEqual(len(column), 2,
                                 msg=f"{qid}: {column!r} is not a "
                                     f"(table, column) pair")
                table, name = column
                self.assertIsInstance(table, str)
                self.assertIsInstance(name, str)
                self.assertTrue(table and name,
                                msg=f"{qid}: empty table/column in {column!r}")
                self.assertIn(".", f"{table}.{name}")


class UnverifiedExclusionTest(unittest.TestCase):
    """The 4 unverified Dev questions stay out of the OOF population."""

    def test_unverified_qids_excluded_from_oof(self):
        for protocol in PROTOCOLS:
            manifest = bird_data.load_manifest(protocol)
            unverified = [
                qid
                for qids in manifest["unverified_qids_by_db"].values()
                for qid in qids
            ]
            self.assertEqual(len(unverified), 4,
                             msg=f"{protocol}: expected 4 unverified qids")
            oof = set(manifest["oof_population"][protocol])
            self.assertEqual(len(oof), 1530)
            for qid in unverified:
                self.assertNotIn(qid, oof,
                                 msg=f"{protocol}: unverified {qid} must "
                                     f"not be evaluated")
            # They remain present in the shipped gold (raw identity).
            gold = bird_data.load_dev_gold()
            for qid in unverified:
                self.assertEqual(gold[qid]["parse_status"], "unverified")


class TrainDevTest(unittest.TestCase):
    """The Train development inputs cover all 69 databases."""

    def test_schema_covers_69_databases(self):
        schema_root = BIRD_ROOT / "train_dev" / "schema"
        databases = sorted(p.name for p in schema_root.iterdir())
        self.assertEqual(len(databases), 69)

    def test_train_rows_cover_69_databases(self):
        rows_root = BIRD_ROOT / "train_dev" / "train_rows"
        databases = sorted(p.name for p in rows_root.iterdir())
        self.assertEqual(len(databases), 69)
        for db in databases[:3] + databases[-3:]:
            rows = bird_data.train_dev_rows(db)
            self.assertTrue(rows["opt_train"])
            self.assertTrue(rows["opt_dev"])

    def test_opt_manifest_verified_count(self):
        with open(BIRD_ROOT / "train_dev" / "opt_manifest.json",
                  encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["counts"]["n_databases"], 69)
        self.assertEqual(manifest["counts"]["n_questions"], 9428)
        self.assertEqual(manifest["counts"]["n_verified"], 9068)
        self.assertEqual(manifest["counts"]["n_unverified"], 360)


class HSplitsTest(unittest.TestCase):
    """The Train-side held-out-table split manifest and its rows agree."""

    def test_h_manifest_declares_its_own_counts(self):
        manifest = bird_data.h_manifest()
        counts = manifest["counts"]
        # The file is the source of truth; these are its recorded values.
        self.assertEqual(counts["n_applicable"], 55)
        self.assertEqual(counts["n_dev_questions"], 4387)
        self.assertEqual(counts["n_dev_population"], 4387)
        self.assertEqual(len(manifest["dev_population_qids"]), 4387)
        applicable = sorted(db for db, entry in manifest["databases"].items()
                             if entry["applicable"])
        self.assertEqual(len(applicable), 55)

    def test_h_train_rows_exist_for_applicable_databases_only(self):
        manifest = bird_data.h_manifest()
        applicable = sorted(db for db, entry in manifest["databases"].items()
                             if entry["applicable"])
        shipped = sorted(p.name for p in
                         (BIRD_ROOT / "train_dev" / "h_splits" / "train_rows")
                         .iterdir())
        self.assertEqual(shipped, applicable)


class PackageHygieneTest(unittest.TestCase):
    """No regenerable caches ship under data/bird."""

    def test_no_engine_cache_directories(self):
        caches = [p for p in BIRD_ROOT.rglob("engine_cache")]
        self.assertEqual(caches, [])

    def test_no_ds_store_files(self):
        strays = [p for p in BIRD_ROOT.rglob(".DS_Store")]
        self.assertEqual(strays, [])


class MissingInputTest(unittest.TestCase):
    """Missing files raise clear errors instead of being rebuilt."""

    def test_train_rows_missing_file_names_the_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            bird_data.train_rows("question", 0, "no_such_db")
        self.assertIn("no_such_db", str(ctx.exception))

    def test_schema_dir_missing_db_names_the_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            bird_data.schema_dir("no_such_db", "processed")
        self.assertIn("no_such_db", str(ctx.exception))
        with self.assertRaises(ValueError):
            bird_data.schema_dir("financial", "no_such_split")


if __name__ == "__main__":
    unittest.main()
