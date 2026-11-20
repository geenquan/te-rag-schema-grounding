# BIRD data: source, license, and provenance

This directory ships the complete retrieval-side inputs for the BIRD
experiments (question-level `Q` and held-out-table `H` protocols, Dev
evaluation, Train development). It contains the raw public benchmark files
as locally frozen, plus the derived gold/fold/development artifacts that
the paper's frozen configurations reference. The SQLite database archives
themselves are NOT shipped (see "SQLite databases" below) — schema
grounding in this method needs only the JSON/CSV inputs included here.

## Official source

BIRD is a public text-to-SQL benchmark:

- Project page: https://bird-bench.github.io/
- Publication: "Can LLM Serve as Data Analyst? ..." — a benchmark for
  large-scale database grounded text-to-SQL, NeurIPS 2023 Datasets and
  Benchmarks Track.
- DOI (dataset deposit): 10.52202/075280-1835

## License

The BIRD authors changed the dataset license to **Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** on 2024-04-27
(per the announcement on the project page). The full legal code is shipped
verbatim in [`LICENSE-BIRD-DATA.txt`](LICENSE-BIRD-DATA.txt) (source:
https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt).

The raw files in `raw/` are unmodified copies of the official
distribution and remain CC BY-SA 4.0. The derived files under
`processed/` and `train_dev/` are adaptations of BIRD material and are
likewise made available under CC BY-SA 4.0 (ShareAlike applies to them).
The repository's code license is separate and does not apply to the
contents of this directory.

## Local-copy identity

The copies shipped here were pinned by file hash at migration time. The
upstream download distribution does not publish a per-file release
manifest for the exact snapshot these bytes came from, so the SHA256
values below — not a download date — are the authoritative identity of
the frozen inputs. Do not overwrite them with a fresh "latest" download;
doing so changes the experiment inputs.

### Raw files (`raw/`)

| File | Content | Size (bytes) | SHA256 |
|---|---|---:|---|
| `raw/train/train.json` | 9,428 questions (raw official Train split) | 4,321,661 | `abf17d3def22f56f41b3e139b94bba3fa4a986280abcbf6df2d7f30150695520` |
| `raw/train/train_tables.json` | 69 Train database schema/description entries | 726,087 | `85300c745b263db88a5e0cdf083e8b2427d90272e954bc8c5f92aa1dc66c836b` |
| `raw/dev/dev.json` | 1,534 questions (raw official Dev split, 2024-06-27 snapshot) | 741,332 | `630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06` |
| `raw/dev/dev_tables.json` | 11 Dev database schema/description entries | 158,350 | `35e7f73f2bc3a3bc4def9466215f7617ff8b37e9b2837f03539a42c9e9c69d97` |

## Question counts (raw vs. evaluated)

The raw files are shipped untouched; populations used by experiments are
defined by the derived manifests:

- **Dev**: 1,534 raw questions. The frozen five-fold construction
  verified 1,530 of them (`q5_manifest.json` / `h5_manifest.json` record
  `counts: {gold_entries: 1534, verified: 1530, unverified: 4}`). The
  main evaluation population is therefore the 1,530 verified questions
  (the `oof_population` of both manifests); the raw file is not truncated
  to 1,530 and must not be described as containing only 1,530 questions.
- **Train**: 9,428 raw questions. The Train development manifest
  (`train_dev/opt_manifest.json`) records
  `counts: {n_questions: 9428, n_verified: 9068, n_unverified: 360}`;
  development splits are built over verified questions only.

### The 4 excluded Dev questions

Exactly what `q5_manifest.json` records under
`unverified_qids_by_db` (identical block in `h5_manifest.json`):

```json
{
 "card_games": ["card_games::518"],
 "codebase_community": ["codebase_community::604"],
 "debit_card_specializing": ["debit_card_specializing::1499"],
 "european_football_2": ["european_football_2::1058"]
}
```

The manifests list the qids grouped by database; the corresponding
entries in `processed/gold/gold.json` carry
`parse_status: "unverified"` (their gold SQL could not be fully
verified during gold construction), which is why they are kept out of
the verified evaluation population while remaining present in the raw
data and in the gold file.

## Directory contents and provenance

### `raw/`

Byte-identical local copies of the official BIRD distribution files
listed above.

### `processed/` — frozen Dev five-fold construction run

Derived by the frozen Dev five-fold construction run (question-level
`question_v5` and held-out-table `heldout_table_v5` protocols, seed 42,
five folds per database):

- `gold/gold.json` — derived gold for all 1,534 Dev questions
  (physical tables/columns parsed from the official gold SQL, duplicate
  groups, parse status; entries keyed `<db_id>::<index>`).
- `q5_manifest.json` — question-level five-fold manifest: per-database
  duplicate-group fold assignment, per-(db, fold) train/test qids,
  applicability gate `train >= 5 AND test >= 1 verified questions per
  (db, fold)`, 55 applicable cells, out-of-fold population of 1,530
  qids.
- `h5_manifest.json` — held-out-table five-fold manifest (tables as
  hold-out units, anchor rule and fixed tie permutation recorded
  inside), 50 applicable cells, out-of-fold population of 1,530 qids.
- `schema/<db_id>/{field_schema.csv,table_schema.csv}` — per-database
  schema CSVs for the 11 Dev databases, extracted from
  `raw/dev/dev_tables.json`.
- `train/{question,heldout}_f<fold>_<db>.jsonl` — 105 files, one row
  per training question for every applicable (protocol, fold, database)
  cell (55 question + 50 heldout).

### `train_dev/` — frozen Train development construction run

Derived by the frozen Train development construction run over the
official Train split (per-db duplicate-group 80/20 split, seed 42):

- `opt_gold.json` — derived gold for the Train questions used in
  development.
- `opt_manifest.json` — the frozen Train development manifest: 69
  databases, 9,428 questions of which 9,068 verified, per-database
  80/20 `opt_train`/`opt_dev` qid lists over verified questions
  (duplicate groups never span sides; unverified questions never enter
  the populations).
- `BUILD_MANIFEST.json` — build-time manifest of the construction
  (relative paths and SHA256 of the generator inputs/outputs).
- `schema/<db_id>/{field_schema.csv,table_schema.csv}` — per-database
  schema CSVs for all 69 Train databases, extracted from
  `raw/train/train_tables.json`.
- `train_rows/<db_id>/{opt_train.jsonl,opt_dev.jsonl}` — training and
  development rows per Train database (69 databases).
- `h_splits/h_manifest.json` and `h_splits/train_rows/<db>/h_train.jsonl`
  — the frozen Train-side held-out-table inner two-way split
  (`heldout_table_inner_2way`, seed 42): 55 of the 69 databases pass the
  `inner_train >= 5 AND inner_dev >= 1` gate (14 do not and are recorded
  as inapplicable with reasons), declaring 4,387 dev-population and
  2,154 inner-train questions (1,907 dropped for reference crossing,
  360 unverified kept out). `h_train.jsonl` rows exist for exactly the
  55 applicable databases.

Regenerable engine caches and control-arm outputs from the construction
runs are intentionally not shipped.

## SQLite databases (NOT shipped)

The official SQLite archives (`train_databases.zip` ~9.35 GB,
`dev_databases.zip` ~346 MB for this snapshot) are far beyond normal
Git limits and are not included. Schema grounding in this method reads
only the JSON/CSV inputs above — no database SQL is executed and no
table rows are read — so the full reproduction path is complete without
them. To obtain the database archives, use the official download links
at https://bird-bench.github.io/ .

## Changes made relative to the official distribution

No raw file was modified. The derived artifacts under `processed/` and
`train_dev/` were constructed as follows (all derivations are
deterministic given the recorded seeds/rules and are fully specified in
the shipped manifests):

1. **Per-database schema CSV extraction** — `field_schema.csv` /
   `table_schema.csv` per database, extracted from the official
   `*_tables.json` descriptions.
2. **qid assignment** — stable question identifiers
   `<db_id>::<index>` assigned in raw file order.
3. **Gold derivation** — physical gold tables/columns parsed from the
   official gold SQL, with duplicate groups and parse status recorded.
4. **Five-fold Dev manifests** — duplicate-group-preserving five-fold
   assignment per database (question-level; and table-level hold-out
   with anchor/tie rules recorded in the manifests), 55/50 applicable
   cells, out-of-fold populations of 1,530 verified questions.
5. **Train duplicate-group 80/20 split** — per-database
   connected-component duplicate groups greedily filling an 80/20
   opt-train/opt-dev split over verified questions (plus the
   held-out-table inner two-way split for H development).

Derived data is marked ShareAlike under CC BY-SA 4.0 as stated above.
