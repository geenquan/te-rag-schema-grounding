# BIRD reference results (historical frozen predictions)

This directory contains the **normalized per-question predictions, per-question
metrics, and aggregate summaries of the final BIRD evaluation** reported in
the paper. They let anyone **recompute every BIRD table value offline** — no
model download, no API key, no engine run.

**What these are.** Historical reference predictions produced by the frozen
final evaluation (unified execution entry) over the Dev five-fold population
shipped under `data/bird/`. Methods rerun through the unified entry (engine +
H-reliability gating + Q field rerank/gate, with per-question execution
evidence) are marked `unified_entry_rerun` in `MANIFEST.json`; the reused
original-run methods (baselines, `ORIGINAL_FULL`, the three `route*`
exploratory rows) were hash-verified against their declared frozen sources
before reuse and are marked `reused_hash_verified`.

**What these are NOT.** They are not outputs of a run you just executed, and
they are not a substitute for running the methods yourself
(`python -m terag_release run-bird ...` reproduces them from code; see
docs/reproduction.md). Look-up tables never masquerade as algorithm
execution: the paper's claims rest on the engine runs, and these files are
the archived evidence.

## Contents

```
predictions/<protocol>/<method>.jsonl.gz   one sanitized record per question:
    protocol, fold, db_id, qid, variant, status,
    retrieved_tables (<=5), retrieved_column_keys (<=10 per table),
    field_rerank_applied (Q-gate marker where applicable)
metrics/<protocol>/<method>_per_question.csv    the 9 metrics per qid
aggregates/<protocol>_summary.csv               per-method means (percent)
MANIFEST.json        per-file SHA256 + source kind + reuse-group digests
```

104 prediction files cover every method×protocol of the final evaluation
(question: 54 methods incl. 10 baselines + `ORIGINAL_FULL` (Core) +
`qB_rerank_gate_v1` (Gate) + fixed/optimized ablations; heldout: 50 methods
incl. `full_href_1_8` (H-opt)). Prompts, raw responses, request/response ids,
API ledgers, timings, error strings, score details, and machine paths were
removed during normalization.

## Recompute check

```sh
python -m terag_release evaluate-bird-reference --output runs/refcheck
```

recomputes the metrics from `predictions/` against `data/bird/processed/gold`
and compares them to `metrics/` and `aggregates/` (hard-fails on any
mismatch). Key identities (percent, paper Table 2/appendix):

| Method | protocol | Covered@5 | ColF1 | FGCS |
|---|---|---:|---:|---:|
| ORIGINAL_FULL (Q Core) | question | 93.79 | 25.30 | 75.62 |
| qB_rerank_gate_v1 (Q + Gate) | question | 93.79 | 27.87 | 77.91 |
| ORIGINAL_FULL (H core) | heldout | 76.54 | 21.99 | 51.70 |
| full_href_1_8 (H-opt) | heldout | 82.29 | 22.98 | 57.97 |

The rerank-only stage of the Q three-stage table is
`qB_rerank_gate_v1-field_filter_gate_off` (gate disabled, rerank axes kept);
its identity is documented in docs/experiment_catalog.md.

## Provenance

`MANIFEST.json` records the source identity (digests of the frozen metric
files and of every reuse group). The materials were assembled from the frozen
research evaluation trees read-only; see docs/release_mapping.md for the
logical identity chain. Enterprise (power-grid) reference predictions are
**not** and will not be published here.
