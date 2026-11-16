"""TE-RAG public release orchestration package.

Modules:
* config       — layered YAML configuration (extends/resolved export/SHA256)
* protocols    — deterministic fold construction + AST isolation (paper v2)
* supervision  — full-SQL training supervision injection
* variants     — the paper's method/variant matrix (single source of truth)
* runner       — fit/retrieve/save/summarize pipeline
* validation   — input-contract checks (validate-data)
* benchmark    — fresh timing of the current build
* reporting    — run_meta provenance snapshots
* cli          — `python -m terag_release <command>`
"""

__version__ = "0.1.0"
