"""BIRD API method adapters (DenseDeepSeek, KaSLA-Adapted, LinkAlign).

Frozen research call semantics (deepseek-flash, thinking disabled, one
app-level retry, conservative budget broker, HTTP response cache) with the
public-port boundaries documented in ``config.py``: env-only credentials,
user-set budget cap, per-run evidence directories, and a pinned-revision
MiniLM encoder with a configurable cache. See docs/reproduction.md before
running any online method.
"""
