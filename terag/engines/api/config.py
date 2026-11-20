"""Shared configuration for the BIRD API methods (DenseDeepSeek, KaSLA-Adapted,
LinkAlign-SingleDB-Adapted).

PUBLIC-PORT adaptations vs the research implementation (behavioral contract
unchanged, research-environment IO removed):
- Credentials are read from environment variables ONLY (DEEPSEEK_API_KEY /
  DEEPSEEK_BASE_URL). The research run's config.yaml file fallback is removed:
  no credential file is ever read, printed, or copied.
- The budget cap is NOT baked in. The historical research budget (CNY 20 per
  run, later a cumulative CNY 60 continuation cap) is not silently reused: a
  user MUST pass an explicit cap for this run via configure() (the CLI forwards
  --budget-cap-cny / TERAG_BIRD_BUDGET_CNY); refusing to run without one is
  intentional.
- Evidence/cache/ledger directories live under the CURRENT run's output
  directory instead of the research run tree.
- The MiniLM encoder is identified by model id + pinned revision with a
  configurable local cache directory (the research machine's absolute snapshot
  path is removed).

Kept from the research identity: deepseek-flash requests with thinking
explicitly disabled, one app-level retry on transient errors, SDK retries
disabled, peak/off-peak price buckets, and the conservative
reservation/settlement accounting in budget_broker.
"""
import os
from pathlib import Path

MINILM_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MINILM_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

# ---- budget (CNY per 1M tokens; research contract values, kept frozen) ----
PRICE_CNY_PER_M = {
    'peak': {'input': 2.0, 'output': 8.0},
    'offpeak': {'input': 1.0, 'output': 4.0},
}
OFFPEAK_UTC8 = (0, 30)   # off-peak window starts 00:30 UTC+8 ...
OFFPEAK_END_UTC8 = (8, 30)  # ... and ends 08:30 UTC+8 (research rule)

# ---- model wiring ----
REQUESTED_MODEL = 'deepseek-flash'
# Documented DeepSeek chat.completions thinking toggle; the official docs'
# example shows {"thinking": {"type": "enabled"}}, so "disabled" is the
# documented way to force the non-thinking mode.
THINKING_CONTROL = {'thinking': {'type': 'disabled'}}
THINKING_CONTROL_NOTE = (
    "extra_body={'thinking': {'type': 'disabled'}} passed to "
    "chat.completions.create; per api-docs.deepseek.com (guides/first_api_call "
    "example and api/create-chat-completion reference) this is the documented "
    "thinking-mode toggle; deepseek-flash defaults to thinking ON, so it is "
    "explicitly disabled for every call.")
REQUEST_TIMEOUT_S = 60.0     # same as the reference implementations
MAX_APP_RETRIES = 1          # one app-level retry on transient errors only


class ApiRuntime:
    """Per-run mutable configuration (evidence dirs, budget cap, caches).

    The research modules read module-level constants tied to the research run
    tree; the public port funnels them through ONE runtime object that the
    runner configures before any API activity. Accessor properties keep the
    historical names available at the call sites.
    """

    def __init__(self):
        self.run_dir = None            # Path: current run output directory
        self.budget_cap_cny = None     # float: REQUIRED user-set cap (CNY)
        self.minilm_cache_dir = None   # Path or None -> HF default cache
        self.allow_download = False    # MiniLM download requires opt-in

    # ---- evidence / cache directories (always under the run directory) ----
    @property
    def api_evidence_dir(self):
        self._require_run_dir()
        return self.run_dir / 'api_evidence'

    @property
    def http_cache_dir(self):
        return self.api_evidence_dir / 'http_cache'

    @property
    def dense_cache_dir(self):
        return self.api_evidence_dir / 'dense_cache'

    @property
    def ledger_path(self):
        return self.api_evidence_dir / 'budget_ledger.json'

    @property
    def ledger_lock_path(self):
        return self.api_evidence_dir / 'budget_ledger.lock'

    @property
    def audit_path(self):
        return self.api_evidence_dir / 'budget_audit.jsonl'

    @property
    def budget_state_path(self):
        return self.api_evidence_dir / 'budget_state.json'

    @property
    def model_selection_path(self):
        return self.api_evidence_dir / 'model_selection.json'

    def _require_run_dir(self):
        if self.run_dir is None:
            raise RuntimeError(
                'terag.engines.api runtime not configured: call '
                'configure(run_dir=..., budget_cap_cny=...) before any API '
                'activity (the runner does this automatically)')

    def require_budget_cap(self):
        if self.budget_cap_cny is None:
            raise RuntimeError(
                'no API budget cap configured: online API methods refuse to '
                'run without an explicit user-set cap (--budget-cap-cny or '
                'TERAG_BIRD_BUDGET_CNY); the historical research budget is '
                'never reused silently')


RUNTIME = ApiRuntime()


def configure(run_dir, budget_cap_cny=None, minilm_cache_dir=None,
              allow_download=False):
    """Set the per-run API configuration. Must run before any API call."""
    RUNTIME.run_dir = Path(run_dir)
    RUNTIME.budget_cap_cny = (None if budget_cap_cny is None
                              else float(budget_cap_cny))
    RUNTIME.minilm_cache_dir = (Path(minilm_cache_dir)
                                if minilm_cache_dir else None)
    RUNTIME.allow_download = bool(allow_download)


def load_credentials():
    """Return (api_key, base_url) from ENVIRONMENT VARIABLES ONLY.

    The key is only ever returned to the OpenAI client constructor; it is
    never printed, logged, or persisted by any caller. There is deliberately
    NO file fallback (the research project-root config.yaml reader is removed
    from the public port).
    """
    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip() or None
    base_url = os.environ.get('DEEPSEEK_BASE_URL', '').strip() or None
    if not base_url:
        base_url = 'https://api.deepseek.com'
    return api_key, base_url


def price_bucket_now(now=None):
    """Research price bucket from the response time: UTC+8 00:30-08:30 is
    off-peak, everything else is peak."""
    import datetime as _dt
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    utc8 = (now + _dt.timedelta(hours=8))
    minutes = utc8.hour * 60 + utc8.minute
    start = OFFPEAK_UTC8[0] * 60 + OFFPEAK_UTC8[1]
    end = OFFPEAK_END_UTC8[0] * 60 + OFFPEAK_END_UTC8[1]
    return 'offpeak' if start <= minutes < end else 'peak'
