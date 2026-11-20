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
import math
import os
from pathlib import Path

MINILM_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MINILM_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


class InvalidBudgetCap(ValueError):
    """The configured API budget cap is not a finite positive number."""


def validate_budget_cap(value, where='budget cap'):
    """Validate an API budget cap; return it as a float.

    The cap is the hard invariant every reservation checks against, so it
    must be a FINITE POSITIVE number. Rejected: None (unset — use
    require_budget_cap for that case), booleans (float(True) == 1.0 would
    otherwise pass), NaN and +/-inf (every comparison against them is False,
    which would silently disable the cap), zero and negative numbers, and
    non-numeric strings. Raises InvalidBudgetCap with a message naming the
    offending value; callers must treat it as a configuration error BEFORE
    any network request or reservation.
    """
    if isinstance(value, bool):
        raise InvalidBudgetCap(
            f'{where}: boolean {value!r} is not a valid budget cap '
            '(CNY); pass a finite positive number')
    try:
        cap = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidBudgetCap(
            f'{where}: {value!r} is not a number (CNY); pass a finite '
            'positive number') from exc
    if not math.isfinite(cap):
        raise InvalidBudgetCap(
            f'{where}: {value!r} is not finite; NaN/inf caps make every '
            'budget comparison false and would disable the limit')
    if cap <= 0:
        raise InvalidBudgetCap(
            f'{where}: {cap!r} must be positive (CNY); zero or negative '
            'caps forbid all spending and are rejected as configuration '
            'errors')
    return cap

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

    ``budget_cap_cny`` is a validating property: None means "not set"
    (require_budget_cap rejects API activity), and any non-None assignment
    must pass validate_budget_cap — NaN/inf/non-positive/boolean values are
    rejected at the assignment site, not at first use.
    """

    def __init__(self):
        self.run_dir = None            # Path: current run output directory
        self._budget_cap_cny = None    # float: REQUIRED user-set cap (CNY)
        self.minilm_cache_dir = None   # Path or None -> HF default cache
        self.allow_download = False    # MiniLM download requires opt-in

    @property
    def budget_cap_cny(self):
        return self._budget_cap_cny

    @budget_cap_cny.setter
    def budget_cap_cny(self, value):
        if value is None:
            self._budget_cap_cny = None
        else:
            self._budget_cap_cny = validate_budget_cap(
                value, 'RUNTIME.budget_cap_cny')

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
        # a non-None value already passed validate_budget_cap at assignment;
        # re-assert in case the attribute was set through __dict__ tricks
        return validate_budget_cap(self.budget_cap_cny,
                                   'RUNTIME.budget_cap_cny')


RUNTIME = ApiRuntime()


def configure(run_dir, budget_cap_cny=None, minilm_cache_dir=None,
              allow_download=False):
    """Set the per-run API configuration. Must run before any API call.

    A non-None ``budget_cap_cny`` must be a finite positive number
    (InvalidBudgetCap otherwise) — the same rule every other entry point
    (CLI flag, TERAG_BIRD_BUDGET_CNY, BudgetBroker) enforces.
    """
    RUNTIME.run_dir = Path(run_dir)
    RUNTIME.budget_cap_cny = budget_cap_cny  # validating property
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
