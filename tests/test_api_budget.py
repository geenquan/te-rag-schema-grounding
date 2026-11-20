"""API budget-cap validation regressions (offline; no network, no key).

Historical defect (independent review PROBES.json): a NaN cap was accepted
and the broker successfully reserved, because every comparison against NaN
is False — the spending limit could be silently disabled. These tests pin
the fix: the cap must be a finite positive number at EVERY entry point
(validator, ApiRuntime property, configure, require_budget_cap, direct
BudgetBroker construction, and the CLI/env parsing), while a valid cap
keeps the original reservation/settlement semantics.
"""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from terag.engines.api.config import (RUNTIME, InvalidBudgetCap,
                                      validate_budget_cap)

BAD_VALUES = [float('nan'), float('inf'), float('-inf'), 0.0, -0.0, -1.0,
              -1e-9, True, False, 'nan', 'inf', '-inf', 'abc', '', '0',
              '-5', None]
GOOD_VALUES = [5.0, 0.01, 1e6, 3, '2.5', ' 7 ']


class TestValidator(unittest.TestCase):
    def test_good_values_accepted(self):
        for v in GOOD_VALUES:
            cap = validate_budget_cap(v)
            self.assertTrue(math.isfinite(cap) and cap > 0, repr(v))

    def test_bad_values_rejected(self):
        for v in BAD_VALUES:
            with self.assertRaises(InvalidBudgetCap, msg=repr(v)):
                validate_budget_cap(v)
            # booleans must be rejected even though float(True) == 1.0
            if isinstance(v, bool):
                with self.assertRaises(InvalidBudgetCap):
                    validate_budget_cap(v)


class TestRuntimeEntries(unittest.TestCase):
    def setUp(self):
        self._saved = (RUNTIME.run_dir, RUNTIME._budget_cap_cny)

    def tearDown(self):
        RUNTIME.run_dir, RUNTIME._budget_cap_cny = self._saved

    def test_property_setter_rejects_invalid(self):
        for v in (float('nan'), float('inf'), -1.0, 0.0, True):
            with self.assertRaises(InvalidBudgetCap, msg=repr(v)):
                RUNTIME.budget_cap_cny = v
        self.assertIsNone(RUNTIME.budget_cap_cny)  # unchanged by failures

    def test_configure_rejects_invalid(self):
        from terag.engines.api import config as api_config
        with tempfile.TemporaryDirectory() as tmp:
            for v in (float('nan'), float('inf'), -1.0):
                with self.assertRaises(InvalidBudgetCap, msg=repr(v)):
                    api_config.configure(run_dir=tmp, budget_cap_cny=v)

    def test_require_budget_cap_reasserts_finiteness(self):
        with tempfile.TemporaryDirectory() as tmp:
            RUNTIME.run_dir = Path(tmp)
            # bypass the property via __dict__ (the re-assert path)
            RUNTIME.__dict__['_budget_cap_cny'] = float('nan')
            with self.assertRaises(InvalidBudgetCap):
                RUNTIME.require_budget_cap()
            RUNTIME.__dict__['_budget_cap_cny'] = None
            with self.assertRaises(RuntimeError):
                RUNTIME.require_budget_cap()


class TestBudgetBrokerBoundary(unittest.TestCase):
    def _broker(self, cap):
        from terag.engines.api.budget_broker import BudgetBroker
        tmp = tempfile.mkdtemp(prefix='broker_test_')
        return BudgetBroker(cap_cny=cap, evidence_dir=tmp)

    def test_invalid_caps_rejected_at_construction(self):
        from terag.engines.api.config import InvalidBudgetCap
        for v in (float('nan'), float('inf'), float('-inf'), 0.0, -1.0,
                  True, 'nan'):
            with self.assertRaises(InvalidBudgetCap, msg=repr(v)):
                self._broker(v)

    def test_valid_cap_reserves_and_enforces_offline(self):
        """No network: a legal cap keeps the reservation semantics, and the
        invariant still refuses an over-cap reservation."""
        from terag.engines.api.budget_broker import BudgetExceeded
        broker = self._broker(0.01)   # 1e-2 CNY cap
        res = broker.reserve('test', 'stage', est_input_tokens=100,
                             max_tokens=100)
        self.assertIn('req_id', res)
        self.assertGreater(res['reserve_cny'], 0)
        # a reservation that would exceed the cap must be refused
        with self.assertRaises(BudgetExceeded):
            broker.reserve('test', 'stage', est_input_tokens=100_000_000,
                           max_tokens=100_000_000)
        state = broker.state()
        self.assertEqual(state['cap_cny'], 0.01)
        self.assertGreater(state['committed_cny'], 0)


class TestCliEntry(unittest.TestCase):
    def test_cli_rejects_nan_budget_before_any_run(self):
        """`--budget-cap-cny nan` must exit(2) with a clear error before
        run_bird (and therefore before any network/API activity)."""
        from terag_release import cli
        with mock.patch.object(cli, '_fail', side_effect=cli._fail):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(['run-bird', '--methods', 'Core',
                          '--budget-cap-cny', 'nan',
                          '--output', '/tmp/never_created_budget_test'])
            self.assertEqual(ctx.exception.code, 2)

    def test_cli_rejects_nonpositive_env_budget(self):
        from terag_release import cli
        with mock.patch.dict('os.environ',
                             {'TERAG_BIRD_BUDGET_CNY': '-3'}):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(['run-bird', '--methods', 'Core',
                          '--output', '/tmp/never_created_budget_test2'])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
