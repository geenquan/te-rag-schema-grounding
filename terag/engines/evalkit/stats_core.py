"""Pure statistical core for the review correction.

`sign_flip` is ported VERBATIM (logic-preserving, numpy formatting only) from
research_runs/targeted_ablation_optimization_v2/analyze.py::sign_flip — the accepted
cluster sign-flip test of the prior round. Nothing here imports the old research
entry points (they load powergrid data); only the pure function is reused.

Semantics (must match the accepted implementation):
- cluster sums with |value| <= 1e-12 are dropped (zero tolerance);
- no nonzero cluster sum -> (None, 'zero_cluster_sums');
- two-sided statistic: |sum of signed nonzero cluster sums|;
- exhaustive enumeration over all 2^m sign combinations when m <= 20
  (p = #(combos with |signed sum| >= target - 1e-12) / 2^m), exact;
- Monte Carlo fallback (100000 draws, seed 42) only when m > 20 (not reachable
  with 11 databases, kept for fidelity to the reference).

PUBLIC-PORT: the research copy also defined an ORIG_RUN constant pointing at
a research run directory; it was unused by the statistical core and is not
carried over.
"""
import numpy as np

ZERO_TOL = 1e-12
PROTOCOLS = ['question', 'heldout']
METRICS = ['colf1', 'covered@5']
FAMILY_K = 22  # pre-registered: 11 opponents x 2 primary metrics, never shrunk


def sign_flip(sums):
    """Exact two-sided cluster sign-flip p-value. Ported verbatim from analyze.py."""
    x = np.array([value for value in sums if abs(value) > ZERO_TOL], dtype=float)
    if not len(x):
        return None, 'zero_cluster_sums'
    target = abs(x.sum()); count = 0; total = 0
    if len(x) <= 20:
        for start in range(0, 2 ** len(x), 32768):
            integers = np.arange(start, min(start + 32768, 2 ** len(x)), dtype=np.uint32)
            signs = 2 * ((integers[:, None] >> np.arange(len(x), dtype=np.uint32)) & 1).astype(float) - 1
            values = np.abs(signs @ x); count += int(np.sum(values >= target - ZERO_TOL)); total += len(values)
        return count / total, 'exact_cluster_sign_flip'
    rng = np.random.default_rng(42); signs = rng.choice([-1, 1], size=(100000, len(x)))
    return (1 + np.sum(np.abs(signs @ x) >= target - ZERO_TOL)) / 100001, 'monte_carlo_cluster_sign_flip_100000'


def classify_difference(diffs, cluster_sums):
    """Distinguish exact zero difference from cancellation, per accepted taxonomy.

    diffs: per-question differences (1-D array).
    cluster_sums: per-db-cluster sums of the same differences.
    Returns one of:
      'zero_difference'          — no per-question difference exceeds tolerance;
      'cancels_within_cluster'   — question-level differences exist but every
                                    cluster sum is within tolerance;
      'computed'                 — at least one nonzero cluster sum, test ran.
    """
    nonzero_q = np.any(np.abs(diffs) > ZERO_TOL)
    nonzero_c = np.any(np.abs(cluster_sums) > ZERO_TOL)
    if not nonzero_q:
        return 'zero_difference'
    if not nonzero_c:
        return 'cancels_within_cluster'
    return 'computed'


def cluster_bootstrap_ci(diffs, clusters, n_resamples=10000, seed=42):
    """Db-cluster paired bootstrap CI for the mean per-question difference.

    Resample clusters with replacement; pool per-question values (sum/sum of sizes).
    Unadjusted 95% percentile interval — descriptive uncertainty of the effect,
    NOT a family significance statement and NOT the formal test.
    """
    by = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(float(d))
    names = sorted(by)
    sums = np.array([sum(by[c]) for c in names])
    sizes = np.array([len(by[c]) for c in names])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(names), size=(n_resamples, len(names)))
    boots = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return [float(v) for v in np.quantile(boots, [0.025, 0.975])], names
