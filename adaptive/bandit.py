"""Plan v3 learning rule (section 7): the weekly coach's allocation across strategies.

Each Sunday every active strategy is scored on its market-adjusted return (return minus beta times the
benchmark, after costs) over a trailing window. Target weights are a softmax of the scores scaled by their
volatility, mixed with equal weight so every strategy keeps a floor. Weights then move toward the target by
at most `max_step` (10 percentage points) per strategy per week. Deterministic: same inputs, same weights.
"""
import numpy as np


def market_adjusted(returns, bench):
    """Daily returns minus beta x benchmark (OLS beta on the same days; beta 0 under 20 days)."""
    r, b = np.asarray(returns, float), np.asarray(bench, float)
    ok = np.isfinite(r) & np.isfinite(b)
    r, b = r[ok], b[ok]
    if len(r) < 20 or np.var(b) == 0:
        return r
    beta = np.cov(r, b, ddof=0)[0, 1] / np.var(b)
    return r - beta * b


def score(returns, bench):
    """Mean market-adjusted daily return over its standard error (0 with under 5 days)."""
    x = market_adjusted(returns, bench)
    if len(x) < 5 or np.std(x, ddof=1) == 0:
        return 0.0
    return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))


def target(scores, temperature=1.0, floor_mix=0.3):
    s = np.asarray(scores, float) / temperature
    w = np.exp(s - s.max())
    w = w / w.sum()
    return (1 - floor_mix) * w + floor_mix / len(w)


def step(current, tgt, max_step=0.10):
    """Move each weight toward target by at most max_step, then renormalise to sum to 1."""
    cur, tgt = np.asarray(current, float), np.asarray(tgt, float)
    new = cur + np.clip(tgt - cur, -max_step, max_step)
    new = np.clip(new, 0, None)
    return new / new.sum()


def weekly_update(current, returns_by_strategy, bench, max_step=0.10):
    """current: weights in strategy order; returns_by_strategy: list of daily-return arrays aligned with
    bench. Returns (new weights, scores)."""
    sc = [score(r, bench) for r in returns_by_strategy]
    return step(current, target(sc), max_step), sc
