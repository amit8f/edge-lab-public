"""Plan v3 promotion of shadow rules (Q18, Q37, Q48).

A shadow rule is promoted when, over at least 4 weeks and 30 closed trades after costs, its daily
market-adjusted return beats the frozen copy's with one-sided confidence of at least `required(n)`.
The bar starts at 80% and rises linearly to 90% as the number of shadow rules tried reaches 50.
"""
from statistics import NormalDist

import numpy as np

MIN_DAYS = 20          # 4 trading weeks
MIN_TRADES = 30


def required(n_tried):
    return 0.80 + 0.10 * min(max(n_tried, 1) - 1, 49) / 49


def confidence(rule_daily, frozen_daily):
    d = np.asarray(rule_daily, float) - np.asarray(frozen_daily, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return 0.5
    t = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
    return NormalDist().cdf(t)


def decide(rule_daily, frozen_daily, trades, n_tried):
    days = int(np.isfinite(np.asarray(rule_daily, float)).sum())
    if days < MIN_DAYS or trades < MIN_TRADES:
        return {"promote": False, "why": f"needs {MIN_DAYS} days and {MIN_TRADES} trades", "days": days,
                "trades": trades}
    c, need = confidence(rule_daily, frozen_daily), required(n_tried)
    return {"promote": c >= need, "confidence": round(c, 4), "required": round(need, 4), "days": days,
            "trades": trades}
