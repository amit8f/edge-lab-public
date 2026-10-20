"""Plan v3 books (section 7): the learning book, its frozen copy, and the same-day random control.

All three see the same candidate list each day. The learning book sizes each strategy's candidates from
its current weekly weights; the frozen copy uses the weights it started with; the random control takes,
for each strategy, the same number of candidates drawn at random from that day's eligible universe, on
the same side, sized the same way. Comparing the three answers "does the learning help?" and "is it
better than luck?".
"""
import numpy as np

from adaptive import sizing


def allocate(candidates, weights, equity, gross_now, lim):
    """candidates: dicts with strategy, symbol, side, price, stop_dist, liq_dollars, score (higher first).
    weights: {strategy: weight}. Each strategy may use weight x equity of notional. Returns sized orders."""
    out, used = [], {}
    gross = gross_now
    for c in sorted(candidates, key=lambda c: (-c.get("score", 0), c["symbol"])):
        w = weights.get(c["strategy"], 0.0)
        room = w * equity - used.get(c["strategy"], 0.0)
        if room <= 0:
            continue
        shares, why = sizing.size(equity, c["side"], c["stop_dist"], c["price"], c["liq_dollars"], gross, lim)
        shares = min(shares, int(room // c["price"]))
        if shares <= 0:
            continue
        notional = shares * c["price"]
        used[c["strategy"]] = used.get(c["strategy"], 0.0) + notional
        gross += notional
        out.append(dict(c, shares=shares, notional=notional, binding=why))
    return out


def random_control(candidates, universe, day_seed):
    """Same-day random twin of each candidate: a random eligible symbol (not already a candidate), same
    strategy, side, stop distance and score. universe: dicts with symbol, price, liq_dollars."""
    rng = np.random.default_rng(day_seed)
    taken = {c["symbol"] for c in candidates}
    pool = [u for u in universe if u["symbol"] not in taken]
    out = []
    for c in candidates:
        if not pool:
            break
        u = pool.pop(int(rng.integers(len(pool))))
        out.append(dict(c, symbol=u["symbol"], price=u["price"], liq_dollars=u["liq_dollars"], control=True))
    return out
