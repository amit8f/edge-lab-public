"""Plain-language reasons, calibrated expectations and price ranges for each pick. Pure functions."""
import math

import numpy as np

from tracks.main.longterm.score import FAMILY_NAMES


def _top(p):
    return f"top {max(1, round((1 - p) * 100))}%"


def reasons(row):
    """(strengths, watch-outs): short sentences from the factor values, strongest first."""
    good, bad = [], []
    fam = {f: row.get(f) for f in FAMILY_NAMES}

    def add(f, hi_text, lo_text):
        p = fam.get(f)
        if p is None or not np.isfinite(p):
            return
        if p >= 0.7:
            good.append((p, f"{hi_text} ({_top(p)} on {FAMILY_NAMES[f]})"))
        elif p <= 0.3 and lo_text:
            bad.append((p, lo_text))

    m, v = row.get("mom"), row.get("vol")
    if m is not None and np.isfinite(m):
        add("f_mom", f"Strong trend: up {m:.0%} over the 12 months before last month",
            f"Weak trend: {m:+.0%} over the 12 months before last month")
    if v is not None and np.isfinite(v):
        add("f_lowvol", f"Steadier than most: {v:.0%} annualised volatility",
            f"Volatile: {v:.0%} annualised, expect big swings")
    gpa, roa = row.get("gpa"), row.get("roa")
    q = []
    if gpa is not None and np.isfinite(gpa):
        q.append(f"gross profit {gpa:.0%} of assets")
    if roa is not None and np.isfinite(roa):
        q.append(f"return on assets {roa:.0%}")
    if q:
        add("f_quality", "Profitable business: " + ", ".join(q), "Low profitability: " + ", ".join(q))
    ep, bm = row.get("ep"), row.get("bm")
    if ep is not None and np.isfinite(ep):
        pe = f"P/E about {1 / ep:.0f}" if ep > 0 else "loss-making on the last annual report"
        add("f_value", f"Reasonably priced: {pe}", f"Expensive: {pe}")
    iss = row.get("issuance")
    if iss is not None and np.isfinite(iss):
        ch = math.expm1(iss)
        add("f_issuance", f"Share count {'down' if ch < 0 else 'up'} {abs(ch):.1%} in a year"
            + (" (buybacks)" if ch < 0 else ""), f"Dilution: share count up {ch:.1%} in a year")
    ag = row.get("assetgrowth")
    if ag is not None and np.isfinite(ag):
        add("f_assetgrowth", f"Disciplined growth: assets {ag:+.0%} last year",
            f"Fast expansion: assets {ag:+.0%} last year (historically a drag on returns)")
    good.sort(reverse=True)
    bad.sort()
    return [t for _, t in good], [t for _, t in bad]


def bucket_for(cal, score_pct):
    for b in cal.get("buckets", []):
        if b["lo"] < score_pct <= b["hi"] or (score_pct <= b["hi"] and b["lo"] <= 0):
            return b
    return None


def expectation(cal, score_pct, price, vol):
    """Historical 6-month outcome for stocks ranked like this one, and a one-sigma price band for this
    stock's own volatility centred on that median."""
    b = bucket_for(cal, score_pct) if cal else None
    out = {"bucket": None, "median": None, "p25": None, "p75": None, "beat_spy": None, "n": None,
           "band_low": None, "band_high": None}
    if b is None:
        return out
    out.update({"bucket": f"{b['lo']:.0%}-{b['hi']:.0%}", "median": b["median"], "p25": b["p25"],
                "p75": b["p75"], "beat_spy": b["beat_spy"], "n": b["n"]})
    if price and vol and np.isfinite(vol):
        mid = price * (1 + b["median"])
        s = vol * math.sqrt(0.5)
        out["band_low"], out["band_high"] = mid * math.exp(-s), mid * math.exp(s)
    return out


def confidence_label(beat_spy):
    if beat_spy is None:
        return "untested"
    if beat_spy >= 0.6:
        return "higher"
    if beat_spy >= 0.5:
        return "moderate"
    return "low"
