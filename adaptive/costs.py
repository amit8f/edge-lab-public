"""Plan v3 cost model (PLAN_V3 section 3, addition 5): what one fill really costs at a $5k account.

Every paper fill pays, on top of Alpaca's simulated price:
- half the measured quoted spread (V4: 0.41% round trip for main picks, 0.65% for penny), each side;
- an IBKR Pro tiered-style commission: $0.0035 a share, at least $0.35 and at most 1% of the trade value;
- on sells only, the SEC fee and FINRA's trading activity fee;
- on shorts, borrow at the logged IBKR rate (or the track default) for each calendar day held.
"""

SPREAD_ROUND_TRIP = {"main": 0.0041, "penny": 0.0065}
COMMISSION_PER_SHARE = 0.0035
COMMISSION_MIN = 0.35
COMMISSION_MAX_FRAC = 0.01
SEC_FEE_RATE = 0.0000278          # of sale value (rate since 2025-05-14); refreshed when the SEC changes it
FINRA_TAF_PER_SHARE = 0.000166
FINRA_TAF_MAX = 8.30
BORROW_DEFAULT = {"main": 0.03, "penny": 0.20}


def commission(shares, price):
    value = abs(shares) * price
    c = max(COMMISSION_MIN, abs(shares) * COMMISSION_PER_SHARE)
    return min(c, COMMISSION_MAX_FRAC * value) if value > 0 else 0.0


def regulatory(shares, price, is_sale):
    if not is_sale:
        return 0.0
    return abs(shares) * price * SEC_FEE_RATE + min(FINRA_TAF_MAX, abs(shares) * FINRA_TAF_PER_SHARE)


def fill_cost(book, shares, price, is_sale, spread=None):
    """Dollars one fill costs: half spread + commission + sale fees."""
    rt = SPREAD_ROUND_TRIP[book] if spread is None else spread
    return abs(shares) * price * rt / 2 + commission(shares, price) + regulatory(shares, price, is_sale)


def borrow_cost(book, notional, days, rate=None):
    """Borrow on a short for `days` calendar days at an annual `rate` (IBKR's logged fee when known)."""
    r = BORROW_DEFAULT[book] if rate is None else rate
    return abs(notional) * r * days / 365
