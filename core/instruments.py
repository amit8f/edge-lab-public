# ORIGIN: copied from breakout-scanner scanner/instruments.py at commit 111b71d on 2026-10-11.
"""Instrument-type classification of Alpaca assets (name + symbol conventions).

Each asset gets exactly one `excl_reason` (first matching rule wins, in RULES order) or
None if it looks like common stock. Name rules are applied to the Alpaca `name`; symbol rules
catch assets with blank names (mostly inactive) using exchange suffix conventions
(`.WS`/`.U`/`.RT`/`.PR*`, and NASDAQ 5-letter W/U/R suffixes).
"""
import re

import numpy as np
import pandas as pd

LISTED = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}

_I = re.IGNORECASE
RULES = [
    ("adr", re.compile(r"\bamerican depositary\b|\bADRs?\b|\bADSs?\b|\bdepositary receipts?\b", _I)),
    ("warrant", re.compile(r"\bwarrants?\b|\bwts?\b", _I)),
    ("right", re.compile(r"\brights?\b(?! ?(?:bio|side|way|scale|now|point))|\bcontingent value\b|\bCVR\b", _I)),
    ("unit", re.compile(r"\bunits?\b(?!.*\b(?:limited partner|common units representing)\b)(?=.*(?:consisting|each|warrant|right|acquisition|capital|holdings|merger|spac|corp|inc|ltd|\)|$))", _I)),
    ("subunit", re.compile(r"\bsub-?units?\b", _I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpref\b|\bpfd\b|\bdepositary shares?\b|\bdep(?:ositary)? shs\b|\bdepositary shares?,? each|\bperpetual\b|\bcumulative\b|\bseries [a-z]\b.*%|%.*\bseries [a-z]\b|\bfixed[- ]to[- ]floating\b|\bfixed[- ]rate reset\b", _I)),
    ("debt", re.compile(r"\bnotes? due\b|\bsenior notes?\b|\bsubordinated\b|\bdebentures?\b|\bbonds?\b|%\s*notes?\b|\bbaby bond", _I)),
    ("etf_fund", re.compile(
        r"\bETFs?\b|\bETNs?\b|\bexchange[- ]traded\b|\bfunds?\b|\bi ?shares\b|\bSPDR\b|\bproshares\b|\bdirexion\b|"
        r"\binvesco\b|\bvanguard\b|\bwisdomtree\b|\bglobal x\b|\bvaneck\b|\bfirst trust\b|\bschwab\b.*\b(?:index|etf)|"
        r"\bindex\b|\bportfolio\b|\bstrategy shares\b|\bdaily (?:bull|bear|2x|3x|inverse|target)|\b[23]x\b|\binverse\b|\bleveraged\b|"
        r"\bbuffer\b|\bincome (?:shares|trust)\b|\bmunicipal\b|\bmuni\b|\bcommodity\b|\bbitcoin trust\b|\bether(?:eum)? trust\b|"
        r"\bgold trust\b|\bsilver trust\b|\boil fund\b|\bvolatility\b|\bVIX\b|\bgrayscale\b|\bclosed[- ]end\b|\bcef\b|"
        r"\btarget (?:term|date)\b|\bmunicipal\b|\bnuveen\b|\bpimco\b|\bblackrock\b.*\b(?:trust|income|municipal)\b|"
        r"\beaton vance\b|\bcalamos\b|\bcohen & steers\b|\bgabelli\b|\baberdeen\b|\btekla\b|\bkayne\b|\bflaherty\b|\bdoubleline\b|"
        r"\bmacquarie\b.*\bincome\b|\bdividend (?:and|&) income\b|\bopportunit(?:y|ies) trust\b|\bhigh income\b|\bhigh yield\b|"
        r"\bcredit (?:strategies|income|fund)\b|\bsector\b.*\bshares\b|\btrust (?:ii|iii|iv|v|vi)\b|\bbdc income\b|"
        r"ADRhedged|\bphysical\b|\bgraniteshares\b|\bmarket vectors\b|\bSTRATS\b|\bsynthetic fixed|\bcorts\b|"
        r"\bsaturns?\b|\bpplus\b|\bescrow\b|^ESC |\bOP,? L\.?P\.?\b|\bL\.P\.,? series\b|\bsolana trust\b|\bcrypto\b.*\btrust\b|"
        r"\bcarbon allowance\b|\bequity strategy\b|\bunconstrained\b|\bdynamic\b.*\b(?:income|allocation)\b|\bdimensions group\b", _I)),
    ("when_issued", re.compile(r"\bwhen[- ]issued\b|\bwhen issued\b|\bwi\b$", _I)),
]

# Tickers with a class-share dot (BRK.B, BF.B, HEI.A) are common stock; other dot-suffixes are not.
_SYM_DOT_EXCL = [
    ("warrant", re.compile(r"\.(?:WS|W|WT|WSA|WSB)(?:\.[A-Z])?$")),
    ("unit", re.compile(r"\.U$|\.UN$")),
    ("right", re.compile(r"\.(?:RT|R|RTW)$")),
    ("preferred", re.compile(r"\.P(?:R)?[A-Z]?$|\.PR[A-Z]*$|p[A-Z]?$")),
    ("when_issued", re.compile(r"\.WI$|\.WD$")),
]
_NASDAQ5 = [("warrant", "W"), ("unit", "U"), ("right", "R"), ("preferred", "P")]


def classify(assets):
    """Returns a copy of `assets` with columns `base_symbol` (without any `_DELISTED` suffix),
    `excl_reason` (None = common-stock-like) and `name_blank`."""
    a = assets.copy()
    a["name"] = a["name"].fillna("").str.strip()
    a["name_blank"] = a["name"] == ""
    a["base_symbol"] = a["symbol"].str.replace(r"_DELISTED$", "", regex=True)
    reasons = pd.Series([None] * len(a), index=a.index, dtype=object)

    reasons[~a["exchange"].isin(LISTED)] = "otc_or_unlisted"
    reasons[a["base_symbol"].str.contains(r"TEST|ZVZZT|ZXZZT|ZWZZT|NTEST|ATEST", regex=True) & reasons.isna()] = "test_symbol"
    reasons[a["base_symbol"].str.contains(r"\d") & reasons.isna()] = "non_standard_symbol"  # CUSIP-like escrow lines

    for reason, rx in RULES:
        m = reasons.isna() & a["name"].str.contains(rx)
        reasons[m] = reason

    # MLP / LP common units are operating businesses and trade like common stock: kept, tagged.
    a["is_lp_units"] = a["name"].str.contains(r"limited partner|common units|\bL\.?P\.? units?\b|\bLP common", case=False)
    reasons[(reasons == "unit") & a["is_lp_units"]] = None

    for reason, rx in _SYM_DOT_EXCL:
        m = reasons.isna() & a["base_symbol"].str.contains(rx)
        reasons[m] = reason
    # NASDAQ 5-letter conventions only for blank-name assets (named ones are handled by name rules,
    # and plenty of 5-letter common stocks legitimately end in W/U/R, e.g. BLDR is 4 letters but
    # ZUMZW would not be).
    blank5 = reasons.isna() & a["name_blank"] & (a["exchange"] == "NASDAQ") & (a["base_symbol"].str.len() == 5)
    for reason, suf in _NASDAQ5:
        m = blank5 & a["base_symbol"].str.endswith(suf)
        reasons[m] = reason
    # ARCA-listed assets are overwhelmingly ETPs; a blank-named ARCA asset cannot be shown to be common stock.
    reasons[reasons.isna() & a["name_blank"] & (a["exchange"] == "ARCA")] = "etf_fund"

    a["excl_reason"] = reasons
    return a


def symbol_convention_reason(sym):
    """Instrument type implied by the ticker alone (for symbols with no Alpaca name, e.g.
    DERA-only tickers): dot suffixes, and NASDAQ-style 5-letter W/U/R suffixes."""
    for reason, rx in _SYM_DOT_EXCL:
        if rx.search(sym):
            return reason
    if len(sym) == 5 and "." not in sym:
        for reason, suf in _NASDAQ5:
            if sym.endswith(suf):
                return reason
    return None
