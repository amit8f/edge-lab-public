"""Fama-French 12-industry codes from SEC SIC codes. The SIC comes from each company's EDGAR submissions
record (its current code, not point-in-time). The ranges are French's Siccodes12 definitions."""
import json
from pathlib import Path

import pandas as pd

from core import sec

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FF12 = {
    1: [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    2: [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
        (3792, 3792), (3900, 3939), (3990, 3999)],
    3: [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709),
        (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    4: [(1200, 1399), (2900, 2999)],
    5: [(2800, 2829), (2840, 2899)],
    6: [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    7: [(4800, 4899)],
    8: [(4900, 4949)],
    9: [(5000, 5999), (7200, 7299), (7600, 7699)],
    10: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    11: [(6000, 6999)],
}
NAMES = {1: "Consumer nondurables", 2: "Consumer durables", 3: "Manufacturing", 4: "Energy", 5: "Chemicals",
         6: "Business equipment", 7: "Telecom", 8: "Utilities", 9: "Shops", 10: "Health", 11: "Money", 12: "Other"}


def ff12(sic):
    """French's 12-industry code for a SIC code; 12 (Other) when it matches no range or is missing."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return 12
    for k, ranges in FF12.items():
        if any(lo <= s <= hi for lo, hi in ranges):
            return k
    return 12


def sic_codes(ciks, cache_dir, get=sec.get):
    """{cik: SIC} from EDGAR submissions, cached in cache_dir/sic.json (written after every 200 lookups)."""
    path = Path(cache_dir) / "sic.json"
    known = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    todo = [int(c) for c in sorted(set(ciks)) if int(c) not in known]
    for k, c in enumerate(todo, 1):
        r = get(SUBMISSIONS.format(cik=c))
        known[c] = (r.json().get("sic") or None) if r is not None else None
        if k % 200 == 0 or k == len(todo):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(known))
            tmp.replace(path)
    return known


def industries(ciks, cache_dir, get=sec.get):
    """Series cik -> FF12 code."""
    sic = sic_codes(ciks, cache_dir, get)
    return pd.Series({c: ff12(s) for c, s in sic.items()}, dtype="int64")
