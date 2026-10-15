"""Kenneth French's data library: download once per cache, keep the sha256, parse daily or monthly
factor files into decimals."""
import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "strev_monthly": "F-F_ST_Reversal_Factor_CSV.zip",
}


def fetch(name, cache, get=requests.get):
    """(zip bytes, sha256) for one library file, cached under cache/french/."""
    path = Path(cache) / "french" / FILES[name]
    if not path.exists():
        r = get(BASE + FILES[name], timeout=120, headers={"User-Agent": "edge-lab research"})
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _text(raw_zip):
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def parse(raw_zip, freq):
    """First table of a French CSV as decimals. freq 'D' keys rows by date (8-digit yyyymmdd), 'M' by
    month period (6-digit yyyymm); the annual table and footers are skipped. Columns are lower-cased
    with '-' removed (mkt-rf -> mktrf)."""
    width = 8 if freq == "D" else 6
    cols, rows = None, []
    for ln in _text(raw_zip).splitlines():
        p = [x.strip() for x in ln.split(",")]
        if cols is None:
            if len(p) > 1 and p[0] == "" and all(p[1:]):
                cols = [c.lower().replace("-", "") for c in p[1:]]
            continue
        if len(p) == len(cols) + 1 and len(p[0]) == width and p[0].isdigit():
            key = pd.Timestamp(p[0]) if freq == "D" else pd.Period(f"{p[0][:4]}-{p[0][4:]}", "M")
            rows.append([key] + [float(x) / 100 for x in p[1:]])
        elif rows:
            break                              # end of the first table
    df = pd.DataFrame(rows, columns=["key"] + cols).drop_duplicates("key").set_index("key")
    return df.sort_index()
