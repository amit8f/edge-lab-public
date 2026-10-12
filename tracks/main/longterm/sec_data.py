"""SEC downloads for M4: historical ticker -> CIK map (DERA insider data sets), today's ticker map, and
the companyfacts bulk archive. All requests go through SEC's fair-access limiter in core.sec."""
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from core import sec
from core.env import sec_headers
from tracks.main.longterm.fundamentals import extract

DERA = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
_PREFIX = re.compile(r"^(?:NYSE|NASDAQ|NASD|AMEX|NYSEAMERICAN|NYSE AMERICAN|NYSEARCA|ARCA|BATS|OTC|OTCBB)\s*[:/\- ]\s*", re.I)


def norm_symbol(s):
    """Filing ticker -> Alpaca style: strip exchange prefixes, '/' and '-' class separators become '.'."""
    if not isinstance(s, str):
        return None
    s = _PREFIX.sub("", s.strip().upper())
    s = re.sub(r"[/\-]", ".", s)
    if not re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?", s):
        return None
    return s


def _dates(col):
    """DERA dates are DD-MON-YYYY; fall back to pandas' parser if a quarter uses another format."""
    d = pd.to_datetime(col, format="%d-%b-%Y", errors="coerce")
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(col, errors="coerce", format="mixed")
    return d


def parse_dera_submission(tsv_bytes):
    d = pd.read_csv(io.BytesIO(tsv_bytes), sep="\t", dtype=str, usecols=["ISSUERCIK", "ISSUERTRADINGSYMBOL",
                                                                        "FILING_DATE"])
    out = pd.DataFrame({"cik": pd.to_numeric(d["ISSUERCIK"], errors="coerce"),
                        "symbol": d["ISSUERTRADINGSYMBOL"].map(norm_symbol),
                        "filed": _dates(d["FILING_DATE"])})
    return out.dropna().astype({"cik": "int64"}).drop_duplicates()


def dera_ticker_map(start_year, end_quarter, cache_dir):
    """(cik, symbol, filed) rows from every quarterly Form 3/4/5 data set from start_year Q1 through
    end_quarter (a pd.Period). Each quarter is cached as parquet."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for per in pd.period_range(pd.Period(f"{start_year}Q1"), end_quarter, freq="Q"):
        f = cache / f"dera_{per.year}q{per.quarter}.parquet"
        if not f.exists():
            r = sec.get(DERA.format(y=per.year, q=per.quarter))
            if r is None:
                print(f"[dera] {per} not published")
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = next(n for n in z.namelist() if n.upper().endswith("SUBMISSION.TSV"))
                part = parse_dera_submission(z.read(name))
            tmp = f.with_suffix(".tmp")
            part.to_parquet(tmp)
            tmp.replace(f)
        frames.append(pd.read_parquet(f))
    m = pd.concat(frames, ignore_index=True).drop_duplicates()
    print(f"[dera] {len(m)} (cik, symbol, date) rows, {m['symbol'].nunique()} symbols, {m['cik'].nunique()} CIKs")
    return m


def map_ciks(keys, tmap, max_age_days=730):
    """CIK for each (symbol, date) in keys: the CIK of the latest filing on or before the date naming the
    symbol, within max_age_days. Unmapped rows get NaN."""
    k = keys.reset_index(drop=True).copy()
    k["_row"] = range(len(k))
    t = tmap.rename(columns={"filed": "date_f"}).sort_values("date_f")
    m = pd.merge_asof(k.sort_values("date"), t, left_on="date", right_on="date_f", by="symbol",
                      direction="backward", tolerance=pd.Timedelta(days=max_age_days))
    return m.sort_values("_row")["cik"].to_numpy()


def current_ticker_map():
    """Today's SEC ticker -> CIK map (right for today's ranking only, never for history)."""
    j = sec.get(TICKERS).json()
    d = pd.DataFrame(j.values())
    return pd.DataFrame({"symbol": d["ticker"].map(norm_symbol), "cik": d["cik_str"].astype("int64"),
                         "title": d["title"]}).dropna(subset=["symbol"]).drop_duplicates("symbol")


def download_companyfacts(path):
    """Stream the companyfacts bulk zip (~1.3 GB) to `path` atomically."""
    path = Path(path)
    tmp = path.with_suffix(".part")
    sec._wait_turn()
    with requests.get(COMPANYFACTS, headers=sec_headers(), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.replace(path)
    return path


def facts_for(zip_path, ciks, out_dir, batch=500):
    """Long fact rows for the given CIKs, read from the bulk zip in batches of `batch` companies; each
    batch is written as parquet (atomic) and the concatenation of the compact rows is returned."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ciks = sorted({int(c) for c in ciks})
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for b in range(0, len(ciks), batch):
            f = out / f"facts_{b // batch:04d}.parquet"
            rows = []
            for c in ciks[b:b + batch]:
                n = f"CIK{c:010d}.json"
                if n in names:
                    rows.append(extract(json.loads(z.read(n))))
            part = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            tmp = f.with_suffix(".tmp")
            part.to_parquet(tmp)
            tmp.replace(f)
            parts.append(part)
    facts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"[facts] {len(facts)} rows for {facts['cik'].nunique() if len(facts) else 0} of {len(ciks)} CIKs")
    return facts
