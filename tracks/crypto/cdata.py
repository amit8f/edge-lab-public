"""Crypto history from Binance's public bulk archive (research/prereg/CRYPTO.md). Free, no key, read only.

Stages (each resumable, files written atomically under <cache>/crypto):
  rank   the quarterly universe: the 10 USDT perpetuals with the highest quote volume in the last month of the
         previous quarter -> universe.json
  bars   15-minute USD-M candles and funding rates of every coin that is ever in the universe -> um15/<SYM>.parquet,
         funding/<SYM>.parquet
  daily  spot daily candles of BTCUSDT and ETHUSDT -> spot1d/<SYM>.parquet

Nothing after 2025-09 is requested (the phase-2 holdout).

Usage: python -m tracks.crypto.cdata --stage rank|bars|daily --cache cache
"""
import argparse
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FIRST, LAST = "2020-01", "2025-09"
SPOT_FIRST = "2017-08"
Q_FIRST, Q_LAST = "2020Q2", "2025Q3"
TOP_N = 10
THREADS = 8
STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "UST", "USTC", "EUR"}
INDEX = {"BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}
KCOLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ignore"]
KEEP = ["t", "o", "h", "l", "c", "v", "qv", "n", "tbqv"]
# used only if the archive's folder listing cannot be read
FALLBACK = ["BTC", "ETH", "BCH", "XRP", "EOS", "LTC", "TRX", "ETC", "LINK", "XLM", "ADA", "XMR", "DASH", "ZEC", "XTZ",
            "BNB", "ATOM", "ONT", "IOTA", "BAT", "VET", "NEO", "QTUM", "IOST", "THETA", "ALGO", "ZIL", "KNC", "ZRX",
            "COMP", "OMG", "DOGE", "SXP", "KAVA", "BAND", "RLC", "WAVES", "MKR", "SNX", "DOT", "DEFI", "YFI", "BAL",
            "CRV", "TRB", "RUNE", "SUSHI", "SRM", "EGLD", "SOL", "ICX", "STORJ", "BLZ", "UNI", "AVAX", "FTM", "HNT",
            "ENJ", "FLM", "TOMO", "REN", "KSM", "NEAR", "AAVE", "FIL", "RSR", "LRC", "MATIC", "OCEAN", "CVC", "BEL",
            "CTK", "AXS", "ALPHA", "ZEN", "SKL", "GRT", "1INCH", "CHZ", "SAND", "ANKR", "LUNA", "FTT", "MANA", "SHIB",
            "1000SHIB", "ICP", "GALA", "APE", "GMT", "OP", "LUNA2", "1000LUNC", "APT", "ARB", "SUI", "1000PEPE", "WLD",
            "SEI", "TIA", "ORDI", "WIF", "1000BONK", "ENA", "TON", "NOT", "PEPE", "HYPE", "TRUMP"]


def cdir(cache):
    return Path(cache) / "crypto"


def atomic(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)


# ---------- pure helpers (tested) ----------

def eligible(sym):
    """A plain USDT perpetual: no dated contract, stablecoin pair or index product."""
    if not sym.endswith("USDT") or "_" in sym:
        return False
    base = sym[:-4]
    return bool(base) and base not in STABLE and base not in INDEX


def parse_klines(raw):
    """Zip bytes of one archive file -> candles (t = bar open, UTC, naive). Old files have no header row; spot
    files switch to microseconds in 2025."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=KCOLS, dtype=str)
    df = df[df["open_time"].str.fullmatch(r"\d+")].copy()
    ts = df["open_time"].astype("int64")
    ts = np.where(ts > 10**14, ts // 1000, ts)
    df["t"] = pd.to_datetime(ts, unit="ms")
    for c in ["o", "h", "l", "c", "v", "qv", "n", "tbqv"]:
        df[c] = df[c].astype(float)
    return df[KEEP].sort_values("t").drop_duplicates("t").reset_index(drop=True)


def parse_funding(raw):
    """Zip bytes of one funding file -> (t, rate)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)
    df = df[df[0].str.fullmatch(r"\d+")]
    out = pd.DataFrame({"t": pd.to_datetime(df[0].astype("int64"), unit="ms").dt.floor("min"),
                        "rate": df[df.columns[-1]].astype(float)})
    return out.sort_values("t").drop_duplicates("t").reset_index(drop=True)


def rank_month(quarter):
    """The month whose volume ranks a quarter's universe: the last month of the quarter before."""
    return str(pd.Period(quarter, freq="Q").asfreq("M", how="start") - 1)


def top_symbols(volumes, n=TOP_N):
    """{symbol: quote volume} -> the n eligible symbols with the most volume (ties by name)."""
    ok = [(s, v) for s, v in volumes.items() if eligible(s) and v > 0]
    return [s for s, _ in sorted(ok, key=lambda x: (-x[1], x[0]))[:n]]


def quarter_of(t):
    return pd.PeriodIndex(pd.DatetimeIndex(t), freq="Q").astype(str)


# ---------- network ----------

def fetch(url):
    """Bytes, or None when the archive has no such file."""
    for k in range(4):
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r.content
        if k == 3:
            r.raise_for_status()
    return None


def kline_url(kind, sym, interval, month):
    return f"{BASE}/{kind}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{month}.zip"


def list_symbols():
    """Every symbol folder of the USD-M monthly candles (delisted coins included); the fallback list if the
    listing cannot be read."""
    out, marker = [], None
    prefix = "data/futures/um/monthly/klines/"
    try:
        while True:
            p = {"delimiter": "/", "prefix": prefix}
            if marker:
                p["marker"] = marker
            r = requests.get(LIST, params=p, timeout=60)
            r.raise_for_status()
            got = re.findall(r"<Prefix>" + re.escape(prefix) + r"([^<]+)/</Prefix>", r.text)
            out += got
            nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text)
            if "<IsTruncated>true</IsTruncated>" not in r.text or not nxt:
                break
            marker = nxt.group(1)
    except requests.RequestException as e:
        print(f"[crypto] listing failed ({e}); using the fallback list", flush=True)
        out = []
    syms = sorted(set(out)) or [f"{b}USDT" for b in FALLBACK]
    return [s for s in syms if eligible(s)]


def stage_rank(cache):
    f = cdir(cache) / "universe.json"
    uni = json.loads(f.read_text()) if f.exists() else {}
    syms = list_symbols()
    print(f"[crypto rank] {len(syms)} candidate symbols", flush=True)
    for q in pd.period_range(Q_FIRST, Q_LAST, freq="Q").astype(str):
        if q in uni:
            continue
        m = rank_month(q)

        def vol(s, m=m):
            raw = fetch(kline_url("futures/um", s, "1d", m))
            return s, (float(parse_klines(raw)["qv"].sum()) if raw else 0.0)
        with ThreadPoolExecutor(THREADS) as ex:
            vols = dict(ex.map(vol, syms))
        uni[q] = top_symbols(vols)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(uni, indent=1))
        tmp.replace(f)
        print(f"[crypto rank] {q} (by {m}): {' '.join(uni[q])}", flush=True)
    return uni


def months(first, last):
    return [str(m) for m in pd.period_range(first, last, freq="M")]


def download(kind, sym, interval, first, f):
    if f.exists():
        return
    with ThreadPoolExecutor(THREADS) as ex:
        raws = list(ex.map(lambda m: fetch(kline_url(kind, sym, interval, m)), months(first, LAST)))
    parts = [parse_klines(r) for r in raws if r]
    if not parts:
        raise RuntimeError(f"no {interval} candles for {sym}")
    df = pd.concat(parts, ignore_index=True).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    assert df["t"].max() < pd.Timestamp("2025-10-01"), "holdout data downloaded"
    atomic(df, f)
    print(f"[crypto] {sym} {interval}: {len(df)} bars {df['t'].min()} .. {df['t'].max()}", flush=True)


def stage_bars(cache):
    uni = json.loads((cdir(cache) / "universe.json").read_text())
    syms = sorted({s for v in uni.values() for s in v} | {"BTCUSDT"})
    for s in syms:
        download("futures/um", s, "15m", FIRST, cdir(cache) / "um15" / f"{s}.parquet")
        f = cdir(cache) / "funding" / f"{s}.parquet"
        if f.exists():
            continue
        with ThreadPoolExecutor(THREADS) as ex:
            raws = list(ex.map(lambda m, s=s: fetch(
                f"{BASE}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"), months(FIRST, LAST)))
        parts = [parse_funding(r) for r in raws if r]
        fr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"t": pd.to_datetime([]), "rate": []})
        atomic(fr.sort_values("t").drop_duplicates("t").reset_index(drop=True), f)
        print(f"[crypto] {s} funding: {len(fr)} payments", flush=True)


def stage_daily(cache):
    for s in ("BTCUSDT", "ETHUSDT"):
        download("spot", s, "1d", SPOT_FIRST, cdir(cache) / "spot1d" / f"{s}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rank", "bars", "daily"])
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    {"rank": stage_rank, "bars": stage_bars, "daily": stage_daily}[a.stage](a.cache)
    print(f"[crypto {a.stage}] done", flush=True)


if __name__ == "__main__":
    main()
