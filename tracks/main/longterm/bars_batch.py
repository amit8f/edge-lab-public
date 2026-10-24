"""Fetch SIP bars for a batch of symbols in the three adjustments M4 needs and turn them into feature
rows and split points. One batch is ~500 symbols; results are cached per batch with atomic writes."""
from pathlib import Path

import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main.longterm.prices import align, features, split_points


def fetch_three(symbols, start, end):
    h = alpaca_headers()
    out = {}
    for adj in ("all", "raw", "split"):
        out[adj] = to_frames(fetch_daily_bars(symbols, start, end, adjustment=adj, headers=h))
    return out


def process(symbols, start, end, dates, ends, fetch=fetch_three):
    """Feature rows and split points for `symbols` (no caching)."""
    fr = fetch(symbols, start, end)
    feats, rts = [], []
    for sym in symbols:
        if sym not in fr["all"] or sym not in fr["raw"] or sym not in fr["split"]:
            continue
        d = align(fr["all"][sym], fr["raw"][sym], fr["split"][sym])
        if d.empty:
            continue
        rts.append(split_points(sym, d))
        f = features(sym, d, dates, ends)
        if len(f):
            feats.append(f)
    f = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    r = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
    return f, r, len(fr["all"])


def run_batches(symbols, start, end, dates, ends, cache_dir, size=500, fetch=fetch_three):
    """Process symbols in batches of `size`, caching each batch's two outputs as parquet."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    feats, rts, with_bars = [], [], 0
    symbols = sorted(set(symbols))
    for b in range(0, len(symbols), size):
        ff, fr = cache / f"feat_{b // size:04d}.parquet", cache / f"rtab_{b // size:04d}.parquet"
        if ff.exists() and fr.exists():
            f, r = pd.read_parquet(ff), pd.read_parquet(fr)
            n = r["symbol"].nunique()
        else:
            f, r, n = process(symbols[b:b + size], start, end, dates, ends, fetch)
            for df, path in ((f, ff), (r, fr)):
                tmp = path.with_suffix(".tmp")
                df.to_parquet(tmp)
                tmp.replace(path)
        with_bars += n
        feats.append(f)
        rts.append(r)
        print(f"[bars] batch {b // size + 1}/{-(-len(symbols) // size)}: {len(f)} feature rows")
    feat = pd.concat([f for f in feats if len(f)], ignore_index=True)
    rtab = pd.concat([r for r in rts if len(r)], ignore_index=True)
    return feat, rtab, with_bars


def assets_all():
    """Alpaca us_equity assets, active and inactive, classified."""
    from core.instruments import classify
    rows = []
    for status in ("active", "inactive"):
        rows += alpaca.get(f"{alpaca.TRADE}/assets", {"status": status, "asset_class": "us_equity"})
    return classify(pd.DataFrame(rows))


def run_live(symbols, start, end, t, keep=None, size=500, fetch=fetch_three, boom=None, panel=None):
    """Today's feature rows (signal date t, no forward returns), split points, and all-adjusted open/close
    frames for every symbol (small: ~1.5 years of daily bars), in batches of `size`. When `boom` is a
    list, the short-term model's feature rows for t (core.shortterm) are appended to it; when `panel` is a
    list, the weekly panel's row for t (tracks.main.weekly_panel.symbol_rows, no forward returns) is."""
    from core import shortterm
    from tracks.main import weekly_panel
    feats, rts, closes = [], [], {}
    symbols = sorted(set(symbols))
    for b in range(0, len(symbols), size):
        batch = symbols[b:b + size]
        fr = fetch(batch, start, end)
        for sym in batch:
            if sym in fr["all"]:
                closes[sym] = fr["all"][sym][["o", "c"]]
            if sym not in fr["raw"] or sym not in fr["split"] or sym not in fr["all"]:
                continue
            d = align(fr["all"][sym], fr["raw"][sym], fr["split"][sym])
            if d.empty:
                continue
            rts.append(split_points(sym, d))
            f = features(sym, d, [t], {})
            if len(f):
                feats.append(f)
            if boom is not None:
                bf = shortterm.features(sym, d, [t], {})
                if len(bf):
                    boom.append(bf)
            if panel is not None:
                panel += weekly_panel.symbol_rows(sym, d, [t], {h: {} for h in weekly_panel.HOLDS})
        print(f"[bars] live batch {b // size + 1}/{-(-len(symbols) // size)}")
    feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    rtab = pd.concat(rts, ignore_index=True) if rts else pd.DataFrame(columns=["symbol", "date", "R"])
    return feat, rtab, closes
