"""Nightly "High risk, experimental" section: today's picks from the refit M10 (main) and P3 (penny)
models, with each screen's verdict. Both screens FAILED; the picks are shown because Amit asked to see
them (research/prereg/M10.md, "After the screen"), never as Buy calls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import shortterm as st
from tracks.main.longterm import analyst
from tracks.main.shortterm import panel as pn
from tracks.main.shortterm.screen_m10 import FEATURES as M10_FEATURES
from tracks.penny.shortterm.screen_p3 import FEATURES as P3_FEATURES, universe as p3_universe

RESULTS = Path(__file__).resolve().parents[3] / "research" / "results"
TOP_N = 20
SHOW = ["r21", "vol21", "maxret21", "news30", "an_up"]


def news_frames(rows):
    """(single-symbol headline days, analyst events) from fetch_window rows, or None."""
    if rows is None:
        return None
    s = rows["symbols"].fillna("")
    m = rows[(s != "") & ~s.str.contains(",")]
    ts = pd.to_datetime(m["created_at"], utc=True).dt.tz_convert("America/New_York")
    days = pd.DataFrame({"symbol": m["symbols"].to_numpy(), "day": ts.dt.tz_localize(None).dt.normalize().to_numpy()})
    acts = m[m["headline"].map(analyst.classify) != 0]
    ev = analyst.events(acts) if len(acts) else pd.DataFrame(columns=["symbol", "day", "direction", "headline"])
    return days, ev


def _with_news(rows, news):
    if news is None:
        out = rows.copy()
        for c in st.NEWS_FEATURES:
            out[c] = np.nan
        return out
    return pn.add_news(rows, *news)


def score(rows, feats, model_path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(model_path)
    r = rows.reset_index(drop=True)
    X = st.rank_within_date(r, feats)[feats].to_numpy(dtype="float32")
    return r.assign(prob=m.predict_proba(X)[:, 1]).sort_values("prob", ascending=False).reset_index(drop=True)


def _verdict(idea):
    f = RESULTS / idea / f"{idea}_screen.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    p = s["primary"]
    return {"mean": p["mean"], "t": p["t_hac"], "passes": p["passes"], "weeks": p["n_dates"],
            "calibration": s.get("calibration"), "label_rate": s["funnel_universe"]["label_rate"]}


def section(boom_rows, uni, tmap, names, news_rows=None):
    """Returns ({idea: section data}, {idea: scored universe}) for M10 and P3; an idea whose model file
    is missing is left out."""
    if not boom_rows:
        return {}, {}
    b = pd.concat(boom_rows, ignore_index=True)
    news = news_frames(news_rows)
    out, scored = {}, {}
    main = b.merge(uni[["symbol", "cik", "mcap"]], on="symbol", how="inner")
    main["lmcap"] = np.log(main["mcap"]).astype("float32")
    penny = p3_universe(b.merge(tmap[["symbol", "cik"]], on="symbol", how="inner"))
    for idea, rows, feats in (("M10", main, M10_FEATURES), ("P3", penny, P3_FEATURES)):
        path = RESULTS / idea / f"{idea}_xgb.json"
        if not path.exists() or len(rows) < TOP_N:
            continue
        s = score(_with_news(rows, news), feats, path)
        scored[idea] = s
        out[idea] = {"verdict": _verdict(idea), "universe": int(len(s)), "news": news is not None,
                     "picks": [{"rank": i + 1, "symbol": r["symbol"], "name": names.get(r["symbol"]) or "",
                                "price": float(r["price"]), "prob": float(r["prob"]),
                                **{k: (None if pd.isna(r[k]) else float(r[k])) for k in SHOW}}
                               for i, r in s.head(TOP_N).iterrows()]}
    return out, scored
