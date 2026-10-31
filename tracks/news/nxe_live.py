"""NXE-EXL: the Gemini exit agent's forward log (research/prereg/NXE.md, level 3), from 2026-11-02. Ledger
only: no orders, and NX3's own exit is unchanged.

After the close, for each of the day's NX3 picks, Gemini (temperature 0) gets the side, NX3's confidence, the
14-session ATR and the last 5 sessions' daily bars before the decision day (SIP, adjustment=all), all as
fractions of the entry price, and returns a stop and a target in ATRs for NXE's 5-session frame. Answers
outside [0.5, 6] ATR or missing are logged as missing. The prompt below is fixed before the first call and
logged by its sha256, with the model id, the raw answer and the token counts (free tier: cost 0).

Kind nxe_exl_day. Settlement (NXE's frame_exit on the 5 sessions after the decision) comes with its first
exit dates.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import llm as L
from tracks.news import nxe_study as E

FIRST_DAY = "2026-11-02"
LOOKBACK_DAYS = 40
CLIP = (0.5, 6.0)
MODELS = ("gemini-3-flash-preview", "gemini-3.5-flash-lite")
SYSTEM = ("You set exits for paper stock trades opened today by a news-driven model. Each numbered block is one "
          "trade: side (long or short), the model's confidence (1 = just at its firing line, higher = "
          "stronger), the stock's 14-day average true range (ATR) as a share of the entry price, and the last 5 "
          "daily bars before today as open/high/low/close relative to the entry price (1.00 = entry). The trade "
          "is held for at most 5 more sessions. Choose a stop-loss and a take-profit distance for each, both in "
          "ATRs from the entry, each between 0.5 and 6. Answer only with lines 'number: stop, target', e.g. "
          "'1: 1.5, 3'.")
PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()


def context(daily, d, entry):
    """(atr as a share of entry, [5 bars relative to entry]) from a daily frame (o, h, l, c by date) up to d - 1."""
    g = daily[daily.index < pd.Timestamp(d)].tail(E.ATR_N + 1)
    if len(g) < E.ATR_N + 1 or not entry > 0:
        return None, []
    a = E.atr(g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy())
    bars = [[round(float(r[k]) / entry, 4) for k in ("o", "h", "l", "c")] for _, r in g.tail(5).iterrows()]
    return a / entry, bars


def text(trades):
    out = []
    for n, x in enumerate(trades, 1):
        bars = "; ".join("/".join(f"{v:.3f}" for v in b) for b in x["bars"])
        out.append(f"{n}. {'long' if x['side'] > 0 else 'short'}, confidence {x['conf']:.2f}, ATR "
                   f"{x['atr_frac']:.4f}, last 5 days {bars}")
    return "\n".join(out)


def parse(answer, n):
    """[(stop, target)] in ATRs per trade; None where missing or outside CLIP."""
    out = [None] * n
    num = r"(\d+(?:\.\d+)?)"
    for m in re.finditer(rf"^\s*(\d+)\s*[:.)-]\s*\**\s*{num}\s*(?:ATRs?)?\s*[,;/ ]\s*{num}", answer or "", re.M | re.I):
        i, s, t = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if 1 <= i <= n and CLIP[0] <= s <= CLIP[1] and CLIP[0] <= t <= CLIP[1]:
            out[i - 1] = (s, t)
    return out


def gemini(model, user):
    key = os.environ["GEMINI_API_KEY"].strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": {"temperature": 0}}
    j = L._post(url, {"x-goog-api-key": key}, body, "gemini")
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts), j.get("usageMetadata", {})


def ask(trades, call=gemini):
    base = {"prompt_sha256": PROMPT_SHA, "cost_usd": 0.0}
    if not trades:
        return base | {"model": None, "levels": []}
    if call is gemini and not os.environ.get("GEMINI_API_KEY"):
        return base | {"model": None, "error": "no GEMINI_API_KEY", "levels": [None] * len(trades)}
    err = None
    for model in MODELS:
        try:
            answer, usage = call(model, text(trades))
        except Exception as ex:
            err = f"{model}: {type(ex).__name__}: {str(ex)[:200]}"
            continue
        lv = parse(answer, len(trades))
        return base | {"model": model, "answer": answer, "usage": usage, "levels": lv,
                       "missing": sum(x is None for x in lv)}
    return base | {"model": None, "error": err, "levels": [None] * len(trades)}


def close_day(root, api, decisions, d, cut, call=gemini):
    """After the close: Gemini's stop and target for each of the day's NX3 picks (kind nxe_exl_day)."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nxe_exl_day", day_s):
        return None
    picks = [x for x in decisions if x.get("status") == "decided" and x.get("side")]
    syms = sorted({x["symbol"] for x in picks})
    first = (pd.Timestamp(d) - pd.Timedelta(days=LOOKBACK_DAYS)).date()
    daily = api.daily(syms, first, pd.Timestamp(d).date(), "all") if syms else {}
    trades = []
    for x in picks:
        entry = float(x.get("entry_iex") or np.nan)
        a, bars = context(daily[x["symbol"]], d, entry) if x["symbol"] in daily else (None, [])
        if a is None:
            continue
        trades.append({"symbol": x["symbol"], "t": x["t"], "side": x["side"], "entry_iex": entry,
                       "conf": abs(x["e"]) / cut, "atr_frac": a, "bars": bars})
    res = ask(trades, call)
    for x, lv in zip(trades, res.pop("levels")):
        x["stop_atr"], x["target_atr"] = (lv if lv else (None, None))
    rec = {"day": day_s, "picks": len(picks), "trades": trades, "llm": res}
    return led.write_json("nxe_exl_day", day_s, f"nxe_exl/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))
