"""NX3-ALL shadow book (research/prereg/NX3ALL.md) and the NXS-LLM sizer log (research/prereg/NXS.md), from
2026-11-02. Ledger only: no orders.

After the close, from NX3's own decided events (nx3_live.decide logs every one with its e, fired flag and
easy-to-borrow flag): every fired event is a pick (a short only where the stock was easy to borrow at
decision time), sized by NX3-ALL's confidence rule (5% of the $5,000 book at |e|/cut = 1 up to 10% at 2 or
more; a day adding up to more than the book is scaled down together). The same-day control draws as many
decided events at random (default_rng(SEED + yyyymmdd + 100)); control trade i takes pick i's side and size.

NXS-LLM: Gemini is asked once a day for every pick's size from the inputs logged at decision time only, with
the fixed prompt below (logged by its sha256, with the model id and the raw answer). An answer outside
5%-10% or missing is logged as missing and that trade keeps the rule's size.

Kinds nx3all_day and nx3all_settle (NX3's settlement code).
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import llm as L
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

FIRST_DAY = "2026-11-02"
BOOK = 5000.0
SEED_ADD = 100
SLIM = ["symbol", "t", "side", "e", "prev_c", "dv20"]
LLM_MODELS = ("gemini-3-flash-preview", "gemini-3.5-flash-lite")
LLM_SYSTEM = ("You size paper trades for a news-driven stock model. Each numbered line is one trade the model "
              "took today: side (long or short), the model's confidence (1 = just at its firing line, higher = "
              "stronger), the stock's typical daily move (20-day volatility of daily returns), minutes after the 9:30 ET "
              "open the news came (negative = before the open), the stock's move from the previous close to "
              "the news time, and how many trades came before it today. Give each trade a size between 5 and "
              "10 percent of a $5,000 book: larger where you expect the trade to be more reliable. Answer only "
              "with lines 'number: percent', e.g. '1: 7.5'.")
PROMPT_SHA = hashlib.sha256(LLM_SYSTEM.encode()).hexdigest()


def rule_sizes(conf):
    return S.conf_sizes(conf)


def book(decided, cut, seed):
    """Picks (with conf and the rule's size w) and the paired control from the day's decided events."""
    dec = sorted(decided, key=lambda x: (pd.Timestamp(x["t"]), x["symbol"]))
    picks = []
    for x in dec:
        e = x.get("e")
        if e is None or not x.get("fired"):
            continue
        side = 1 if e > 0 else -1
        if side < 0 and not x.get("etb"):
            continue
        picks.append(dict(x, side=side, conf=abs(e) / cut))
    w = rule_sizes([p["conf"] for p in picks]) if picks else np.zeros(0)
    for i, (p, wi) in enumerate(zip(picks, w)):
        p.update(i=i, w=float(wi))
    ctrl = []
    if picks and len(dec) >= len(picks):
        k = np.random.default_rng(seed).choice(len(dec), size=len(picks), replace=False)
        ctrl = [dict(dec[j], side=p["side"], i=p["i"], w=p["w"]) for j, p in zip(k, picks)]
    return picks, ctrl


def llm_text(picks):
    lines = []
    for n, p in enumerate(picks, 1):
        x = p.get("inputs") or {}
        atr = x.get("vol20")
        lines.append(f"{n}. {'long' if p['side'] > 0 else 'short'}, confidence {p['conf']:.2f}, typical daily move "
                     f"{atr if atr is None else round(float(atr), 4)}, minutes from open "
                     f"{x.get('mins_open') if x.get('mins_open') is None else round(float(x['mins_open']))}, "
                     f"move to news {x.get('r_open') if x.get('r_open') is None else round(float(x['r_open']), 4)}, "
                     f"trades before {n - 1}")
    return "\n".join(lines)


def llm_parse(text, n):
    """Sizes 1..n as fractions; missing or outside 5-10% is None."""
    out = [None] * n
    for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*\**\s*(\d+(?:\.\d+)?)\s*%?", text or "", re.M):
        i, v = int(m.group(1)), float(m.group(2))
        if 1 <= i <= n and 5.0 <= v <= 10.0:
            out[i - 1] = v / 100
    return out


def gemini(model, text):
    key = os.environ["GEMINI_API_KEY"].strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": LLM_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0}}
    j = L._post(url, {"x-goog-api-key": key}, body, "gemini")
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def ask_llm(picks, call=gemini):
    """{model, answer, sizes (fractions, scaled like the rule; None where missing)} or the error."""
    if not picks:
        return {"model": None, "answer": None, "sizes": [], "prompt_sha256": PROMPT_SHA}
    if call is gemini and not os.environ.get("GEMINI_API_KEY"):
        return {"model": None, "error": "no GEMINI_API_KEY", "sizes": [None] * len(picks), "prompt_sha256": PROMPT_SHA}
    err = None
    for model in LLM_MODELS:
        try:
            text = call(model, llm_text(picks))
        except Exception as ex:
            err = f"{model}: {type(ex).__name__}: {str(ex)[:200]}"
            continue
        raw = llm_parse(text, len(picks))
        w = np.array([r if r is not None else p["w"] for r, p in zip(raw, picks)], float)
        w = w * min(1.0, 1.0 / w.sum()) if w.sum() > 0 else w
        sizes = [float(x) if r is not None else None for x, r in zip(w, raw)]
        return {"model": model, "answer": text, "sizes": sizes, "missing": sum(r is None for r in raw),
                "prompt_sha256": PROMPT_SHA}
    return {"model": None, "error": err, "sizes": [None] * len(picks), "prompt_sha256": PROMPT_SHA}


def close_day(root, decisions, d, seed, cut, call=gemini):
    """After the close: the day's NX3-ALL book and NXS-LLM sizes (kind nx3all_day). None before FIRST_DAY or
    if already logged."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nx3all_day", day_s):
        return None
    dec = [x for x in decisions if x["status"] == "decided"]
    picks, ctrl = book(dec, cut, seed + SEED_ADD)
    llm = ask_llm(picks, call)
    keep = SLIM + ["i", "w"]
    rec = {"day": day_s, "fire_cut": cut, "decided": len(dec), "fired": sum(bool(x.get("fired")) for x in dec),
           "picks": [{k: p.get(k) for k in keep + ["conf"]} | {"w_llm": s} for p, s in zip(picks, llm["sizes"])],
           "control": [{k: c.get(k) for k in keep} for c in ctrl], "llm": {k: v for k, v in llm.items() if k != "sizes"}}
    return led.write_json("nx3all_day", day_s, f"nx3all/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))


def settle(root, api, cal, today):
    """Settles every nx3all_day whose exit session has closed (kind nx3all_settle)."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3all_day"]:
        if led.has("nx3all_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, role=role) for role in ("picks", "control") for p in rec[role]]
        res = {"day": e["date"], "trades": [], "sized_pick_minus_control": None, "llm_sized_pick_minus_control": None}
        if rows:
            ev = pd.DataFrame(rows).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                p = t[t["role"] == "picks"].set_index("i")
                c = t[t["role"] == "control"].set_index("i")
                both = p.index.intersection(c.index)
                if len(both):
                    diff = p.loc[both, "net"] - c.loc[both, "net"]
                    res["sized_pick_minus_control"] = float((p.loc[both, "w"] * diff).sum())
                    wl = p.loc[both, "w_llm"].astype(float).fillna(p.loc[both, "w"])
                    res["llm_sized_pick_minus_control"] = float((wl * diff).sum())
                res["book_pnl"] = float((p["w"] * BOOK * p["net"]).sum())
                cols = ["symbol", "role", "i", "w", "side", "t", "entry", "exit_ac", "ret", "net"]
                res["trades"] = t[[x for x in cols if x in t]].to_dict("records")
        out.append(led.write_json("nx3all_settle", e["date"], f"nx3all/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
