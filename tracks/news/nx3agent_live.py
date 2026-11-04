"""NX3-AGENT shadow book (research/prereg/NX3AGENT.md), from 2026-11-05. Ledger only: no orders.

After the close, Gemini (temperature 0) is shown every event the NX3 tree fired that day (the same events NX3-ALL
trades), with the tree's side and confidence and the numbers logged at decision time, and answers TAKE, SKIP or
FLIP for each. TAKE trades the tree's side, SKIP does not trade, FLIP trades the other side (a short only where the
stock was easy to borrow at decision time). Sizes are NX3-ALL's confidence rule on the tree's fired events, so the
agent only changes the decision. An unreadable or missing answer is logged as missing and counts as TAKE.
The headline text is not shown to the agent (v1: the rows keep no text); only the logged numbers.

Settlement logs the next-close net of the tree's trade, of the agent's trade, and of the same-day random control
(default_rng(SEED + yyyymmdd + 300), as many of the day's decided events as the tree fired, tree sides and sizes).

Kinds nx3agent_day and nx3agent_settle.
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

FIRST_DAY = "2026-11-05"
BOOK = 5000.0
SEED_ADD = 300
MODELS = ("gemini-3-flash-preview", "gemini-3.5-flash-lite")
KEEP = ["symbol", "t", "side", "e", "prev_c", "dv20"]
FLAGS = ["f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger", "f_fda",
         "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive"]
SYSTEM = ("You are the final decision step for paper stock trades. A model trained on past news fired each numbered "
          "event below and proposes a trade: side (long or short) and confidence (1 = just at its firing line, higher "
          "= stronger). For each you see the kinds of news found, the average positive and negative tone of the "
          "headlines (0 to 1), the number of articles, the stock's 20-day volatility of daily returns, its price, "
          "minutes after the 9:30 ET open the news came (negative = before the open), and the stock's move from the "
          "previous close to the news time. Decide TAKE (trade the proposed side), SKIP (no trade) or FLIP (trade the "
          "opposite side). Answer only with lines 'number: TAKE', 'number: SKIP' or 'number: FLIP'.")
PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()
ACTIONS = ("TAKE", "SKIP", "FLIP")


def events(decisions, cut):
    """The tree's fired events in time order with side, confidence, size w (NX3-ALL rule) and a borrow flag."""
    dec = sorted([x for x in decisions if x.get("status") == "decided"], key=lambda x: (pd.Timestamp(x["t"]), x["symbol"]))
    rows = []
    for x in dec:
        e = x.get("e")
        if e is None or not x.get("fired"):
            continue
        side = 1 if e > 0 else -1
        if side < 0 and not x.get("etb"):
            continue
        rows.append(dict(x, side=side, conf=abs(e) / cut))
    w = S.conf_sizes([r["conf"] for r in rows]) if rows else np.zeros(0)
    for i, (r, wi) in enumerate(zip(rows, w)):
        r.update(i=i, w=float(wi))
    return dec, rows


def text(rows):
    out = []
    for n, r in enumerate(rows, 1):
        x = r.get("inputs") or {}
        kinds = [f[2:] for f in FLAGS if x.get(f)] or ["none"]
        num = lambda k, nd=3: None if x.get(k) is None else round(float(x[k]), nd)    # noqa: E731
        out.append(f"{n}. {'long' if r['side'] > 0 else 'short'}, confidence {r['conf']:.2f}, news {'/'.join(kinds)}, "
                   f"articles {num('n_articles', 0)}, positive tone {num('fb_pos')}, negative tone {num('fb_neg')}, "
                   f"20-day volatility {num('vol20', 4)}, price {round(float(r.get('prev_c') or 0), 2)}, minutes from "
                   f"open {num('mins_open', 0)}, move to news {num('r_open', 4)}")
    return "\n".join(out)


def parse(answer, n):
    """TAKE/SKIP/FLIP per event 1..n; None where missing or malformed."""
    out = [None] * n
    for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*\**\s*(TAKE|SKIP|FLIP)\b", answer or "", re.M | re.I):
        i = int(m.group(1))
        if 1 <= i <= n:
            out[i - 1] = m.group(2).upper()
    return out


def gemini(model, user):
    key = os.environ["GEMINI_API_KEY"].strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": {"temperature": 0}}
    j = L._post(url, {"x-goog-api-key": key}, body, "gemini")
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts), j.get("usageMetadata", {})


def ask(rows, call=gemini):
    base = {"prompt_sha256": PROMPT_SHA, "cost_usd": 0.0}
    if not rows:
        return base | {"model": None, "actions": []}
    if call is gemini and not os.environ.get("GEMINI_API_KEY"):
        return base | {"model": None, "error": "no GEMINI_API_KEY", "actions": [None] * len(rows)}
    err = None
    for model in MODELS:
        try:
            answer, usage = call(model, text(rows))
        except Exception as ex:
            err = f"{model}: {type(ex).__name__}: {str(ex)[:200]}"
            continue
        acts = parse(answer, len(rows))
        return base | {"model": model, "answer": answer, "usage": usage, "actions": acts,
                       "missing": sum(a is None for a in acts)}
    return base | {"model": None, "error": err, "actions": [None] * len(rows)}


def close_day(root, decisions, d, seed, cut, call=gemini):
    """After the close: the agent's answers for the day's fired events (kind nx3agent_day)."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nx3agent_day", day_s):
        return None
    dec, rows = events(decisions, cut)
    res = ask(rows, call)
    acts = res.pop("actions")
    ctrl = []
    if rows and len(dec) >= len(rows):
        k = np.random.default_rng(seed + SEED_ADD).choice(len(dec), size=len(rows), replace=False)
        ctrl = [dict(dec[j], side=r["side"], i=r["i"], w=r["w"]) for j, r in zip(k, rows)]
    keep = KEEP + ["i", "w", "conf", "etb"]
    rec = {"day": day_s, "fire_cut": cut, "fired": len(rows), "llm": res,
           "events": [{k: r.get(k) for k in keep} | {"action": a} for r, a in zip(rows, acts)],
           "control": [{k: c.get(k) for k in KEEP + ["i", "w"]} for c in ctrl]}
    return led.write_json("nx3agent_day", day_s, f"nx3agent/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))


def settle(root, api, cal, today):
    """Settles every nx3agent_day whose exit session has closed (kind nx3agent_settle): per fired event the tree's
    net, the agent's action and net (missing = TAKE, FLIP = the other side, a flipped short only where easy to
    borrow), and the control's."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nx3agent_day"]:
        if led.has("nx3agent_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, role="tree") for p in rec["events"]] + [dict(c, role="control") for c in rec["control"]]
        flips = [dict(p, role="flip", side=-p["side"]) for p in rec["events"]
                 if p.get("action") == "FLIP" and (-p["side"] > 0 or p.get("etb"))]
        res = {"day": e["date"], "trades": [], "agent_minus_tree": None}
        if rows:
            ev = pd.DataFrame(rows + flips).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                tree = t[t["role"] == "tree"].set_index("i")
                flip = t[t["role"] == "flip"].set_index("i")
                ctrl = t[t["role"] == "control"].set_index("i")
                act = {p["i"]: p.get("action") for p in rec["events"]}
                agent = {}
                for i, r in tree.iterrows():
                    a = act.get(i)
                    if a == "SKIP":
                        continue
                    if a == "FLIP":
                        if i in flip.index:
                            agent[i] = float(flip.loc[i, "net"])
                        continue
                    agent[i] = float(r["net"])
                w = tree["w"].astype(float)
                tree_pnl = float((w * tree["net"]).sum())
                agent_pnl = float(sum(w[i] * v for i, v in agent.items()))
                both = tree.index.intersection(ctrl.index)
                res |= {"tree_sized_net": tree_pnl, "agent_sized_net": agent_pnl, "agent_minus_tree": agent_pnl - tree_pnl,
                        "agent_minus_control": agent_pnl - float((w[both] * ctrl.loc[both, "net"]).sum()),
                        "book_pnl": agent_pnl * BOOK, "taken": len(agent), "fired": len(tree)}
                cols = ["symbol", "role", "i", "w", "side", "t", "entry", "exit_ac", "ret", "net"]
                res["trades"] = t[[x for x in cols if x in t]].to_dict("records")
                res["actions"] = {str(i): act.get(i) for i in tree.index}
        out.append(led.write_json("nx3agent_settle", e["date"], f"nx3agent/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
