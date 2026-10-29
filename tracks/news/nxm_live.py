"""NXM shadow books (research/prereg/NXM.md "What happens next"): one $5,000 ledger book per arm that passed
(LGBM, LIN, MLP, ENS), on NX3's live events, plus NXR's passing arms (XGBR, RFR; research/prereg/NXR.md) from
2026-10-30. Ledger only: no orders, no extra API calls.

Each arm reads the inputs NX3's scorer logged for every decided event at its decision time (nx3_live.decide),
so its effect is the one it would have had live. After the close, per arm: e = N(x) - N(x0) with the arm's
version 1 models, fired at its frozen cut, up to 3 longs and 3 shorts in time order (a short only where the
stock was easy to borrow at decision time), and the same-day random control (default_rng(SEED + yyyymmdd +
arm number)). Logged as kind nxm_day; settled with NX3's code as kind nxm_settle. Forward test per arm: NX3's
(book minus control, NW t >= 2 at 100 closed trades or 2027-12-31).
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.news import nx1_study as N
from tracks.news import nx3_study as S

RES = Path("research/results/NXM")
RES_R = Path("research/results/NXR")
NXM_ARMS, NXR_ARMS = ("LGBM", "LIN", "MLP", "ENS"), ("XGBR", "RFR")
ARMS = NXM_ARMS + NXR_ARMS              # an arm's control seed adds its number here (1-6)
FIRST = {a: "2026-10-29" for a in NXM_ARMS} | {a: "2026-10-30" for a in NXR_ARMS}
FIRST_DAY = min(FIRST.values())         # 2026-10-29: the first session whose decisions log easy-to-borrow
SLIM = ["symbol", "t", "side", "e", "prev_c", "dv20"]


class Arm:
    def __init__(self, base, news, cut):
        self.news, self.cut = news, float(cut)

    def effect(self, X):
        return self.news.predict(X) - self.news.predict(S.no_news(X))


class Ens:
    def __init__(self, members, cut):
        self.members, self.cut = members, float(cut)

    def effect(self, X):
        return np.mean([m.effect(X) / m.cut for m in self.members], axis=0)


def _reader(res, meta_name):
    meta = json.loads((res / meta_name).read_text())

    def read(key):
        f = res / meta[key]["file"]
        b = f.read_bytes()
        if hashlib.sha256(b).hexdigest() != meta[key]["sha256"]:
            raise RuntimeError(f"{f} does not match its sha256")
        return pickle.loads(b)
    return read, meta


def load(res=RES, res_r=RES_R):
    """{arm: model with .effect and .cut} from the version 1 files, each checked against its sha256."""
    read, _ = _reader(res, "nxm_v1_meta.json")
    out = {}
    for a in NXM_ARMS:
        m = read(a)
        if a == "ENS":
            out[a] = Ens([Arm(**read(f"member_{k}")) for k in m["members"]], m["cut"])
        else:
            out[a] = Arm(**m)
    read_r, _ = _reader(res_r, "nxr_v1_meta.json")
    for a in NXR_ARMS:
        out[a] = Arm(**read_r(a))
    return out


def shas():
    m = json.loads((RES / "nxm_v1_meta.json").read_text()) | json.loads((RES_R / "nxr_v1_meta.json").read_text())
    return {a: m[a]["sha256"] for a in ARMS if a in m}


def book(decided, model, seed):
    """One arm's picks and control from the day's decided events (dicts with inputs, t, symbol, etb)."""
    dec = sorted(decided, key=lambda x: (pd.Timestamp(x["t"]), x["symbol"]))
    if not dec:
        return [], []
    X = np.array([[x["inputs"][k] if x["inputs"][k] is not None else np.nan for k in S.INPUTS] for x in dec], float)
    e = model.effect(X)
    picks, n = [], {1: 0, -1: 0}
    for x, ei in zip(dec, e):
        if not (abs(ei) >= model.cut and abs(ei) > 0):
            continue
        side = 1 if ei > 0 else -1
        if n[side] >= S.PICKS or (side < 0 and not x.get("etb")):
            continue
        n[side] += 1
        picks.append(dict(x, side=side, e=float(ei)))
    ctrl = []
    if picks and len(dec) >= len(picks):
        k = np.random.default_rng(seed).choice(len(dec), size=len(picks), replace=False)
        ctrl = [dict(dec[i], side=p["side"]) for i, p in zip(k, picks)]
    return picks, ctrl


def close_day(root, decisions, d, seed, models=None):
    """After the close: every arm's book for day d (ledger kind nxm_day). None before FIRST_DAY or if done."""
    day_s = f"{pd.Timestamp(d):%Y-%m-%d}"
    led = Ledger(root)
    if day_s < FIRST_DAY or led.has("nxm_day", day_s):
        return None
    models = models or load()
    dec = [x for x in decisions if x["status"] == "decided"]
    arms = {}
    for a, m in models.items():
        if day_s < FIRST.get(a, FIRST_DAY):
            continue
        p, c = book(dec, m, seed + ARMS.index(a) + 1)
        arms[a] = {"fire_cut": m.cut, "picks": [{k: x.get(k) for k in SLIM} for x in p],
                   "control": [{k: x.get(k) for k in SLIM} for x in c],
                   "fired": None if not dec else int(np.sum(np.abs(m.effect(np.array(
                       [[x["inputs"][k] if x["inputs"][k] is not None else np.nan for k in S.INPUTS] for x in dec],
                       float))) >= m.cut))}
    sh = shas()
    rec = {"day": day_s, "models_sha256": {a: sh[a] for a in arms if a in sh},
           "decided": len(dec), "arms": arms}
    return led.write_json("nxm_day", day_s, f"nxm/days/{day_s}.json", json.loads(json.dumps(rec, default=str)))


def settle(root, api, cal, today):
    """Settles every nxm_day whose exit session has closed (kind nxm_settle), with NX3's settlement code."""
    led = Ledger(root)
    caldates = pd.DatetimeIndex(cal["date"])
    out = []
    for e in [x for x in led.entries() if x["kind"] == "nxm_day"]:
        if led.has("nxm_settle", e["date"]):
            continue
        d = pd.Timestamp(e["date"])
        k = caldates.searchsorted(d)
        if k + 1 >= len(caldates) or caldates[k + 1] > today:
            continue
        rec = json.loads((Path(root) / e["file"]).read_text())
        rows = [dict(p, arm=a, role=role) for a, b in rec["arms"].items() for role in ("picks", "control")
                for p in b[role]]
        res = {"day": e["date"], "arms": {a: {"trades": [], "pick_minus_control": None} for a in rec["arms"]}}
        if rows:
            ev = pd.DataFrame(rows).assign(d=d, next=caldates[k + 1])
            ev["t"] = pd.to_datetime(ev["t"])
            t = api.settle_trades(ev, cal)
            if len(t):
                t["net"] = N.net(t)
                for a, g in t.groupby("arm"):
                    p, c = g[g["role"] == "picks"], g[g["role"] == "control"]
                    res["arms"][a] = {
                        "trades": g[["symbol", "role", "side", "t", "entry", "exit_ac", "ret", "spy_ret", "net"]]
                        .to_dict("records"),
                        "pick_minus_control": float(p["net"].mean() - c["net"].mean()) if len(p) and len(c) else None,
                        "book_pnl": float((p["net"] * N.NOTIONAL).sum())}
        out.append(led.write_json("nxm_settle", e["date"], f"nxm/settled/{e['date']}.json",
                                  json.loads(json.dumps(res, default=str))))
    return out
