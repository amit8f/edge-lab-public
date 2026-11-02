"""CHR-P live book (tracks/main/chr_live.py): synthetic data and a fake Alpaca, no network."""
import numpy as np
import pandas as pd

from core.ledger import Ledger
from tracks.main import ch_build as B
from tracks.main import chr_live as L

DAY = pd.Timestamp("2026-03-03")


def _frames(n=120, price=2.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=DAY - pd.Timedelta(days=1), periods=n)
    c = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    f = pd.DataFrame({"o": c, "h": c * 1.02, "l": c * 0.98, "c": c, "v": rng.uniform(5e5, 9e5, n)}, index=idx)
    return f


def test_universe_keeps_pennies_and_levels_come_from_prior_days_only():
    raw = {"AAA": _frames(), "BIG": _frames(price=40.0, seed=1)}
    adj = {k: v.copy() for k, v in raw.items()}
    ctx, prior = L.universe_ctx(raw, adj, DAY)
    assert set(ctx) == {"AAA"}                                       # the $40 stock is not a penny stock
    f = raw["AAA"]
    assert np.isclose(ctx["AAA"]["prev_c"], f["c"].iloc[-1])
    assert np.isclose(ctx["AAA"]["hi20"], f["h"].iloc[-20:].max())  # today's row is not in its own levels
    assert prior["AAA"]["date"].max() < DAY and len(prior["AAA"]) <= 300


def test_controls_are_deterministic_and_never_the_pick():
    pool = ["A", "B", "C", "D", "E"]
    c1, c2 = L.controls(DAY, pool, ["B", "D"]), L.controls(DAY, pool, ["D", "B"])
    assert c1 == c2 and c1["B"] != "B" and c1["D"] != "D" and set(c1.values()) <= set(pool)
    assert L.controls(DAY + pd.Timedelta(days=1), pool, ["B", "D"]) is not None


def test_exposure_cap_counts_open_picks():
    assert L.exposure_ok([0.5, 0.4], 0.1) and not L.exposure_ok([0.5, 0.45], 0.1)


def test_choose_uses_the_frozen_cut_and_scale():
    fz = {"cut": 0.01, "scale": np.linspace(0, 0.05, 101)}
    p = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])
    lab, trade, size = L.choose(p, fz)
    assert lab == "long_2.0" and trade and 0.02 <= size <= 0.10
    assert L.choose(np.zeros(6), fz)[1] is False


class _Api:
    def __init__(self, bars, trades=None):
        self.bars, self.trades = bars, trades or {}

    def assets(self):
        return ["AAA"], {"AAA"}

    def bars5(self, syms, day, end_utc=None):
        return self.bars[self.bars["symbol"].isin(syms)].copy()

    def daily(self, syms, start, end, adjustment):
        return {s: _frames(30, 400.0, seed=3) for s in syms}

    def last_trades(self, syms):
        return {s: self.trades[s] for s in syms if s in self.trades}


def _bars_for_trigger():
    t0 = DAY + B.OPEN
    rows = []
    for k in range(0, 24):                                           # 9:30 .. 11:30, heavy volume from the start
        t = t0 + k * B.SLOT
        rows.append(("AAA", t, 2.0 + 0.01 * k, 2.02 + 0.01 * k, 1.99 + 0.01 * k, 2.01 + 0.01 * k, 2000.0, 2.0))
    return pd.DataFrame(rows, columns=["symbol", "t", "o", "h", "l", "c", "v", "vw"])


def _setup(monkeypatch, tmp_path, now_str="2026-03-03 10:20"):
    ctx = {"AAA": {"prev_c": 2.0, "prev_dv20": 2e6, "prev_ac": 2.0, "vol20": 78000.0, "hi20": 9.0, "lo20": 0.1,
                   "hi252": 9.0, "lo252": 0.1}}
    prior = {"AAA": _frames(100)}
    monkeypatch.setattr(L, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.bdate_range("2026-01-01", "2026-03-02"), "vix": 18.0}).to_csv(tmp_path / "cache" / "vix.csv", index=False)
    monkeypatch.setattr(L, "build_universe", lambda api, day: (ctx, prior))
    monkeypatch.setattr(L, "sic_maps", lambda syms: ({}, {}))
    cal = pd.DataFrame({"date": [DAY], "open": ["09:30"], "close": ["16:00"], "early_close": [False]})
    monkeypatch.setattr(L.alpaca, "trading_calendar", lambda a, b: cal)
    fz = {"features": ["x"], "cut": 0.01, "scale": np.linspace(0, 0.05, 101), "curve": [(k + 1) / 78 for k in range(78)]}
    monkeypatch.setattr(L, "predict", lambda fz_, feats: np.array([[0.03] + [0.0] * 5 for _ in feats]))
    return fz, L.State(tmp_path / "state"), pd.Timestamp(now_str)


def test_scan_takes_a_trigger_once_and_logs_the_decision(monkeypatch, tmp_path):
    fz, st, now = _setup(monkeypatch, tmp_path)
    api = _Api(_bars_for_trigger(), {"AAA": (2.2, "2026-03-03T15:00:00Z")})
    assert L.scan(st, api, fz, now).endswith("new")
    trig = st.load("2026-03-03/triggers.json", {})
    assert trig["AAA"]["trade"] and trig["AAA"]["label"] == "long_1.0" and trig["AAA"]["entry"] == 2.2
    assert len(st.load("open.json", [])) == 1
    L.scan(st, api, fz, now + pd.Timedelta(minutes=5))               # the same stock is not decided twice
    assert len(st.load("open.json", [])) == 1


def test_scan_logs_why_a_trigger_is_not_taken(monkeypatch, tmp_path):
    fz, st, now = _setup(monkeypatch, tmp_path)
    L.scan(st, _Api(_bars_for_trigger()), fz, now)                   # no quote
    assert st.load("2026-03-03/triggers.json", {})["AAA"]["reason"] == "no quote"
    fz2, st2, now2 = _setup(monkeypatch, tmp_path / "b")
    L.scan(st2, _Api(_bars_for_trigger(), {"AAA": (2.2, "t")}), fz2, now2 + pd.Timedelta(hours=3))
    assert st2.load("2026-03-03/triggers.json", {})["AAA"]["reason"] == "late"


def test_settle_one_uses_the_training_labels(monkeypatch):
    t0 = DAY + B.OPEN
    b = pd.DataFrame({"symbol": "AAA", "t": [t0 + k * B.SLOT for k in range(10, 30)], "o": 2.0, "h": 2.5, "l": 1.99,
                      "c": 2.4, "v": 1e3, "vw": 2.2})
    api = _Api(b)
    days = list(pd.bdate_range(DAY + pd.Timedelta(days=1), periods=4))

    def daily(syms, start, end, adjustment):
        idx = pd.bdate_range(DAY, periods=6)
        f = pd.DataFrame({"o": 2.3, "h": 2.6, "l": 2.2, "c": 2.4, "v": 1e6}, index=idx)
        return {s: f for s in syms}
    api.daily = daily
    p = {"symbol": "AAA", "date": str(DAY.date()), "label": "long_1.0", "entry": 2.0, "atr": 0.1,
         "decided": str(DAY + pd.Timedelta(hours=10)), "role": "pick"}
    net = L.settle_one(api, p, DAY, days)
    assert net is not None and net > 0                              # the target (entry + 2 x ATR) is touched


def test_after_close_logs_the_day_once_and_adds_the_controls(monkeypatch, tmp_path):
    fz, st, now = _setup(monkeypatch, tmp_path, "2026-03-03 16:30")
    rec = lambda s, trade: {"symbol": s, "trade": trade, "entry": 2.0, "atr": 0.1, "pred": [0.0] * 6,
                            "label": "long_1.0", "size": 0.05, "decided": "2026-03-03 10:20:00"}
    st.save("2026-03-03/triggers.json", {"AAA": rec("AAA", True), "BBB": rec("BBB", False), "CCC": rec("CCC", False)})
    st.save("open.json", [{"symbol": "AAA", "date": "2026-03-03", "label": "long_1.0", "size": 0.05, "entry": 2.0,
                           "atr": 0.1, "decided": "2026-03-03 10:20:00", "role": "pick"}])
    L.after_close(st, _Api(_bars_for_trigger()), now)
    roles = sorted((p["symbol"], p["role"]) for p in st.load("open.json", []))
    assert len(roles) == 2 and ("AAA", "pick") in roles and roles[1][0] in {"BBB", "CCC"} or roles[0][0] in {"BBB", "CCC"}
    assert st.ledger.has("chrp_day", "2026-03-03") and st.ledger.verify() == []
    L.after_close(st, _Api(_bars_for_trigger()), now)                # a second call neither logs nor adds again
    assert len(st.load("open.json", [])) == 2
