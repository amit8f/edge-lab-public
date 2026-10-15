"""P1/P2 nightly paper job on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from core.ledger import Ledger
from tracks.penny import nightly as N

DAYS = pd.bdate_range("2026-03-02", "2026-12-31")
PENNY = ["XPEN", "YPEN", "CTA", "CTB", "CTC", "CTD"]
CIK = {s: 1000 + i for i, s in enumerate(PENNY + ["BIG"])}
NIGHT = pd.Timestamp("2026-11-02")


def bars_frame():
    rows = []
    for s in PENNY + ["BIG"]:
        px = 50.0 if s == "BIG" else 3.0
        for d in DAYS:
            rows.append({"symbol": s, "date": d, "o": px, "h": px * 1.03, "l": px * 0.97, "c": px, "v": 500_000})
    return pd.DataFrame(rows)


class FakeApi:
    def __init__(self, news=None, filings=None, subs=None, index_until=None):
        self.b = bars_frame()
        self.news_rows = news or []
        self.filings = filings or {}
        self.subs = subs or {}
        self.index_until = index_until
        self.sent = []

    def calendar(self, start, end):
        d = DAYS[(DAYS >= pd.Timestamp(start).normalize()) & (DAYS <= pd.Timestamp(end))]
        return pd.DataFrame({"date": d, "open": "09:30", "close": "16:00", "early_close": False})

    def assets(self):
        return pd.DataFrame({"symbol": PENNY + ["BIG"], "name": [f"{s} Inc" for s in PENNY + ["BIG"]],
                             "excl_reason": None, "shortable": True, "easy_to_borrow": True})

    def tickers(self):
        return pd.DataFrame({"symbol": list(CIK), "cik": list(CIK.values()), "title": [f"{s} Inc" for s in CIK]})

    def news(self, start, end):
        d = pd.DataFrame(self.news_rows, columns=["id", "created_at", "headline", "summary", "symbols", "url"])
        d["created_at"] = pd.to_datetime(d["created_at"])
        return d[(d["created_at"] >= start) & (d["created_at"] <= end)]

    def bars(self, symbols, start, end, adjustment):
        b = self.b
        return b[b["symbol"].isin(set(symbols)) & (b["date"] >= start) & (b["date"] <= end)].reset_index(drop=True)

    def form_index(self, d):
        if self.index_until is not None and d > self.index_until:
            return None
        rows = [{"form": f, "company": s, "cik": CIK[s], "filed": d, "file": f"edgar/data/{CIK[s]}/0001-26-{k:06d}.txt"}
                for k, (s, f, _, _) in enumerate(self.filings.get(d, []))]
        return pd.DataFrame(rows, columns=["form", "company", "cik", "filed", "file"])

    def header(self, cik, file):
        for d, lst in self.filings.items():
            for k, (s, f, acc, items) in enumerate(lst):
                if CIK[s] == cik and file.endswith(f"{k:06d}.txt"):
                    return f"0001-26-{k:06d}", acc, items
        return None, None, []

    def submissions(self, cik):
        return self.subs.get(cik)

    def positions(self):
        return {}

    def order(self, body):
        self.sent.append(body)
        return 200, {"status": "accepted"}

    def order_by_client_id(self, cid):
        return {"status": "filled", "filled_qty": "10", "filled_avg_price": "3.03", "filled_at": "x"}


def et(ts):
    return pd.Timestamp(ts, tz=N.ET).tz_convert("UTC")


def standard_api(**kw):
    news = [{"id": 1, "created_at": "2026-11-02 14:00", "headline": "YPEN raises full-year revenue guidance above estimates",
             "summary": "", "symbols": "YPEN", "url": "https://x/news/1"}]
    filings = {NIGHT: [("XPEN", "8-K", "2026-11-02 16:30:00", ["3.02", "9.01"])]}
    return FakeApi(news=news, filings=filings, **kw)


def test_nothing_runs_before_the_first_night(tmp_path):
    class Boom:
        def __getattr__(self, k):
            raise AssertionError("no network before START")
    assert N.run(tmp_path, now=et("2026-11-02 20:00"), api=Boom()) == 0
    assert not (tmp_path / "paper").exists()


def test_universe_rules():
    b = bars_frame()
    b = b[~((b["symbol"] == "CTA") & (b["date"] > "2026-09-01") & (b["date"] < "2026-09-20"))]   # >15-day gap
    b.loc[b["symbol"] == "CTB", "v"] = 100_000                                                      # dv20 $300k
    u = N.universe_on(N.annotate(b), NIGHT, CIK)
    assert set(u["symbol"]) == {"XPEN", "YPEN", "CTC", "CTD"}     # CTA: 30 bars since the gap; BIG: $50
    assert (u["cik"] == u["symbol"].map(CIK)).all()


def test_first_night_signals_controls_and_mirror(tmp_path):
    api = standard_api()
    n = N.run(tmp_path, now=et("2026-11-03 05:00"), api=api)
    led = Ledger(tmp_path / "paper")
    assert led.verify() == []
    days = [e["date"] for e in led.entries() if e["kind"] == "paper_day"]
    assert days[0] == "2026-10-05" and days[-1] == "2026-11-02" and n == len(days)
    rec = N._read(led, "paper_day", "2026-11-02")
    assert {(s["idea"], s["symbol"], s["side"]) for s in rec["signals"]} == {("P1", "XPEN", "short"), ("P2", "YPEN", "long")}
    assert {(e["idea"], e["entry_date"]) for e in rec["book"]["pending_entries"]} == {("P1", "2026-11-03"), ("P2", "2026-11-03")}
    ctrl = {c["idea"]: c for c in rec["controls"]}
    assert ctrl["P1c"]["symbol"] in {"CTA", "CTB", "CTC", "CTD"} and ctrl["P1c"]["side"] == "short"
    assert ctrl["P2c"]["symbol"] not in {"YPEN"} and ctrl["P2c"]["pair_id"] == next(
        s["event_id"] for s in rec["signals"] if s["idea"] == "P2")
    assert len(rec["control_book"]["pending_entries"]) == 2
    assert N._read(led, "paper_day", "2026-10-30")["warmup"] is True
    lim = {e["symbol"]: e["limit"] for e in rec["limit_book"]["pending_entries"]}
    assert lim == {"XPEN": pytest.approx(2.97), "YPEN": pytest.approx(3.03)}
    assert all(s["borrow_day"] is None for s in rec["signals"])
    sides = {b["symbol"]: (b["side"], b["time_in_force"]) for b in api.sent}
    assert sides == {"XPEN": ("sell", "opg"), "YPEN": ("buy", "opg")}
    assert all(b["client_order_id"].startswith("el-20261102-") and int(b["qty"]) >= 1 for b in api.sent)

    # same night again: nothing new, no order sent twice
    assert N.run(tmp_path, now=et("2026-11-03 06:00"), api=api) == 0
    assert len(api.sent) == 2

    # next night: entries filled at the open, Alpaca fills read back with their shortfall
    N.run(tmp_path, now=et("2026-11-04 05:00"), api=api)
    rec2 = N._read(led, "paper_day", "2026-11-03")
    assert {p["symbol"] for p in rec2["book"]["open"]} == {"XPEN", "YPEN"}
    # flat synthetic bars (open 3.00, range 2.91-3.09): both limits trade through, at the open
    lb = {p["symbol"]: p for p in rec2["limit_book"]["open"]}
    assert {k: p["entry_price"] for k, p in lb.items()} == {"XPEN": 3.0, "YPEN": 3.0}
    assert lb["YPEN"]["costs"] == 0                     # no entry cost; the short pays one day of borrow
    assert lb["XPEN"]["costs"] == pytest.approx(lb["XPEN"]["notional"] * 0.20 / 252)
    m = N._read(led, "paper_mirror", "2026-11-03")
    assert len(m["fills"]) == 2
    assert sorted(x["is_vs_book_open"] for x in m["shortfall"]) == pytest.approx([-0.01, 0.01])
    assert led.verify() == []


def test_mirror_not_sent_after_the_open(tmp_path):
    api = standard_api()
    N.run(tmp_path, now=et("2026-11-03 10:00"), api=api)
    m = N._read(Ledger(tmp_path / "paper"), "paper_mirror", "2026-11-02")
    assert m["status"].startswith("not sent") and api.sent == []


def test_stops_when_the_sec_index_is_missing(tmp_path):
    api = standard_api(index_until=pd.Timestamp("2026-10-30"))
    N.run(tmp_path, now=et("2026-11-03 05:00"), api=api)
    led = Ledger(tmp_path / "paper")
    assert led.entries()[-1]["date"] == "2026-10-30" and api.sent == []


def test_p2_skips_after_dilution(tmp_path):
    subs = {CIK["YPEN"]: {"filings": {"recent": {"form": ["424B5"], "filingDate": ["2026-09-15"], "items": [""]}}}}
    api = standard_api(subs=subs)
    N.run(tmp_path, now=et("2026-11-03 05:00"), api=api, mirror=False)
    rec = N._read(Ledger(tmp_path / "paper"), "paper_day", "2026-11-02")
    assert [s["idea"] for s in rec["signals"]] == ["P1"]
    assert rec["skipped"] == [{"event_id": rec["skipped"][0]["event_id"], "idea": "P2", "symbol": "YPEN",
                               "reason": "dilution: A8 in the prior 90 days"}]


def test_p2_pump_and_split_rules():
    b = bars_frame()
    days = list(DAYS)
    i = days.index(NIGHT)
    up = b["symbol"].eq("YPEN") & b["date"].isin(days[i - 4:i + 1])
    b.loc[up, "c"] = 4.8                                    # +60% in 5 days
    ev = {"symbol": "YPEN", "series_id": "YPEN"}
    empty = pd.DataFrame(columns=N.EVENT_COLS)
    assert N.p2_filter(ev, NIGHT, b, b, empty, None, days).startswith("pump")
    # a 1-for-10 reverse split on day k: raw x10, adjusted history rescaled
    raw, adj = bars_frame(), bars_frame()
    k = days[i - 30]
    raw.loc[raw["symbol"].eq("YPEN") & (raw["date"] < k), "v"] *= 10
    raw.loc[raw["symbol"].eq("YPEN") & (raw["date"] < k), ["o", "h", "l", "c"]] /= 10
    assert N.a9_dates(raw, adj, "YPEN", days[i - 60], NIGHT) == [k]
    assert N.p2_filter(ev, NIGHT, raw, adj, empty, None, days).startswith("reverse split")
    assert N.split_ratio(raw, adj, "YPEN", k) == pytest.approx(0.1)
    assert N.split_ratio(raw, adj, "YPEN", days[i]) is None


def test_recent_a8_reads_items():
    subs = {"filings": {"recent": {"form": ["8-K", "8-K", "S-3"], "filingDate": ["2026-10-01", "2026-10-02", "2026-01-01"],
                                   "items": ["3.02,9.01", "2.02", ""]}}}
    got = N.recent_a8(subs, pd.Timestamp("2026-08-04"), NIGHT)
    assert got == [{"form": "8-K", "filed": "2026-10-01"}]


def test_mirror_orders_for_exits():
    st = {"pending_entries": [], "open": [
        {"track": "penny", "symbol": "A", "side": "short", "event_id": "e1", "days_held": 19, "hold_days": 20,
         "pending_exit": None},
        {"track": "penny", "symbol": "B", "side": "long", "event_id": "e2", "days_held": 2, "hold_days": 5,
         "pending_exit": {"date": "2026-11-05", "reason": "stop"}},
        {"track": "penny", "symbol": "C", "side": "long", "event_id": "e3", "days_held": 1, "hold_days": 5,
         "pending_exit": None}]}
    o = {cid: (b["side"], b["time_in_force"]) for cid, b, _ in N.mirror_orders(st, "2026-11-05")}
    assert o == {"el-e1-time": ("buy", "cls"), "el-e2-stop": ("sell", "opg")}


def test_control_pool_excludes_recent_events():
    u = pd.DataFrame({"symbol": ["A", "B", "C"], "series_id": ["A", "B", "C"], "atr_ratio": [0.05, 0.05, np.nan],
                      "shortable": [True, True, True], "etb": [True, False, True]})
    days = list(DAYS)
    store = pd.DataFrame([{"night_day": f"{days[days.index(NIGHT) - 19]:%Y-%m-%d}", "cls": "A6", "series_id": "A"}])
    assert list(N.control_pool(u, store, N.P1_CLASSES, NIGHT, days, set(), True)["symbol"]) == []
    assert list(N.control_pool(u, store, N.P1_CLASSES, NIGHT, days, set(), False)["symbol"]) == ["B"]
