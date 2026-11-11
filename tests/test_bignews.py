"""BIGNEWS study on synthetic data (no network): event type, universe, novelty, sessions, minute entries, resume."""
import time

import numpy as np
import pandas as pd

from tracks.news import bignews_study as B
from tracks.news import nx1_study as N
from tracks.news import w1_data as W

ET = "America/New_York"
FOREVER = time.monotonic() + 1e9


def utc(et):
    return pd.Timestamp(et, tz=ET).tz_convert("UTC").isoformat()


class FakeApi:
    """Same bars_1min as Api: naive-ET start..end on known bars, one call per date recorded."""

    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def bars_1min(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        out = {}
        for s in symbols:
            df = self.bars.get(s)
            if df is not None:
                d = df[(df["t"] >= start) & (df["t"] < end)].reset_index(drop=True)
                if len(d):
                    out[s] = d
        return out


def minute_day(day, base=10.0, over=None):
    """Bars 4:00-19:59 of one day at a flat price, with {'HH:MM': {col: value}} overrides."""
    t = pd.date_range(f"{day} 04:00", f"{day} 19:59", freq="min")
    df = pd.DataFrame({"t": t, "o": base, "h": base + 0.1, "l": base - 0.1, "c": base, "v": 1000.0})
    for hm, vals in (over or {}).items():
        i = df.index[df["t"] == pd.Timestamp(f"{day} {hm}")][0]
        for k, v in vals.items():
            df.loc[i, k] = v
    return df


def write_cache(cache, days, daily_sets, news=None, news_month="2019-03"):
    """Synthetic NX1 cache: calendar, one news month, and W1 daily batch rows for the given symbols.
    daily_sets: {symbol: (closes, volumes)} indexed by date."""
    N.atomic_parquet(pd.DataFrame({"date": days, "open": "09:30", "close": "16:00"}), cache / "cal.parquet")
    if news is not None:
        N.atomic_parquet(news, W.news_dir(cache) / f"news_{news_month}.parquet")
    rows = []
    for sym, (c, v) in daily_sets.items():
        raw = pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": v}, index=days)
        rows.append(W.daily_rows(sym, raw, raw[["o", "c"]]))
    N.atomic_parquet(pd.concat(rows, ignore_index=True), W.w1dir(cache) / "daily" / "d_0000.parquet")


def test_event_type_examples():
    cases = {
        "Acme announces positive Phase 3 trial results": "trial",
        "FDA approves Acme's new drug": "fda",
        "Acme to be acquired for $10 a share": "merger",
        "Acme raises full-year guidance": "guidance",
        "Short seller report alleges accounting fraud": "fraud_short",
        "Acme files for Chapter 11 bankruptcy": "bankruptcy",
        "Acme prices $100 million public offering": "offering",
        "Acme wins $50 million contract from DoD": "contract",
        "Acme receives Nasdaq delisting notice": "delisting",
        "Acme names a new CFO": "other",
    }
    for headline, expected in cases.items():
        assert B.event_type(headline) == expected, headline


def test_session_mapping_including_after_hours_to_next_session():
    dates = pd.DatetimeIndex(pd.bdate_range("2019-03-04", "2019-03-15"))  # Mon 3-04 .. Fri 3-15
    t = pd.Series(pd.to_datetime([
        "2019-03-04 08:00", "2019-03-04 10:00", "2019-03-04 16:00", "2019-03-04 17:30",   # pre, regular, after, after
        "2019-03-04 02:00", "2019-03-04 21:00", "2019-03-08 17:00",                      # early, late, Friday after
        "2019-03-09 10:00"]))                                                             # Saturday
    sess, when = B.session_of(t, dates)
    assert list(sess.dt.strftime("%Y-%m-%d")) == ["2019-03-04", "2019-03-04", "2019-03-05", "2019-03-05",
                                                  "2019-03-04", "2019-03-05", "2019-03-11", "2019-03-11"]
    assert list(when) == ["pre", "regular", "after", "after", "overnight", "overnight", "after", "overnight"]


def test_novelty_same_symbol_same_type_in_three_days():
    rows = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "AAA"],
        "t": pd.to_datetime(["2019-03-04 08:00", "2019-03-06 08:00", "2019-03-08 09:00", "2019-03-12 08:00",
                             "2019-03-06 08:00", "2019-03-06 08:00"]),
        "event_type": ["fda", "fda", "fda", "fda", "fda", "trial"]})
    got = dict(zip(rows["id"], B.novelty(rows)))
    assert got == {1: True, 2: False, 3: False, 4: True, 5: True, 6: True}


def test_universe_filter_uses_prior_sessions_only(tmp_path):
    cache, out = tmp_path / "nx1", tmp_path / "out"
    days = pd.bdate_range("2018-12-03", "2019-03-15")
    idx = days
    pre = idx < pd.Timestamp("2019-03-04")
    aaa_v = np.full(len(idx), 1_000_000.0)
    ddd_v = np.where(pre, 1000.0, 1_000_000.0)               # huge volume only on the headline's session
    sets = {"AAA": (np.full(len(idx), 10.0), aaa_v),
            "BBB": (np.full(len(idx), 0.8), np.full(len(idx), 1_000_000.0)),   # under $1
            "CCC": (np.full(len(idx), 10.0), np.full(len(idx), 1000.0)),       # dollar volume ~$10k
            "DDD": (np.full(len(idx), 10.0), ddd_v)}
    news = pd.DataFrame({
        "id": [1, 2, 3], "created_at": [utc("2019-03-04 08:00"), utc("2019-03-04 08:00"), utc("2019-03-04 08:00")],
        "headline": ["Acme FDA approval", "Penny news", "Low volume news"],
        "symbols": ["AAA,BBB", "CCC", "DDD"]})
    write_cache(cache, days, sets, news)
    B.stage_events(cache, out, FOREVER, months=[pd.Period("2019-03", "M")])
    ev = pd.read_parquet(out / "events" / "ev_2019-03.parquet")
    assert list(ev["id"]) == [1]                             # BBB ($0.80), CCC (volume), DDD (prior volume) out
    assert ev.iloc[0]["symbol"] == "AAA" and ev.iloc[0]["tier"] == "main"


def test_stage_events_novelty_session_and_resume(tmp_path):
    cache, out = tmp_path / "nx1", tmp_path / "out"
    days = pd.bdate_range("2018-12-03", "2019-03-15")
    sets = {"AAA": (np.full(len(days), 10.0), np.full(len(days), 1_000_000.0))}
    news = pd.DataFrame({
        "id": [1, 2, 3, 4], "headline": ["Acme FDA approval", "Acme trial results", "Acme FDA again",
                                         "Acme wins contract"],
        "created_at": [utc("2019-03-04 08:00"), utc("2019-03-04 08:30"), utc("2019-03-05 10:00"),
                       utc("2019-03-04 17:00")],
        "symbols": ["AAA"] * 4})
    write_cache(cache, days, sets, news)
    m = [pd.Period("2019-03", "M")]
    B.stage_events(cache, out, FOREVER, months=m)
    ev = pd.read_parquet(out / "events" / "ev_2019-03.parquet").set_index("id")
    assert list(ev.index) == [1, 2, 3, 4]
    assert ev.loc[1, "novel"] and ev.loc[2, "novel"]         # trial is a different type from fda
    assert not ev.loc[3, "novel"]                            # FDA again within 3 days of the first FDA
    assert ev.loc[4, "when"] == "after" and ev.loc[4, "session_date"] == pd.Timestamp("2019-03-05")
    assert ev.loc[1, "session_date"] == pd.Timestamp("2019-03-04") and ev.loc[1, "when"] == "pre"
    # resume: the existing month is kept (no news file is needed to skip it)
    (cache / "g1" / "news" / "news_2019-03.parquet").unlink()
    assert B.stage_events(cache, out, FOREVER, months=m)
    assert len(pd.read_parquet(out / "events" / "ev_2019-03.parquet")) == 4


def test_minutes_entries_moves_and_premarket(tmp_path):
    cache, out = tmp_path / "nx1", tmp_path / "out"
    days = pd.bdate_range("2018-12-03", "2019-03-15")
    px = pd.Series(10.0, index=days)
    px[pd.Timestamp("2019-03-05")] = 9.5
    px[pd.Timestamp("2019-03-11")] = 8.0
    sets = {"AAA": (px.to_numpy(), np.full(len(days), 1_000_000.0))}
    write_cache(cache, days, sets)
    s = pd.Timestamp("2019-03-04")
    ev = pd.DataFrame({
        "id": [1, 2, 3], "symbol": "AAA",
        "t": pd.to_datetime(["2019-03-04 08:00", "2019-03-04 09:29:30", "2019-03-04 17:00"], format="mixed"),
        "session_date": [s, s, pd.Timestamp("2019-03-05")],
        "when": ["pre", "pre", "after"], "tier": "main", "prev_c": 10.0,
        "dv20": 1e7, "event_type": "fda", "novel": True, "headline": "x"})
    N.atomic_parquet(ev, out / "events" / "ev_2019-03.parquet")
    over = {"08:01": {"c": 11.0, "h": 11.2, "l": 10.9}, "08:02": {"v": 3000.0}, "08:05": {"c": 12.0},
            "09:30": {"c": 10.5}, "09:45": {"h": 13.0}, "12:00": {"l": 9.0}, "15:55": {"c": 10.4}}
    bars = pd.concat([minute_day("2019-03-04", over=over), minute_day("2019-03-05", over={"09:30": {"c": 10.5}})],
                     ignore_index=True)
    api = FakeApi({"AAA": bars})
    B.stage_minutes(cache, out, FOREVER, months=[pd.Period("2019-03", "M")], api=api)
    m = pd.read_parquet(out / "minutes" / "m_2019-03.parquet").set_index("id")

    r = m.loc[1]                                             # pre-market headline at 08:00
    assert r["px_m1"] == 11.0 and r["t_m1"] == pd.Timestamp("2019-03-04 08:01")
    assert np.isclose(r["mfe_m1"], 13.0 / 11.0 - 1) and np.isclose(r["mae_m1"], 9.0 / 11.0 - 1)
    assert np.isclose(r["jump_m1"], 0.1)
    assert r["px_m5"] == 12.0 and np.isclose(r["mfe_m5"], 13.0 / 12.0 - 1) and np.isclose(r["mae_m5"], 9.0 / 12 - 1)
    assert r["px_op"] == 10.5 and np.isclose(r["mae_op"], 9.0 / 10.5 - 1)
    assert r["px_pre"] == 11.0 and r["t_pre"] == pd.Timestamp("2019-03-04 08:01")   # pre-market entry
    assert r["p1555"] == 10.4
    assert r["v5"] == 1000 * 4 + 3000                        # bars 08:00-08:04
    assert r["c_next"] == 9.5 and r["c_5"] == 8.0 and r["adj_f"] == 1.0

    r2 = m.loc[2]                                            # trigger 09:29:30: first bar is after the open
    assert r2["px_m1"] == 10.0 and pd.isna(r2["t_pre"]) and np.isnan(r2["px_pre"])
    assert r2["px_op"] == 10.5                               # the 9:30 open is still an entry

    r3 = m.loc[3]                                            # after-hours headline: session is the next day
    assert r3["px_m1"] == 10.0 and pd.isna(r3["px_pre"]) and r3["px_op"] == 10.5
    assert r3["t_op"] == pd.Timestamp("2019-03-05 09:30")


def test_minutes_resume_skips_existing_month(tmp_path):
    cache, out = tmp_path / "nx1", tmp_path / "out"
    days = pd.bdate_range("2018-12-03", "2019-03-15")
    write_cache(cache, days, {"AAA": (np.full(len(days), 10.0), np.full(len(days), 1_000_000.0))})
    N.atomic_parquet(pd.DataFrame({"id": [1]}), out / "minutes" / "m_2019-03.parquet")
    api = FakeApi({})
    assert B.stage_minutes(cache, out, FOREVER, months=[pd.Period("2019-03", "M")], api=api)
    assert api.calls == []
    assert list(pd.read_parquet(out / "minutes" / "m_2019-03.parquet")["id"]) == [1]


def test_novelty_other_type_only_blocks_reposts():
    rows = pd.DataFrame({
        "id": [1, 2, 3],
        "symbol": ["AAA"] * 3,
        "t": pd.to_datetime(["2019-03-04 08:00", "2019-03-04 09:00", "2019-03-05 08:00"]),
        "event_type": ["other"] * 3,
        "headline": ["AAA names new CFO", "AAA opens plant in Ohio", "AAA names new CFO"]})
    got = dict(zip(rows["id"], B.novelty(rows)))
    assert got == {1: True, 2: True, 3: False}
