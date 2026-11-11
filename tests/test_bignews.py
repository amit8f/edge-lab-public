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


SHOCK = pd.Timestamp("2019-01-15")
SHOCK_JAN = pd.Period("2019-01", "M")
DAYS = pd.bdate_range("2018-10-01", "2019-02-28")


class FakeBars:
    """Same bars_5min as ShockApi: {symbol: 5-minute bars} for the symbols that have bars on that day."""

    def __init__(self, gen):
        self.gen = gen
        self.calls = []

    def bars_5min(self, symbols, day):
        self.calls.append((tuple(symbols), pd.Timestamp(day)))
        out = {}
        for s in symbols:
            b = self.gen(s, pd.Timestamp(day))
            if b is not None:
                out[s] = b
        return out


def bars5(day, close, vol=1000.0, shock=None):
    """78 regular-session 5-minute bars flat at close with volume vol; shock = {slot: (volume, close)}."""
    t = pd.date_range(f"{day.date()} 09:30", periods=78, freq="5min")
    df = pd.DataFrame({"t": t, "o": close, "h": close, "l": close, "c": close, "v": vol})
    for slot, (v, c) in (shock or {}).items():
        df.loc[slot, ["v", "c", "h"]] = [v, c, max(c, close)]
    return df


def run_shocks(tmp_path, daily, overrides=None, vols=None, months=(SHOCK_JAN,)):
    """daily: {symbol: closes over DAYS}; vols: {symbol: daily volumes} (default 1e6 a session);
    overrides: {day: {symbol: {slot: (volume, close)}}} for the 5-minute bars."""
    cache, out = tmp_path / "nx1", tmp_path / "out"
    vols = vols or {}
    write_cache(cache, DAYS, {s: (c, vols.get(s, np.full(len(DAYS), 1e6))) for s, c in daily.items()})

    def gen(s, day):
        i = DAYS.get_loc(day)
        return bars5(day, daily[s][i], shock=(overrides or {}).get(day, {}).get(s))

    api = FakeBars(gen)
    assert B.stage_shocks(cache, out, FOREVER, months=list(months), api=api)
    return pd.read_parquet(out / "shocks" / f"s_{months[-1]}.parquet"), api


def test_shock_flags_main_at_4pct_not_2pct(tmp_path):
    daily = {"AAA": np.full(len(DAYS), 10.0), "BBB": np.full(len(DAYS), 10.0)}
    over = {SHOCK: {"AAA": {10: (15000, 10.4)}, "BBB": {10: (15000, 10.2)}}}
    rows, _ = run_shocks(tmp_path, daily, over)
    assert list(rows["symbol"]) == ["AAA"]
    r = rows.iloc[0]
    assert r["tier"] == "main" and r["t"] == pd.Timestamp("2019-01-15 10:25")   # slot 10 = 10:20 bar, t = its end
    assert r["v"] == 15000 and r["v_med"] == 1000 and r["ratio"] == 15
    assert np.isclose(r["move"], 0.04) and r["prev_c"] == 10.0 and r["c"] == 10.4


def test_shock_penny_needs_5pct(tmp_path):
    daily = {"CCC": np.full(len(DAYS), 3.0), "DDD": np.full(len(DAYS), 3.0)}
    over = {SHOCK: {"CCC": {10: (15000, 3.12)}, "DDD": {10: (15000, 3.18)}}}     # 4% and 6%
    rows, _ = run_shocks(tmp_path, daily, over)
    assert list(rows["symbol"]) == ["DDD"] and rows.iloc[0]["tier"] == "penny"


def test_needs_twenty_prior_sessions(tmp_path):
    s = DAYS.get_loc(SHOCK)
    for n, expect in ((19, []), (20, ["FFF"])):
        g = s - 1 - n                                   # FFF trades (is in the universe) from index g + 1
        daily = {"FFF": np.where(np.arange(len(DAYS)) >= g, 10.0, 0.5)}
        vols = {"FFF": np.full(len(DAYS), 4e6)}          # keeps dv20 above $1M while the close is under $1
        over = {SHOCK: {"FFF": {10: (15000, 10.4)}}}
        rows, _ = run_shocks(tmp_path / f"n{n}", daily, over, vols)
        assert list(rows["symbol"]) == expect, n


def test_window_excludes_current_day(tmp_path):
    s = DAYS.get_loc(SHOCK)
    over = {DAYS[i]: {"AAA": {10: (1000.0 if i < s - 10 else 3000.0, 10.0)}} for i in range(s - 20, s)}
    over[SHOCK] = {"AAA": {10: (25000, 10.4)}}
    rows, _ = run_shocks(tmp_path, {"AAA": np.full(len(DAYS), 10.0)}, over)
    # median of the 20 prior values (10 x 1000, 10 x 3000) is 2000; with today's 25000 in it, it would be 3000
    assert list(rows["symbol"]) == ["AAA"]
    assert rows.iloc[0]["v_med"] == 2000 and rows.iloc[0]["ratio"] == 12.5


def test_gap_over_15_days_resets_window(tmp_path):
    bad = (DAYS >= pd.Timestamp("2018-12-03")) & (DAYS <= pd.Timestamp("2018-12-28"))   # 4 weeks out of the universe
    closes = np.where(bad, 0.5, 10.0)
    vols = {"AAA": np.where(bad, 4e6, 1e6)}
    over = {SHOCK: {"AAA": {10: (15000, 10.4)}}, pd.Timestamp("2019-01-31"): {"AAA": {10: (15000, 10.4)}}}
    rows, _ = run_shocks(tmp_path, {"AAA": closes}, over, vols)
    # 1-15 has only 10 sessions since the gap (window restarted: no flag); 1-31 has 22 and is flagged
    assert list(rows["session_date"]) == [pd.Timestamp("2019-01-31")]


def test_shocks_resume_from_saved_state(tmp_path):
    feb = pd.Timestamp("2019-02-12")
    daily = {"AAA": np.full(len(DAYS), 10.0)}
    over = {SHOCK: {"AAA": {10: (15000, 10.4)}}, feb: {"AAA": {10: (15000, 10.4)}}}
    cache, out = tmp_path / "nx1", tmp_path / "out"
    write_cache(cache, DAYS, {"AAA": (daily["AAA"], np.full(len(DAYS), 1e6))})

    def gen(s, day):
        return bars5(day, 10.0, shock=over.get(day, {}).get(s))

    first = FakeBars(gen)
    assert B.stage_shocks(cache, out, FOREVER, months=[SHOCK_JAN], api=first)
    assert (out / "shocks" / "state.pkl").exists() and (out / "shocks" / "s_2019-01.parquet").exists()
    second = FakeBars(gen)
    assert B.stage_shocks(cache, out, FOREVER, months=[SHOCK_JAN, pd.Period("2019-02", "M")], api=second)
    assert second.calls and {pd.Timestamp(d).month for _, d in second.calls} == {2}   # Dec and Jan not refetched
    resumed = pd.read_parquet(out / "shocks" / "s_2019-02.parquet")
    once = tmp_path / "once"
    assert B.stage_shocks(cache, once, FOREVER, months=[SHOCK_JAN, pd.Period("2019-02", "M")], api=FakeBars(gen))
    pd.testing.assert_frame_equal(resumed, pd.read_parquet(once / "shocks" / "s_2019-02.parquet"))
    assert list(resumed["session_date"]) == [feb]
