# ORIGIN: copied from breakout-scanner catalyst/rules.py at commit 111b71d on 2026-10-15. Only the import
# lines changed (the frozen text rules now come from tracks/penny/frozen/p27c.py); every line from
# OFFER_FORMS on is unchanged.
"""Event detection for the catalyst forward test (Amendment 19, rules frozen from Amendment 18 after
clarification 8). Pure functions over frames, so the tests can drive them with fixtures."""
import numpy as np
import pandas as pd

from tracks.penny.frozen.p27c import (FIXED_SIGN, SUBJECT_CLASSES, classify, day_session, not_news, sign_of,  # noqa: F401
                                      subject_in_headline)

OFFER_FORMS = {"S-1", "S-1/A", "S-3", "424B1", "424B2", "424B3", "424B4", "424B5"}
ITEM_CLASS = {"3.02": "A8", "3.01": "A6"}
CLUSTER_DAYS = 2
SIGN_WINDOW = pd.Timedelta(minutes=15)

# Amendment 19 clarification 1: test cells and descriptive-only rows (class, sign, group)
I12 = {("A8", "-", "PENNY"), ("A8", "-", "MAIN"), ("A6", "-", "PENNY"), ("A9", "-", "PENNY"), ("C4", "-", "PENNY"),
       ("A9", "-", "MAIN")}
I13 = {("C4", "+", "PENNY")}
DESCRIPTIVE = {("C1", "+", "PENNY"), ("C1", "-", "PENNY"), ("C1", "+", "MAIN"), ("C1", "-", "MAIN"),
               ("C4", "+", "MAIN"), ("C4", "-", "MAIN")}


def cell_of(cls, sign, group):
    k = (cls, sign, group)
    return "I-12" if k in I12 else "I-13" if k in I13 else "descriptive" if k in DESCRIPTIVE else "other"


def symbol_names(alpaca_names, sec_titles):
    """Clarification 1.4: symbol -> [Alpaca asset name, SEC company_tickers title] (non-empty)."""
    out = {}
    for s in set(alpaca_names) | set(sec_titles):
        out[s] = [n for n in (alpaca_names.get(s), sec_titles.get(s)) if isinstance(n, str) and n.strip()]
    return out


def article_rows(arts, names):
    """One row per (article, symbol) for articles with 1-3 symbols that are not 'not news': the frozen text
    classes after the subject rule (C1, C4, A8 text) and the earnings-word flag is not needed here."""
    rows = []
    for r in arts.itertuples(index=False):
        syms = [s for s in (r.symbols or "").split(",") if s]
        if not (1 <= len(syms) <= 3) or not_news(r.headline, r.url):
            continue
        cls, text = classify(r.headline, r.summary)
        cls = [c for c in cls if c in ("C1", "C4", "A8t")]
        for s in syms:
            c = [x for x in cls if not (x in SUBJECT_CLASSES and not subject_in_headline(r.headline, s, names))]
            rows.append({"id": r.id, "created_at": r.created_at, "symbol": s, "classes": c, "text": text,
                         "headline": r.headline})
    return pd.DataFrame(rows, columns=["id", "created_at", "symbol", "classes", "text", "headline"])


def text_items(rows):
    """Text items (cls, symbol, T, article id) from the classified rows; A8 text applies to MAIN only (later)."""
    out = []
    for r in rows.itertuples(index=False):
        for c in r.classes:
            out.append({"cls": "A8" if c == "A8t" else c, "symbol": r.symbol, "T": r.created_at, "source": "benzinga",
                        "id": r.id, "src": "text", "main_only": c == "A8t", "penny_only": False})
    return pd.DataFrame(out, columns=["cls", "symbol", "T", "source", "id", "src", "main_only", "penny_only"])


def filing_items(headers, universe_cik_group):
    """Filing items from parsed SEC headers: 8-K item 3.02 -> A8, 3.01 -> A6; offering forms of PENNY filers
    -> A8 (PENNY only). universe_cik_group: cik -> (series_id, symbol, group)."""
    out = []
    for h in headers:
        if not h.get("found") or h.get("accepted_et") is None:
            continue
        ciks = [c for c in h["ciks"] if c in universe_cik_group]
        for c in ciks:
            sid, sym, grp = universe_cik_group[c]
            if h["type"] == "8-K":
                for it, cls in ITEM_CLASS.items():
                    if it in h["items"]:
                        out.append({"cls": cls, "symbol": sym, "t_item": h["accepted_et"], "src": f"8-K {it}",
                                    "accession": h["accession"], "penny_only": False})
            elif h["type"] in OFFER_FORMS:
                out.append({"cls": "A8", "symbol": sym, "t_item": h["accepted_et"], "src": h["type"],
                            "accession": h["accession"], "penny_only": True})
    return pd.DataFrame(out, columns=["cls", "symbol", "t_item", "src", "accession", "penny_only"])


def first_article_time(symbol, t_item, all_rows):
    """Earliest article (any non-'not news' article tagging the symbol) in [t - 24 h, t + 2 h]: (T, id) or None."""
    x = all_rows[(all_rows["symbol"] == symbol) & (all_rows["created_at"] >= t_item - pd.Timedelta(hours=24))
                 & (all_rows["created_at"] <= t_item + pd.Timedelta(hours=2))]
    if not len(x):
        return None
    r = x.sort_values("created_at").iloc[0]
    return r["created_at"], r["id"]


def a9_items(raw, adj, t):
    """p26 E2 reverse-split rule on day t: k = (c_raw/c)(t) / (c_raw/c)(prev bar) >= 1.5 and the volume
    ratio (v_raw/v_adj)(prev) / (v_raw/v_adj)(t) within 10% of k. Timed at 09:29:59 of t."""
    m = raw.merge(adj, on=["symbol", "date"], suffixes=("_r", "_a")).sort_values(["symbol", "date"])
    m = m[(m["v_r"] > 0) & (m["v_a"] > 0)]
    g = m.groupby("symbol", sort=False)
    ratio = m["c_r"] / m["c_a"]
    k = ratio / ratio.groupby(m["symbol"]).shift(1)
    vq = m["v_r"] / m["v_a"]
    vr = vq.groupby(m["symbol"]).shift(1) / vq
    hit = (m["date"] == pd.Timestamp(t)) & (k >= 1.5) & ((vr - k).abs() <= 0.10 * k)
    x = m[hit]
    return pd.DataFrame({"cls": "A9", "symbol": x["symbol"].values,
                         "T": pd.Timestamp(t) + pd.Timedelta(hours=9, minutes=29, seconds=59), "source": "bars",
                         "id": None, "src": "bars E2", "main_only": False, "penny_only": False,
                         "k": k[hit].values})


def cluster_new(items, store):
    """Drop items that join an existing (series, cls) cluster: within CLUSTER_DAYS calendar days after the
    cluster's first item (store: earlier events with series_id, cls, T). Returns the new first items and the
    joined ones (for the 15-minute sign text)."""
    items = items.sort_values(["series_id", "cls", "T"]).reset_index(drop=True)
    firsts = {(s, c): t for s, c, t in zip(store["series_id"], store["cls"], store["T"])} if len(store) else {}
    keep = np.zeros(len(items), bool)
    for i, (s, c, t) in enumerate(zip(items["series_id"], items["cls"], items["T"])):
        f = firsts.get((s, c))
        if f is None or t > f + pd.Timedelta(days=CLUSTER_DAYS):
            firsts[(s, c)] = t
            keep[i] = True
    return items[keep].reset_index(drop=True), items[~keep].reset_index(drop=True)


def signs(events, text_by_id, rows):
    """Sign per event: fixed classes by FIXED_SIGN; C4 from its first article only; C1 from the texts of
    the symbol's C1 articles created in [T, T + 15 min]."""
    out = []
    for e in events.itertuples(index=False):
        if e.cls in FIXED_SIGN:
            out.append(FIXED_SIGN[e.cls])
        elif e.cls == "C4":
            out.append(sign_of("C4", text_by_id.get(e.id, "")))
        elif e.cls == "C1":
            w = rows[(rows["symbol"] == e.symbol) & rows["classes"].map(lambda c: "C1" in c)
                     & (rows["created_at"] >= e.T) & (rows["created_at"] <= e.T + SIGN_WINDOW)]
            out.append(sign_of("C1", " ".join(w["text"])))
        else:
            out.append("")
    return [{"0": "unsigned"}.get(s, s) for s in out]


def sessions(T, cal_dates):
    d, sess, nontrading = day_session(pd.Series(T), pd.DatetimeIndex(cal_dates))
    return d, sess, nontrading


def entry_day(T, cal):
    """Entry = the official open of the first trading day whose open is after T (cal: date, open)."""
    opens = pd.to_datetime(cal["date"].dt.strftime("%Y-%m-%d") + " " + cal["open"])
    i = int(np.searchsorted(opens.values, np.datetime64(pd.Timestamp(T)), side="right"))
    return cal["date"].iloc[i] if i < len(cal) else pd.NaT
