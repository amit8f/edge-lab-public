"""HI3K (research/prereg/HI3K.md): HI3's 18 news arms fed SEC 8-K press releases instead of Benzinga.

Stages (cache under <cache>, default cache/hi3k):
  fetch   8-Ks from EDGAR's daily form indexes -> index page (acceptance time, documents) -> first EX-99 title
          -> cache/news/news_<m>.parquet in Alpaca's article shape (id, created_at, headline, symbols)
  build   HI3's replay code on that feed (headline rows, events, inputs from SIP) -> cache/days/<d>.json
  report  HI3's settle and summary, split into the gap and overlap parts -> <out>/HI3K.json

    python -m tracks.news.hi3k_8k --stage fetch|build|report --cache cache/hi3k --out research/results/HI3K
"""
import argparse
import html
import json
import re
import time
from pathlib import Path

import pandas as pd

from core import sec
from core.env import alpaca_headers
from tracks.news import hi3_replay as R
from tracks.news import nx1_study as N
from tracks.news import nx3_live as L
from tracks.news import nx3_study as S

ET = S.ET
FIRST_DAY, LAST_DAY = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-10-28")
GAP = (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-05-29"))
OVERLAP = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-28"))
EXCHANGES = {"Nasdaq", "NYSE", "NYSE American"}
TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
MIN_WORDS, MAX_WORDS = 5, 40
BOILER = re.compile(r"^(exhibit\b|ex-?\s?99|for immediate release|immediate release|press release|news release|"
                    r"news\b|release\b|media release|investor release|contacts?\b|investor (relations|contact)|"
                    r"media (relations|contact)|source:|page \d|table of contents|forward[- ]looking)", re.IGNORECASE)
MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_LINE = re.compile(rf"^({MONTHS}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\W*$",
                       re.IGNORECASE)


# ---------- pure helpers (tested) ----------

def ticker_by_cik(j):
    """SEC's company_tickers_exchange.json -> {cik: first listed ticker on Nasdaq / NYSE / NYSE American}."""
    f = j["fields"]
    i_c, i_t, i_x = f.index("cik"), f.index("ticker"), f.index("exchange")
    out = {}
    for row in j["data"]:
        if row[i_t] and row[i_x] in EXCHANGES:
            out.setdefault(int(row[i_c]), str(row[i_t]).upper().replace("-", "."))
    return out


def parse_index(text):
    """A filing's -index.htm -> (acceptance time as a UTC Timestamp or None, [(type, href)] of its documents)."""
    m = re.search(r"Accepted</div>\s*<div[^>]*>\s*([\d-]+ [\d:]+)", text)
    acc = pd.Timestamp(m.group(1)).tz_localize(ET).tz_convert("UTC") if m else None
    docs = []
    tab = re.search(r'(?is)<table[^>]*summary="Document Format Files"[^>]*>(.*?)</table>', text)
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", tab.group(1) if tab else ""):
        cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
        if len(cells) < 4:
            continue
        href = re.search(r'href="([^"]+)"', cells[2])
        if href:
            docs.append((re.sub(r"(?s)<[^>]+>", "", cells[3]).strip().upper(),
                         href.group(1).replace("/ix?doc=", "")))
    return acc, docs


def first_ex99(docs):
    """The href of the first EX-99 document (EX-99, EX-99.1, ...), or None."""
    return next((h for t, h in docs if re.match(r"EX-99(\.\d+)?$", t)), None)


def text_lines(raw):
    """HTML or text document -> its non-empty lines, block tags turned into line breaks."""
    t = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", raw)
    t = re.sub(r"(?i)<br\s*/?>|</?(p|div|tr|h[1-6]|li|table|center)\b[^>]*>", "\n", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t)).replace("\xa0", " ")
    return [" ".join(ln.split()) for ln in t.splitlines() if ln.strip()]


PHONE = re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-.]\d{4}")
BODY = re.compile(r"\((nasdaq|nyse|otc)|today announced|today reported|exhibit 99|^item \d|^(investors?|media|analysts?):",
                  re.IGNORECASE)
JOIN_END = re.compile(r"\b(of|for|to|and|or|with|the|a|an|in|on|at|by|from|approximately|its|up)$|[,;:&–-]$", re.IGNORECASE)


def skip(ln):
    return bool(BOILER.match(ln) or DATE_LINE.match(ln) or BODY.search(ln) or PHONE.search(ln) or "@" in ln
                or "http" in ln.lower())


def nwords(ln):
    return len(re.findall(r"[A-Za-z][A-Za-z'&.-]*", ln))


def title(raw):
    """A press release's title: the first line with MIN_WORDS..MAX_WORDS words after boilerplate, or None. A title
    broken over lines (ending in "of", "for", a comma, ...) takes the next line when the whole stays <= MAX_WORDS."""
    lines = text_lines(raw)[:80]
    for i, ln in enumerate(lines):
        if skip(ln) or len(ln.split()) > MAX_WORDS:
            continue
        while JOIN_END.search(ln) and i + 1 < len(lines) and not skip(lines[i + 1]) \
                and len((ln + " " + lines[i + 1]).split()) <= MAX_WORDS:
            i += 1
            ln = ln + " " + lines[i]
        if nwords(ln) >= MIN_WORDS:
            return ln
    return None


# ---------- fetch ----------

def fetch_day(d, tick):
    """One day's 8-K press releases -> (articles, funnel counts)."""
    try:
        idx = sec.daily_form_index(d)
    except RuntimeError as e:     # extra federal closures (e.g. 2025-12-24) have no index and answer 403
        if "HTTP 403" not in str(e):
            raise
        idx = None
    if idx is None:
        return None, None
    k = idx[(idx["form"] == "8-K") & idx["cik"].isin(tick)]
    fun = {"8k": len(k), "ex99": 0, "titled": 0}
    rows = []
    for r in k.itertuples():
        acc = r.file.rsplit("/", 1)[-1].removesuffix(".txt")
        base = f"https://www.sec.gov/Archives/edgar/data/{r.cik}/{acc.replace('-', '')}/"
        page = sec.get(base + f"{acc}-index.htm")
        if page is None:
            continue
        t, docs = parse_index(page.content.decode("latin-1"))
        h = first_ex99(docs)
        if t is None or h is None:
            continue
        fun["ex99"] += 1
        doc = sec.get(("https://www.sec.gov" + h) if h.startswith("/") else base + h)
        head = title(doc.content.decode("utf-8", "replace")) if doc is not None else None
        if not head:
            continue
        fun["titled"] += 1
        rows.append({"id": acc, "created_at": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "headline": head,
                     "symbols": tick[r.cik]})
    return pd.DataFrame(rows, columns=["id", "created_at", "headline", "symbols"]), fun


def stage_fetch(cache, budget_min):
    """Day files cache/8k/<d>.parquet (resumable), then month files cache/news/news_<m>.parquet."""
    deadline = time.monotonic() + budget_min * 60
    dd, nd = Path(cache) / "8k", Path(cache) / "news"
    dd.mkdir(parents=True, exist_ok=True)
    nd.mkdir(parents=True, exist_ok=True)
    tick = ticker_by_cik(sec.get(TICKERS).json())
    print(f"[hi3k] {len(tick)} listed CIKs", flush=True)
    funnel_f = Path(cache) / "funnel.json"
    funnel = json.loads(funnel_f.read_text()) if funnel_f.exists() else {}
    for d in pd.bdate_range(FIRST_DAY, LAST_DAY):
        f = dd / f"{d:%Y-%m-%d}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        arts, fun = fetch_day(d, tick)
        if arts is None:
            print(f"[hi3k day] {d:%Y-%m-%d}: no index", flush=True)
            arts, fun = pd.DataFrame(columns=["id", "created_at", "headline", "symbols"]), {}
        N.atomic_parquet(arts, f)
        funnel[f"{d:%Y-%m-%d}"] = fun
        tmp = funnel_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(funnel))
        tmp.replace(funnel_f)
        print(f"[hi3k day] {d:%Y-%m-%d}: {fun}", flush=True)
        for x in arts.head(3).itertuples():     # public SEC filings: a few titles show the parser works
            print(f"[hi3k title] {x.symbols} {x.created_at}: {x.headline}", flush=True)
    for m in pd.period_range(FIRST_DAY, LAST_DAY, freq="M"):
        parts = [pd.read_parquet(g) for g in sorted(dd.glob(f"{m}-*.parquet"))]
        N.atomic_parquet(pd.concat(parts, ignore_index=True).drop_duplicates("id"), nd / f"news_{m}.parquet")
    return True


# ---------- build ----------

def stage_build(cache, budget_min):
    deadline = time.monotonic() + budget_min * 60
    info = {str(m): (0, 0) for m in pd.period_range(FIRST_DAY, LAST_DAY, freq="M")}
    if any(not (Path(cache) / "news" / f"news_{m}.parquet").exists() for m in info):
        raise SystemExit("[hi3k] run the fetch stage first")
    (Path(cache) / "window.json").write_text(json.dumps({"feed": "SEC 8-K EX-99 titles", "gap": [str(x.date()) for x in GAP],
                                                          "overlap": [str(x.date()) for x in OVERLAP]}))
    rows = R.headline_file(cache, info, deadline)
    if rows is None:
        return False
    api = R.SipApi()
    cal = api.calendar(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2026-12-31").date())
    cal["date"] = pd.to_datetime(cal["date"])
    ses = S.sessions(cal)
    days = ses[(ses["date"] >= GAP[0]) & (ses["date"] <= OVERLAP[1])]
    nd = R.news_days_all(cache, info)
    ev_all = S.session_events(rows, days.reset_index(drop=True))
    out = Path(cache) / "days"
    out.mkdir(parents=True, exist_ok=True)
    for s in days.to_dict("records"):
        f = out / f"{s['date']:%Y-%m-%d}.json"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        ev = ev_all[ev_all["d"] == s["date"]]
        dec = R.replay_day(api, cal, s, ev, nd) if len(ev) else []
        part = "gap" if s["date"] <= GAP[1] else "overlap"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": f"{s['date']:%Y-%m-%d}", "part": part, "events": dec}, default=L._js))
        tmp.replace(f)
        print(f"[hi3k day] {s['date']:%Y-%m-%d} ({part}): {len(ev)} events, "
              f"{sum(x['status'] == 'decided' for x in dec)} decided", flush=True)
    return True


def overlap_share(trades, hi3_days):
    """Per arm: share of overlap-part 8-K picks whose stock also had a decided Benzinga event that session."""
    bz = {}
    for f in sorted(Path(hi3_days).glob("*.json")):
        rec = json.loads(f.read_text())
        bz[rec["day"]] = {x["symbol"] for x in rec["events"] if x.get("status") == "decided"}
    p = trades[(trades["part"] == "overlap") & (trades["role"] == "picks") & trades["day"].isin(bz)]
    return {a: float(g.apply(lambda r: r["symbol"] in bz[r["day"]], axis=1).mean()) if len(g) else None
            for a, g in p.groupby("arm")}


def stage_report(cache, out, hi3_cache):
    res = R.stage_report(cache, out, parts=("gap", "overlap"), name="HI3K")
    tr = pd.read_parquet(Path(cache) / "trades.parquet")
    res["overlap_pick_share_with_benzinga_event"] = overlap_share(tr, Path(hi3_cache) / "days")
    res["model_counts"] = {m: bool(res["arms"][f"{m}-plain"]["gap"].get("passes")
                                   and res["arms"][f"{m}-HI"]["gap"].get("passes")) for m in R.MODELS}
    f = Path(cache) / "funnel.json"
    res["funnel"] = ({k: sum(v.get(k, 0) for v in json.loads(f.read_text()).values()) for k in ("8k", "ex99", "titled")}
                     if f.exists() else None)
    tmp = Path(out) / "HI3K.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(Path(out) / "HI3K.json")
    print(f"[hi3k] funnel {res['funnel']}; models counting: {[m for m, v in res['model_counts'].items() if v]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "build", "report"])
    ap.add_argument("--cache", default="cache/hi3k")
    ap.add_argument("--out", default="research/results/HI3K")
    ap.add_argument("--hi3-cache", default="cache/hi3")
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        alpaca_headers()
        stage_report(a.cache, a.out, a.hi3_cache)
        return
    if a.stage == "fetch":
        done = stage_fetch(a.cache, a.budget_min)
    else:
        alpaca_headers()
        done = stage_build(a.cache, a.budget_min)
    print(f"[hi3k] {a.stage} {'done' if done else 'not finished (run again)'}", flush=True)


if __name__ == "__main__":
    main()
