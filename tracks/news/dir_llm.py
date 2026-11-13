"""DIR-LLM (research/prereg/DIRECTION.md): a forced up/down answer from Gemini 2.5 Flash on masked headlines, with
the model's own confidence.

Model: gemini-2.5-flash (Google's model card, checked 2026-11-13: knowledge cutoff January 2025). Window
2025-02-01..2025-09-30, after the cutoff and before the holdout. Confidence is the answer token's probability
when the API returns log-probabilities; otherwise the share of 5 answers at temperature 1 (logged which).

Stages (dir-llm workflow):
  items  BOOM1 picks with a single-symbol headline from the previous close to 9:14 ET, NX3's long picks, and 300
         date-only probes -> masked prompts and, in a separate file, their outcomes (cache only: headline text and
         prompts never leave the cache)
  calls  asks the model, one prompt per call, until the daily quota runs out; one JSON line per answer with the
         model ID, prompt hash, raw probabilities and time (rerun daily until done)
  test   the confidence cut from the answers alone, then the outcomes joined -> research/results/DIR/DIRLLM.json
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

IDEA = 1
MODEL = "gemini-2.5-flash"
CUTOFF_NOTE = "gemini-2.5-flash knowledge cutoff January 2025 (ai.google.dev model card, checked 2026-11-13)"
FIRST, LAST = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-09-30")
PROBES, VOTES, MAX_HEADLINES = 300, 5, 20
MIN_BOOM, MIN_NX3, BOOM_GAIN, NX3_GAIN, PROBE_BAND = 200, 150, 0.0025, 0.001, 0.03
MIN_GAP_SEC = 7.0                         # free tier: about 10 requests a minute
SEED = 20261113
BOOT, BOOT_SEED = 2000, 0

Q_BOOM = "Will the stock trade higher (A) or lower (B) at the close than at 9:51 ET? Answer with one letter."
Q_NX3 = ("Will the stock trade higher (A) or lower (B) at the next trading day's close than one minute after the "
         "latest headline? Answer with one letter.")
SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|"
                    r"technologies|therapeutics|pharmaceuticals|sa|nv|ag|lp|llc|trust|class [a-c]|common stock)\b\.?",
                    re.IGNORECASE)
GENERIC = {"american", "first", "united", "general", "national", "global", "international", "the", "new", "north",
           "south", "west", "east", "bank", "capital", "energy", "health", "royal", "great", "southern", "western",
           "eastern", "northern", "pacific", "atlantic", "digital", "advanced", "applied", "standard", "central"}


# ---------- pure helpers (tested) ----------

def clean_name(name):
    n = str(name or "").strip()
    for _ in range(3):
        n = SUFFIX.sub("", n).strip(" ,.")
    return n


def mask(text, symbol, name):
    """Replaces the ticker, the company's name (without Inc/Corp/...) and its first word (when distinctive) by
    'Company X'."""
    t = str(text)
    n = clean_name(name)
    forms = [n] if len(n) >= 3 else []
    first = n.split()[0] if n else ""
    if len(first) >= 4 and first.lower() not in GENERIC and first.isalpha():
        forms.append(first)
    for f in sorted(set(forms), key=len, reverse=True):
        t = re.sub(rf"(?<![\w]){re.escape(f)}(?:'s)?(?![\w])", "Company X", t, flags=re.IGNORECASE)
    if symbol:
        t = re.sub(rf"(?<![\w$]){re.escape(symbol)}(?![\w])|\${re.escape(symbol)}\b", "Company X", t)
    return t


def prompt(headlines, question):
    hs = list(headlines)[-MAX_HEADLINES:]
    return "Headlines about Company X:\n" + "\n".join(f"- {' '.join(str(h).split())}" for h in hs) + "\n" + question


def probe_prompt(d):
    return f"Date: {pd.Timestamp(d):%Y-%m-%d}. Company X is a US-listed stock.\n{Q_BOOM}"


def phash(p):
    return hashlib.sha256(p.encode()).hexdigest()[:16]


def p_up_from_logprobs(cands):
    """[{token, logProbability}] of the answer position -> P(A) / (P(A) + P(B)); NaN when neither appears."""
    pa = sum(np.exp(c["logProbability"]) for c in cands if c.get("token", "").strip().upper().startswith("A"))
    pb = sum(np.exp(c["logProbability"]) for c in cands if c.get("token", "").strip().upper().startswith("B"))
    return float(pa / (pa + pb)) if pa + pb > 0 else float("nan")


def letter(text):
    m = re.search(r"\b([AB])\b", str(text or "").upper())
    return m.group(1) if m else None


def confidence(p_up):
    p = np.asarray(p_up, float)
    return np.maximum(p, 1 - p)


def top_third(conf):
    """Mask of the top third by confidence; the cut comes from the confidence values alone."""
    c = np.asarray(conf, float)
    return c >= np.nanquantile(c, 2 / 3) if np.isfinite(c).any() else np.zeros(len(c), bool)


def random_sides(days, idea=IDEA):
    days = pd.DatetimeIndex(days)
    out = np.empty(len(days))
    for d in np.unique(days):
        ix = np.flatnonzero(days == d)
        out[ix] = np.random.default_rng(int(pd.Timestamp(d).strftime("%Y%m%d")) + idea).choice([-1.0, 1.0], len(ix))
    return out


def boot_days(x, n=BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "days": len(x)}
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)),
            "days": len(x)}


def probe_result(pr):
    """pr: probes with answer side (+1 A, -1 B) and the real direction (sign of close vs 9:51)."""
    x = pr[(pr["side"] != 0) & (pr["truth"] != 0)]
    hit = float((x["side"] == x["truth"]).mean()) if len(x) else float("nan")
    return {"answered": len(x), "hit_rate": hit, "a_share": float((pr["side"] == 1).mean()) if len(pr) else None,
            "within_band": bool(np.isfinite(hit) and abs(hit - 0.5) <= PROBE_BAND)}


def boom_result(b, probe_ok, cost_net):
    """b: answered BOOM items (d, conf, side, entry, exitp, lab, open). cost_net(side, entry, exit, price)."""
    b = b[np.isfinite(b["conf"])].reset_index(drop=True)
    top = b[top_third(b["conf"])]
    x = top[(top["lab"] != 2) & np.isfinite(top["exitp"]) & np.isfinite(top["entry"])].copy()
    if x.empty:
        return {"trades": 0, "verdict": "too few"}
    x["ctrl"] = random_sides(x["d"])
    pr = x["open"].to_numpy(float)
    x["net"] = cost_net(x["side"], x["entry"], x["exitp"], pr)
    x["ctrl_net"] = cost_net(x["ctrl"], x["entry"], x["exitp"], pr)
    gross = x["side"] * (x["exitp"] / x["entry"] - 1)
    pmc = boot_days(x.assign(_d=x["net"] - x["ctrl_net"]).groupby("d")["_d"].mean().to_numpy())
    n = len(x)
    touched = x["lab"] != 0
    ok = (n >= MIN_BOOM and pmc["mean"] >= BOOM_GAIN and pmc["lo"] is not None and pmc["lo"] > 0
          and x["net"].mean() > 0 and probe_ok)
    lab, side = x["lab"].to_numpy(), x["side"].to_numpy()
    rec = [float((side[lab == k] == k).mean()) for k in (1, -1) if (lab == k).any()]
    return {"verdict": "too few" if n < MIN_BOOM else "passed" if ok else "failed", "trades": n,
            "answered": len(b), "conf_cut": float(top["conf"].min()),
            "mean_net": float(x["net"].mean()), "mean_net_2x_cost": float((2 * x["net"] - gross).mean()),
            "control_mean_net": float(x["ctrl_net"].mean()), "arm_minus_control_per_day": pmc,
            "hit_rate_touched": float((x.loc[touched, "side"] == x.loc[touched, "lab"]).mean()) if touched.any() else None,
            "balanced_accuracy": float(np.mean(rec)) if len(rec) == 2 else None,
            "per_side": {s: {"n": int((x["side"] == v).sum()),
                             "mean_net": float(x.loc[x["side"] == v, "net"].mean()) if (x["side"] == v).any() else None}
                         for s, v in (("long", 1), ("short", -1))}}


def nx3_result(a, all_longs, probe_ok, pooled_diff):
    """a: answered NX3 longs (d, conf, side, net); all_longs: every NX3 long in the window (d, net)."""
    ya = a[(a["side"] == 1) & np.isfinite(a["conf"])].reset_index(drop=True)
    top = ya[top_third(ya["conf"])] if len(ya) else ya
    n = len(top)
    if n == 0:
        return {"trades": 0, "verdict": "too few", "answered": len(a)}
    gain = pooled_diff(top, all_longs)
    ok = n >= MIN_NX3 and gain["mean"] >= NX3_GAIN and gain["lo"] > 0 and top["net"].mean() > 0 and probe_ok
    return {"verdict": "too few" if n < MIN_NX3 else "passed" if ok else "failed", "trades": n,
            "answered": len(a), "a_share": float((a["side"] == 1).mean()), "conf_cut": float(top["conf"].min()),
            "mean_net": float(top["net"].mean()), "all_longs_mean_net": float(all_longs["net"].mean()),
            "all_longs": len(all_longs), "gain_over_all_longs": gain,
            "b_answers_mean_net": float(a.loc[a["side"] == -1, "net"].mean()) if (a["side"] == -1).any() else None}


# ---------- the model call ----------

class QuotaExhausted(Exception):
    pass


def _post(body):
    key = os.environ["GEMINI_API_KEY"].strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    for attempt in range(6):
        try:
            r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            low = r.text.lower()
            if "perday" in low or "per day" in low or attempt == 5:
                raise QuotaExhausted(r.text[:200])
            time.sleep(30)
            continue
        if r.status_code >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        return r
    raise RuntimeError("gemini: no answer after retries")


def ask_logprobs(p):
    body = {"contents": [{"role": "user", "parts": [{"text": p}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4, "responseLogprobs": True, "logprobs": 5,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    r = _post(body)
    if r.status_code == 400 and "logprob" in r.text.lower():
        return None                                            # not offered on this key: use votes
    r.raise_for_status()
    c = (r.json().get("candidates") or [{}])[0]
    txt = "".join(x.get("text", "") for x in c.get("content", {}).get("parts", []))
    tops = (c.get("logprobsResult") or {}).get("topCandidates") or []
    pos = next((t["candidates"] for t in tops if any(letter(x.get("token")) for x in t.get("candidates", []))), [])
    return {"text": txt, "top": pos, "p_up": p_up_from_logprobs(pos)}


def ask_votes(p):
    body = {"contents": [{"role": "user", "parts": [{"text": p}]}],
            "generationConfig": {"temperature": 1, "maxOutputTokens": 4, "candidateCount": 1,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    ans = []
    for _ in range(VOTES):
        r = _post(body)
        r.raise_for_status()
        c = (r.json().get("candidates") or [{}])[0]
        ans.append(letter("".join(x.get("text", "") for x in c.get("content", {}).get("parts", []))))
        time.sleep(MIN_GAP_SEC)
    got = [a for a in ans if a]
    return {"votes": ans, "p_up": float(sum(a == "A" for a in got) / len(got)) if got else float("nan")}


# ---------- stages ----------

def stage_items(nx1_cache, state):
    """Builds items.parquet (id, kind, d, prompt, hash) and outcomes.parquet (id and outcome columns)."""
    from core.alpaca import TRADE, get
    from tracks.main import boom2_study as B2
    from tracks.news import dir_gate as G
    from tracks.news import nx1_study as N
    from tracks.news import nx3_study as X
    from tracks.news import w1_data as W
    st = Path(state)
    st.mkdir(parents=True, exist_ok=True)
    names = {}
    for status in ("inactive", "active"):          # active last: a reused ticker takes its current company's name
        for a in get(f"{TRADE}/assets", {"status": status, "asset_class": "us_equity"}):
            names[a["symbol"]] = a.get("name") or ""
    ses = X.sessions(X.calendar(nx1_cache)).set_index("date")
    news = []
    for per in pd.period_range("2025-01", "2025-09", freq="M"):
        n = W.news_month(nx1_cache, per)
        n = n[n["symbols"].fillna("").ne("") & ~n["symbols"].fillna("").str.contains(",")]
        n = n[n["headline"].fillna("").str.strip().ne("")]
        news.append(pd.DataFrame({"symbol": n["symbols"].to_numpy(), "t": W.et_time(n["created_at"]).to_numpy(),
                                  "headline": n["headline"].to_numpy()}))
    news = pd.concat(news, ignore_index=True).sort_values("t")
    by = {k: g for k, g in news.groupby("symbol")}
    items, outs = [], []
    picks = B2.load_all(nx1_cache)
    picks = picks[(picks["nxt"] >= FIRST) & (picks["nxt"] <= LAST) & (picks["open"] >= 5.0)]
    for r in picks.itertuples(index=False):
        g = by.get(r.symbol)
        if g is None or r.nxt not in ses.index:
            continue
        lo, hi = ses.loc[r.nxt, "prev_close_t"], r.nxt + pd.Timedelta(hours=9, minutes=14)
        h = g[(g["t"] > lo) & (g["t"] <= hi)]["headline"].tolist()
        if not h:
            continue
        p = prompt([mask(x, r.symbol, names.get(r.symbol)) for x in h], Q_BOOM)
        iid = f"boom|{r.nxt:%Y-%m-%d}|{r.symbol}"
        items.append({"id": iid, "kind": "boom", "d": r.nxt, "prompt": p, "hash": phash(p)})
        outs.append({"id": iid, "d": r.nxt, "open": r.open, "entry": r.e951, "exitp": r.exit951, "lab": r.lab951,
                     "close": r.close})
    longs, _ = G.nx3_longs(nx1_cache, years=[2025])
    longs = longs[(longs["d"] >= FIRST) & (longs["d"] <= LAST)]
    for r in longs.itertuples(index=False):
        g = by.get(r.symbol)
        if g is None or r.d not in ses.index:
            continue
        h = g[(g["t"] > ses.loc[r.d, "prev_close_t"]) & (g["t"] <= r.t)]["headline"].tolist()
        iid = f"nx3|{r.d:%Y-%m-%d}|{r.symbol}"
        p = prompt([mask(x, r.symbol, names.get(r.symbol)) for x in h], Q_NX3) if h else None
        if p:
            items.append({"id": iid, "kind": "nx3", "d": r.d, "prompt": p, "hash": phash(p)})
        outs.append({"id": iid, "d": r.d, "net": r.net, "has_headline": bool(p)})
    bo = [o for o in outs if o["id"].startswith("boom|")]
    rng = np.random.default_rng(SEED)
    for k in rng.choice(len(bo), size=min(PROBES, len(bo)), replace=False):
        o = bo[int(k)]
        p = probe_prompt(o["d"])
        iid = "probe|" + o["id"].split("|", 1)[1]
        items.append({"id": iid, "kind": "probe", "d": o["d"], "prompt": p, "hash": phash(p)})
        outs.append({"id": iid, "d": o["d"], "truth": float(np.sign(o["close"] / o["entry"] - 1))
                     if np.isfinite(o["entry"]) else 0.0})
    it = pd.DataFrame(items)
    order = np.r_[np.flatnonzero(it["kind"] == "probe"), rng.permutation(np.flatnonzero(it["kind"] != "probe"))]
    N.atomic_parquet(it.iloc[order].reset_index(drop=True), st / "items.parquet")
    N.atomic_parquet(pd.DataFrame(outs), st / "outcomes.parquet")
    meta = {"model": MODEL, "cutoff": CUTOFF_NOTE, "questions": {"boom": Q_BOOM, "nx3": Q_NX3},
            "question_hashes": {"boom": phash(Q_BOOM), "nx3": phash(Q_NX3)},
            "items": {k: int((it["kind"] == k).sum()) for k in ("boom", "nx3", "probe")},
            "nx3_longs_in_window": len(longs)}
    (st / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def stage_calls(state, deadline, max_requests=None):
    st = Path(state)
    it = pd.read_parquet(st / "items.parquet")
    f = st / "answers.jsonl"
    done = set()
    if f.exists():
        done = {json.loads(x)["id"] for x in f.read_text().splitlines() if x.strip()}
    mf = st / "mode.json"
    mode = json.loads(mf.read_text())["mode"] if mf.exists() else None
    sent = 0
    with f.open("a") as fh:
        for r in it.itertuples(index=False):
            if r.id in done:
                continue
            if time.monotonic() > deadline or (max_requests is not None and sent >= max_requests):
                break
            try:
                res = None
                if mode in (None, "logprobs"):
                    res = ask_logprobs(r.prompt)
                    sent += 1
                    if res is None:
                        mode = "votes"
                    elif mode is None:
                        mode = "logprobs" if np.isfinite(res["p_up"]) else "votes"
                    if mode == "votes":
                        res = None
                    mf.write_text(json.dumps({"mode": mode, "decided": pd.Timestamp.utcnow().isoformat()}))
                if res is None:
                    res = ask_votes(r.prompt)
                    sent += VOTES
                time.sleep(MIN_GAP_SEC)
            except QuotaExhausted as e:
                print(f"[dirllm] daily quota used ({sent} requests this run): {str(e)[:120]}")
                break
            fh.write(json.dumps({"id": r.id, "kind": r.kind, "model": MODEL, "prompt_hash": r.hash, "mode": mode,
                                 "t": pd.Timestamp.utcnow().isoformat()} | res, default=float) + "\n")
            fh.flush()
    n = len(f.read_text().splitlines()) if f.exists() else 0
    print(f"[dirllm] {n} of {len(it)} answered (mode {mode}); {sent} requests this run")
    return n >= len(it)


def run_test(state, out):
    from tracks.main import boom2_study as B2
    from tracks.news import dir_gate as G
    st = Path(state)
    ans = pd.DataFrame([json.loads(x) for x in (st / "answers.jsonl").read_text().splitlines() if x.strip()])
    ans["conf"] = confidence(ans["p_up"])
    ans["side"] = np.where(ans["p_up"] > 0.5, 1.0, np.where(ans["p_up"] < 0.5, -1.0, 0.0))
    # the confidence cuts are fixed inside boom_result / nx3_result from `conf` before outcomes matter
    oc = pd.read_parquet(st / "outcomes.parquet")
    j = ans.merge(oc, on="id", how="inner")
    j["d"] = pd.to_datetime(j["d"])
    pr = probe_result(j[j["kind"] == "probe"])
    meta = json.loads((st / "meta.json").read_text())
    rep = {"idea_count": "145 (160 with phase 1)", "model": MODEL, "cutoff": CUTOFF_NOTE,
           "mode": json.loads((st / "mode.json").read_text())["mode"] if (st / "mode.json").exists() else None,
           "question_hashes": meta["question_hashes"], "items": meta["items"], "answered": len(ans),
           "probe": pr,
           "boom": boom_result(j[j["kind"] == "boom"], pr["within_band"], B2.net),
           "nx3": nx3_result(j[j["kind"] == "nx3"], oc[oc["id"].str.startswith("nx3|")].assign(d=lambda x: pd.to_datetime(x["d"])),
                             pr["within_band"], G.boot_pooled_diff)}
    rep["complete"] = bool(len(ans) >= sum(meta["items"].values()))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "DIRLLM.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["items", "calls", "test"], required=True)
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--state", default="cache/dirllm")
    ap.add_argument("--out", default="research/results/DIR")
    ap.add_argument("--budget-min", type=float, default=100)
    ap.add_argument("--max-requests", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "items":
        stage_items(a.cache, a.state)
    elif a.stage == "calls":
        done = stage_calls(a.state, time.monotonic() + a.budget_min * 60, a.max_requests)
        print("[dirllm] all answered" if done else "[dirllm] not finished; the next run continues")
    else:
        run_test(a.state, a.out)


if __name__ == "__main__":
    main()
