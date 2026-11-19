"""CRY-NEWMEME (research/prereg/CRYNEWMEME.md): watch brand-new Solana and Base meme tokens, log a snapshot of every
tracked token (the dead ones too), and run the fresh-rule $1,000 paper book with a random-token control.
Paper only: no wallet, nothing is ever sent on chain.

One call runs for --minutes (the workflow loops it for about 5.5 hours and saves the state between calls).

    python -m tracks.crypto.newmeme_live --state state --minutes 25
"""
import argparse
import io
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.ledger import Ledger

CHAINS = {"solana": "solana", "base": "base"}           # GeckoTerminal network id -> DexScreener chain id
GT = "https://api.geckoterminal.com/api/v2/networks"
DS = "https://api.dexscreener.com"
RUGCHECK = "https://api.rugcheck.xyz/v1/tokens/{}/report/summary"
HONEYPOT = "https://api.honeypot.is/v2/IsHoneypot?address={}&chainID=8453"

TRACK_LIQ, DEAD_LIQ, TRACK_DAYS, MAX_TRACKED = 10_000, 1_000, 7, 1_500
FAST_AGE_H, FAST_S, SLOW_S, DISCOVER_S, TRENDING_S = 24, 300, 1800, 60, 300
START_USD, SIZE, MAX_OPEN, SIDE_COST = 1000.0, 25.0, 10, 0.015
AGE_MIN, AGE_MAX = 15 / 60, 6.0                         # hours
MIN_LIQ, MIN_VOL1H = 20_000, 50_000
TP, SL, HOLD_H, RUG_LIQ, MISS_MAX = 2.0, 0.5, 24, 0.10, 3
THEME_N, THEME_H = 5, 6
STOP = {"the", "and", "coin", "token", "sol", "solana", "base", "inu", "meme", "official", "for", "you", "new", "of",
        "on", "ai", "usd", "wif", "dog", "cat", "pump", "fun", "moon", "eth", "baby", "mini", "elon", "real"}
SNAP_COLS = ["ts", "chain", "token", "pair", "symbol", "name", "age_h", "price", "liq", "fdv", "mcap", "buys5", "sells5",
             "buys1h", "sells1h", "vol5", "vol1h", "ch5", "ch1h", "ch6h"]


# ---------- pure helpers (tested) ----------

def words(*texts):
    out = set()
    for t in texts:
        for w in re.findall(r"[a-z0-9]+", str(t or "").lower()):
            if len(w) >= 3 and w not in STOP and not w.isdigit():
                out.add(w)
    return out


def hot_words(seen, now, n=THEME_N, hours=THEME_H):
    """Words shared by at least n different tokens first seen in the last `hours`."""
    cnt = {}
    for tok in seen.values():
        if now - tok["first_seen"] <= hours * 3600:
            for w in tok.get("words", []):
                cnt[w] = cnt.get(w, 0) + 1
    return {w for w, c in cnt.items() if c >= n}


def gt_pools(js):
    """GeckoTerminal pools response -> [{token, symbol, name, created, liq}]."""
    out = []
    for p in (js or {}).get("data", []) or []:
        a, r = p.get("attributes", {}), p.get("relationships", {})
        tid = (((r.get("base_token") or {}).get("data") or {}).get("id") or "")
        if "_" not in tid:
            continue
        name = a.get("name") or ""
        try:
            created = datetime.fromisoformat(str(a.get("pool_created_at")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            created = None
        out.append({"token": tid.split("_", 1)[1], "symbol": name.split(" / ")[0].strip(), "name": name,
                    "created": created, "liq": _f(a.get("reserve_in_usd"))})
    return out


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def best_pairs(pairs):
    """DexScreener pairs -> {token address: the pair with the most liquidity where the token is the base}."""
    out = {}
    for p in pairs or []:
        tok = ((p.get("baseToken") or {}).get("address") or "")
        liq = _f((p.get("liquidity") or {}).get("usd"))
        if tok and (tok not in out or np.nan_to_num(liq) > np.nan_to_num(_f((out[tok].get("liquidity") or {}).get("usd")))):
            out[tok] = p
    return out


def snap_row(p, chain, now):
    tx, vol, ch = p.get("txns") or {}, p.get("volume") or {}, p.get("priceChange") or {}
    created = p.get("pairCreatedAt")
    return {"ts": now, "chain": chain, "token": p["baseToken"]["address"], "pair": p.get("pairAddress"),
            "symbol": p["baseToken"].get("symbol"), "name": p["baseToken"].get("name"),
            "age_h": (now - created / 1000) / 3600 if created else np.nan, "price": _f(p.get("priceUsd")),
            "liq": _f((p.get("liquidity") or {}).get("usd")), "fdv": _f(p.get("fdv")), "mcap": _f(p.get("marketCap")),
            "buys5": _f((tx.get("m5") or {}).get("buys")), "sells5": _f((tx.get("m5") or {}).get("sells")),
            "buys1h": _f((tx.get("h1") or {}).get("buys")), "sells1h": _f((tx.get("h1") or {}).get("sells")),
            "vol5": _f(vol.get("m5")), "vol1h": _f(vol.get("h1")), "ch5": _f(ch.get("m5")), "ch1h": _f(ch.get("h1")),
            "ch6h": _f(ch.get("h6"))}


def fresh_ok(r):
    """The fresh rule's market conditions (the rug check is separate)."""
    return bool(AGE_MIN <= r["age_h"] <= AGE_MAX and r["liq"] >= MIN_LIQ and r["vol1h"] >= MIN_VOL1H
                and r["buys1h"] > r["sells1h"] and r["ch1h"] > 0 and r["price"] > 0)


def control_ok(r):
    return bool(AGE_MIN <= r["age_h"] <= AGE_MAX and r["liq"] >= MIN_LIQ and r["price"] > 0)


def rug_pass(chain, js):
    """True / False from the checker's answer, None when it can't be read."""
    if not js:
        return None
    if chain == "solana":
        risks = js.get("risks")
        if risks is None:
            return None
        return not any(str(x.get("level", "")).lower() == "danger" for x in risks)
    hp = js.get("honeypotResult") or {}
    if "isHoneypot" not in hp:
        return None
    tax = _f((js.get("simulationResult") or {}).get("sellTax"))
    return bool(not hp["isHoneypot"] and (np.isnan(tax) or tax <= 10))


def open_trade(token, chain, r, now, arm):
    return {"arm": arm, "token": token, "chain": chain, "symbol": r["symbol"], "entry_ts": now, "entry_px": r["price"],
            "entry_liq": r["liq"], "last_px": r["price"], "miss": 0, "usd": SIZE}


def exit_reason(tr, r, now):
    """(reason, exit price) or None. r is the token's snapshot this refresh, or None when the pair couldn't be read."""
    if r is None or not (r["price"] > 0):
        if tr["miss"] + 1 >= MISS_MAX:
            return "rug_missing", tr["last_px"] if tr.get("seen_after") else 0.0
        return None
    x = r["price"] / tr["entry_px"]
    if x >= TP:
        return "take_profit", r["price"]
    if x <= SL:
        return "stop", r["price"]
    if r["liq"] < RUG_LIQ * tr["entry_liq"]:
        return "rug_liquidity", r["price"]
    if now - tr["entry_ts"] >= HOLD_H * 3600:
        return "time", r["price"]
    return None


def net_return(entry_px, exit_px):
    return (exit_px * (1 - SIDE_COST)) / (entry_px * (1 + SIDE_COST)) - 1


def clustered_t(net, day):
    net = np.asarray(net, float)
    if len(net) < 2:
        return float("nan")
    e = pd.Series(net - net.mean()).groupby(np.asarray(day)).sum().to_numpy()
    se = np.sqrt((e ** 2).sum()) / len(net)
    return float(net.mean() / se) if se > 0 else float("nan")


def summarize(closed, open_trades, now):
    out = {"updated": now, "open": sum(1 for t in open_trades if t["arm"] == "fresh"), "arms": {}}
    for arm in ("fresh", "control"):
        c = [t for t in closed if t["arm"] == arm]
        net = np.array([t["net"] for t in c])
        out["arms"][arm] = {
            "closed": len(c), "mean_net": float(net.mean()) if len(c) else None,
            "win": float((net > 0).mean()) if len(c) else None,
            "pnl_usd": float(sum(t["net"] * t["usd"] for t in c)),
            "t_day": clustered_t(net, [datetime.fromtimestamp(t["entry_ts"], timezone.utc).date() for t in c]),
            "reasons": {k: sum(1 for t in c if t["reason"] == k) for k in sorted({t["reason"] for t in c})}}
    return out


# ---------- the watcher ----------

class Http:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["Accept"] = "application/json"
        self.errors = 0

    def __call__(self, url):
        for k in range(3):
            try:
                r = self.s.get(url, timeout=20)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (404, 400):
                    return None
                time.sleep(2 * (k + 1))
            except (requests.RequestException, ValueError):
                time.sleep(2 * (k + 1))
        self.errors += 1
        return None


def load_work(root):
    f = root / "work.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"seen": {}, "tracked": {}, "open": [], "next_id": 1, "last": {}}


def save_work(root, w):
    tmp = root / "work.tmp"
    tmp.write_text(json.dumps(w))
    tmp.replace(root / "work.json")


def run(state, minutes=25, get=None, clock=time.time, sleep=time.sleep, rng=None):
    root = Path(state) / "newmeme"
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(root / "ledger")
    get = get or Http()
    rng = rng or random.Random()
    w = load_work(root)
    seen, tracked, last = w["seen"], w["tracked"], w["last"]
    t0 = clock()
    chunk = datetime.fromtimestamp(t0, timezone.utc).strftime("%Y%m%dT%H%M%S")
    snaps, events = [], {"chunk": chunk, "entries": [], "exits": [], "rug_checks": [], "dropped": 0, "gaps": []}
    if last.get("refresh") and t0 - last["refresh"] > 2 * FAST_S:
        events["gaps"].append([last["refresh"], t0])
    while clock() - t0 < minutes * 60:
        now = clock()
        if now - last.get("discover", 0) >= DISCOVER_S:
            last["discover"] = now
            found = []
            for net in CHAINS:
                found += [(net, p) for p in gt_pools(get(f"{GT}/{net}/new_pools?page=1"))]
            if now - last.get("trending", 0) >= TRENDING_S:
                last["trending"] = now
                for net in CHAINS:
                    found += [(net, p) for p in gt_pools(get(f"{GT}/{net}/trending_pools?page=1"))]
            for net, p in found:
                key = f"{CHAINS[net]}:{p['token']}"
                if key not in seen:
                    seen[key] = {"first_seen": now, "words": sorted(words(p["name"], p["symbol"])), "created": p["created"]}
                if p["liq"] >= TRACK_LIQ and key not in tracked:
                    tracked[key] = {"since": now, "last": 0, "miss": 0}
            for pr in get(f"{DS}/token-profiles/latest/v1") or []:
                ch, tok = pr.get("chainId"), pr.get("tokenAddress")
                if ch in CHAINS.values() and tok:
                    key = f"{ch}:{tok}"
                    if key not in seen:
                        seen[key] = {"first_seen": now, "words": sorted(words(pr.get("description", "")[:80])), "created": None}
                    tracked.setdefault(key, {"since": now, "last": 0, "miss": 0, "probe": True})
        due = {}
        for key, tk in tracked.items():
            age_h = (now - (seen.get(key, {}).get("created") or tk["since"])) / 3600
            if now - tk["last"] >= (FAST_S if age_h < FAST_AGE_H else SLOW_S) - 5:
                due.setdefault(key.split(":", 1)[0], []).append(key.split(":", 1)[1])
        held = {t["chain"] + ":" + t["token"] for t in w["open"]}
        for t in w["open"]:              # open trades refresh every 5 minutes whatever the token's age
            ch, tok = t["chain"], t["token"]
            tk = tracked.setdefault(f"{ch}:{tok}", {"since": now, "last": 0, "miss": 0})
            if now - tk["last"] >= FAST_S - 5 and tok not in due.get(ch, []):
                due.setdefault(ch, []).append(tok)
        refreshed = {f"{ch}:{tok}" for ch, toks in due.items() for tok in toks}
        rows = {}
        for ch, toks in due.items():
            for i in range(0, len(toks), 30):
                part = toks[i:i + 30]
                best = best_pairs(get(f"{DS}/tokens/v1/{ch}/{','.join(part)}"))
                for tok in part:
                    key = f"{ch}:{tok}"
                    if key in tracked:
                        tracked[key]["last"] = now
                    if tok in best:
                        r = snap_row(best[tok], ch, now)
                        rows[key] = r
                        snaps.append(r)
        if due:
            last["refresh"] = now
        for key in list(tracked):        # stop tracking the old, the dead and the probes that never qualified
            r, tk = rows.get(key), tracked[key]
            if key in held:
                continue
            if r is not None:
                tk["miss"] = 0
                if tk.pop("probe", False) and not r["liq"] >= TRACK_LIQ:
                    del tracked[key]
                    continue
                if r["liq"] < DEAD_LIQ or now - tk["since"] > TRACK_DAYS * 86400:
                    del tracked[key]
            elif tk["last"] == now:
                tk["miss"] += 1
                if tk["miss"] >= 6:
                    del tracked[key]
        if len(tracked) > MAX_TRACKED:
            keep = sorted(tracked, key=lambda k: -(rows[k]["liq"] if k in rows else 0))[:MAX_TRACKED]
            events["dropped"] += len(tracked) - len(keep)
            for k in set(tracked) - set(keep) - held:
                del tracked[k]
        # exits
        still = []
        for tr in w["open"]:
            key = tr["chain"] + ":" + tr["token"]
            if key not in refreshed:
                still.append(tr)
                continue
            r = rows.get(key)
            ex = exit_reason(tr, r, now)
            if ex is None:
                if r is None:
                    tr["miss"] += 1
                else:
                    tr["miss"], tr["last_px"], tr["seen_after"] = 0, r["price"], True
                still.append(tr)
                continue
            tr.update(reason=ex[0], exit_ts=now, exit_px=ex[1], net=net_return(tr["entry_px"], ex[1]))
            events["exits"].append(tr)
        w["open"] = still
        # entries
        hot = hot_words(seen, now)
        held = {t["chain"] + ":" + t["token"] for t in w["open"]}
        traded = set(w.setdefault("traded", []))
        for key, r in sorted(rows.items(), key=lambda kv: -kv[1]["vol1h"] if np.isfinite(kv[1]["vol1h"]) else 0):
            if sum(1 for t in w["open"] if t["arm"] == "fresh") >= MAX_OPEN:
                break
            if key in held or key in traded or not fresh_ok(r):
                continue
            ch, tok = key.split(":", 1)
            js = get(RUGCHECK.format(tok)) if ch == "solana" else get(HONEYPOT.format(tok))
            ok = rug_pass(ch, js)
            events["rug_checks"].append([key, ok])
            if not ok:
                continue
            tr = open_trade(tok, ch, r, now, "fresh")
            tr["id"], tr["hot"] = w["next_id"], sorted(set(seen.get(key, {}).get("words", [])) & hot)
            w["next_id"] += 1
            pool = [k for k, x in rows.items() if k != key and k not in held and control_ok(x)]
            if pool:
                ck = rng.choice(sorted(pool))
                c = open_trade(ck.split(":", 1)[1], ck.split(":", 1)[0], rows[ck], now, "control")
                c["id"], c["pair_of"] = w["next_id"], tr["id"]
                w["next_id"] += 1
                w["open"].append(c)
                held.add(ck)
            w["open"].append(tr)
            held.add(key)
            traded.add(key)
            events["entries"].append(tr)
        w["traded"] = sorted(traded)
        for k in [k for k, v in seen.items() if now - v["first_seen"] > TRACK_DAYS * 86400 and k not in tracked]:
            del seen[k]
        rest = DISCOVER_S - (clock() - now)
        if rest > 0 and clock() - t0 + rest < minutes * 60:
            sleep(rest)
        elif rest > 0:
            break
    # log this chunk (append-only) and the working state
    events["http_errors"] = getattr(get, "errors", 0)
    if snaps:
        buf = io.BytesIO()
        pd.DataFrame(snaps, columns=SNAP_COLS).to_parquet(buf, compression="zstd", index=False)
        led.write_bytes("snaps", chunk, f"snaps/{chunk[:8]}/{chunk}.parquet", buf.getvalue())
    led.write_json("events", chunk, f"events/{chunk[:8]}/{chunk}.json", events)
    closed = []
    for e in led.entries():
        if e["kind"] == "events":
            closed += json.loads((led.root / e["file"]).read_text())["exits"]
    s = summarize(closed, w["open"], clock())
    s.update(tracked=len(tracked), seen=len(seen), start_usd=START_USD, size_usd=SIZE,
             hot=sorted(hot_words(seen, clock()))[:20])
    (root / "summary.tmp").write_text(json.dumps(s, indent=1))
    (root / "summary.tmp").replace(root / "summary.json")
    save_work(root, w)
    probs = led.verify()
    if probs:
        raise RuntimeError(f"ledger problems: {probs[:3]}")
    print(f"[newmeme] {chunk}: {len(snaps)} snapshots, {len(tracked)} tracked, {len(events['entries'])} entries, "
          f"{len(events['exits'])} exits, {events['http_errors']} http errors", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--minutes", type=float, default=25)
    a = ap.parse_args()
    run(a.state, a.minutes)


if __name__ == "__main__":
    main()
