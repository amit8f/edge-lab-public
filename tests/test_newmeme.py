import json

import numpy as np

from core.ledger import Ledger
from tracks.crypto import newmeme_live as N

T0 = 1_800_000_000.0


def _pool(tok, liq, name):
    return {"attributes": {"name": f"{name} / SOL", "reserve_in_usd": str(liq), "pool_created_at": "2027-01-15T07:00:00Z"},
            "relationships": {"base_token": {"data": {"id": f"solana_{tok}"}}}}


def _pair(tok, px, liq, created, vol=80_000, buys=300, sells=200, ch1h=12.0):
    return {"pairAddress": "P" + tok, "baseToken": {"address": tok, "symbol": tok.upper(), "name": f"gta six {tok}"},
            "priceUsd": str(px), "liquidity": {"usd": liq}, "fdv": 1e6, "marketCap": 1e6, "pairCreatedAt": created * 1000,
            "txns": {"m5": {"buys": 10, "sells": 5}, "h1": {"buys": buys, "sells": sells}},
            "volume": {"m5": 5000, "h1": vol}, "priceChange": {"m5": 1, "h1": ch1h, "h6": 30}}


class World:
    def __init__(self):
        self.t = T0
        self.px = {"aaa": 1.0, "bbb": 1.0}
        self.calls = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def get(self, url):
        self.calls.append(url)
        if "new_pools" in url and "solana" in url:
            return {"data": [_pool("aaa", 50_000, "GTA6 Moon"), _pool("bbb", 40_000, "Other"), _pool("tiny", 500, "x")]}
        if "geckoterminal" in url or "token-profiles" in url:
            return {"data": []} if "geckoterminal" in url else []
        if "rugcheck" in url:
            return {"risks": [{"name": "x", "level": "warn"}]}
        if "/tokens/v1/solana/" in url:
            toks = url.rsplit("/", 1)[1].split(",")
            made = T0 - 3600
            return [_pair(t, self.px[t], 50_000, made, vol=80_000 if t == "aaa" else 1_000) for t in toks if t in self.px]
        return None


def test_pure_rules():
    assert N.words("GTA 6 Moon", "GTA6") == {"gta", "gta6"}
    seen = {str(i): {"first_seen": 0, "words": ["gta"]} for i in range(5)}
    assert N.hot_words(seen, 3600) == {"gta"} and N.hot_words(seen, 7 * 3600) == set()
    assert N.rug_pass("solana", {"risks": [{"level": "danger"}]}) is False
    assert N.rug_pass("solana", {"risks": []}) is True and N.rug_pass("solana", None) is None
    assert N.rug_pass("base", {"honeypotResult": {"isHoneypot": False}, "simulationResult": {"sellTax": 12}}) is False
    assert N.rug_pass("base", {"honeypotResult": {"isHoneypot": False}, "simulationResult": {"sellTax": 3}}) is True
    assert np.isclose(N.net_return(1.0, 2.0), 2 * 0.985 / 1.015 - 1)
    tr = {"entry_px": 1.0, "entry_liq": 50_000, "entry_ts": 0, "miss": 2, "last_px": 1.0}
    assert N.exit_reason(tr, None, 10) == ("rug_missing", 0.0)
    r = {"price": 0.4, "liq": 50_000}
    assert N.exit_reason(dict(tr, miss=0), r, 10) == ("stop", 0.4)
    assert N.exit_reason(dict(tr, miss=0), {"price": 1.1, "liq": 1000}, 10)[0] == "rug_liquidity"
    assert N.exit_reason(dict(tr, miss=0), {"price": 1.1, "liq": 50_000}, 25 * 3600)[0] == "time"


def test_watcher_enters_with_a_control_and_takes_profit_across_runs(tmp_path):
    w = World()
    s = N.run(tmp_path, minutes=3, get=w.get, clock=w.clock, sleep=w.sleep, rng=N.random.Random(0))
    assert s["open"] == 1 and s["tracked"] == 2                 # aaa entered, bbb is the control, tiny not tracked
    work = json.loads((tmp_path / "newmeme" / "work.json").read_text())
    arms = sorted((t["arm"], t["token"]) for t in work["open"])
    assert arms == [("control", "bbb"), ("fresh", "aaa")]
    assert any("rugcheck" in u and "aaa" in u for u in w.calls)
    w.px["aaa"], w.px["bbb"] = 2.5, 0.3
    w.t += 600
    s = N.run(tmp_path, minutes=3, get=w.get, clock=w.clock, sleep=w.sleep, rng=N.random.Random(0))
    f, c = s["arms"]["fresh"], s["arms"]["control"]
    assert f["closed"] == 1 and f["reasons"] == {"take_profit": 1} and np.isclose(f["mean_net"], N.net_return(1, 2.5))
    assert c["closed"] == 1 and c["reasons"] == {"stop": 1}
    led = Ledger(tmp_path / "newmeme" / "ledger")
    assert led.verify() == [] and {e["kind"] for e in led.entries()} == {"snaps", "events"}
