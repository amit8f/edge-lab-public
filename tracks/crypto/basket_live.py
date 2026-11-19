"""CRY-BIG and CRY-MEME forward books (research/prereg/CRYBASKETS.md): CRY-TREND's rule (hold a coin while its daily
close is above its close 28 days earlier, 1/k of the book per coin) on a big-coin and a meme-coin basket, each a
$1,000 ledger book next to a buy-and-hold twin. Ledger only: nothing is ever sent to an exchange.

    python -m tracks.crypto.basket_live --state state
"""
import argparse

from tracks.crypto import trend_live as L

BOOKS = {
    "crybig": {"coins": {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD", "BNB": "BNBUSD"},
               "cost": 0.005},
    "crymeme": {"coins": {"DOGE": "XDGUSD", "SHIB": "SHIBUSD", "PEPE": "PEPEUSD", "BONK": "BONKUSD", "WIF": "WIFUSD",
                          "FLOKI": "FLOKIUSD"}, "cost": 0.01},
}
MIN_ROUND_TRIPS = 15


def run(state, fetch=L.fetch):
    bad = []
    for name, cfg in BOOKS.items():
        try:
            L.run(state, fetch=lambda c=cfg["coins"]: fetch(c), name=name, coins=cfg["coins"], cost=cfg["cost"],
                  min_round_trips=MIN_ROUND_TRIPS)
        except Exception as e:      # one basket's bad candle day must not stop the other book
            print(f"[{name}] FAILED: {e}", flush=True)
            bad.append(name)
    if bad:
        raise RuntimeError(f"books failed: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
