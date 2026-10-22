"""Plan v3 paper accounts: one per funded strategy (SYS1 clarification 1), plus EL-DATA for prices and news.

`python -m adaptive.accounts` checks every account whose secrets are set: it prints the account's status,
equity, last 4 digits of its account number and whether Alpaca still flags pattern-day-trader limits.
It never prints a key.
"""
import os
from datetime import datetime, timedelta, timezone

import requests

from core.alpaca import TRADE

ACCOUNTS = {          # Alpaca account name -> secret prefix (<prefix>_KEY_ID, <prefix>_SECRET_KEY)
    "EL-DATA": "ALPACA_API",
    "EL-P1P2": "ALPACA_P1P2",
    "EL-M3": "ALPACA_M3",
    "EL-M4": "ALPACA_M4",
    "EL-M6": "ALPACA_M6",
    "EL-P4": "ALPACA_P4",
    "EL-PICK-MAIN": "ALPACA_PICK_MAIN",
    "EL-PICK-PENNY": "ALPACA_PICK_PENNY",
}


def headers(name):
    p = ACCOUNTS[name]
    key, secret = (os.environ.get(f"{p}_KEY_ID") or "").strip(), (os.environ.get(f"{p}_SECRET_KEY") or "").strip()
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def check(name, get=requests.get):
    h = headers(name)
    if h is None:
        return f"{name}: secrets not set"
    try:
        r = get(f"{TRADE}/account", headers=h, timeout=30)
    except requests.RequestException as e:
        return f"{name}: no answer ({type(e).__name__})"
    if r.status_code != 200:
        return f"{name}: HTTP {r.status_code} (wrong key or not a paper account)"
    a = r.json()
    return (f"{name}: OK, status {a.get('status')}, equity ${float(a.get('equity', 0)):,.0f}, "
            f"account ...{str(a.get('account_number', ''))[-4:]}, pattern_day_trader={a.get('pattern_day_trader')}, "
            f"shorting_enabled={a.get('shorting_enabled')}")


def activity(name, get=requests.get):
    """Lines for each open position (unrealized P&L in %) and each order of the last 7 days."""
    h = headers(name)
    if h is None:
        return []
    out = []
    try:
        for p in get(f"{TRADE}/positions", headers=h, timeout=30).json():
            out.append(f"  position {p['symbol']} qty {p['qty']} entry {float(p['avg_entry_price']):.4f} "
                       f"now {float(p['current_price']):.4f} P&L ${float(p['unrealized_pl']):.2f} "
                       f"({100 * float(p['unrealized_plpc']):.2f}%)")
        after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for o in get(f"{TRADE}/orders", params={"status": "all", "after": after, "limit": 100}, headers=h,
                     timeout=30).json():
            out.append(f"  order {o['symbol']} {o['side']} {o['qty']} {o['type']}/{o['time_in_force']} "
                       f"status {o['status']} filled {o.get('filled_qty')} @ {o.get('filled_avg_price')}")
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        out.append(f"  activity unavailable ({type(e).__name__})")
    return out


def main():
    for name in ACCOUNTS:
        print(check(name))
        for line in activity(name):
            print(line)


if __name__ == "__main__":
    main()
