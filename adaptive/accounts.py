"""Plan v3 paper accounts: one per funded strategy (SYS1 clarification 1), plus EL-DATA for prices and news.

`python -m adaptive.accounts` checks every account whose secrets are set: it prints the account's status,
equity, last 4 digits of its account number and whether Alpaca still flags pattern-day-trader limits.
It never prints a key.
"""
import os

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


def main():
    for name in ACCOUNTS:
        print(check(name))


if __name__ == "__main__":
    main()
