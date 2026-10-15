"""Nightly short-borrow fees and shares available, from Interactive Brokers' public stock-loan file.

IBKR publishes its shortable-stock list on a public FTP server (user "shortstock", no password),
updated through the day: one pipe-separated line per stock with the rebate rate, the fee rate (% a
year) and the shares available to borrow. Alpaca only says yes/no to "easy to borrow", and P1 assumes
a flat 20% a year; this log gives each P1 trade the real fee on the night it was decided (P1
clarification 11). Each night's slim copy (symbol, fee, rebate, available) is written once to the
hash-chained ledger under <state>/borrow/.

Usage: python -m core.borrow --state <dir>
"""
import argparse
import ftplib
import gzip
import io

import pandas as pd

from core.ledger import Ledger

HOST, USER, FILE = "ftp3.interactivebrokers.com", "shortstock", "usa.txt"
ET = "America/New_York"


def fetch(host=HOST):
    buf = io.BytesIO()
    with ftplib.FTP(host, timeout=120) as f:
        f.login(USER, "")
        f.retrbinary(f"RETR {FILE}", buf.write)
    return buf.getvalue().decode("latin-1")


def parse(text):
    """(file time 'YYYY.MM.DD HH:MM:SS' or None, frame symbol, fee, rebate, available). The column
    order comes from the '#SYM|...' header line. Available '>10000000' reads as 10000000; 'NA' and
    blanks are missing."""
    when, cols, rows = None, None, []
    for ln in text.splitlines():
        p = ln.rstrip("|").split("|")
        if p[0] == "#BOF" and len(p) >= 3:
            when = f"{p[1]} {p[2]}"
        elif p[0] == "#SYM":
            cols = [c.strip().upper() for c in p]
            cols[0] = "SYM"
        elif cols and p[0] and not p[0].startswith("#") and len(p) >= len(cols) - 1:
            rows.append(dict(zip(cols, p)))
    d = pd.DataFrame(rows, columns=cols or ["SYM", "FEERATE", "REBATERATE", "AVAILABLE"])

    def num(s):
        return pd.to_numeric(s.astype(str).str.strip().str.lstrip(">"), errors="coerce")
    out = pd.DataFrame({"symbol": d["SYM"].str.strip().str.replace(" ", ".", regex=False),
                        "fee": num(d["FEERATE"]) / 100, "rebate": num(d["REBATERATE"]) / 100,
                        "available": num(d["AVAILABLE"])})
    return when, out.drop_duplicates("symbol").reset_index(drop=True)


def run(state_dir, now=None, get=fetch):
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    day = f"{now.tz_convert(ET):%Y-%m-%d}"
    led = Ledger(f"{state_dir}/borrow")
    if led.has("ibkr_borrow", day):
        print(f"borrow: {day} already logged")
        return None
    when, d = parse(get())
    if not len(d):
        raise RuntimeError("borrow: empty or unreadable IBKR file")
    body = f"# ibkr {FILE} as of {when}; fetched {now:%Y-%m-%dT%H:%M:%SZ}\n" + d.to_csv(index=False)
    led.write_bytes("ibkr_borrow", day, f"{day}.csv.gz", gzip.compress(body.encode(), mtime=0))
    print(f"borrow: {day}: {len(d)} symbols (file time {when})")
    return d


def latest(state_dir):
    """The newest logged night's frame (symbol, fee, rebate, available) and its day, or (None, None)."""
    led = Ledger(f"{state_dir}/borrow")
    es = [e for e in led.entries() if e["kind"] == "ibkr_borrow"]
    if not es:
        return None, None
    e = es[-1]
    txt = gzip.decompress((led.root / e["file"]).read_bytes()).decode()
    return pd.read_csv(io.StringIO(txt), comment="#"), e["date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
