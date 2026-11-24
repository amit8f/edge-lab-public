"""Nightly short-borrow log from IBKR's public short-stock file (no account, no login).

IBKR publishes availability and borrow fees for every US stock it lends on its anonymous FTP
(user `shortstock`, file `usa.txt`, pipe-separated, header line starting `#SYM`). This saves one
compact gzip CSV per day (symbol, fee rate %/yr, rebate %/yr, shares available) under
<state>/borrow/, so studies can charge each short the fee it actually had (Amit, 2026-11-24).
"""
from __future__ import annotations

import argparse
import ftplib
import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HOSTS = ("ftp3.interactivebrokers.com", "ftp2.interactivebrokers.com")
COLS = {"#SYM": "symbol", "REBATERATE": "rebate", "FEERATE": "fee", "AVAILABLE": "available"}


def fetch(hosts=HOSTS) -> str:
    err = None
    for h in hosts:
        try:
            with ftplib.FTP(h, timeout=60) as f:
                f.login("shortstock", "")
                buf = io.BytesIO()
                f.retrbinary("RETR usa.txt", buf.write)
                return buf.getvalue().decode("utf-8", "replace")
        except (OSError, ftplib.Error) as e:
            err = e
    raise RuntimeError(f"IBKR short-stock file unreachable: {err}")


def parse(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = next(i for i, ln in enumerate(lines) if ln.startswith("#SYM"))
    names = lines[head].split("|")
    rows = [ln.split("|") for ln in lines[head + 1:] if not ln.startswith("#")]
    df = pd.DataFrame([r[:len(names)] for r in rows if len(r) >= len(names)], columns=names)
    df = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    for c in ("rebate", "fee"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["available"] = pd.to_numeric(df["available"].astype(str).str.lstrip(">"), errors="coerce")
    return df.dropna(subset=["symbol"]).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def save(df: pd.DataFrame, root: Path, day: str) -> Path:
    d = root / "borrow"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{day}.csv.gz"
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    df = parse(fetch())
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"borrow: {len(df)} symbols, {int((df['fee'] > 5).sum())} with fee > 5%/yr, "
          f"{int((df['available'].fillna(0) == 0).sum())} with none available")
    if not a.print_only:
        print("saved", save(df, Path(a.state), day))


if __name__ == "__main__":
    main()
