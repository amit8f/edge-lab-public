# ORIGIN: adapted from breakout-scanner scanner/shared/env.py on 2026-10-11. Reads the environment first
# (GitHub Actions secrets), then .env. Never prints a value.
"""Credentials for Alpaca and SEC EDGAR. Every function raises when a value is missing instead of
sending a placeholder."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _need(*names):
    _load_dotenv()
    vals = [os.environ.get(n) for n in names]
    missing = [n for n, v in zip(names, vals) if not v]
    if missing:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(missing)} (set them as secrets or in .env)")
    return vals


def sec_headers():
    """SEC's fair-access policy needs a real name and email in the User-Agent."""
    (ua,) = _need("SEC_USER_AGENT")
    return {"User-Agent": ua}


def alpaca_headers():
    key, secret = _need("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
