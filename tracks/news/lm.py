"""Loughran-McDonald dictionary labels for headlines: the baseline an LLM label has to beat (N1's
amendment of 2026-10-15). Word lists: tracks/news/lm_words.json (origin and sha256 inside)."""
import json
import re
from functools import lru_cache
from pathlib import Path

WORDS = Path(__file__).with_name("lm_words.json")
TOKEN = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def lists():
    j = json.loads(WORDS.read_text())
    return frozenset(j["negative"]), frozenset(j["positive"])


def label(headline):
    """NEG when the headline has more LM negative than positive words (at least one), POS the other way,
    else NEU. Words are matched whole, case-insensitive; no negation handling."""
    neg, pos = lists()
    toks = [t.upper() for t in TOKEN.findall(headline or "")]
    n, p = sum(t in neg for t in toks), sum(t in pos for t in toks)
    return "NEG" if n > p else "POS" if p > n else "NEU"
