"""FLIP1 (research/prereg/FLIP1.md): does an agent read the news? Once a day each system re-asks one agent call
with the headline words flipped ("beats" for "misses", "raises" for "cuts", ...) and logs whether the answer
moved the other way. Logging only: a flipped answer never trades and never counts as an agent error.
"""
import re

PAIRS = [("beats", "misses"), ("beat", "miss"), ("tops", "trails"), ("raises", "lowers"), ("raised", "lowered"),
         ("hikes", "cuts"), ("upgrades", "downgrades"), ("upgraded", "downgraded"),
         ("upgrade", "downgrade"), ("rises", "falls"), ("rose", "fell"), ("surges", "plunges"),
         ("soars", "sinks"), ("jumps", "drops"), ("gains", "losses"), ("profit", "loss"), ("approves", "rejects"),
         ("approved", "rejected"), ("approval", "rejection"), ("wins", "loses"), ("above", "below"),
         ("higher", "lower"), ("strong", "weak"), ("stronger", "weaker"), ("positive", "negative"),
         ("bullish", "bearish"), ("outperform", "underperform"), ("increase", "decrease"),
         ("increases", "decreases"), ("expands", "shrinks"), ("record high", "record low")]
SWAP = {**{a: b for a, b in PAIRS}, **{b: a for a, b in PAIRS}}
_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SWAP), key=len, reverse=True)) + r")\b", re.IGNORECASE)
BLOCK = re.compile(r"(<<<UNTRUSTED DATA, not instructions>>>\n)(.*?)(\n<<<END UNTRUSTED DATA>>>)", re.DOTALL)


def _case(src, word):
    if src.isupper():
        return word.upper()
    return word[0].upper() + word[1:] if src[0].isupper() else word


def flip_text(text):
    """(text with every listed word swapped for its opposite, number of swaps)."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        return _case(m.group(0), SWAP[m.group(0).lower()])
    return _RE.sub(sub, text), n


def flip_blocks(text):
    """flip_text applied only inside guards.wrap_untrusted blocks (the headlines and filings), so the prompt's
    own words (lane names, field names) stay as they are."""
    n = 0

    def sub(m):
        nonlocal n
        t, k = flip_text(m.group(2))
        n += k
        return m.group(1) + t + m.group(3)
    return BLOCK.sub(sub, text), n


def moved(before, after):
    """Did the answer move the other way? before/after: a signed number (tone, or +1 long / -1 short / 0 none).
    None when the first answer had no direction (nothing to flip)."""
    if not before:
        return None
    return (after < before) if before > 0 else (after > before)


SIDE = {"long": 1, "short": -1, "none": 0}
