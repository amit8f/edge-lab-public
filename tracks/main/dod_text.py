"""Reading the Pentagon's daily contract announcements (the "Contracts for <date>" page or its RSS text).

Shared by DOD1's history study and its forward log (research/prereg/DOD1.md). One award per paragraph: the
awardee is the text before "is awarded" / "was awarded" / "has been awarded" (and similar), the value is the
first dollar figure. Paragraphs naming a modification or several awardees are flagged, not dropped, so the
study decides.
"""
import html as _html
import re

VERB = re.compile(r",?\s+(?:is|was|has been|have been|are|were|is being|has received|received|was issued|is issued)"
                  r"\s+(?:awarded|an? |a\s)", re.I)
MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion)?", re.I)
MOD = re.compile(r"\bmodification\b|\bP000\d+\b|previously awarded|exercis\w+ (?:an )?option", re.I)
MULTI = re.compile(r"following (?:\d+ |\w+ )?(?:companies|firms|contractors)|multiple[- ]award|"
                   r"each (?:awardee|contractor) will|combined (?:estimated |maximum )?value", re.I)
SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|ltd|limited|lp|l\.p|plc|"
                    r"holdings?|group|the)\b\.?", re.I)


def paragraphs(page):
    """Plain-text paragraphs of an announcement page or RSS description."""
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page)
    parts = re.split(r"(?i)</p>|<br\s*/?>", t)
    out = []
    for p in parts:
        p = _html.unescape(re.sub(r"<[^>]+>", " ", p))
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) > 40:
            out.append(p)
    return out


def amount(text):
    m = MONEY.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    return v * (1e9 if unit == "billion" else 1e6 if unit == "million" else 1)


def awards(page):
    """[{name, value, modification, multiple, text}] for the award paragraphs of one page."""
    out = []
    for p in paragraphs(page):
        m = VERB.search(p)
        if not m or m.start() == 0 or "$" not in p:
            continue
        head = p[:m.start()]
        out.append({"name": head.split(",")[0].strip(), "awardee": head.strip(), "value": amount(p),
                    "modification": bool(MOD.search(p)), "multiple": bool(MULTI.search(p)), "text": p})
    return out


def clean_name(name):
    """Upper-case name without punctuation and legal suffixes, for exact matching to SEC company names."""
    s = SUFFIX.sub(" ", name.replace("&", " AND "))
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip().upper()
