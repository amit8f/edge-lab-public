# ORIGIN: copied from breakout-scanner pipeline/p27c_classify.py at commit 111b71d on 2026-10-15, lines
# unchanged: rx, the frozen rule constants, text_sign, sign_of, subject_in_headline, not_news, classify,
# simple_name and day_session. Left out: the batch-pipeline code (paths, stages, main), which holds no rules.
"""Phase 1's frozen catalyst text rules (Amendment 18 with clarification 8), used by P1 and P2."""
import re

import numpy as np
import pandas as pd

F = re.IGNORECASE


def rx(p):
    return re.compile(p, F)


# ------------------------------------------------------------------ frozen rules (Amendment 18)
NOT_NEWS_URL = ("/movers/", "/analyst-ratings/", "/options/", "/earnings/previews", "/markets/")
NOT_NEWS_HEAD = rx(r"shares are trading (higher|lower)|stock is (moving|trading|falling|rising|soaring|sliding|plunging)|"
                   r"why |what's going on|movers|mid-day|pre-market|after-hours|stocks moving|top \d+|price target|"
                   r"upgrade|downgrade|maintains|initiates coverage|options activity|unusual options|whale|"
                   r"earnings preview|what to expect|earnings scheduled|here's|these \d* ?stocks|52-week|short interest|"
                   r"call options|put options")
C1_A = rx(r"topline|top-line|primary endpoint|interim analysis|pivotal")
C1_B = rx(r"phase (2/3|2b|iib|2|ii|3|iii)\b")
C1_HOLD = rx(r"clinical hold")
C1_POS = rx(r"met |meets|achieved|achieves|positive|statistically significant|successful|superior|exceeded|improvement")
C1_NEG = rx(r"did not meet|not meet|fail|missed|did not achieve|not statistically significant|clinical hold|halt|"
            r"discontinu|futility|terminat")
C2_FDA = rx(r"\bfda\b")
C2_A = rx(r"approv|complete response|\bcrl\b|advisory committee|adcomm")
C2_NOT = rx(r"\bind |investigational new drug|breakthrough|fast track|orphan|priority review|accepts|acceptance of|"
            r"submission|submits|filing")
C2_POS = rx(r"approv|voted in favor|vote in favor|recommend|positive")
C2_NEG = rx(r"complete response|\bcrl\b|reject|voted against|not approv|negative vote|refus")
C3_TGT = rx(r"to be acquired|agrees to be acquired|definitive agreement to be acquired|to be taken private|go-private|"
            r"take-private|acquired by")
C3_DEF = rx(r"definitive (merger )?agreement")
C3_CASH = rx(r"per share in cash")
C3_BEFORE = rx(r"to be acquired|agrees to be acquired|acquired by")
C3_AFTER = rx(r"to acquire|acquires|to buy|buys")
C3_STRAT = rx(r"strategic alternatives|strategic review|exploring a sale|explore a sale|exploring strategic")
C4_A = rx(r"guidance|outlook|forecast|sees fy|sees q[1-4]|sees full-year|preliminary (revenue|results|sales)|"
          r"expects (q[1-4]|fourth-quarter|third-quarter|second-quarter|first-quarter|full-year|fy|annual) "
          r"(revenue|sales|eps|earnings)")
C4_POS = rx(r"raises|raised|boosts|lifts|increases|ups |above|higher|tops")
C4_NEG = rx(r"lowers|lowered|cuts|reduces|slashes|below|withdraws|suspends|weaker|warns")
C5_VERB = rx(r"awarded|wins|win |secures|signs|lands|receives")
C5_OBJ = rx(r"contract|order|agreement|deal")
C5_USD = rx(r"\$\s?\d[\d,.]*\s?(million|billion|m|b)\b")
C5_PART = rx(r"partnership|collaboration|license agreement|licensing agreement|strategic agreement")
C5_PAY = rx(r"upfront|milestone|\$\s?\d")
C5_NOT = rx(r"terminat|ends |cancel")
C7_GOV = rx(r"department of (defense|energy|commerce|war)|pentagon|\bdod\b|u\.s\. government|federal government|\barmy\b|"
            r"\bnavy\b|air force|\bnasa\b|chips act|doe loan|darpa|space force")
C7_OBJ = rx(r"equity stake|stake|investment|contract|award|grant|loan|agreement")
A2_WHO = rx(r"\bceo\b|\bcfo\b|chief executive|chief financial")
A2_ACT = rx(r"resign|depart|step(s|ping)? down|terminat|leave|exit|ousted|fired")
A2_NOW = rx(r"effective immediately")
A3_TXT = rx(r"cybersecurity incident|ransomware|data breach|cyber ?attack|unauthorized access")
A4_TXT = rx(r"chapter 11|files for bankruptcy|bankruptcy protection")
A5_TXT = rx(r"subpoena|sec investigation|doj investigation|department of justice investigation|grand jury|wells notice|"
            r"under investigation|criminal investigation")
A5_NOT = rx(r"on behalf of|investors|shareholders|class action|law firm|lawsuit|attorney|pomerantz|rosen|levi|bragar|"
            r"faruqi|kessler|glancy|bronstein|schall|portnoy|howard g\. smith|investigation of")
A7_A = rx(r"short report|short seller|short-seller|short position")
A7_B = rx(r"hindenburg|muddy waters|citron|spruce point|culper|grizzly|wolfpack|fuzzy panda|kerrisdale|iceberg|"
          r"blue orca|bonitas|viceroy|j capital|scorpion|morpheus|white diamond|gotham city|hunterbrook|bleecker street|"
          r"edwin dorsey|the bear cave")
A8_TXT = rx(r"prices (\$[\d.]+ (million|billion) )?(public|registered direct|underwritten|secondary|follow-on) offering|"
            r"announces (proposed )?(public|registered direct|underwritten) offering|at-the-market")
C6_HEAD = rx(r"eps|earnings|results|revenue|sales")
# ---- clarification 8 (2026-10-03): the one fix per failing class (A3, A4, A5, A7, C1, C2, C3, C4, C5)
SUBJECT_CLASSES = {"A3", "A4", "A5", "A7", "C1", "C2", "C3", "C4", "C5"}
A3_NOT = rx(r"testif|senator|lawsuit|sued|responds|recovery|anniversary|patent|study|tools|warns|cautions")
A4_REQ = rx(r"files|filed|commences|voluntary petition|enters chapter 11")
A4_NOT = rx(r"subsidiar|\bunits?\b|tenant|exposure|emerg|exit|not considering|no plans|trustee|settlement|history")
A5_NOT2 = rx(r"resolution|resolve|concluded|terminat|closes|ended|no action|without")
A7_NOT = rx(r"responds|response|refutes|rejects|addresses|denies|statement")
C1_RES = rx(r"topline|top-line|primary endpoint|results|data")
C1_NOT = rx(r"enrol|initiat|first patient|design|sites|to present|will present|database lock")
C1_LIFT = rx(r"lift|remov|clear")
C1_NEG8 = rx(r"did not meet|not meet|not met|did not show|failed to|fails to|failure to meet|missed|did not achieve|"
             r"not statistically significant|discontinu|futility|terminat|halt")
C2_NOT2 = rx(r"convene|scheduled|schedules|meeting date|briefing|postpone|reschedul|will review|will meet|to discuss|"
             r"conference call|warning letter|leadership|preps|loans|milestone payment")
C2_POS8 = rx(r"approv|voted in favor|vote in favor|recommend|positive|votes in favor|supports|unanimously")
C2_NEG8 = rx(r"complete response|\bcrl\b|reject|voted against|not approv|negative vote|refus|thumbs down|"
             r"not provided sufficient|votes against")
C3_UNIT = rx(r"(strategic alternatives|strategic review|exploring a sale|explore a sale|exploring strategic)[^.;]*"
             r"\b(business|unit|segment|division)\b")
C3_NOT = rx(r"continues|ongoing|completed|concluded|no plans|disappointment")
C4_NOT = rx(r"capex|capital expenditure")
C5_NOT2 = rx(r"acquir|acquisition|merger|\btax\b|\bbids?\b|\blease\b")
NAME_DROP = rx(r"\b(inc|corp|corporation|co|ltd|plc|holdings|group|class a|common stock|ordinary shares)\b|[^\w\s]")
ITEM_CLASS = {"2.02": "C6", "4.02": "A1", "1.05": "A3", "1.03": "A4", "3.01": "A6", "3.02": "A8"}
FIXED_SIGN = {"C3": "+", "C5": "+", "C7": "+", "A1": "-", "A2": "-", "A3": "-", "A4": "-", "A5": "-", "A6": "-", "A7": "-",
              "A8": "-", "A9": "-"}
TEXT_SIGNED = {"C1", "C2", "C4"}


def text_sign(t, pos, neg):
    return "-" if neg.search(t) else ("+" if pos.search(t) else "0")


def sign_of(cls, t):
    """Text sign of class C1, C2 or C4 (clarification 8f, 8g): '+', '-' or '0' (unsigned)."""
    if cls == "C1":
        if C1_HOLD.search(t):
            return "+" if C1_LIFT.search(t) else "-"
        return text_sign(t, C1_POS, C1_NEG8)
    if cls == "C2":
        return text_sign(t, C2_POS8, C2_NEG8)
    return text_sign(t, C4_POS, C4_NEG)


def subject_in_headline(head, s, names):
    """Clarification 8a: the subject's ticker (2+ characters, whole word, case-sensitive) or one of its
    simplified company names appears in the headline."""
    h = head or ""
    if len(s) >= 2 and re.search(rf"\b{re.escape(s)}\b", h):
        return True
    hl = h.lower()
    return any(n and n in hl for n in {simple_name(x) for x in names.get(s, [])})


def not_news(head, url):
    return any(s in (url or "") for s in NOT_NEWS_URL) or bool(NOT_NEWS_HEAD.search(head or ""))


def classify(head, summ):
    """Text classes of one article (without C3 role, A2's 8-K condition and A8's MAIN condition)."""
    t = f"{head or ''} {summ or ''}"
    out = []
    if ((C1_RES.search(t) and C1_B.search(t)) or C1_HOLD.search(t)) and not C1_NOT.search(t):
        out.append("C1")
    if C2_FDA.search(t) and C2_A.search(t) and not C2_NOT.search(t) and not C2_NOT2.search(t):
        out.append("C2")
    c3_excl = C3_UNIT.search(t) or C3_NOT.search(t)
    if (C3_TGT.search(t) or (C3_DEF.search(t) and C3_CASH.search(t))) and not c3_excl:
        out.append("C3a")
    if C3_STRAT.search(t) and not c3_excl:
        out.append("C3b")
    if C4_A.search(t) and (C4_POS.search(t) or C4_NEG.search(t)) and not C4_NOT.search(t):
        out.append("C4")
    gov = bool(C7_GOV.search(t))
    if not gov and not C5_NOT.search(t) and not C5_NOT2.search(t) and (
            (C5_VERB.search(t) and C5_OBJ.search(t) and C5_USD.search(t)) or (C5_PART.search(t) and C5_PAY.search(t))):
        out.append("C5")
    if gov and C7_OBJ.search(t):
        out.append("C7")
    if A2_WHO.search(t) and A2_ACT.search(t) and A2_NOW.search(t):
        out.append("A2t")
    if A3_TXT.search(t) and not A3_NOT.search(t):
        out.append("A3")
    if A4_TXT.search(t) and A4_REQ.search(t) and not A4_NOT.search(t):
        out.append("A4")
    if A5_TXT.search(t) and not A5_NOT.search(t) and not A5_NOT2.search(t):
        out.append("A5")
    if A7_A.search(t) and A7_B.search(t) and not A7_NOT.search(t):
        out.append("A7")
    if A8_TXT.search(t):
        out.append("A8t")
    return out, t


def simple_name(name):
    w = NAME_DROP.sub(" ", (name or "").lower()).split()
    return " ".join(w[:2])



def day_session(T, cal):
    """Event day d and session from ET times T (naive)."""
    T = pd.Series(pd.to_datetime(T)).reset_index(drop=True)
    date = T.dt.normalize()
    m = T.dt.hour * 60 + T.dt.minute
    trading = pd.Series(np.isin(date.values, cal.values))
    nxt = pd.Series(cal[np.minimum(cal.searchsorted(date.values, side="right"), len(cal) - 1)])
    reg = trading & (m >= 570) & (m < 960)
    pre = trading & (m < 570)
    d = pd.Series(np.where(reg | pre, date.values, nxt.values))
    sess = np.select([reg, pre, trading & (m >= 960)], ["regular", "pre", "after"], "pre")
    return pd.to_datetime(d), sess, (~trading).values
