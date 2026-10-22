"""P4 live job on synthetic data (no network)."""
import json

import numpy as np
import pandas as pd

from tracks.penny import p4_live as L

DAYS = pd.bdate_range("2026-04-01", "2026-12-31")
F = pd.Timestamp("2026-10-23")


def xml(owner, rel="isDirector", code="P", shares=10000, price=2.0, issuer=1):
    return f"""<?xml version="1.0"?><ownershipDocument><documentType>4</documentType>
<issuer><issuerCik>{issuer:010d}</issuerCik></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>{owner:010d}</rptOwnerCik></reportingOwnerId>
<reportingOwnerRelationship><{rel}>1</{rel}></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-10-20</value></transactionDate>
<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>{shares}</value></transactionShares>
<transactionPricePerShare><value>{price}</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""


class Fake:
    def __init__(self, subs_extra=()):
        self.sent, self.subs_extra = [], list(subs_extra)

    def calendar(self, start, end):
        d = DAYS[(DAYS >= start) & (DAYS <= end)]
        return pd.DataFrame({"date": d, "open": "09:30", "close": "16:00"})

    def assets(self):
        syms = [f"S{i}" for i in range(12)]
        return pd.DataFrame({"symbol": syms, "exchange": "NASDAQ", "excl_reason": None})

    def tickers(self):
        return pd.DataFrame({"symbol": [f"S{i}" for i in range(12)], "cik": list(range(1, 13))})

    def bars(self, symbols, start, end, adjustment):
        d = DAYS[(DAYS >= start) & (DAYS <= end)]
        parts = []
        for s in sorted(symbols):
            rng = np.random.default_rng(int(s[1:]))
            c = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d))))
            parts.append(pd.DataFrame({"symbol": s, "date": d, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c,
                                       "v": 1e6}))
        return pd.concat(parts, ignore_index=True)

    def submissions(self, cik, first, last):
        if cik != 1:
            return []
        return [{"accession": "0001-26-000001", "form": "4", "filed": "2026-10-23", "items": [], "doc": "x/a.xml"},
                {"accession": "0001-26-000002", "form": "4", "filed": "2026-10-21", "items": [], "doc": "x/b.xml"},
                *self.subs_extra]

    def form4_xml(self, cik, accn, doc):
        return xml(101 if accn.endswith("1") else 102, rel="isOfficer" if accn.endswith("2") else "isDirector")

    def history(self, year, cache):
        return pd.DataFrame(columns=["cik", "filed", "accn", "sk", "owner", "tdate", "value"])

    def order(self, body):
        self.sent.append(body)
        return 200, {"status": "accepted"}

    def positions(self):
        return {}

    def open_orders(self):
        return []


def _state(tmp_path):
    sec = tmp_path / "leads" / "sec"
    sec.mkdir(parents=True)
    (sec / "2026-10-23.jsonl").write_text(json.dumps({"form": "4", "cik": 1, "file": "x"}) + "\n"
                                          + json.dumps({"form": "4", "cik": 9001, "file": "x"}) + "\n")
    return tmp_path


def test_parse_form4_keeps_insider_open_market_buys():
    r = L.parse_form4(xml(7), "a", "2026-10-23")
    assert len(r) == 1 and r[0]["owner"] == 7 and r[0]["value"] == 20000.0 and r[0]["cik"] == 1
    assert L.parse_form4(xml(7, code="S"), "a", "2026-10-23") == []
    assert L.parse_form4(xml(7, rel="isOther"), "a", "2026-10-23") == []


def test_cluster_signal_orders_a_bracket(tmp_path):
    st = _state(tmp_path)
    api = Fake()
    now = pd.Timestamp("2026-10-24T03:00", tz="UTC")      # 23:00 ET on F
    assert L.run(st, tmp_path / "cache", now=now, api=api, send=True, start=F) == 1
    rec = json.loads((st / "p4" / "days" / "2026-10-23.json").read_text())
    assert rec["form4_issuers"] == 1 and len(rec["purchases"]) == 2
    assert [s["symbol"] for s in rec["signals"]] == ["S0"] and rec["signals"][0]["excl"] is None
    assert len(rec["controls"]) == 1 and rec["controls"][0]["symbol"] not in (None, "S0")
    (o,) = rec["orders"]
    assert o["kind"] == "entry" and o["http"] == 200 and api.sent[0]["order_class"] == "bracket"
    lim = float(o["body"]["limit_price"])
    assert float(o["body"]["stop_loss"]["stop_price"]) < lim < float(o["body"]["take_profit"]["limit_price"])
    assert int(o["body"]["qty"]) * lim <= 25.5          # 1% of $500 at risk to a 20% stop
    assert L.run(st, tmp_path / "cache", now=now, api=api, send=True, start=F) == 0      # once per day


def test_offering_excludes_and_dry_run_sends_nothing(tmp_path):
    st = _state(tmp_path)
    api = Fake([{"accession": "0001-26-000003", "form": "424B5", "filed": "2026-09-30", "items": []}])
    L.run(st, tmp_path / "cache", now=pd.Timestamp("2026-10-24T03:00", tz="UTC"), api=api, send=False, start=F)
    rec = json.loads((st / "p4" / "days" / "2026-10-23.json").read_text())
    assert rec["signals"][0]["excl"] == "offering_90d" and rec["orders"] == [] and api.sent == []


def test_waits_for_the_form_index(tmp_path):
    api = Fake()
    assert L.run(tmp_path, tmp_path / "c", now=pd.Timestamp("2026-10-24T03:00", tz="UTC"), api=api, send=True,
                 start=F) == 0
    assert not (tmp_path / "p4" / "days").exists()


def test_held_days_counts_trading_days():
    tdays = list(DAYS)
    assert L.held_days("2026-10-26T13:30:00Z", pd.Timestamp("2026-10-27"), tdays) == 1
