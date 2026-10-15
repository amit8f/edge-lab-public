"""The dashboard's research scorecard (prompt 6): every finished idea's raw, factor-adjusted and
luck-adjusted verdict, the pipeline's known-answer check, how long each forward test needs, and the
forward ledger's outside timestamps. Reads committed result files only; no network."""
import json
from pathlib import Path

NAMES = {
    "M4": "Six-factor long-term ranking",
    "M5": "Insider buying plus buybacks",
    "M6": "Analyst upgrades and target raises",
    "M7": "Earnings jumps from SEC filings",
    "M10": "Boom model, regular stocks",
    "P3": "Boom model, penny stocks",
    "M11": "Boom picks filtered for quality and value",
    "M12": "Boom model with volatility removed",
    "M1": "Filing changes (Lazy Prices)",
    "M3": "Residual momentum",
}


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def ideas(results):
    v2 = (_load(Path(results) / "V2" / "V2_results.json") or {}).get("ideas", {})
    rows = []
    for idea, name in NAMES.items():
        s = _load(Path(results) / idea / f"{idea}_screen.json")
        if s is None or "primary" not in s:
            continue
        p, a = s["primary"], v2.get(idea, {})
        raw_mean = p.get("mean", p.get("raw_mean"))
        raw_t = p.get("t_hac", p.get("alpha_t"))
        rows.append({"idea": idea, "name": name, "hold": "6 mo" if idea in ("M4", "M5", "M6", "M7") else "1 mo",
                     "raw_mean": raw_mean, "raw_t": raw_t,
                     "raw_passes": bool(p.get("passes")),
                     "alpha": a.get("alpha"), "alpha_t": a.get("alpha_t"), "p_romano_wolf": a.get("p_romano_wolf"),
                     "dsr": a.get("dsr"), "survives": bool(a.get("survives", False)),
                     "rejudged": bool(a)})
    return rows


def timestamps(state):
    d = Path(state) / "ots"
    stamps = sorted(d.glob("head-*.txt")) if d.exists() else []
    if not stamps:
        return {"count": 0}
    last = stamps[-1]
    return {"count": len(stamps), "latest": last.stem.replace("head-", ""),
            "latest_bitcoin_confirmed": (d / f"{last.name}.ots.btc").exists(),
            "confirmed": sum((d / f"{s.name}.ots.btc").exists() for s in stamps)}


def build(results, state):
    v1 = _load(Path(results) / "V1" / "V1_results.json")
    v3 = _load(Path(results) / "V3" / "V3_results.json") or _load(Path(results) / "V3" / "V3_part_a.json")
    pipeline = None if v1 is None else {
        "all_pass": bool(v1.get("all_pass")),
        "checks": {k: {"pass": v1[k]["pass"], "corr": v1[k].get("corr")}
                   for k in ("mkt", "smb", "hml", "umd", "strev", "max_effect") if k in v1},
        "prices_pass": v1.get("prices", {}).get("pass"), "sec_pass": v1.get("sec_values", {}).get("pass")}
    forward = None
    if v3 is not None:
        a, b = v3.get("part_a", {}), v3.get("part_b") or {}
        forward = {"m8_years_needed": a.get("years_for_80pct_power"),
                   "m8_p_pass_52w": (a.get("weeks_52") or {}).get("p_t_ge_2"),
                   "best_alternative_years": min((b[k]["years_for_80pct_power"] for k in ("ic", "net_spread") if k in b),
                                                 default=None),
                   "decision": v3.get("decision")}
    rows = ideas(results)
    return {"pipeline": pipeline, "ideas": rows, "any_survives": any(r["survives"] for r in rows),
            "forward": forward, "timestamps": timestamps(state)}
