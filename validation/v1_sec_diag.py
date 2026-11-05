"""V1 diagnosis (post hoc, not a new test): for the SEC values V1 could not find in SEC's Financial Statement Data
Sets, print every num.txt row of the same filing and tag, whatever its period, duration, unit or dimensions, so the
cause of the miss (period mismatch, tag, dimension, comparative year) is visible. Writes research/results/V1/V1_sec_diag.json.
Uses the same cache as V1 and changes no V1 result."""
import argparse
import json
from pathlib import Path

import pandas as pd

from validation.v1_known_answers import fs_values

ROOT = Path(__file__).resolve().parents[1]


def diagnose(res, cache):
    out = []
    for r in res["sec_values"]["rows"]:
        if r["ok"]:
            continue
        num = fs_values(r["quarter"], [r["accn"]], cache)
        same = num[num["tag"] == r["concept"]]
        cols = [c for c in ("ddate", "qtrs", "uom", "coreg", "segments", "value") if c in same]
        out.append({"cik": r["cik"], "concept": r["concept"], "end": r["end"], "accn": r["accn"], "ours": r["ours"],
                    "filing_rows": int(len(num)), "same_tag_rows": same[cols].astype(str).to_dict("records")[:12],
                    "other_tag_sample": sorted(num["tag"].unique().tolist())[:5]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "V1"))
    ap.add_argument("--budget-min", type=float, default=0)    # accepted for the shared study workflow, unused
    a = ap.parse_args()
    res = json.loads((Path(a.out) / "V1_results.json").read_text())
    d = diagnose(res, a.cache)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "V1_sec_diag.json").write_text(json.dumps(d, indent=1))
    for x in d:
        print(x["concept"], x["end"], x["accn"], "filing rows", x["filing_rows"], "same-tag rows", len(x["same_tag_rows"]))
        for row in x["same_tag_rows"]:
            print("   ", row)


if __name__ == "__main__":
    main()
