"""E1: macro-release blackout test on finished screens' trade lists (prereg/E1.md)."""
import numpy as np, pandas as pd

COST = {"P4": 0.02}
cal = pd.read_csv("research/results/E1/macro_days.csv", parse_dates=["date"])
t = pd.read_csv("research/results/P4/P4_trades.csv", usecols=["entry_day", "gross"], parse_dates=["entry_day"])
t = t.dropna()
t = t[(t.entry_day >= "2016-01-01") & (t.entry_day <= "2025-09-30")]
t["idea"] = "P4"
t["net"] = t.gross - t.idea.map(COST)
t["dm"] = t.net - t.groupby("idea").net.transform("mean")
kinds = cal.groupby("date").kind.apply(set)
t["macro"] = t.entry_day.isin(kinds.index)


def diff_t(df, flag):
    # OLS of demeaned return on the flag, SEs clustered by entry date
    x = np.column_stack([np.ones(len(df)), flag.astype(float)])
    y = df.dm.values
    xtx = np.linalg.inv(x.T @ x)
    b = xtx @ x.T @ y
    e = y - x @ b
    meat = np.zeros((2, 2))
    for _, idx in df.groupby("entry_day").indices.items():
        s = x[idx].T @ e[idx]
        meat += np.outer(s, s)
    g = df.entry_day.nunique()
    v = xtx @ meat @ xtx * g / (g - 1)
    return b[1], b[1] / np.sqrt(v[1, 1])


out = []
d, tt = diff_t(t, t.macro.values)
out.append(f"PRIMARY pooled: trades {len(t)}, macro-day trades {t.macro.sum()}, "
           f"mean net macro {t.net[t.macro].mean():+.4f} vs other {t.net[~t.macro].mean():+.4f}, "
           f"diff {d:+.4f}, t {tt:+.2f}")
ok = d < 0 and abs(tt) >= 3 and t.macro.sum() >= 100
out.append(f"PASS (filter helps): {ok}")
out.append("Secondary (descriptive):")
for k in ["cpi", "jobs", "fomc"]:
    f = t.entry_day.map(lambda x: k in kinds.get(x, set())).values
    d, tt = diff_t(t, f)
    out.append(f"  {k}: n {f.sum()}, diff {d:+.4f}, t {tt:+.2f}")
bdays = pd.bdate_range("2015-12-01", "2025-10-31")
pos = {x: i for i, x in enumerate(bdays)}
before = {bdays[pos[x] - 1] for x in kinds.index if x in pos}
f = t.entry_day.isin(before).values
d, tt = diff_t(t, f)
out.append(f"  day before release: n {f.sum()}, diff {d:+.4f}, t {tt:+.2f}")
print("\n".join(out))
open("research/results/E1/E1_output.txt", "w").write("\n".join(out) + "\n")
