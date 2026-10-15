"""Blind re-implementation of pre-registration V2 (research/prereg/V2.md).

Written from the V2 text and the input CSV columns only, so that it can be compared with the main V2
implementation. It re-scores every finished screen against the Fama-French 5 factors plus momentum and
against luck (Holm, Romano-Wolf stepdown, deflated Sharpe ratio), and runs the 52-week placebo.

Usage::

    python -m validation.v2_blind --cache <dir> [--budget-min N]

Inputs: every ``research/results/<ID>/<ID>_primary_series.csv`` (columns ``date, pick_mean, uni_mean,
excess, n_picks, n_universe``) and Kenneth French's daily FF5 (2x3) and daily momentum zip files,
downloaded into ``<cache>/french_blind/`` with their sha256 saved.

Outputs (``research/results/V2_blind/``): ``V2_blind_results.json``, ``V2_blind_table.csv`` (one row per
idea), ``V2_blind_window_factors.csv`` (the window factor table) and ``V2_blind_placebo.csv``.

Interpretation choices
----------------------
Where V2.md leaves room, the most standard reading was taken:

1. Window. On French's daily calendar, with p = the number of French days on or before the signal
   date t, the window is the h days with positions p .. p+h-1 (entry = first French day after t,
   exit = the h-th French day after t). Factor and RF window returns are plain sums of daily returns
   (not compounded), entry day included. Dates whose window runs past the end of French's data are
   dropped (counted in the output).
2. French files are in percent and are divided by 100; FF5 and momentum are inner-joined on date;
   rows with the -99.99 / -999 missing codes are dropped.
3. Regression: OLS of the per-date series on a constant and the six window factor sums. Newey-West
   covariance uses the Bartlett kernel, weights 1 - l/(L+1), the sandwich (X'X)^-1 S (X'X)^-1 with no
   small-sample degrees-of-freedom correction. R^2 is the centred, unadjusted R^2.
4. p-values are one-sided (H1: alpha > 0), from the standard normal applied to the Newey-West t, to
   match V2's one-sided Romano-Wolf null. Holm is applied to these one-sided p-values across all
   ideas in the inputs. The two-sided p is also reported.
5. Picks' own alpha: dependent variable = pick_mean minus the idea's cost minus window RF. The cost is
   not a column; it is inferred per row as pick_mean - uni_mean - excess (constant within each input
   file: 0.002 for M ideas, 0.02 for P3), so the picks series is net of the same costs as the excess.
6. Romano-Wolf: the factor-adjusted series is intercept + residual of the primary regression, per
   date. Every idea is averaged to calendar months (a no-op for one-date-per-month ideas), and ideas
   are aligned on the intersection of their months. Moving-block bootstrap: ceil(n/6) block starts
   drawn uniformly from 0..n-6, blocks of 6 consecutive months concatenated and cut to n, the same
   indices for every idea (joint draws); numpy default_rng(20261015); 10,000 draws. Studentization
   uses the Newey-West standard error of the mean with 5 lags (block length - 1) for every idea, in
   the original sample and identically in each bootstrap sample. Bootstrap statistics are centred at
   the sample mean: t*_j = (mean*_j - mean_j) / se*_j. Stepdown: ideas ordered by t descending; the
   p-value at each step is the share of draws whose max t* over the ideas not yet rejected is >= that
   idea's t, then made monotone (running maximum).
7. Deflated Sharpe ratio, per idea, on the raw primary series (the excess column, picks minus universe
   minus costs), i.e. an information ratio:
   - non-overlapping subsamples take every k-th date, k = round(h / period length in trading days)
     (6 for monthly 126-day ideas, 4 for weekly 21-day ideas, 1 for monthly 21-day ideas), for each
     offset 0..k-1; per-period IR = mean / std (ddof=1) per offset, averaged over offsets;
   - annualisation by sqrt(periods per year / k) with 12 periods a year for monthly and 52 for weekly
     signals (so sqrt(2) and sqrt(13));
   - skew and (non-excess, Pearson) kurtosis of the full series, population moments;
   - T = average subsample length across offsets;
   - V[SR] = sample variance (ddof=1) of the annualised IRs of all ideas in the inputs (all phase 2);
   - expected maximum SR under the null with N = 28 trials, Bailey and Lopez de Prado (2014):
     SR0 = sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))), g = Euler-Mascheroni,
     converted to each idea's per-period units by dividing by its annualisation factor;
   - DSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2)) in per-period units.
8. Placebo: each signal date is moved 364 calendar days (52 weeks) later before the window is located
   on French's calendar; everything else, including the series values, is unchanged. Dates whose
   shifted window runs past French's data are dropped.
9. Survives = alpha > 0 and Newey-West t >= the idea's bar and Romano-Wolf p < 0.05.
10. ``--budget-min N``: an optional wall-clock budget in minutes. The run stops with an error at the
    next checkpoint (after download, after regressions, after the bootstrap) once it is exceeded; it
    never shortens the bootstrap.
11. An idea with no entry in IDEA_SPECS (holding days, lags, bar) is an error rather than a guess.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")
FF5_FILE = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_FILE = "F-F_Momentum_Factor_daily_CSV.zip"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# idea -> (holding trading days, Newey-West lags, t bar)
IDEA_SPECS: dict[str, tuple[int, int, float]] = {
    "M4": (126, 5, 2.0), "M5": (126, 5, 2.0), "M6": (126, 5, 2.0), "M7": (126, 5, 2.0),
    "M10": (21, 6, 3.0), "P3": (21, 6, 3.0), "M11": (21, 6, 3.0),
    "M1": (21, 3, 2.0), "M3": (21, 3, 2.0),
}

N_TRIALS = 28
RW_BLOCK = 6
RW_DRAWS = 10_000
RW_SEED = 20261015
RW_LAGS = RW_BLOCK - 1
RW_ALPHA = 0.05
PLACEBO_SHIFT_DAYS = 364
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------------------------- utils

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return x


class Budget:
    def __init__(self, minutes: float | None):
        self.minutes = minutes
        self.t0 = time.monotonic()

    def check(self, stage: str) -> None:
        if self.minutes is None:
            return
        used = (time.monotonic() - self.t0) / 60.0
        if used > self.minutes:
            raise RuntimeError(f"time budget of {self.minutes} min exceeded after {stage} "
                               f"({used:.1f} min used)")


# ------------------------------------------------------------------------------------- French data

def download_french(cache: Path, timeout: float = 120.0) -> dict[str, str]:
    """Download both zips into <cache>/french_blind/ and return {file name: sha256}."""
    out = Path(cache) / "french_blind"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for url, name in ((FF5_URL, FF5_FILE), (MOM_URL, MOM_FILE)):
        req = urllib.request.Request(url, headers={"User-Agent": "edge-lab-v2-blind"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        _atomic_write_bytes(out / name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    _atomic_write_text(out / "sha256.json", json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def hash_cached(cache: Path) -> dict[str, str]:
    d = Path(cache) / "french_blind"
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in (FF5_FILE, MOM_FILE)}


_DATE_RE = re.compile(r"^\d{8}$")


def parse_french_csv_text(text: str) -> pd.DataFrame:
    """Parse the first daily table of a French CSV: header row with an empty first field, then
    YYYYMMDD rows. Values are converted from percent to decimals."""
    header = None
    rows: list[list[str]] = []
    last_blank_first: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if _DATE_RE.match(parts[0]):
            if header is None:
                if last_blank_first is None:
                    raise ValueError("French file: data rows before any header row")
                header = [c for c in last_blank_first[1:]]
            rows.append(parts)
            continue
        if header is not None and rows:
            break  # end of the daily table (copyright line or another section)
        if parts[0] == "" and len(parts) > 1 and any(parts[1:]):
            last_blank_first = parts
    if header is None or not rows:
        raise ValueError("French file: no daily table found")
    ncol = len(header)
    idx = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    vals = np.array([[float(v) if v not in ("", None) else np.nan for v in r[1:ncol + 1]]
                     for r in rows], dtype=float)
    df = pd.DataFrame(vals, index=idx, columns=header)
    df = df.mask(df <= -99.99)  # French missing-value codes (-99.99, -999)
    return df / 100.0


def _read_zip_csv(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the zip, found {names}")
        return z.read(names[0]).decode("latin-1")


def load_french(ff5_zip: bytes, mom_zip: bytes) -> pd.DataFrame:
    ff5 = parse_french_csv_text(_read_zip_csv(ff5_zip))
    mom = parse_french_csv_text(_read_zip_csv(mom_zip))
    need = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing = [c for c in need if c not in ff5.columns]
    if missing:
        raise ValueError(f"FF5 file is missing columns {missing}")
    mom_col = [c for c in mom.columns if c.lower().startswith("mom")]
    if len(mom_col) != 1:
        raise ValueError(f"momentum file: cannot find the Mom column in {list(mom.columns)}")
    df = ff5[need].join(mom[[mom_col[0]]].rename(columns={mom_col[0]: "Mom"}), how="inner")
    df = df.dropna().sort_index()
    if not df.index.is_unique:
        raise ValueError("duplicate dates in French data")
    return df[FACTORS + ["RF"]]


# ----------------------------------------------------------------------------------- window factors

def window_factor_sums(daily: pd.DataFrame, signal_dates, h: int,
                       shift_days: int = 0) -> pd.DataFrame:
    """Sum of each daily column over the h French days after each (shifted) signal date.

    Rows whose window runs past the end of the data are NaN."""
    days = daily.index.values.astype("datetime64[ns]")
    vals = daily.to_numpy(dtype=float)
    csum = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, axis=0)])
    sig = pd.DatetimeIndex(pd.to_datetime(signal_dates))
    anchor = (sig + pd.Timedelta(days=shift_days)).values.astype("datetime64[ns]")
    p = np.searchsorted(days, anchor, side="right")  # number of French days <= anchor
    ok = p + h <= len(days)
    out = np.full((len(sig), vals.shape[1]), np.nan)
    out[ok] = csum[p[ok] + h] - csum[p[ok]]
    df = pd.DataFrame(out, index=sig, columns=daily.columns)
    entry = np.full(len(sig), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_ = entry.copy()
    entry[ok] = days[p[ok]]
    exit_[ok] = days[p[ok] + h - 1]
    df.insert(0, "entry", entry)
    df.insert(1, "exit", exit_)
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------------------- regression

def newey_west_cov(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel HAC covariance of OLS coefficients (no df correction)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


@dataclass
class RegResult:
    alpha: float
    alpha_se: float
    alpha_t: float
    p_one_sided: float
    p_two_sided: float
    loadings: dict
    loading_t: dict
    r2: float
    n: int
    fitted_adjusted: pd.Series  # intercept + residual per date

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "alpha_se": self.alpha_se, "alpha_t": self.alpha_t,
                "p_one_sided": self.p_one_sided, "p_two_sided": self.p_two_sided,
                "loadings": self.loadings, "loading_t": self.loading_t, "r2": self.r2,
                "n": self.n}


def factor_regression(y: pd.Series, F: pd.DataFrame, lags: int) -> RegResult:
    df = pd.concat([y.rename("y"), F[FACTORS]], axis=1).dropna()
    yv = df["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df[FACTORS].to_numpy(float)])
    n, k = X.shape
    if n <= k:
        raise ValueError(f"too few observations ({n}) for the regression")
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    e = yv - X @ beta
    V = newey_west_cov(X, e, lags)
    se = np.sqrt(np.diag(V))
    t = beta / se
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / sst if sst > 0 else float("nan")
    adj = pd.Series(beta[0] + e, index=df.index, name="adjusted")
    return RegResult(
        alpha=float(beta[0]), alpha_se=float(se[0]), alpha_t=float(t[0]),
        p_one_sided=float(stats.norm.sf(t[0])),
        p_two_sided=float(2 * stats.norm.sf(abs(t[0]))),
        loadings={f: float(b) for f, b in zip(FACTORS, beta[1:])},
        loading_t={f: float(v) for f, v in zip(FACTORS, t[1:])},
        r2=r2, n=n, fitted_adjusted=adj)


# --------------------------------------------------------------------------------- multiple testing

def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys], dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[j]))
        adj[j] = running
    return {k: float(a) for k, a in zip(keys, adj)}


def nw_se_of_mean(Z: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West standard error of the column means. Z: (..., n, m)."""
    n = Z.shape[-2]
    d = Z - Z.mean(axis=-2, keepdims=True)
    var = (d * d).sum(axis=-2) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var = var + 2.0 * w * (d[..., l:, :] * d[..., :-l, :]).sum(axis=-2) / n
    var = np.where(var > 0, var, (d * d).sum(axis=-2) / n)
    return np.sqrt(var / n)


def romano_wolf(Z: pd.DataFrame, block: int = RW_BLOCK, draws: int = RW_DRAWS,
                seed: int = RW_SEED, lags: int = RW_LAGS, chunk: int = 1000) -> dict:
    """One-sided studentized Romano-Wolf stepdown with a joint moving-block bootstrap."""
    X = Z.to_numpy(float)
    n, m = X.shape
    if n < block:
        raise ValueError("fewer common months than the block length")
    mean = X.mean(axis=0)
    t = mean / nw_se_of_mean(X, lags)
    rng = np.random.default_rng(seed)
    nblocks = math.ceil(n / block)
    tstar = np.empty((draws, m))
    offs = np.arange(block)
    done = 0
    while done < draws:
        b = min(chunk, draws - done)
        starts = rng.integers(0, n - block + 1, size=(b, nblocks))
        idx = (starts[:, :, None] + offs).reshape(b, -1)[:, :n]
        Xb = X[idx]  # (b, n, m)
        tstar[done:done + b] = (Xb.mean(axis=1) - mean) / nw_se_of_mean(Xb, lags)
        done += b
    order = np.argsort(-t, kind="mergesort")
    p_adj = np.empty(m)
    running = 0.0
    for k, j in enumerate(order):
        maxstar = tstar[:, order[k:]].max(axis=1)
        running = max(running, float(np.mean(maxstar >= t[j])))
        p_adj[j] = running
    cols = list(Z.columns)
    return {"t": dict(zip(cols, t.tolist())), "mean": dict(zip(cols, mean.tolist())),
            "p_adjusted": dict(zip(cols, p_adj.tolist())), "n_months": int(n),
            "first_month": str(Z.index[0]), "last_month": str(Z.index[-1])}


# ------------------------------------------------------------------------------------ deflated SR

def infer_frequency(dates: pd.DatetimeIndex) -> str:
    gap = float(np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float)))
    return "weekly" if gap <= 10 else "monthly"


def subsample_ir(x: np.ndarray, step: int) -> tuple[float, float]:
    """Mean over offsets of the per-period IR on every step-th value; also the mean length."""
    srs, lens = [], []
    for off in range(step):
        s = x[off::step]
        if len(s) >= 2 and np.std(s, ddof=1) > 0:
            srs.append(np.mean(s) / np.std(s, ddof=1))
            lens.append(len(s))
    if not srs:
        return float("nan"), float("nan")
    return float(np.mean(srs)), float(np.mean(lens))


def expected_max_sr(var_sr: float, n_trials: int) -> float:
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sr(sr: float, sr0: float, T: float, skew: float, kurt: float) -> float:
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if not denom > 0 or not T > 1:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1.0) / math.sqrt(denom)))


# --------------------------------------------------------------------------------------- the study

def find_series(results_root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(Path(results_root).glob("*/*_primary_series.csv")):
        idea = p.parent.name
        if idea.startswith("V2") or p.name != f"{idea}_primary_series.csv":
            continue
        out[idea] = p
    return out


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date", "pick_mean", "uni_mean", "excess"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: missing columns {need - set(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if not df.index.is_unique:
        raise ValueError(f"{path}: duplicate dates")
    df["cost"] = df["pick_mean"] - df["uni_mean"] - df["excess"]
    df["picks_net"] = df["pick_mean"] - df["cost"]
    return df


def run_study(daily: pd.DataFrame, series: dict[str, pd.DataFrame], draws: int = RW_DRAWS,
              budget: Budget | None = None) -> dict:
    budget = budget or Budget(None)
    ideas = sorted(series, key=lambda s: (s[0], int(re.sub(r"\D", "", s) or 0)))
    unknown = [i for i in ideas if i not in IDEA_SPECS]
    if unknown:
        raise ValueError(f"no holding/lag/bar spec for ideas {unknown}")

    per: dict[str, dict] = {}
    windows: list[pd.DataFrame] = []
    adjusted_monthly: dict[str, pd.Series] = {}
    for idea in ideas:
        h, lags, bar = IDEA_SPECS[idea]
        s = series[idea]
        W = window_factor_sums(daily, s.index, h)
        dropped = int(W["Mkt-RF"].isna().sum())
        y_ex = s["excess"]
        y_pk = s["picks_net"] - W["RF"]
        prim = factor_regression(y_ex, W, lags)
        own = factor_regression(y_pk, W, lags)
        Wp = window_factor_sums(daily, s.index, h, shift_days=PLACEBO_SHIFT_DAYS)
        plac = factor_regression(y_ex, Wp, lags)
        dropped_placebo = int(Wp["Mkt-RF"].isna().sum())

        freq = infer_frequency(s.index)
        period_td = 21 if freq == "monthly" else 5
        per_year = 12 if freq == "monthly" else 52
        step = max(1, int(round(h / period_td)))
        ann = math.sqrt(per_year / step)
        x = s["excess"].to_numpy(float)
        sr_p, T = subsample_ir(x, step)

        per[idea] = {
            "hold_days": h, "nw_lags": lags, "t_bar": bar, "frequency": freq,
            "n_dates": int(len(s)), "n_dropped_window_past_data": dropped,
            "n_dropped_placebo_window_past_data": dropped_placebo,
            "first_date": s.index[0], "last_date": s.index[-1],
            "cost_per_window": float(s["cost"].median()),
            "raw_mean_excess": float(s["excess"].mean()),
            "primary": prim.as_dict(), "picks_own": own.as_dict(), "placebo": plac.as_dict(),
            "dsr_inputs": {"subsample_step": step, "annualisation": ann,
                           "ir_per_period": sr_p, "T": T,
                           "skew": float(stats.skew(x, bias=True)),
                           "kurtosis": float(stats.kurtosis(x, fisher=False, bias=True))},
            "annual_ir": sr_p * ann,
        }
        w = W.copy()
        w.insert(0, "idea", idea)
        windows.append(w.reset_index())
        adj = prim.fitted_adjusted
        adjusted_monthly[idea] = adj.groupby(adj.index.to_period("M")).mean()
    budget.check("regressions")

    # Holm on one-sided NW p-values
    holm_p = holm({i: per[i]["primary"]["p_one_sided"] for i in ideas})

    # Romano-Wolf on common months
    Z = pd.DataFrame(adjusted_monthly).dropna(how="any")
    rw = romano_wolf(Z[ideas], draws=draws)
    budget.check("bootstrap")

    # Deflated Sharpe ratio
    ann_irs = np.array([per[i]["annual_ir"] for i in ideas], dtype=float)
    var_sr = float(np.var(ann_irs[np.isfinite(ann_irs)], ddof=1)) if len(ideas) > 1 else float("nan")
    sr0_ann = expected_max_sr(var_sr, N_TRIALS) if math.isfinite(var_sr) else float("nan")
    for i in ideas:
        d = per[i]["dsr_inputs"]
        sr0_p = sr0_ann / d["annualisation"]
        d["sr0_per_period"] = sr0_p
        per[i]["dsr"] = deflated_sr(d["ir_per_period"], sr0_p, d["T"], d["skew"], d["kurtosis"])

    for i in ideas:
        prim = per[i]["primary"]
        per[i]["holm_p"] = holm_p[i]
        per[i]["rw_p"] = rw["p_adjusted"][i]
        per[i]["rw_t"] = rw["t"][i]
        per[i]["survives"] = bool(prim["alpha"] > 0 and prim["alpha_t"] >= per[i]["t_bar"]
                                  and rw["p_adjusted"][i] < RW_ALPHA)

    mean_abs = lambda key: float(np.mean([abs(v) for i in ideas
                                          for v in per[i][key]["loadings"].values()]))
    placebo_summary = {"mean_abs_loading_aligned": mean_abs("primary"),
                       "mean_abs_loading_placebo": mean_abs("placebo"),
                       "mean_r2_aligned": float(np.mean([per[i]["primary"]["r2"] for i in ideas])),
                       "mean_r2_placebo": float(np.mean([per[i]["placebo"]["r2"] for i in ideas])),
                       "shift_calendar_days": PLACEBO_SHIFT_DAYS}

    return {
        "ideas": ideas, "per_idea": per,
        "romano_wolf": {k: v for k, v in rw.items() if k != "mean"} | {
            "block": RW_BLOCK, "draws": draws, "seed": RW_SEED, "studentization_lags": RW_LAGS,
            "mean_adjusted": rw["mean"]},
        "dsr_common": {"n_trials": N_TRIALS, "var_annual_ir": var_sr, "sr0_annual": sr0_ann},
        "placebo_summary": placebo_summary,
        "french_calendar": {"first": daily.index[0], "last": daily.index[-1],
                            "n_days": int(len(daily))},
        "n_survivors": int(sum(per[i]["survives"] for i in ideas)),
        "_windows": pd.concat(windows, ignore_index=True),
    }


def results_table(res: dict) -> pd.DataFrame:
    rows = []
    for i in res["ideas"]:
        d = res["per_idea"][i]
        p, o = d["primary"], d["picks_own"]
        row = {"idea": i, "hold_days": d["hold_days"], "nw_lags": d["nw_lags"],
               "t_bar": d["t_bar"], "n": p["n"], "raw_mean_excess": d["raw_mean_excess"],
               "alpha": p["alpha"], "alpha_t": p["alpha_t"], "r2": p["r2"]}
        row.update({f"beta_{f}": p["loadings"][f] for f in FACTORS})
        row.update({"p_one_sided": p["p_one_sided"], "holm_p": d["holm_p"], "rw_p": d["rw_p"],
                    "picks_alpha": o["alpha"], "picks_alpha_t": o["alpha_t"], "picks_r2": o["r2"],
                    "annual_ir": d["annual_ir"], "dsr": d["dsr"], "survives": d["survives"]})
        rows.append(row)
    return pd.DataFrame(rows)


def placebo_table(res: dict) -> pd.DataFrame:
    rows = []
    for i in res["ideas"]:
        d = res["per_idea"][i]
        for kind in ("primary", "placebo"):
            r = d[kind]
            row = {"idea": i, "calendar": "aligned" if kind == "primary" else "shifted_52w",
                   "n": r["n"], "alpha": r["alpha"], "alpha_t": r["alpha_t"], "r2": r["r2"]}
            row.update({f"beta_{f}": r["loadings"][f] for f in FACTORS})
            rows.append(row)
    return pd.DataFrame(rows)


def run(cache: Path, results_root: Path | None = None, out_dir: Path | None = None,
        budget_min: float | None = None, download: bool = True, draws: int = RW_DRAWS) -> dict:
    budget = Budget(budget_min)
    results_root = Path(results_root) if results_root else REPO / "research" / "results"
    out_dir = Path(out_dir) if out_dir else results_root / "V2_blind"
    cache = Path(cache)
    hashes = download_french(cache) if download else hash_cached(cache)
    budget.check("download")
    fdir = cache / "french_blind"
    daily = load_french((fdir / FF5_FILE).read_bytes(), (fdir / MOM_FILE).read_bytes())

    paths = find_series(results_root)
    if not paths:
        raise FileNotFoundError(f"no *_primary_series.csv under {results_root}")
    series = {i: load_series(p) for i, p in paths.items()}
    res = run_study(daily, series, draws=draws, budget=budget)
    windows = res.pop("_windows")
    res["french_sha256"] = hashes
    res["french_urls"] = {FF5_FILE: FF5_URL, MOM_FILE: MOM_URL}
    res["inputs"] = {i: str(p.relative_to(results_root)) for i, p in paths.items()}
    res["run_utc"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_dir / "V2_blind_results.json",
                       json.dumps(_jsonable(res), indent=2) + "\n")
    _atomic_write_text(out_dir / "V2_blind_table.csv", results_table(res).to_csv(index=False))
    _atomic_write_text(out_dir / "V2_blind_placebo.csv", placebo_table(res).to_csv(index=False))
    _atomic_write_text(out_dir / "V2_blind_window_factors.csv", windows.to_csv(index=False))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args(argv)
    res = run(a.cache, budget_min=a.budget_min)
    tab = results_table(res)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(tab[["idea", "n", "raw_mean_excess", "alpha", "alpha_t", "r2", "holm_p", "rw_p",
                   "annual_ir", "dsr", "survives"]].to_string(index=False))
    print(f"survivors: {res['n_survivors']} of {len(res['ideas'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
