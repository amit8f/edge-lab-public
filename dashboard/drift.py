"""Drift alarm for live books (Amit 2026-11-24): a one-sided CUSUM on each closed trade's net, against the same
model's walk-forward history. It warns when live trades run below what history led us to expect; it never changes
a book (no rule is tuned on it)."""
import numpy as np

CLIP = 0.5          # one bad price print would swamp the sums (as the history simulation)
K = 0.25            # tuned to catch a drop of half a standard deviation per trade (a broken book, not a slow fade)
N_CAL, P_FALSE, SIMS, SEED = 300, 0.05, 2000, 20261124


def cusum_path(x, mu, sd, k):
    """Downward CUSUM: S_t = max(0, S_{t-1} - (x_t - mu) / sd - k)."""
    s, out = 0.0, []
    for v in (np.asarray(x, float) - mu) / sd:
        s = max(0.0, s - v - k)
        out.append(s)
    return np.array(out)


def drift(live, hist, n_cal=N_CAL, p_false=P_FALSE, sims=SIMS, seed=SEED):
    """live, hist: per-trade nets. Per-trade noise is many times NX3's edge, so no test can tell a +0.1% book
    from a 0% one in a few hundred trades; this one looks for a break: live trades about half a standard deviation
    below history (about 4% a trade for NX3, whose trades swing about 9%). The threshold h is set so that trades
    drawn from the history itself raise a false alarm within max(n_cal, live trades) trades with chance p_false."""
    live = np.clip(np.asarray(live, float), -CLIP, CLIP)
    hist = np.clip(np.asarray(hist, float), -CLIP, CLIP)
    hist = hist[np.isfinite(hist)]
    live = live[np.isfinite(live)]
    if len(hist) < 100:
        return None
    mu, sd = float(hist.mean()), float(hist.std())
    k = K
    n = max(n_cal, len(live))
    rng = np.random.default_rng(seed)
    peaks = [cusum_path(rng.choice(hist, n), mu, sd, k).max() for _ in range(sims)]
    h = float(np.quantile(peaks, 1 - p_false))
    s = cusum_path(live, mu, sd, k) if len(live) else np.array([0.0])
    return {"trades": int(len(live)), "live_mean": float(live.mean()) if len(live) else None, "hist_mean": mu,
            "s": float(s[-1]), "s_max": float(s.max()), "h": h, "alarm": bool(s.max() > h)}
