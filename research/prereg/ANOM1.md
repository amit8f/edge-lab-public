# ANOM1: anomaly scores to find big movers before they move

Registered 2026-11-24, before any number of this idea is computed. Source: Amit, 2026-11-24 ("for now do all 5", picking from his model-families list: GARCH, HMM regimes, survival models, Isolation Forest / CUSUM, quantile regression / NGBoost). One idea. Phase-2 count 182 -> 183 (phase 1 + phase 2 = 198).

## Mechanism (rule 10)

BOOM1 already finds stocks likely to move 8%+ the next day (55% of picks vs 3% random). Unusual activity the
day before (volume, range, gaps that are odd for *that* stock) may show informed or attention-driven trading
before news is public. An anomaly detector scores how unlike its own recent history a stock's day is, which
BOOM1's absolute inputs don't measure. Nobody pays for this directly; it's a better filter for BOOM picks.
Published in spirit (abnormal volume: Gervais, Kaniel & Mingelgrin 2001); not as an Isolation Forest.

## Rules

Universe and data as BOOM1 (prereg BOOM1.md): its nightly candidates and scores on 2019-01 to 2025-09, from its
walk-forward (re-run as in `boom2_study.stage_picks`). Per stock per day, at the close, an Isolation Forest (`sklearn`, 200 trees, seed 7) fitted
on that stock's prior 60 sessions of [log volume z, (high-low)/close z, gap z, return z, dollar-volume change]
scores the day; plus a two-sided CUSUM alarm on log volume (k = 0.5 sd, h = 4 sd, prior 60 sessions). Anomaly
score = Isolation Forest score, +1 if the CUSUM alarm fired. Combined pick score = BOOM1 rank + w x anomaly rank,
w in {0.25, 0.5, 1.0} chosen on 2019-2023. Each night take the top 20, as BOOM1.

## Primary test (one look, 2024-01 to 2025-09)

Per day, the share of the top 20 that move 8%+ (BOOM1's own measure), ANOM1 minus BOOM1 alone: mean > 0 with
Newey-West t >= 2. Pass: ANOM1's picks are logged nightly beside BOOM1's and BOOM-SHORT may use them only under
a new registration.

## Reported, no rule

Anomaly score alone (no BOOM1) vs random; CUSUM alone; how many picks change per night.

## Prediction (written first)

BOOM1 already leans on range and volume, so the extra score changes few picks. P(pass) about 0.2.

## Amendment 1 (2026-11-24, code details, before any result)

`tracks/main/anom1_study.py`. Each night's pool is BOOM1's top 100 by score (scoring every universe stock is too slow);
the combined rank and BOOM1's own top 20 both come from that pool, so BOOM1 alone is unchanged. The five inputs come
from the chart cache: log of volume / its 20-day mean (the cache keeps relative, not raw, volume), range / close,
gap, return, and the change in log 20-day dollar volume. The z-scoring is dropped because an Isolation Forest is
unchanged by rescaling each input. The forest is our own numpy version of the same algorithm (Liu et al. 2008: 200
trees, all 60 sessions per tree, depth limit 6, seed 7), growing only the scored day's path, because sklearn's needs
0.2 s a fit (about 10 hours here); a test checks it ranks points as sklearn does. CUSUM: mean and sd of the 60 prior
sessions' log relative volume; the two sums run over the last 5 sessions ending on the scored day. A stock without 60
complete prior sessions in one segment gets an ordinary score (0.5, no alarm). Reported "anomaly alone" and "random"
are within the same pool.
