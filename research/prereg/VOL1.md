# VOL1: GARCH volatility sizing for NX3-ALL-HI

Registered 2026-11-24, before any number of this idea is computed. Source: Amit, 2026-11-24 ("for now do all 5", picking from his model-families list: GARCH, HMM regimes, survival models, Isolation Forest / CUSUM, quantile regression / NGBoost). One idea. Phase-2 count 179 -> 180 (phase 1 + phase 2 = 195).

## Mechanism (rule 10)

Not a new source of profit: sizing. NX3's trades differ hugely in risk (a $300 stock and a $2 stock get similar
dollars), and a few big trades make the average (LESSONS 10). Equalising risk per trade should raise return per
unit of risk if volatility is forecastable, which it is (BOOM: atr14 AUC 0.89). Nobody is on the other side; the
gain, if any, is lower noise. Published (Moreira & Muir 2017 volatility-managed portfolios); no decay expected for
risk scaling itself.

## Rules

Base events: NX3-ALL-HI (prereg NX3ALLHI.md) as logged in `cache/nx1/nx3/trades_v2` (same models, seeds, confidence >= 1.5, uncapped, costs `N.cost_frac`), 2019-01 to 2025-09. Daily bars from Alpaca SIP (`adjustment=all` for returns, `raw` for prices). Choose years 2019-2023; one look at 2024-01 to 2025-09; nothing from 2025-10-01 on (holdout). Every input uses data known before the entry.

For each trade, fit GARCH(1,1) with Student-t errors (`arch` package) on that stock's 500 prior daily returns
(at least 250, else ATR fallback below) and forecast next-day volatility s. Weight = (target / s), clipped to
[0.25, 2.0] times NX3-ALL-HI's normal size, with target = the median s over 2019-2023 trades. The book's daily
gross is then rescaled to NX3-ALL-HI's daily gross, so only the mix changes, not total exposure.

Simple baseline (must also be beaten): the same weights with s = ATR(14)/price.

## Primary test (one look, 2024-01 to 2025-09)

Daily book return per dollar of gross, after costs. VOL1 passes if (a) its daily Sharpe minus plain NX3-ALL-HI's
is > 0 with a month-block bootstrap 95% CI above 0, (b) the same difference vs the ATR baseline is > 0, and (c)
VOL1's own mean daily net is > 0. Pass: a $5,000 ledger shadow book (no orders).

## Reported, no rule

Per year; longs and shorts; worst 1% day; how often the clip binds; GARCH fit failures.

## Prediction (written first)

GARCH and ATR weights rank risk almost the same, so VOL1 beats plain sizing a little and ties ATR. P(pass) about 0.2.
