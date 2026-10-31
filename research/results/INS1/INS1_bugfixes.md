# INS1 bug fixes (logged before any return was computed)

## 2026-10-30: hand check round 1 failed, 16 of 20 right (`research/checks/INS1_round1.csv`)

- Filings with several reporting owners (a fund, its manager and the person, as in AES, BW, HHH) list each
  trade once, but joining owners to trades counted the trade once per owner, so values (and trade histories)
  were multiplied. Fix: one owner per trade (an officer/director first, then the lowest CIK).
- GAM's CEO bought 5.95% preferred stock. The registration is about buys of the company's stock we trade.
  Fix: only common equity titles (drop titles naming preferred, %, notes, debentures, bonds, warrants, rights).

Both are data bugs, not method changes. The Form 4 data were rebuilt (`QDIR = "q2"`), the events redrawn and a
new hand check of 20 drawn; it must again have at least 19 right before returns are computed.

## 2026-10-30: hand check round 2 after the fixes: 20 of 20 right (`research/checks/INS1.csv`)

## 2026-10-30: factor regression included the risk-free rate as a factor

The first measure run regressed on all Ken French columns, RF among them (beta 4.3 on RF). Fix: the six
factors only (Mkt-RF, SMB, HML, RMW, CMA, momentum), as `core/stats.factor_alpha` does. Rule 12's numbers
(pick minus control, t 0.60) don't use the regression and were already failing; measure rerun for the alpha.
