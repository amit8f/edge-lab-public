# INS1 bug fixes (logged before any return was computed)

## 2026-10-30: hand check round 1 failed, 16 of 20 right (`research/checks/INS1_round1.csv`)

- Filings with several reporting owners (a fund, its manager and the person, as in AES, BW, HHH) list each
  trade once, but joining owners to trades counted the trade once per owner, so values (and trade histories)
  were multiplied. Fix: one owner per trade (an officer/director first, then the lowest CIK).
- GAM's CEO bought 5.95% preferred stock. The registration is about buys of the company's stock we trade.
  Fix: only common equity titles (drop titles naming preferred, %, notes, debentures, bonds, warrants, rights).

Both are data bugs, not method changes. The Form 4 data were rebuilt (`QDIR = "q2"`), the events redrawn and a
new hand check of 20 drawn; it must again have at least 19 right before returns are computed.
