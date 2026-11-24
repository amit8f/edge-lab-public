# Lessons from the failures (phase 2)

Written 2026-11-20 from the outcomes in [IDEAS.md](IDEAS.md), where each idea's own result and notes stay. This page
groups them so the next idea starts from what we already know. It adds no new result.

## 1. Costs kill small moves
Most failures beat their random control but not costs: SHORT-FAST, SHORT-ML, SHORT-FIX2, DIR-BRKT, DIR-OFI, PICK-MAIN,
FAST1. A trade needs an expected move several times its round-trip cost (about 0.5% main, 2% penny at $500).
Next time: before building, check the gross move from a descriptive run (as S0 did) against costs.

## 2. Direction on big movers is close to a coin flip
BOOM2 (6 models), DIR-GATE, DIR-PEER, DIR-AGE, DIR-FLOW, DIR-OFI, BOOMX: 48-52% balanced accuracy. Predicting
*that* a stock will move works (BOOM1, AUC 0.89-0.92); predicting *which way* has not. The one survivor is shorting the
most bearish tenth (BOOM-SHORT). Next time: a new direction idea needs a new kind of information, not the same inputs
in a new model.

## 3. News carries the edge; charts add nothing
NX3 and its variants pass; CHART1 (5 models), NX3TA, M-track chart and fundamental screens (M3-M13, R1, NR1, RL1-RL2)
fail. Next time: start from a news event, then add filters.

## 4. Tight stops and brackets hurt
DIR-SSTOP, DIR-BRKT, PICK (55% stopped), FAST1 brackets: normal noise hits the stop before the move. Next time:
a stop wider than the stock's usual 30-60 minute range, or a time exit.

## 5. Exiting early throws the move away
FAST1 (-0.19% at 60 minutes vs +0.18% at the next close), SHORT-FAST. The news move keeps going into the next day.

## 6. Shorts are the hard side
SHORT-FIX, SHORT-ML, SHORT-ALL, NX3 shorts ~0: squeezes and borrow. Shorts only help in weak markets (SHORT-FIX
post hoc) or on the most bearish BOOM picks.

## 7. Models that memorize
TB1 (in-sample AUC 1.00, out 0.50), RL1, META1-2, NXS-ML: rule 21 (in-sample vs out-of-sample gap) catches them.
Next time: always report both.

## 8. Data checks first
G1, W1a, SP1, PD1, INS1 stopped at the hand check (wrong links, UTC read as ET, double-counted owners). The check
saved us from trading on bad data. Next time: hand check before any return, every time.

## 9. Crypto short horizons flip sign
CRY-DAY, CRY-XS, CRY-FUND and CRY-INTRA pass their choice year and fail the next. Only slow long-only trend rules
survive (CRY-TREND).

## 10. A few trades make the average
M5 (GME 2020), A1 (two squeezed shorts), the live NXM books (top 3 trades = half the profit). Next time: report the
median trade and the result without the top 3 beside the mean.

## 11. Don't trim NX3's tails (VOL1, REGIME1, SURV1, QRANGE1, ANOM1; 2026-11-24)
Four ways of reshaping NX3-ALL-HI's trades all lost to the plain next-close trade, for one reason: its profit sits
in a few big moves in volatile stocks on volatile days. GARCH sizing shrank exactly those trades (Sharpe 0.79 ->
0.15); the HMM storm switch cut the worst 1% day from -12.1% to -8.5% but halved the return, because rebounds after
crashes are where it earns; quantile take-profits (median 7%) capped the winners (+0.05% vs +0.19% a trade). Holding
longer helped longs (+0.39% a trade) and hurt shorts (-0.13%), the same split as lessons 5 and 6. ANOM1 failed for a
different reason: BOOM1's inputs already hold the range and volume an anomaly score sees (alone it picks 41% movers
against 37% at random from the same pool). Next time: risk tools are a choice of less loss for less return, not a
free gain; a fix chosen from these splits (e.g. hold only longs) is a new idea tested forward only.
