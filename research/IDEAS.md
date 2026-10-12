# Ideas log (phase 2)

**Phase 2 count: 5** (main 5, penny 0, news 0). Phase 1 tested 15 ideas (I-01 .. I-15, see
`breakout-scanner/IDEAS.md`); phase 1 + phase 2 = 20.

Planned, not registered (see `docs/PLAN_V2.md`): M1 filing changes, M2 options direction on big movers,
M3 residual momentum, P1 penny event shorts, P2 penny guidance raises, N1 two-LLM headline labels.
An idea is counted when its pre-registration is committed.

| ID | Track | Registered | Idea | Source | Pre-registration | Outcome |
|---|---|---|---|---|---|---|
| M4 | main | 2026-10-12 | Long-term ranking: equal-weight composite of momentum, low volatility, quality, value, net issuance and asset growth; top 20 vs universe over 6 months (feeds the picks dashboard) | Amit's dashboard request; published factor literature | [prereg/M4.md](prereg/M4.md) | 2026-10-12 screen FAILED: +1.2% per 6m vs universe after costs, t 0.94, CI -1.4%..+3.8% (99 dates, 2017-01..2025-03). Not paper traded; dashboard labelled "no measured edge". [Report](results/M4/REPORT.md) |
| M5 | main | 2026-10-12 | Insider open-market buying (officers/directors, 6 months, scaled by market cap) plus buybacks; top 20 vs universe over 6 months | Amit's request; buyback half is post-hoc from M4's exploratory table (disclosed) | [prereg/M5.md](prereg/M5.md) | 2026-10-12 screen FAILED: +2.5% per 6m vs universe after costs, t 0.72, CI -2.9%..+10.8% (99 dates); median date -0.7%, mean carried by GME 2020. Found an XBRL share-scale bug (also in M4's issuance). Dashboard unchanged. [Report](results/M5/REPORT.md) |
| M6 | main | 2026-10-12 | Net analyst upgrades/target raises minus downgrades/cuts over 90 days, read from Benzinga headlines; top 20 vs universe over 6 months | Amit's request (free substitute for paid analyst data); analyst-revision literature | [prereg/M6.md](prereg/M6.md) | 2026-10-12 screen FAILED (closest so far): +3.0% per 6m vs universe after costs, t 1.61, CI -0.6%..+6.8% (99 dates); median date +3.2%, beat universe on 72% of dates, calibration monotonic. Classifier sample 50/50 correct. Dashboard unchanged. [Report](results/M6/REPORT.md) |
| M7 | main | 2026-10-12 | Standardized earnings surprise (quarterly net income vs the same quarter a year earlier, scaled by its own history) from SEC filings; top 20 vs universe over 6 months | Amit's request (free substitute for earnings-estimate data); post-earnings drift literature | [prereg/M7.md](prereg/M7.md) | 2026-10-12 screen FAILED: -0.7% per 6m vs universe after costs, t -0.58, CI -3.2%..+1.3% (99 dates); deciles flat; top picks skewed to one-off gains. Dashboard unchanged. [Report](results/M7/REPORT.md) |
| M8 | main | 2026-10-12 | Forward paper test of M6's analyst-action ranking: top 20 logged each month-end from 2026-10-30 | Amit chose a live test after M6 narrowly failed (post-hoc, disclosed) | [prereg/M8.md](prereg/M8.md) | |
