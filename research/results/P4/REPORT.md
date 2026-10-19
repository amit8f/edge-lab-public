# P4 report: insider cluster buys in penny stocks, stop -20% / take-profit +40% (penny track)

Pre-registration: `research/prereg/P4.md`. Phase-2 count 31 (penny 5); phase 1 + phase 2 = 46.

## Verdict: FAILED

| | Mean weekly (trade - control), net 2.0% | Newey-West t (4 lags) | 95% block CI (8 weeks) | Weeks | Trades |
|---|---|---|---|---|---|
| **Primary** | -0.0005 | -0.02 | -0.0341 .. 0.0411 | 148 | 261 |

FF5 + momentum monthly alpha: -0.0058 (t -0.45). Largest single stock's share of net profit: inf. Deflated Sharpe (46 trials): {"sr": -0.0020701900080757852, "sr0": 0.18509609047045966, "n_trials": 46, "t_obs": 148, "dsr": 0.011562802704022046}. Measured penny spread (C0) 0.65% beside the 2.0% cost.

Pass rule: mean > 0, t >= 3.0, CI lower bound > 0, alpha > 0 with t >= 2.0, no stock > 20% of net profit; fewer than 100 trades is inconclusive.

## Funnel

```
{
 "universe_rows": 286039,
 "universe_days": 2176,
 "universe_median_per_day": 132.0,
 "cluster_rows_in_universe": 3158,
 "signals": 1539,
 "excluded_offering_90d": 370,
 "excluded_max_top_decile": 181,
 "excluded_runup_30pct_5d": 60,
 "excluded_delisting_notice_90d": 12,
 "excluded_reverse_split_90d": 10,
 "no_entry_bar": 0,
 "skipped_entry_above_1pct": 467,
 "skipped_open_trade": 177,
 "trades": 262,
 "trades_without_control": 1
}
```

## Exclusions: what could be applied from our data

- `runup_30pct_5d`: applied (all-adjusted closes, 5 trading days ending F)
- `max_top_decile`: applied (largest one-day all-adjusted return over the 21 trading days ending F, top decile of that day's penny universe)
- `reverse_split_90d`: applied from bars, not from a filing: raw/all-adjusted close ratio jumps >= 1.5x between consecutive bars in the 90 days ending F (no SEC form type marks a reverse split)
- `offering_424b_90d`: applied (any 424B* form in the issuer's submissions record, 90 days ending F)
- `offering_s1_s3_effective_90d`: approximated: SEC EFFECT notices in the 90 days ending F; the notice does not name the form, so it also fires for S-4/S-11/F-1/F-3/10-12B etc.; S-8 and automatic shelves get no EFFECT notice (their 424B prospectus is caught)
- `delisting_notice_8k_301_90d`: applied (8-K or 8-K/A with item 3.01 in the submissions record)

## Implementation notes

- Opportunistic (rule 3) is coded as registered: a buyer is opportunistic unless that owner bought the same issuer in the same calendar month (trade date) in each of the three prior years. Cohen, Malloy & Pomorski also leave owners without three years of history unclassified; P4.md does not, so they count as opportunistic here.
- Owners are distinct reporting-owner CIKs on original Forms 4 and 5 (amendments ignored); a joint filing's owners count separately, and each transaction's value is counted once.
- Entry limit compared on all-adjusted prices (equal to raw except across a split or ex-dividend day between F and the entry day). Paths use all-adjusted open/high/low/close.
- Controls are drawn among same-day, same-decile companies whose entry would fill, so every trade with a candidate gets a traded control. Weeks with no trade are not in the series.
- Exchange (no OTC) comes from Alpaca's current asset record, as in P3.

## Exploratory (pre-listed; none is a finding)

```
{
 "label": "EXPLORATORY (pre-listed in P4.md; none is a finding)",
 "hit_rates": {
  "trades": {
   "day21": 0.5402298850574713,
   "stop": 0.2988505747126437,
   "tp": 0.16091954022988506
  },
  "controls": {
   "day21": 0.5210727969348659,
   "stop": 0.3486590038314176,
   "tp": 0.12643678160919541,
   "last_close": 0.0038314176245210726
  },
  "gap_fills_trades": 12
 },
 "hold21_no_stop_no_tp": {
  "weeks": 146,
  "mean": 0.015772172628606092,
  "se_nw": 0.02578806156820015,
  "t_hac": 0.6116075295886186,
  "ci95": [
   -0.027374138789558092,
   0.07053648035409048
  ],
  "trades": 249,
  "trade_net_mean": 0.03636913939094429,
  "control_net_mean": 0.009597245308030864,
  "funnel": {
   "signals": 1539,
   "excluded_offering_90d": 370,
   "excluded_max_top_decile": 181,
   "excluded_runup_30pct_5d": 60,
   "excluded_delisting_notice_90d": 12,
   "excluded_reverse_split_90d": 10,
   "no_entry_bar": 0,
   "skipped_entry_above_1pct": 467,
   "skipped_open_trade": 189,
   "trades": 250
  }
 },
 "stop25_tp50": {
  "weeks": 147,
  "mean": 0.0020684420551590308,
  "se_nw": 0.023032215623402224,
  "t_hac": 0.08980647320170794,
  "ci95": [
   -0.037623675914963794,
   0.04790505103614635
  ],
  "trades": 255,
  "trade_net_mean": 0.002047710477807096,
  "control_net_mean": -0.004756751210944007,
  "funnel": {
   "signals": 1539,
   "excluded_offering_90d": 370,
   "excluded_max_top_decile": 181,
   "excluded_runup_30pct_5d": 60,
   "excluded_delisting_notice_90d": 12,
   "excluded_reverse_split_90d": 10,
   "no_entry_bar": 0,
   "skipped_entry_above_1pct": 467,
   "skipped_open_trade": 183,
   "trades": 256
  }
 },
 "one_insider": {
  "weeks": 154,
  "mean": 0.01621303717485478,
  "se_nw": 0.019425827107094503,
  "t_hac": 0.8346124510154638,
  "ci95": [
   -0.02198240713788039,
   0.05525852379540466
  ],
  "trades": 229,
  "trade_net_mean": 2.0675272421749965e-05,
  "control_net_mean": -0.008820250259945442,
  "funnel": {
   "signals": 1149,
   "excluded_offering_90d": 253,
   "excluded_max_top_decile": 94,
   "excluded_runup_30pct_5d": 45,
   "excluded_delisting_notice_90d": 7,
   "excluded_reverse_split_90d": 5,
   "no_entry_bar": 0,
   "skipped_entry_above_1pct": 446,
   "skipped_open_trade": 70,
   "trades": 229
  }
 },
 "by_year": {
  "2017": {
   "n": 10,
   "diff_mean": 0.018786267208990393,
   "net_mean": 0.03992299397065089
  },
  "2018": {
   "n": 23,
   "diff_mean": 0.06936633553565763,
   "net_mean": 0.058630655208163775
  },
  "2019": {
   "n": 33,
   "diff_mean": 0.011973706644229344,
   "net_mean": 0.053521177374467106
  },
  "2020": {
   "n": 57,
   "diff_mean": 0.016597926896635445,
   "net_mean": 0.02547682234817618
  },
  "2021": {
   "n": 19,
   "diff_mean": 0.037414331276058914,
   "net_mean": 0.004535421443411442
  },
  "2022": {
   "n": 40,
   "diff_mean": -0.03459053930297733,
   "net_mean": -0.08197833875411495
  },
  "2023": {
   "n": 23,
   "diff_mean": -0.04535377582560928,
   "net_mean": -0.06532288129465258
  },
  "2024": {
   "n": 25,
   "diff_mean": 0.027721135883042095,
   "net_mean": -0.03298119562055258
  },
  "2025": {
   "n": 31,
   "diff_mean": -0.028330050749045058,
   "net_mean": -0.06275911675494561
  }
 },
 "median_trade_net": -0.06709495776810857,
 "median_diff": 0.0,
 "without_best_trade": {
  "weeks": 147,
  "mean": -0.004899453722617969,
  "se_nw": 0.018986571002733224,
  "t_hac": -0.25804837123631563,
  "ci95": [
   -0.036849481968322426,
   0.03177547330446855
  ],
  "trades": 260,
  "trade_net_mean": -0.011973076738741612,
  "control_net_mean": -0.014194725451482539
 },
 "cost_0_65pct": {
  "weeks": 148,
  "mean": -0.0005029399916688832,
  "se_nw": 0.02034137773433593,
  "t_hac": -0.024724971840030693,
  "ci95": [
   -0.03409275614692026,
   0.04105831964036261
  ],
  "trades": 261,
  "trade_net_mean": 0.003923997755800057,
  "control_net_mean": -0.000763409308454683,
  "note": "the difference series is the same (trade and control pay the same cost)"
 },
 "car_market_model_kp": {
  "n": 259,
  "t_bmp": 4.017375693251967,
  "rbar": 0.001036079930377859,
  "t_kp": 3.566778616245836,
  "car_mean": 0.06266205866730425,
  "window": "entry day .. 21st trading day after, close to close; est -250..-30"
 }
}
```
