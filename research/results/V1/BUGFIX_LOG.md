# V1 bug fixes (separate from the pre-registered method)

## 2026-11-05: the SEC-value check had two lookup errors (run 1: 13 of 20 equal, 7 "not found")

Diagnosis (`V1_sec_diag.json`, post hoc): in all 7 misses the filing was in SEC's data set but our lookup missed it.
- 3 values (Assets 2021-10-01, StockholdersEquity 2019-03-30, one net income for the year ending 2020-03-28): SEC files
  52/53-week fiscal years under the month-end date (20210930, 20190331, 20200331); we looked only for the exact date.
  The Assets value matched exactly once the date was right (1,147,500,000).
- 4 values (net income): the sample included quarterly net income reported inside a 10-K (for example a quarter ending
  2018-03-31 in a December-year 10-K); the check then asked for a 4-quarter row. The screens already read net income only
  as a 330 to 400 day flow (`fundamentals._first_filed`), so those facts were never what the screens use.

Fix (no pass bar, seed or method change): `sec_ddates` accepts the exact date or its month-end; `sample_sec` keeps
`NetIncomeLoss` only for 330 to 400 day durations. The sample is redrawn from the corrected pool (same seed), so the 20 values
differ from run 1. Run 1's results are kept as `V1_results_run1.json`. Everything else in V1 is unchanged and is re-run.
