# BORROW1: does heavy short pressure at entry hurt NX3's longs? (descriptive, not counted)

Registered 2026-11-24, before any BORROW1 number exists. Source: Amit, 2026-11-24 ("do all of this", the days-to-cover
veto from the status list). Tests no edge and is not counted. Count stays 191 (206 with phase 1).

## Why this and not days-to-cover on history

Days-to-cover is short interest divided by average daily volume. The only free source is FINRA's bi-monthly short
interest file. FINRA's terms of use (modified 2023-11-09) allow "your own non-commercial personal or professional
use" but forbid creating "a database of data", copying the site "in bulk", and use "in conjunction with any machine
learning". Downloading about 170 files for 2019-2025 and using them next to XGBoost models would break those terms,
so no history test is run from FINRA data. Permission can be asked at FINRA's copyright-use page; if it comes, BORROW1
gets a dated amendment adding a days-to-cover history test.

What we hold legally: IBKR's public short-stock file, logged nightly from 2026-11-24 (`tracks/tools/borrow_log.py`,
no account): fee rate, rebate rate and shares available per stock. A high fee and low availability are the same
squeeze pressure that days-to-cover measures, observed directly. It starts today, so only forward data exists.

## Question (answered once 60 trading days are logged, about 2027-02-20)

For NX3-ALL-HI's long trades with a logged borrow row on the entry day, is the net per trade lower for stocks in the
top tertile of borrow fee (or bottom tertile of shares available, computed within each day) than for the rest?

## Test (written first, run once at 60 trading days)

Live NX3-ALL-HI longs, settled, net of the flat cost. Tertile cuts fixed on the borrow rows of the first 30 logged
days (not on outcomes). Report: net per trade by tertile, difference top minus rest with a bootstrap CI clustered by
day, number of trades per cell (cells under 30 are flagged too thin). No rule and no book unless a difference has a CI
that excludes 0 with at least 30 trades per side, in which case a veto is registered as a new idea on its own and tested
forward.

## Prediction (written first)

Too few trades by then (about 15 longs a day, but only those with a logged row): no cell reaches 30 with a CI that
excludes 0. P(any signal) about 0.15.
