# NXR: NX3 with tamer trees (news track)

Registered 2026-10-29, before any NXR model is fit or scored. Source: Amit, 2026-10-29, "do all the fixes you
suggested", after the NXM result (`prereg/NXM.md`): NXM's random forest failed rule 21 (in-sample 2.9x
out-of-sample) and NX3's own XGBoost sits at 2.3x. Two arms, each counted: phase-2 count at registration 65
(main 28, penny 10, news 16, cross-track 10, crypto 1). Phase 1 + phase 2 = 80.

The fix (stronger regularisation) was chosen after seeing NXM's in-sample and out-of-sample numbers, so by
PROTOCOL rules 4 and 23 this history can only reject an arm; the forward book decides.

## Question

Do trees held back from memorising (shallow, big leaves, heavy sampling, an L2 penalty) keep NX3's edge with
an in-sample score no more than twice out-of-sample?

## What stays NX3's

Everything but the learner, exactly as NXM: NX3's cached trades (`trades_v2`), 37 inputs and 13 baseline inputs
(no inputs dropped), label, walk-forward (2019 to 2025-09, 5-session purge), baseline B and news model N,
e = N(x) - N(x0), the fire cut at the training years' 90th percentile of |e|, picks, the same-day control, the
shuffled-news test, seeds and costs. Nothing from the 2025-10-01 holdout is read. Code: `tracks/news/nxr_study.py`
on NXM's functions.

## Arms (settings fixed now, no tuning)

- `XGBR`: XGBoost 3.4.1, 300 trees, depth 2, learning rate 0.03, row sampling 0.5, column sampling 0.5, at
  least 500 events per leaf (min_child_weight 500), L2 penalty 10, Huber loss with slope 1.0, hist.
- `RFR`: random forest, 300 trees, depth at most 6, at least 1,000 events per leaf, each tree on a 20% sample,
  a third of the inputs per split.

## Primary test (per arm), as NXM

1. Rule 21 first: in-sample mean daily pick-minus-control (each year's model on its own training events) more
   than twice the pooled out-of-sample mean fails the arm.
2. NX3's gate: pooled pick-minus-control and pick-minus-shuffled, both mean > 0 and Newey-West t (5 lags) >= 2.

Reported, no rule: the in-sample/out-of-sample ratio beside XGB's and NXM-RF's, head-to-head with XGB on days
both pick (Holm across the two arms), per-year numbers, pick overlap with XGB.

## What happens next (fixed now)

A passing arm gets version 1 trained on all of 2016-10 .. 2025-09-23 and a $5,000 ledger shadow book beside
NX3's (as NXM's books: same events and logged inputs, own cut, picks and same-day control), with NX3's forward
test (book minus control, NW t >= 2 at 100 closed trades or 2027-12-31). No arm replaces NX3.

## Prediction (written first)

XGBR passes rule 21 (ratio about 1.5) with a lower mean than XGB (about +0.4%/day) and passes the gate. RFR
passes rule 21 and the control test but is weaker (t about 2.5). About 0.3 that either fails the gate.

## Live book details (2026-10-29, after the result and before any forward data; `tracks/news/nxm_live.py`)

XGBR and RFR join NXM's books with NXM's live details unchanged (logged inputs at decision time, easy to
borrow for shorts, sha256 checks, after-close scoring), from 2026-10-30, the first session after their
version 1 files exist. Control per arm: default_rng(NX3's day seed + the arm's number), XGBR 5 and RFR 6.
