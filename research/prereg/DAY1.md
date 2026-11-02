# DAY1: the all-day $1,000 multi-sleeve bot (cross-track)

Registered 2026-11-02, before the bot exists and before any forward data. Source: Amit's answers in the
2026-11-02 grill (Q1-Q20) and his yes to the plan's three questions ("1) yes 2) yes but do it every 2 weeks
3) yes"). Plan: `/mnt/project-files/planning/DAYBOT_PLAN_2026-11-02.md`; research behind it:
`/mnt/project-files/reports/All day multi model trading bot.md`. With FIL1 and AIR1 (registered in the same
commit) the phase-2 count goes 94 -> 97 (phase 1 + phase 2 = 112). DAY1 itself is one idea.

## Question

Does one paper bot that pools the project's live signals under fixed, written rules (budgets, a cost check,
risk limits) earn more after costs than a twin that takes the same number of trades, of the same sizes and
holds, picked at random from the same candidates?

## Account, data, hosting

- Alpaca paper account EL-DAY1 (secrets `ALPACA_DAY1_KEY_ID`, `ALPACA_DAY1_SECRET_KEY` in edge-lab-public). Real
  paper orders, but the bot's equity is its own ledger at $1,000 start, flat (no compounding into sizes until a
  2-week reweighting, below). Alpaca's paper fills ignore spread size, slippage and borrow, so every trade is also
  charged the cost model below; the primary test uses the ledger with that cost model, not Alpaca's fills.
- Orders are long only. Shorts a sleeve wants are logged as hypothetical trades in a separate short log, priced
  the same way, never sent (Amit 2026-11-02: a $1,000 real account cannot short; Alpaca needs $2,000).
- Decisions use Alpaca's free real-time IEX feed (latest trade and quote) for entry prices and stop checks;
  scoring and settlement use SIP bars ending 16 minutes before now (CLAUDE.md). Main-track names are bought by
  dollar amount (fractional market orders); pennies in whole shares with a marketable limit (ask + 1%).
- Runs in edge-lab-public on GitHub Actions: two chained jobs each trading day (about 9:00 to 14:55 and 14:55 to
  16:30 ET), state on branch `state-day1`, `core/kick.py` restarts skipped runs. Amit's PC is the backup host.

## Sleeves at the start

Each sleeve's rule, inputs and exit stay exactly as its own registration says; DAY1 only chooses which of its
candidates to fund and how much. Budgets are shares of the bot's $1,000.

| Sleeve | Registration | Candidates | Exit (its own) | Budget |
|---|---|---|---|---|
| News | NX3 (`prereg/NX3.md`), uncapped as NX3-ALL | NX3's fired long events (no 3-a-day cap) | next session's close | 50% |
| Penny charts | CHR-P (`prereg/CHR.md`), near-miss | CHR-P's longs above its cut | first stop/target or 5th session close | 25% |
| Filings | FIL1 (`prereg/FIL1.md`) | buyback-authorization 8-Ks | 5 sessions | 0% (shadow) |
| AI reader | AIR1 (`prereg/AIR1.md`) | headlines scored by an LLM | next session's close | 0% (shadow) |

25% stays as cash. R1 (weekly reversal, a rule 19 near-miss) was in the plan but is left out before any data: its
history mean of +0.53% a trade is below twice the cost check's round trip under any fee basis, so the boss could
never fund it (2026-11-02, Claude; Amit told on the decision card). Its own FWD1 shadow book continues.
A shadow sleeve's candidates are logged with the same entry, cost and exit rules and its own same-day random
control. It gets a budget only after PROTOCOL rule 20's numbers (60 trading days, 30 closed trades, beats its
control after costs at NW t >= 2, own mean > 0); the budget then comes out of the cash share first, then pro rata.

## The boss (fixed rules)

1. **Expected move (gross, in %).** NX3: e x vol20 x sqrt(1.5) (NX3's label is the excess return divided by
   vol20 x sqrt(1.5), so this undoes the scaling). CHR-P: its prediction for the chosen long label plus the cost
   CH's label subtracted (`ch_build.cost_frac` at the entry price), since CH's labels are net. Promoted shadow
   sleeves: their forward gross mean at promotion, fixed.
2. **Cost check.** A candidate is funded only if its expected move is at least 2x its round-trip cost in percent
   at its planned dollar size: `adaptive/costs.py`'s measured spread for its band (main 0.41%, penny 0.65%;
   penny = entry under $5) plus commission and sale fees on that fee basis (see "Fee basis" below). The same
   cost is charged in the ledger, for the bot and the twin.
3. **Size.** $50 to $200 a trade (5% to 20%), scaled linearly by the sleeve's own confidence (NX3: |e|/cut from 1
   to 2; CHR-P: its size scale from 2% to 10%; promoted sleeves: $100 flat). Never more than 20% of the bot in
   one stock (all sleeves together), never more than the sleeve's budget held past today's close (a position
   whose time exit is today's close frees its share for today's new entries), never more than 100% invested.
4. **Same stock, two sleeves.** The larger expected move wins; the other is logged "duplicate".
5. **Overnight cap.** At most 50% of the bot held overnight. A multi-day candidate that would push open overnight
   positions above 50% is skipped and logged "overnight cap" (every starting sleeve holds overnight, so in practice
   this caps the bot at $500 invested).
6. **Day stop.** If equity (realised + marked to IEX last trades) is 5% below the prior close, no new entries for
   the rest of that day; open positions keep their exits.
7. **Pause.** At 10% below the bot's peak equity, no new entries until Amit restarts it (workflow input
   `resume=true`, logged).
8. **Stops** registered by a sleeve (CHR-P) are checked every minute on the IEX last trade and closed with a
   marketable order; exits at a session close use market-on-close orders.

## Fee basis

Alpaca's $0 commission (Amit, 2026-11-02, tapped "Alpaca $0 fee" on the card "Count Alpaca's real $0 commission
in the bot's cost check, or keep the stricter IBKR-style fee?"), which is what the paper and a real Alpaca account
charge. Sale fees (SEC and FINRA TAF, `adaptive/costs.regulatory`) and the measured spread apply. Decided before the
first forward day.

## Reweighting every 2 weeks (Amit: "do it every 2 weeks")

Every second Friday after the close, from the first Friday with 10 trading days of data: candidate budgets =
equal risk (inverse of each funded sleeve's forward daily-return volatility, sleeves with fewer than 10 closed
trades keep their current budget), a sleeve whose forward mean after costs is below 0 over 30+ trades gets 0,
and the result is scaled to the funded share (75% at the start, 25% cash). The candidate runs as a shadow copy of the whole bot for the next 2
weeks. At the following checkpoint it replaces the live budgets only if its 2-week net return beat the live
bot's; either way a new candidate is computed. Each change is logged (`day1_weights`) with both returns. This
process is part of DAY1 and adds no idea; any other change to the boss is a new idea.

## Random twin (the control)

For every funded trade, the twin funds one random candidate from the same sleeve's same-day control pool (the
sleeve's own registered control pool: the day's other decided NX3 news events, the day's other CHR-P triggered
stocks), drawn with `default_rng(20261102 + date ordinal)`, same dollar size, entry minute and exit rule,
same costs. The twin obeys the same budgets but not the day stop (its stop would depend on the bot's results).

## Primary test (forward only)

After at least 60 trading days and 100 closed trades: the bot's daily ledger return minus the twin's, Newey-West
t >= 2.0 (5 lags); the bot's own mean daily return after costs > 0; and the bot-minus-twin mean still > 0 after
dropping its 5 best days. Pass -> rule 9 (real money) still needs Amit's written sign-off in PROTOCOL.md.
Fail -> the bot stops trading and gets a DIAGNOSES.md entry (rule 24); its sleeves keep their own books.

## Reported, no rule

Each sleeve's record and its control; the hypothetical short log; skipped candidates by reason (cost check,
size, overnight cap, day stop, duplicate, no quote); Alpaca fill vs IEX quote at decision (slippage); days the
host or data failed (flagged, excluded from nothing). Daily dashboard line (P&L %, trades, vs twin) and the
Sunday report.

## Disclosed

- Every starting sleeve holds overnight, so most trades are multi-day despite the "all day" name; the bot is
  awake all day to catch news and chart triggers and to enforce stops.
- IEX quotes for thin pennies can be stale or missing; such candidates are logged "no quote" and skipped.
- Expected money at this size is small (about $250 a year at +0.1% a trade on five $200 trades a day).
