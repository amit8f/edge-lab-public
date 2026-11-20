# CRY-NEWMEME: a watcher and paper book for brand-new meme tokens (crypto, outside the three tracks)

Registered 2026-11-19, before any data is recorded. Source: Amit, 2026-11-19: "we need also to trade meme coins that
just was created like now there are a lot of new gta 6 meme coins that we could take advantage of", then yes to
brainstorm round 2 (DexScreener / GeckoTerminal discovery, record first, rug filters, theme signal, 3% round trip
and $25-50 per paper trade, paper only) with the request that brand-new tokens are traded from their first hours.
One idea: phase-2 count 174 -> 175 (phase 1 + phase 2 = 190). Paper only: no wallet, nothing is ever sent on chain.

Why: hype tokens (a game launch, a celebrity, an event) can run hard in their first hours. The research
(`/mnt/project-files/reports/Crypto day trading big and meme coins.md`) found no evidence that a bot profits from
this after costs: 69% of pump.fun tokens die on day one and most pools have their liquidity pulled. No free history
of dead tokens exists, so the watcher records its own, and the paper book tests one simple rule forward from day one.
The prior is low.

Mechanism (rule 10). 1. Counterparty: later buyers drawn by the hype in the hours after launch. 2. Funds can't trade
these at all. 3. Sniper bots take the first seconds; this rule enters 15 minutes to 6 hours after launch, so it only
works if hype keeps running for hours. The rug filters try to remove the tokens built to be drained.

## Watcher (fixed now)

- A GitHub job runs about 5.5 hours at a time and is restarted every hour (one at a time), so watching is close to
  continuous. Gaps (GitHub skips, API errors) are logged, never filled.
- Discovery, every minute: GeckoTerminal `new_pools` (page 1) for Solana and Base, and DexScreener's latest token
  profiles. Every 5 minutes: GeckoTerminal `trending_pools` for both chains.
- Tracking: a token whose pool shows at least $10,000 of liquidity is tracked; every 5 minutes its best pair
  (most liquidity) is read from DexScreener: price, liquidity, FDV, market cap, buys and sells and volume over 5
  minutes and 1 hour, price change over 5 minutes, 1 hour and 6 hours, pair age. Tracking stops after 7 days or once
  liquidity falls under $1,000 (a last snapshot is kept). Every snapshot is logged, the dead tokens included.
- Theme: a token's name and symbol are split into words (3+ letters, common words removed). A word is a hot theme
  when at least 5 different tokens first seen in the last 6 hours contain it. The token's hot words are logged.

## Paper book: the fresh rule (fixed now)

- $1,000 ledger book, $25 per trade, at most 10 open trades. Cost: 1.5% on each side (3% round trip), charged on the
  logged price.
- Entry, checked at each 5-minute refresh, for a token not traded before: pair age 15 minutes to 6 hours; liquidity
  at least $20,000; 1-hour volume at least $50,000; more buys than sells over the last hour; 1-hour price change above
  0; and it passes the rug check. Rug check, Solana: the RugCheck summary lists no risk at level "danger"; Base:
  honeypot.is says it is not a honeypot and the sell tax is at most 10%. A token whose rug check can't be read is not
  entered.
- Exit at the first refresh where: the price is at least 2x the entry price (take profit), or at most 0.5x (stop; the
  logged price is used even if it gapped lower), or 24 hours have passed, or liquidity is below 10% of its entry
  value, or the pair can't be read 3 times in a row (both counted as a rug: exit at the last logged price, or at
  zero if none since entry).
- Random control: at each entry, one other tracked token with pair age 15 minutes to 6 hours and liquidity at least
  $20,000 (no volume, momentum or rug filter) is drawn at random and given the same $25 trade and exits.

## Judgement (forward only)

At 100 closed trades and again at 300 (or 2027-06-30, whichever comes first): pass needs mean net return per trade
> 0, entry-day-clustered t >= 2, and a mean above the random control's. Reported beside it: share of trades lost
to rugs, hot-theme trades vs the rest (secondary, descriptive), and the hit rate of the take profit. After about 4
weeks of recording, any model fitted on the watcher's data is a new idea with its own registration; the
fresh-rule book keeps running unchanged.

## Prediction (written first)

Passes at 100 trades: 0.1. Most likely outcome: a few large winners, many stops and rugs, mean below zero after 3%.

## Bug fix (2026-11-20, logged before the book is judged)

The control arm showed +106,000% a trade because DexScreener returned one bad quote (DCTY, $0.0000475 to $1.50 in
five minutes) and take-profits were paid at whatever price the next read showed. From now on a take-profit fills at
its limit (entry x 2) and a read more than 20 times the last price counts as a missed read. Summaries apply the
same take-profit rule to trades already logged; the ledger itself is unchanged. Rules, sizes and judging dates stay
as registered.
