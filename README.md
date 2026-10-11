# edge-lab

Phase 2 of a pre-registered search for a tradable edge in US stocks, using only free data. Paper trading
only. Phase 1 (15 tested ideas, all failing after costs or still in forward tests) lives in
[breakout-scanner](https://github.com/amit8f/breakout-scanner) and
[market-news-signal-proj](https://github.com/amit8f/market-news-signal-proj).

## What phase 1 taught

- A model can find which stocks will make big moves (15.3% of its daily top 20 rose 20%+ within 5 days,
  against a 1.6% base rate), but not which way they go, and option prices already price the size.
- News is priced before a free-data trader can act: most of the reaction sits in the overnight gap.
- Costs decide short-horizon results. Gross edges of 0.1-0.5% per trade vanish at realistic spreads.

The full review is in [docs/RETHINK.md](docs/RETHINK.md) and the plan in [docs/PLAN_V2.md](docs/PLAN_V2.md).

## Layout

| Path | What |
|---|---|
| `core/` | Data access: Alpaca SIP bars (explicit adjustment), news, calendar, SEC EDGAR with rate limiting, a hash-chained append-only ledger |
| `paper/` | The paper book: next-open fills, ATR stops checked on the close, time exits, costs, borrow fees, drawdown brakes |
| `tracks/main`, `tracks/penny`, `tracks/news` | Three separate tracks that share data but never models |
| `research/` | Protocol, idea ledger, pre-registrations |
| `config/risk.yaml` | Position sizing, stops, costs per track |

## Rules that don't bend

Every idea is written down with its test before any result exists, negative results stay in the ledger,
every result is shown with the number of ideas tried so far, and every forward log is hash-chained so it
can't be edited after the fact. See [research/PROTOCOL.md](research/PROTOCOL.md).

## Running

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

Secrets (`ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT`) come from the environment or a
local `.env`. The GitHub Actions jobs keep their forward data on the `state` branch.

Not investment advice.
