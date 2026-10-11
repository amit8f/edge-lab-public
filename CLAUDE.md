# edge-lab: hard rules

Phase 2 of a pre-registered US-equity research project. Paper trading only: no live-money orders, ever,
until the owner writes otherwise in `research/PROTOCOL.md`. Plan: `docs/PLAN_V2.md`. Phase 1 lives in the
`breakout-scanner` and `market-news-signal-proj` repos and is read-only history.

## Tracks
- Three tracks: `tracks/main` ($5+, $300M+ cap), `tracks/penny` ($1-5, liquid), `tracks/news` (event trades).
- Tracks share `core/` (data) and `paper/` (ledger). They never share models, thresholds or fitted
  parameters. A track may use another track's logged outputs only if a pre-registration says so.

## Data
- Alpaca bars: always `feed=sip`, always pass `adjustment` explicitly (`all` for labels and returns,
  `raw` for point-in-time price, dollar volume, market cap and paper fills). Bar requests end at now - 16
  minutes.
- Keep delisted stocks. A >15-calendar-day gap ends a series. Drop zero-volume bars.
- No look-ahead: point-in-time data only. SEC facts count from their filing date. Nightly jobs run after
  10:00pm ET (Forms 3/4/5 accepted until 10pm keep the same day's date).
- LLM outputs used as features are validated only on dates after that model's training cutoff.

## Research
- Every idea is pre-registered in `research/PROTOCOL.md`'s format and logged in `research/IDEAS.md`
  before any result exists. Negative results stay. Each result is reported with the running idea count.
- Phase 2 holdout: label dates from 2025-10-01 on are untouched for historical screens. Forward (paper)
  data is the confirmatory test.
- Every idea gets a same-day random control and the costs in `config/risk.yaml`.

## Engineering
- Process data in ~500-symbol batches with atomic writes; never load the full history at once.
- Forward logs are append-only and hash-chained (`core/ledger.py`). Never edit or delete a logged file.
- Never print or log secrets. Secrets come from the environment (GitHub Actions secrets) or `.env`.
- Tests run with `python -m pytest -q` and use synthetic data only (no network).
- Use the `preregistered-study` and `equity-data-qa` skills for any study.
