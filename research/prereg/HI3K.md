# HI3K: the 18 HI3 news arms on SEC 8-K press releases, Oct 2025 - Oct 2026 (news track)

Registered 2026-11-27, before any HI3K number is computed. Source: Amit, 2026-11-27, after the Benzinga trial key
turned out to return only about 5% of each day's articles (probe run 46002071353: 57-91 a day against about 1,900 on
Alpaca): "do the free substitute also". New idea because the input feed is new: Phase-2 count 193 -> 194 (209 with
phase 1).

## Why a substitute

No free source carries Benzinga's full feed for Oct 2025 - Apr 2026 (HI3.md). Many Benzinga headlines are company
press releases, and US companies file most material press releases with the SEC as an exhibit 99 to a Form 8-K,
with an exact EDGAR acceptance time. The models never saw this feed, so HI3K asks a narrower question: do the frozen
news models still pick better than random when fed only filed press releases? It can't stand in for HI3's verdict.

## What was seen before this was written

Everything in HI3.md's list, plus the trial-key probe counts above. No 8-K headline has been scored by any model.

## Feed (written so it can't see the future)

- Filings: every form "8-K" (not 8-K/A) in EDGAR's daily form indexes, 2025-09-01 to 2026-10-28, whose filer CIK maps
  to a ticker on Nasdaq, NYSE or NYSE American in SEC's company_tickers_exchange.json (today's map: companies delisted
  since are missing, a survivorship gap reported as a caveat).
- Each filing's index page gives the acceptance time and the documents. A filing with no EX-99 document is dropped.
- Headline: the title of the first EX-99 document, taken as the first line of its text with 5 to 40 words after
  dropping boilerplate lines (exhibit labels, "for immediate release", "press release", "news release", dates,
  contact and address lines). No title found -> dropped.
- Article = {id: accession, created_at: acceptance time, headline, symbols: ticker}. From there everything is HI3's
  replay code unchanged: FinBERT at NX1's pinned commit, headline rows, 3-day dedup, session events, inputs at the
  previous close and from SIP bars at decision, novelty from the same 8-K feed, entry and exit, costs, control seeds.
  The acceptance time is later than the press wire, so the feed can only be late, never early.

## Parts

- **Gap (decides HI3K):** sessions 2025-10-01 to 2026-05-29 (Sep 2025 only warms novelty).
- **Overlap (reported, checks the substitute):** sessions 2026-06-01 to 2026-10-28, the same days HI3 replays on
  Benzinga. Reported per arm: pick minus control on the 8-K feed next to HI3's Benzinga replay, and the share of 8-K
  picks whose stock also had a Benzinga event that session.

## Arms and test

HI3's 18 arms and arm numbers (HI3.md and its Amendment 1), same seeds rule. Per arm, gap sessions: daily pick book
minus control, mean > 0 with Newey-West (5 lags) t >= 1.5. With 18 tests about one passes by chance, so a model counts
only when both its plain and HI arms pass. HI3K changes no live book and no forward rule; at most it adds a reported
line to HI3's write-up.

## Reported, no rule

HI3's list per arm and part; the funnel (8-Ks, with EX-99, with a title, events, decided); events per session.

## Prediction (written first)

Far fewer events a day than Benzinga (a few hundred 8-Ks against about 1,900 articles), mostly earnings releases.
Picks minus control positive for the stronger HI3 models but under t 1.5 on most arms; overlap agrees in sign with
the Benzinga replay for most models.
