# Who loses money to each trade (written 2026-10-15)

From 2026-10-15 every pre-registration must answer three questions before any data is touched (see
`PROTOCOL.md` rule 10). This file answers them for the 13 ideas registered before the rule, in the same
words. It was written after some results existed (M4-M7, M10, P3); for those ideas it is a post-mortem,
and for the others it changes nothing in their pre-registered tests.

The three questions:
1. **Counterparty.** Who is on the other side of the trade, and why do they keep accepting a bad price?
2. **Why us.** Why can't a fund with $1B take this trade away, while a $5k account can?
3. **Decay.** Is it published? If so, how much is left after publication (McLean & Pontiff 2016: about a
   third of the in-sample return is gone after publication, more where trading is cheap)?

An idea is weak when it has no counterparty answer, or when the "why us" answer is "they can, they just
haven't". That's the pattern behind most of the 25 failures so far.

| ID | Counterparty | Why us | Decay | Verdict |
|---|---|---|---|---|
| M1 | Investors who don't read the full 10-K/10-Q and miss bad news buried in changed text (attention limits; Cohen, Malloy & Nguyen 2020) | Weak: any fund can compute text similarity. Our only edge is trading smaller names they skip | Published 2020; text-similarity signals are now cheap for funds. The paper reports up to 188 bp/month of alpha before costs; expect well under half | Reasonable test, low expectations |
| M3 | Investors who under-react to firm-specific news, so the firm-specific part of momentum keeps going (Blitz, Huij & Martens 2011) | None at our size: large funds trade this | Published 2011, widely used | Weak; kept as a benchmark |
| M4 | Composite of six known factors | None | All published decades ago | Failed (t 0.94); expected |
| M5 | Sellers to insiders who know more about the company (insiders buy cheap) | Some: small-cap insider buys are too small for big funds | Published (Lakonishok & Lee 2001 and later) | Failed (t 0.72) |
| M6 | Investors slow to price analyst revisions | Weak | Published (Womack 1996 and later) | Failed (t 1.61); forward test M8 running |
| M7 | Investors who under-react to earnings surprises (post-earnings drift) | Weak in $300M+ names; drift is concentrated in small, costly names | Published 1968-1989; much smaller since 2000 | Failed (t -0.58) |
| M8 | As M6 | As M6 | As M6 | Forward test, low expectations |
| M10 | Not named: a model trained to find big movers has no counterparty story, only a pattern search | None | n/a | Failed (t 0.51); found volatility, not direction |
| M11 | Not named (M10 plus value and quality filters) | None | n/a | Weak; disclosed post-hoc |
| P3 | Not named (penny twin of M10) | None | n/a | Failed (t -3.99) |
| P1 | Retail buyers of penny stocks that are about to issue shares, reverse split or get delisted. Forced selling and dilution follow | Strong: most of these names are too small and too hard to borrow for funds; borrow availability is the real limit | Dilution effects published (Loughran & Ritter 1995), but penny shorts stay hard to trade | Best-founded idea; costs and borrow decide it |
| P2 | Sellers who under-react to good news in neglected penny names | Some: the names are too small for funds | Under-reaction is published; penny guidance raises aren't studied much | Plausible; costs decide it |
| N1 | Investors who under-react to bad news on day 1 (short-sale constraints slow the price drop) | Weak: news-reading LLM funds trade this in large names. Ours is cheap but not faster | Lopez-Lira & Tang 2023; fast decay expected | Plausible, crowded |

What this says about where to look next: the ideas with a real "why us" answer are the ones where size
is the barrier (P1, P2, and the small-cap end of M5). New ideas should start from forced flows, small
names and slow-to-read public records, which is what the first-public-moments study (L1) and the
burst base-rate study (B0) are designed around.
