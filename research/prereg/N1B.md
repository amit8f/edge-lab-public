# N1B: N1 on a fixed sample of 15,000 headlines with Llama 3.3 70B (news track)

Registered 2026-11-07, before any N1B headline was labelled and before any N1 or N1B return exists.
Source: Amit, 2026-11-07, chose "Sample 15,000" on the decision card after N1's labelling stalled.
Phase-2 count 133 -> 134 (phase 1 + phase 2 = 149).

## Why N1 is replaced

N1 (`research/prereg/N1.md`) fixed Llama 4 Scout on Groq. Groq retired it (404) after Gemini had
labelled 1,050 of 69,938 headlines and before any return was computed. N1's rule says a replacement
model is a new idea, so N1 ends unrun and N1B takes its place. N1's files stay as they are.

## What changes from N1

1. Groq model: `llama-3.3-70b-versatile` (Meta's card: cutoff December 2023). Gemini stays
   `gemini-2.5-flash` (cutoff January 2025), so the screen window is unchanged: 2025-02-01 .. 2025-09-24.
   Fixed in `research/results/N1B/models.json`.
2. Headlines: a fixed random 15,000 of N1's in-universe headlines (pandas `sample`, random_state
   20261107), drawn before any label. All other headlines are ignored by N1B.
3. Quota: labelling runs only after the US close on weekdays (21:30 UTC) and at weekends, with at most
   100 Gemini requests per run, so the live bots keep their share of the shared free Gemini quota.
   Labels are cached separately (`labels_n1b_*.jsonl`).

Everything else is N1 as written, including its clarifications and both amendments: the fixed prompt,
universe, entry 30 minutes after the headline at the next open, 3-day hold, market-adjusted short
return, costs, the per-entry-day primary test (HAC t >= 3.0, mean > 0, block-bootstrap CI above 0, at
least 200 trades), FF5 + momentum alpha t >= 2.0, deflated Sharpe with the running idea count, the
random-stock control, the dictionary baseline, the 100-label hand check before returns, the
exploratory list, and the forward test with the same two models after a pass.

## Prediction (written first)

About a fifth of N1's data, so fewer trades and wider error bars: likely 400-900 both-NEG shorts. The
literature puts most of the LLM edge in the first minutes after the news, and our entry is the next
open at least 30 minutes later (arXiv 2611.03985 finds the AI news signal near zero after the open).
Pass: 0.15. Most likely: a small positive mean that doesn't reach t 3 and doesn't beat the dictionary.
