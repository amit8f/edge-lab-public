# AIR1: an LLM reads the headlines NX3 sees (news track, a DAY1 shadow sleeve)

Registered 2026-11-02 with DAY1 (`prereg/DAY1.md`), before any AIR1 score exists. Source: Amit's yes to the
day-bot plan (Q18: an AI reader as one specialist, trusted only after its training cutoff). One idea; count
94 -> 97 together with DAY1 and FIL1.

## Question

Does a free LLM's reading of a headline (direction and whether it matters) pick stocks that beat same-day random
news events over NX3's holding period, in forward paper data?

## Scoring

- Headlines: the tier-1 (primary mention) headlines NX3's live job already sees, before NX3's model scores them.
  A prefilter keeps the free quota: only headlines whose FinBERT tone is beyond +/-0.3 or that NX3 fires on, at
  most 900 a day (the rest logged "not read").
- Model: Groq `openai/gpt-oss-120b`, temperature 0, the fixed prompt in `tracks/news/air1.py` (its sha256 logged
  with every call). If Groq fails twice, Gemini `gemini-3.5-flash-lite` with the same prompt; the model used is
  logged per headline. Model changes forced by a provider (a model retired) are logged and the book continues;
  any other change is a new idea.
- Output: direction in [-1, +1] and material (yes/no). Look-ahead: every scored headline is dated after both
  models' training cutoffs, so the forward record is clean by construction (CLAUDE.md LLM rule).

## Trade and control

Long when direction >= 0.6 and material; short when <= -0.6 and material (hypothetical, logged only, DAY1
rule). Entry and exit as NX3 (pre-open news at 9:31, else the first minute after scoring; exit at the next
session's close), $100 per trade in the shadow ledger, DAY1's cost model. Control: a random other tier-1 news
event the same day, same side and timing, `default_rng(20261102 + date ordinal + 2)`.

## Primary test (forward only)

Longs: PROTOCOL rule 20's numbers (60 trading days, 30 closed trades, picks minus control after costs at NW
t >= 2.0, own mean > 0). Pass -> a DAY1 budget. Reported beside it, no rule: AIR1 vs NX3 on the same days, the
overlap of their picks, and the hypothetical shorts.
