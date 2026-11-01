# W2 hand check (2026-11-01)

Result: **40 of 50 right (80%). W2's rule says fewer than 80% right stops W2, so W2 goes on.** No return was read.

Each article's text was read from the Alpaca news API (kept in the private repo only; no headline text here). The
per-row verdicts and notes, in my own words, are in `W2_check.csv`. A row counts as right if it is the right company
and a new short report or campaign post by a listed short seller.

My first strict reading gave 38/50 (76%) and stopped W2. Two rows were Citron's Andrew Left talking in interviews
about positions he already held (covering part of CC, still short W). Amit decided the same day to count them as
campaign statements, which makes 40/50, exactly on the line. The call is disclosed here because the margin is nothing:
one more miss would have stopped W2.

The 10 misses:
- 7 are not a short report at all: a lawsuit against Tesla, a court plea, Citron defending GE, Citron closing its GME
  short, a story about a stock recovering after an old report, Muddy Waters saying it is too early to short SNAP, and
  Citron turning away from a long ROKU position.
- 2 are the right short seller but a different company: a post on MBLY tagged NVDA, and Wolfpack saying it is long ISRG
  (its report was on UFPT). This is the same tagging problem W1a had.
- 1 could not be read: its id wasn't returned by the API in the window, so it counts as wrong.
