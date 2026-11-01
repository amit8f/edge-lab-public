# W2 hand check (2026-11-01)

Result: **38 of 50 right (76%). W2's rule says fewer than 80% right stops W2, so W2 stops.** No return was read.

Each article's text was read from the Alpaca news API (kept in the private repo only; no headline text here). The
per-row verdicts and notes, in my own words, are in `W2_check.csv`. A row counts as right only if it is the right
company and a new short report or campaign post by a listed short seller.

The 12 misses:
- 7 are not a short report at all: a lawsuit against Tesla, a court plea, Citron defending GE, Citron closing its GME
  short, a story about a stock recovering after an old report, Muddy Waters saying it is too early to short SNAP, and
  Citron turning away from a long ROKU position.
- 2 are the right short seller but a different company: a post on MBLY tagged NVDA, and Wolfpack saying it is long ISRG
  (its report was on UFPT). This is the same tagging problem W1a had.
- 2 are Citron's Andrew Left talking in interviews about positions he already held (covering part of CC, still short W).
  Not a new report or post.
- 1 could not be read: its id wasn't returned by the API in the window.

The margin is thin. If both interview rows and the unreadable one were counted right, the score would be 41/50, so the
call rests on reading "new report or campaign post" strictly. The two interview rows alone would make it 40/50, exactly
80%. The strict reading is the one used, and it matches how W1a's misses were judged.
