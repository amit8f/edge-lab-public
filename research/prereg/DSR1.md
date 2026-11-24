# DSR1: how much of the NX3 family's history passes survive the number of tries (descriptive, not counted)

Registered 2026-11-24, before any DSR1 number is computed. Source: Amit, 2026-11-24 ("do all of this"), the ChatGPT
review's first suggestion. Tests no edge and is not counted (like S0 and SECTOR1). Count stays 190 (205 with phase 1).

## Question

We have registered 190 ideas and tried many variants of one news model. Some of the passes will be luck. For each pass
in `research/IDEAS.md`, how likely is it that its t-statistic would still be that large if the best of N tries had
been picked from ideas with no edge (Bailey and Lopez de Prado's deflated Sharpe ratio)?

## Method (written first)

For each history pass with a reported daily Newey-West t (pick minus same-day random control, or the sized daily gain),
the deflated value is `DSR = Phi(t - m(N))`, where `m(N)` is the expected largest of N independent standard normal
t-statistics: `m(N) = (1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))`, g = 0.5772. The daily Sharpe is t divided
by the square root of the days, so the test-length factor cancels. Skew and kurtosis are not adjusted for (only summary
numbers are committed), which makes the figure generous: the fat tails in these books (top 3 trades = 24-65% of profit)
would lower it. N is varied: 190 (every counted idea, the harsh case), 40 (the about 40 variants of NX3's news model),
10 (a loose view). The trials are correlated (the same events), so 190 overstates the number of independent tries and
10 understates it; neither is the true value, so all three are shown.

Reported for live books too: each book's day-t over its forward days against N = number of live books (about 25).

## Reading rule

A pass whose DSR at N = 40 is below 0.80 is flagged "not believable on its own"; no ideas are dropped and nothing is
re-run. Whatever the table says, the forward ledger decides.

## Not tested

No new data and no new trades. Values come from the committed results listed in `tracks/tools/deflate.py`.

## Result (2026-11-24)

`research/results/DSR1/DSR1_result.json`. The expected best of N null t-statistics is 2.75 at N = 190, 2.19 at N = 40
and 1.57 at N = 10. Of the 21 history passes, 20 stay above 0.80 at N = 40 and 16 stay above 0.90 even at N = 190;
only NX3-BARS (t 2.85) is flagged (DSR 0.75 at N = 40, 0.54 at N = 190). The weakest others at N = 190: NXM-LIN 0.67,
NXM-VOTE 0.71, NXT-TR 0.83, NXK-KRR 0.83. So the number of tries is not what's wrong with the history passes: their
t-statistics are large enough to survive it. The live books are different: a day-t of 1.6 to 2.6 over 13 to 17 days
deflates to 0.35 to 0.73 against 25 live books, which says nothing yet.

What this does not cover, and matters more: (1) the passes compare picks with a random control, not with zero, and
COST1 showed that with measured spreads NX3-ALL-HI's own net is -0.35% a trade; (2) skew and kurtosis are not
adjusted for, which flatters books whose top 3 trades are a quarter or more of profit; (3) the 21 results share the same
news events, so they're variants of one finding, not 21 findings.
