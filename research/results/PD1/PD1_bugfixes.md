
## 2026-10-30: factor regression included the risk-free rate as a factor

Found while checking INS1: the FF5 + momentum regression also used Ken French's RF column as a regressor.
Fixed to the six factors. The verdict doesn't change: rule 12 (t >= 3) already failed and the near-miss
rule 19 uses the pick-minus-control t, not the regression. The committed alpha numbers are from the old code.
