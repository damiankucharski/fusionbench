# Round 2 · regularize the weights in each native logit tree

Round 1 missed its promotion threshold. Start again from the unchanged native comparator. Its mean loss relative to the hindsight discovery-tree choice rose from 0.00360 on clean test rows to 0.02707 on jittered rows. The discrepancy motivates a test of whether native head fitting gives unstable features too much influence. It does not identify weights as its cause.

Fix `l2_strength=0.25` for the challenger, without the explicit pairwise basis. On each discovery or full-data fit, maximize raw Brier skill minus 0.25 divided by the fit sample count times the sum of squared signed feature weights scaled by their logit variances on that fit. Exclude the intercept. For multiclass models, use row-centered log evidence and average its variance across classes. The same derivative-free native coordinate/pattern search handles the penalized objective. Keep validation on raw skill, share the GP path, and leave all four external controls unchanged. Log each raw score and penalty. No strength sweep.

Use the shared [protocol](protocol.md) with the six new replicate indices rooted at 2110000 and its existing promotion gates. Report changes in chosen candidate, head coefficients and refit loss as secondary diagnostics. Command: `run.py --round 2 --challenger-kwargs '{"l2_strength":0.25}' --workers 4`.
