# Does screening help at the same weighting strength?

## Secondary comparison specified before the round 9 lock

On the nine round 10 focal profiles, measure the effect of applying the source trigger at two fixed strengths. Compare the registered `half` with unconditional `always_half`, and the registered `single` with unconditional `guarded`. All four use the same ESS-guarded weights. The half-strength schemes use `1 + 0.5 * (w - 1)` when active. Reuse the same 24 sources per profile and their four registered target laws. Reuse the registered fit whenever its effective weights equal those of `always_half`. Fit the missing half-strength predictions on the original source rows with the same native model and settings. Check reference forecast hashes by replay on a representative source.

Report paired triggered-minus-unconditional differences on each target, and the mean per-source difference in maximum Bayes-relative regret across the inactive/active shortcut endpoints. Summarize activation and fit reuse. Each source contributes once per contrast. The results are descriptive and do not change the selection rule or confirmatory criteria.
