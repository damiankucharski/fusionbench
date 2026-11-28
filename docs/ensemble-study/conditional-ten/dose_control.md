# Does screening help at the same weighting strength?

## Secondary comparison specified before the round 9 lock

On the nine round 10 focal profiles, measure the effect of applying the source trigger at two fixed strengths. Compare the registered `half` with unconditional `always_half`, and the registered `single` with unconditional `guarded`. All four use the same ESS-guarded weights. The half-strength schemes use `1 + 0.5 * (w - 1)` when active. Reuse the same 24 sources per profile and their four registered target laws. Reuse the registered fit whenever its effective weights equal those of `always_half`. Fit the missing half-strength predictions on the original source rows with the same native model and settings. Check reference forecast hashes by replay on a representative source.

Report paired triggered-minus-unconditional differences on each target, and the mean per-source difference in maximum Bayes-relative regret across the inactive/active shortcut endpoints. Summarize activation and fit reuse. Each source contributes once per contrast. The results are descriptive and do not change the selection rule or confirmatory criteria.

## Confirmation readout

On the mixed source the screen activated in all 24 fits. Mean maximum endpoint regret was 0.08986 for ordinary native OKAPI, 0.08202 at half strength and 0.07312 at full strength. For each strength the triggered and unconditional procedures produced identical predictions. The improvement over native in this profile is attributable to applying weights. This comparison uses the single adjusted screen; the separately locked confirmatory rule also requires replication. [Source-level risks and hashes](results/r10-dose-control-scores.csv) · [Rule selection](selection.json)

| Source profile | Single screen active / 24 | Half strength: triggered minus unconditional | Full strength: triggered minus unconditional |
|---|---:|---:|---:|
| Mixed | 24 | 0 | 0 |
| Case mix only | 0 | −0.00027 | −0.00322 |
| Varied shortcut, fixed region prevalence | 24 | 0 | 0 |
| Reversed association | 24 | 0 | 0 |
| Extreme region proportions | 12 | +0.00518 | +0.00645 |
| Small source (128 rows) | 10 | +0.00502 | +0.00676 |
| Noisy context metadata | 24 | 0 | 0 |
| Homogeneous source | 0 | +0.00205 | −0.00502 |
| Narrow shortcut contrast | 17 | +0.00120 | +0.00165 |

Entries are changes in mean maximum endpoint regret. Positive values favor unconditional weighting. At the extreme region proportions, the screen withheld nonuniform weights on 12 sources, raising average regret at both strengths. The paired 95% intervals were [+0.00163, +0.00926] at half strength and [+0.00096, +0.01214] at full strength. For the small source both intervals included zero. In the case-mix-only source the half-strength interval was [−0.00558, +0.00528]. These intervals resample the 24 paired source seeds within a profile; comparisons across the nine profiles are descriptive. The score table also gives each of the four target-world Brier contrasts separately. [Paired profile and target comparisons](results/r10-dose-control.csv)

Across 216 sources, 134 unconditional half-strength predictions reused the registered half-strength fit and two reused a uniform fit. Only 80 sources needed a new half-strength fit. On mixed seed 1150100, replaying native and triggered half reproduced the archived training and test forecasts, tree and search path hashes. The comparison took 35.5 seconds on one thread. [Verification and timing](results/r10-dose-control.json) · [Reproduction code](dose_control.py)
