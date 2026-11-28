# Resampling the eight common seed indices together

The independent confirmation used the same eight integer data-seed indices for each of the 30 standard scenarios. The registered analysis resampled paired method differences separately inside each fixed scenario. As a sensitivity check we instead kept each seed's complete vector of scenario results together, took its profile-weighted mean, and resampled eight such vectors with replacement. We used 10,000 bootstrap replicates and percentile intervals. Both procedures give the same eight point estimates.

| R10 profile | Locked rule minus ordinary native, joint 95% interval | Locked rule minus logistic stacking, joint 95% interval |
|---|---:|---:|
| All 30 scenarios, equal weights | +.00020 [−.00014, +.00054] | −.02515 [−.02672, −.02359] |
| Seven equally weighted mechanism groups | +.00024 [−.00008, +.00054] | −.02069 [−.02220, −.01926] |
| Multi-site research profile | +.00037 [+.00008, +.00066] | −.01328 [−.01709, −.00995] |
| Rule-design research profile | +.00065 [−.00004, +.00135] | −.04620 [−.04929, −.04327] |

Positive differences mean higher Brier for the locked rule (`replicated`). All eight comparisons retain the registered conclusion about whether the interval includes zero. On the full panel the registered interval against ordinary native OKAPI was [−.00021, +.00061]. Grouping the draws by their shared seed still puts the upper bound well below the +.005 tolerance. On the multi-site profile, both schemes detect a small increase in Brier against native OKAPI. Only eight blocks contribute to the joint intervals, so their endpoints are sensitive to individual draws. This check retains the preregistered comparisons as the basis for the formal decision. [Both interval estimates](results/r10-seed-blocks.csv) · [Reproduction from raw confirmation scores](seed_blocks.py)
