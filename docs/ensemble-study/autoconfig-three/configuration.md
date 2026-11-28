# What the source fingerprint could select

Across the 24 development scenarios, the best fixed choice was `flat`, the regularized native logit head fitted on all source examples. Its mean Brier score was 0.27169. The eight pool measurements did not improve this choice through any of the 32 permitted first splits. On both sides of every split, `flat` still had the lowest average loss. Consequently the frozen rule has a single leaf. Leaving each scenario out in turn also selected `flat` for all 96 development banks. [Frozen policy](results/r1-policy.json) · [Leave-one-family-out decisions](results/r1-leave-family-out.csv)

| Fixed choice | Mean Brier across 24 scenarios |
| --- | ---: |
| Arithmetic mean | 0.38842 |
| Best ranked prefix (Top-N) | 0.36174 |
| Mean with fitted temperature | 0.36424 |
| Regularized native logit head | 0.27169 |
| Mean-seeded, pull-regularized tree with fitted temperature | 0.32863 |

There was room to improve. Choosing the lowest-loss action separately for each bank **after seeing its test outcomes** would have scored 0.25869. For example, on `rare_shortcut_noise` the arithmetic mean scored 0.48186 against 0.69425 for the native head. On `gated_experts`, Top-N scored 0.32502 against 0.35498 for the head. These differences are means over four fresh draws per scenario. [Per-family scores](results/r1-families.csv)

This is a limitation of this fitted selection rule. It considered eight individual measurements and required support from at least four training scenarios on each side of each proposed split. A useful exception affecting fewer banks can be outweighed by other scenarios on the same side of a threshold. An interaction between two measurements could also be useful even if neither first split changes the best action. The experiment does not establish either explanation as the cause. [Splitting protocol](protocol.md) · [Learning and evaluation code](router_study.py)
