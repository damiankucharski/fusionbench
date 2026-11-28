# Sensitivity to the mix of benchmark scenarios

Round 10 used eight fresh draws for each of 30 fixed scenarios. Rankings below use mean full-class Brier, with lower values preferred. Every ranking includes all five native configurations, the arithmetic mean, Top-N and logistic stacking (LR). The method locked in round 9 was `replicated`.

## Does the group winner depend on one scenario?

A positive margin is the reduction in Brier relative to the best other method. Each omission reweights the remaining scenarios equally. Methods within 10⁻¹² in mean risk share the lead. An omitted-scenario range extending below zero means an alternative overtakes the full-group leader.

| Mechanism | Full-group leader | Margin [95% interval] | Retains or shares lead after omission | Margin range after omission |
|---|---|---:|---:|---:|
| evidence_calibration | native | +0.000179 [-0.000963, +0.000369] | 2/3 | -0.000144 to +0.000268 |
| pool_selection | replicated | +0.000355 [-0.00224, +0.00071] | 2/4 | -0.000529 to +0.000473 |
| error_dependence | replicated | +0.000123 [-0.00039, +0.00037] | 3/5 | -0.000349 to +0.000187 |
| structured_rules | native | +0.000251 [-0.0011, +0.00135] | 7/9 | -0.000381 to +0.000661 |
| context_routing | native | +0.000915 [-0.00223, +0.00245] | 2/3 | -0.00401 to +0.00137 |
| transfer_shift | mean | +0.0032 [-0.0149, +0.0203] | 2/3 | -0.0986 to +0.108 |
| grouped_sampling | native, single, half, replicated | 0 [-0.0012, 0] | 2/3 | -5.93e-05 to 0 |

The three omissions that cost a full-group leader the most Brier were

- Omitting `rare_shortcut_noise` from transfer_shift puts native, guarded, single, half, replicated first. The original leader trails by 0.0986.
- Omitting `gated_experts` from context_routing puts lr first. The original leader trails by 0.00401.
- Omitting `needle` from pool_selection puts lr first. The original leader trails by 0.000529.

## What changes across the two illustrative profiles?

λ is the fraction of the fixed rule-design profile. The balance goes to the multi-site profile. These two profiles describe examples of research priorities. Applying a profile to deployment would require evidence about its scenario frequencies.

| λ | Leading methods | Mean Brier | Margin [95% interval] |
|---:|---|---:|---:|
| 0 | native | 0.42160 | +0.000183 [-0.000447, +0.000479] |
| 0.25 | native | 0.36948 | +0.000437 [-0.000289, +0.000696] |
| 0.5 | native | 0.31736 | +0.000507 [-0.00031, +0.000946] |
| 0.75 | native | 0.26524 | +0.000577 [-0.000415, +0.00121] |
| 1 | native | 0.21312 | +0.000647 [-0.000564, +0.00148] |

Each method’s risk changes linearly with λ. We calculated all pairwise intersections and compared the corresponding risks between consecutive intersections.

The point leader throughout 0 ≤ λ ≤ 1 is native.

## Which scenarios move the locked comparison?

Signed changes below are locked minus reference Brier within each scenario. Dividing a change by 30 gives its contribution to the standard macro. Shown are the largest absolute changes in either direction.

- Against `native`, compositional_switch -0.00569, gated_experts +0.00551, hardrescue +0.00531, hardveto +0.00341. Standard macro change +0.000196.
- Against `guarded`, parity -0.0684, two_pathways -0.0201, compositional_switch -0.00884, hardveto -0.00836. Standard macro change -0.00413.

Intervals are the 2.5th and 97.5th percentiles of 10,000 paired seed resamples within each fixed scenario. We recompute the closest competitor in each resample. These descriptive intervals condition on the reported leader and are unadjusted for multiple comparisons. Omissions measure sensitivity to group membership. Crossings describe the risks observed on this panel.

[Omissions by scenario](results/r10-profile-omissions.csv) · [All eight methods at each blend](results/r10-profile-grid.csv) · [Continuous intervals](results/r10-profile-envelope.csv) · [All scenario effects](results/r10-profile-drivers.csv)
