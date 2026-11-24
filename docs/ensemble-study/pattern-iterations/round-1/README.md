# Round 1 · explicit pairwise minima

Six independent replicates in each of 19 v1 scenarios compared the existing native pattern procedure with an explicit shortlist of pairwise MIN features. The deployed predictor in each arm is one OKAPI tree. The [protocol](../protocol.md) fixes the losses, promotion threshold and confirmation plan.

Joint Brier changed from 0.21975 to 0.21906. The paired effect was −0.00069 (95% CI −0.00193 to +0.00055). Clean Brier changed by −0.00092 (−0.00259 to +0.00075). Retain the original incumbent for round 2. In 27 of 114 discovery archives at least one extra pair survived feature screening, and in 12 of those banks validation selected the augmented tree. Screening cost an extra 220 scalar fitness calls per bank on average.

On clean inputs the original validation choice lost 0.00360 Brier relative to the best of the three discovery-fitted candidates in hindsight. This rose to 0.02707 under half-dose logit jitter. Refitting the selected tree on all development rows improved jitter Brier by 0.00601 on average. This motivates testing sensitivity to the sizes of fitted logit weights. The oracle uses test labels and is diagnostic only. On the nine canonical shifts probability stacking scored 0.30799, ahead of both native models.

Full control means, paired contrasts, family means, selection diagnostics, sampled executable trees and per-fit elapsed times are archived here. Fits summed to 182 independent worker-seconds across all methods (114 paired banks, four workers). Each native arm ran the same 600 GP evaluations per bank. Reproduce with `run.py --round 1 --workers 4`, followed by `analyze.py <output>` using FusionBench 2f12065 and OKAPI 6377375.
