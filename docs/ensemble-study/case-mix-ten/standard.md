# Where weighting costs accuracy on the standard panel

The rule chosen before round 10, `guarded`, passed the prespecified standard-panel tolerance of +0.005 Brier. Its mean test Brier was 0.25001 versus 0.24912 for ordinary native OKAPI. Applying unrestricted weights everywhere raised loss to 0.25548. For scale, logistic stacking scored 0.27550, ranked-prefix Top-N 0.35747 and the arithmetic mean 0.37592. The panel contains 30 fixed families, each with six fresh paired draws of 512 training and 512 test rows. [Locked rule](selection.json) · [Raw scores](results/r10.csv)

| Training rule | Mean Brier change from native (paired 95% interval) | Families lower / tied / higher | Individual draws lower / tied / higher |
|---|---:|---:|---:|
| Guarded | +0.00089 [−0.00037, +0.00224] | 8 / 5 / 17 | 68 / 35 / 77 |
| Full | +0.00636 [+0.00344, +0.00956] | 6 / 5 / 19 | 57 / 34 / 89 |

Lower Brier is better. A family counts once, by its mean paired difference. Differences of at most 10⁻¹² in magnitude count as ties. The interval resamples paired draws within each family and keeps all 30 families. [Paired contrasts](results/r10-contrasts.csv)

| Family | Guarded minus native | Full minus native | Guarded minus native by seed 1050400 through 1050405 |
|---|---:|---:|---|
| `parity` | +0.02174 | +0.13408 | +.0423, +.0112, +.0081, +.0428, +.0158, +.0102 |
| `two_pathways` | +0.01773 | +0.01773 | +.0160, +.0133, +.0270, +.0007, +.0490, +.0004 |
| `native_close_switch` | −0.01942 | −0.01942 | −.0105, −.0201, −.0362, −.0116, −.0106, −.0275 |

The first two families contributed +0.00132 to the guarded panel mean, 50.5% of the sum of positive family contributions. The other 28 together contributed −0.00042. For unrestricted weighting, those same two families accounted for +0.00506 of the net +0.00636. The largest confirmation improvement was on `native_close_switch`. Guarded had raised loss there on 13 of 16 earlier draws across rounds 1–8, so its six confirmation gains should be read alongside that reversal. [Confirmation family scores](results/r10-families.csv) · [Round 1–8 scores](results/)

In `parity`, detectors report two hidden bits whose XOR determines the label. In `two_pathways`, each of two pairs of hidden conditions can jointly trigger a positive label. Provider disagreement and consensus confidence can therefore vary with the evidence for the generating rule. These were the two median-split variables used as proxies for environment and context on this panel. Forcing their frequencies toward independence changes how much those input patterns count during search, validation and refitting. The source and test samples for these two families already follow the same law. On `parity` the row-weight effective sample size was 147–206 out of 512 under full weighting and 368–378 under guarded weighting, which applied its cap on all six draws. On `two_pathways` the guard accepted unrestricted weights every time. Both weighted procedures can alter the learned parameters or tree within a family of models. The scores alone do not establish which change caused the regression. [Scenario generators](../../../fusionbench/scenarios/gates.py) · [Proxy construction and protocol](protocol.md) · [Row-weight diagnostics](results/r10-weights.csv)
