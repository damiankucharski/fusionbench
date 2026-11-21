# Inherited gates contribute to frozen predictions on both switching laws

Bypassing every genuinely inherited gate occurrence in a frozen FAR-trained champion raised its conditional-excess Brier from **0.00267 to 0.02437** on fresh FAR examples. The paired all-start difference was **+0.02170** (95% interval +0.00500 to +0.03840). Only 10 of 32 fit-selected winners carried the marker in their actual first archived copy. The other 22 remained unchanged and contributed zero. Resetting only those numeric cutoffs to 0.5, while retaining direction and operands, raised excess by **0.01115** (+0.00317 to +0.01913). [Frozen protocol](../gate-necessity-protocol.md) · [Marked search and source champions](../matched-graft/README.md) · [Paired estimates](paired.csv)

The interventions used frozen programs from 16 independent data banks and two restarts per bank. Each bank supplied 32,768 new feature rows. For a given bank, all programs and all three label mechanisms used the same six anonymous provider outputs and the same underlying random uniforms. No test labels entered training, edits or model selection. Intervals use the 16 bank-level paired mean differences. The scores below are stationary conditional-excess Brier for programs selected on training fitness.

| Prediction on fresh examples | FAR | CLOSE | Logistic |
|---|---:|---:|---:|
| Intact evolved program | **.00267** | **.01771** | .01224 |
| Bypass all inherited occurrences | .02437 | .05139 | .01470 |
| Reset only their cutoffs to 0.5 | .01382 | .03352 | .01213 |
| All-provider mean | .09086 | .04361 | .05805 |
| Ranked-prefix Top-N mean | .04049 | .04192 | .01126 |
| Regularized logistic stacking | .02083 | .03407 | **.00281** |

CLOSE-trained programs lost **0.03368** (+0.01944 to +0.04792) following bypass on the CLOSE law and **0.01581** (+0.00905 to +0.02257) following threshold reset. Under the logistic law, bypass increased the programs' error by 0.00246 (+0.00052 to +0.00441), while resetting their thresholds yielded −0.00010 (−0.00036 to +0.00015). An operator can thus help its trained program while the simpler logistic stack yields substantially better predictions. On the three stationary laws, intact observed Brier was .13618/.22237/.18061 for FAR/CLOSE/logistic, compared with .15785/.25607/.18300 after bypass. The corresponding logistic stacking scores were .15445/.23875/.17105. The three baselines reused the original fit labels and predictions, the individual provider ranking and all six prefix scores were recorded for every Top-N fit. [All programs, shifts and baseline scores](runs.csv.gz)

### Dependence on the mechanism and archive policy

A FAR-trained frozen champion's bypass effect was +0.02170 under its original FAR law, +0.00865 (+0.00194 to +0.01535) under the logistic law, and −0.00058 (−0.00183 to +0.00066) under the CLOSE law, all with the very same feature rows. Independently validation-selected champions gave stationary bypass effects of +0.02315 on FAR, +0.03475 on CLOSE and +0.00168 on logistic. The logistic interval then included zero. These are paired comparisons of frozen models, not a new independent bank of training trajectories.

For the first archived fit winners, bypass changed predictions on 13.72% of all FAR rows, 26.76% of CLOSE rows and 15.99% of logistic rows across their starts. Resetting thresholds changed 4.14%, 8.32% and 3.78%. In each regime all winners carrying a marker had some affected rows. Bypass reduced average program size from 5.47 to 4.47 nodes on FAR, 8.78 to 6.78 on CLOSE and 12.28 to 10.03 on logistic. Absent markers and repeated or nested occurrences were all handled as specified. A separate sensitivity analysis used the first genuinely marked copy observed of the winning structural key. The all-start FAR bypass effect was +0.02164 (+0.00485 to +0.03843) under that definition.

This establishes a contribution of operations actually inherited by these trained predictors under both constructed switching laws. The cutoff matters there even with its surrounding operands and operations fixed. Bypassing an operation also removes its operand subtree, so the numerical bypass effect includes that computation. This program edit does not show whether equally accurate models can be learned without that lineage, or whether the same conclusions hold on observational model outputs. A complementary population experiment substitutes a neutral operator during training and tracks its descendants. [Slot-matched study](../matched-graft/README.md)

### Reproduction

An independent NumPy interpreter reconstructed all 3024 prediction vectors bit for bit, including the baselines, all 6048 reported losses and the 216 paired contrast rows. It reapplied each edit at original ordered-child paths, including synthetic nested and sequential-sibling regressions. The raw 2592 program/edit/law observations include 855 with a change to the intact prediction vector. [Independent verification](verified.json) · [Native cutoff and lineage implementation](https://github.com/damiankucharski/OKAPI/blob/c3c4060/experiments/gate_lineage/README.md)

Use the matched-graft raw file committed with its report, and OKAPI commit c3c4060e1474f502fdcb0784b2274e71d9feb5ea.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-c3c4060"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_gate_necessity.py
python -m fusionbench.eval.gate_necessity --source docs/ensemble-study/search-regret/matched-graft/runs.csv.gz --out /tmp/gate-necessity
python docs/ensemble-study/search-regret/gate-necessity/verify.py --source docs/ensemble-study/search-regret/matched-graft/runs.csv.gz --out /tmp/gate-necessity
```
