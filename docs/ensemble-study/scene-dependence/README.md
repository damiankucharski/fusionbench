# Does correlation of provider errors explain OKAPI's transfer gap?

A scene can give a model several chances to repeat the same mistake. We varied this dependence while preserving the distribution of every individual model prediction on a single observation. The intervention had a measurable effect on the precision of a fixed model comparison. Its effect on the final selection of the evolving ensemble was less certain.

## What varied

In 32 new banks we held the underlying events, labels, checkpoint choices and random draws fixed. The share `s` determines how much of a provider's residual variance is common to the eight observations from one scene. Both the persistent error and the correlated checkpoint noise receive the same share. The law of the entire checkpoint vector on any one row stays fixed. The scene also influences the event probability, including at `s=0`. Fit data were 192 observations, grouped either as 24 scenes with eight observations each or as 192 distinct scenes. The eight checkpoint choices were frozen on an independent calibration block before the intervention. All final scores used 512 fresh scenes. In the signal condition, one provider emitted the true probability; in the equal condition, all eight providers had the same population competence. [Prespecified protocol](protocol.md) · [Generator](../../../fusionbench/scenarios/checkpoint_bank.py)

| Share | Variance ratio, fixed pair, equal law | Variance ratio, fixed pair, signal law | Clustering penalty, seeded GP on signal law |
| ---: | ---: | ---: | ---: |
| 0 | 1.028 | 1.003 | +0.00114 |
| 0.5 | 1.364 | 1.043 | +0.00196 |
| 1 | 2.084 | 1.107 | +0.00234 |

Each variance ratio estimates how much more variable the observed Brier difference between the designated singleton and the simple mean would be with eight rows per scene relative to one row per scene at fixed row count. We estimated it from 1,536 scene groups in each bank. The ratio need not be the same for two different candidates or laws. Under full sharing, the equal-law ratio was 2.084 (95% interval 2.035–2.133), versus 1.107 (1.088–1.128) when the singleton was the true probability. [Variance components and intervals](results/fixed-design-effects.csv)

The GP penalty is its **deployed excess loss on fresh scenes with 24×8 training minus that with 192×1**. Excess loss is twice the mean squared error of a binary probability against the simulator's true probability. The primary change in this penalty from zero share to full share was **+0.00120**, with a two-sided 95% paired interval **−0.00301 to +0.00544**. The estimate permits changes of either sign. On the equal law the change was +0.00431 (−0.00027 to +0.00962). The corresponding secondary interaction for logistic stacking on the signal law was +0.00832 (+0.00297 to +0.01451); the Top-N and regularized-native intervals crossed zero. These are descriptive secondary comparisons without multiplicity adjustment. [Bank-level primary contrasts](results/primary-by-bank.csv) · [All intervals](results/paired-contrasts.csv)

## What the learner could still choose

At full sharing with a perfect provider, that prediction started in every population and won the fit comparison in 30/32 banks for each layout. In the 24×8 fits, it survived in 28/32 final populations and was chosen in six. In the 192×1 fits, it survived in 26/32 and was chosen in nine. Neither retention nor scene dependence alone accounts for those misses. [Generation records](results/retention.csv)

| Audit excess at full sharing | Equal, 24×8 | Equal, 192×1 | Signal, 24×8 | Signal, 192×1 |
| --- | ---: | ---: | ---: | ---: |
| Native GP seeded with the mean | 0.03536 | 0.03204 | 0.01022 | 0.00788 |
| Simple mean | 0.02433 | 0.02433 | 0.01942 | 0.01942 |
| Top-N | 0.03494 | 0.03425 | 0.00695 | 0.00779 |
| Probability logistic stacking | 0.04481 | 0.03423 | 0.02865 | 0.02222 |
| Regularized native-pattern OKAPI | 0.04944 | 0.04499 | 0.02896 | 0.02296 |

## Reproduce

Use the FusionBench research checkout and its pinned optional OKAPI revision 1d7ad2d. From the repository root:

```sh
uv sync --extra okapi
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
study=docs/ensemble-study/scene-dependence
.venv/bin/python "$study/run.py" --workers 2
.venv/bin/python "$study/analysis.py"
```

The science run took 111.28 s on two CPU workers and produced 1,920 scores and 6,528 generation records. Results apply to these laws, the sampled seeds and this search budget. Round two uses new seeds to change how the same native GP evaluates improvements over an initial tree. [Run manifest](results/run.json) · [Round two](../fold-promotion/README.md)
