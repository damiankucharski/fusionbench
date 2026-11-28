# Selecting a provider direction before or after mixing it with the mean

Selecting by the loss of the mixture changed the winning candidate on 122 of 192 banks. In every one of those banks, the new candidate lay farther from the provider mean on the selection rows. The fitted mixture therefore moved farther too, even though both rules gave the chosen candidate exactly 25% weight. This geometric effect persisted on most new samples. It did not give a clear improvement in average test Brier. [Frozen protocol](protocol.md) · [Per-bank winners, displacements and input hashes](results/banks.csv)

## A deterministic consequence of squared loss

Let *b* be the mean of all providers, *T* one candidate, *R* empirical full-class Brier, and *D(T)* the mean squared Euclidean distance from *T* to *b* on those rows. For any fixed mixture weight *w* in (0,1), expanding the squared error gives

```text
R((1−w)b + wT) = (1−w)R(b) + w R(T) − w(1−w) D(T).
```

Thus the in-mixture rule minimizes `R(T) − (1−w)D(T)`. If *U* minimizes raw loss and *A* minimizes mixed loss on the same menu, then

```text
0 ≤ R(A) − R(U) ≤ (1−w) [D(A) − D(U)].
```

This proves *D(A) ≥ D(U)* on the selection sample, including exact ties. The squared displacement of the deployed mixture is *w²D(T)*. Independent examples can change both the ordering by displacement and the ordering by prediction error.

## Paired results on standard-v3

The experiment used eight seeds for each of the 24 standard scenarios. The prior, all individual providers and all unweighted pairs formed a fixed menu of 4 to 5,051 candidates depending on the scenario. Each method received the same 192 labeled selection rows and the standard 1,024-example held-out test split. The rules chose by raw or mixed Brier on selection rows. Both headline predictions mixed their winner with the prior at *w*=0.25. The two menus and inference weights were identical. [All candidate losses and raw displacements](results/candidates.csv.gz) · [Per-bank method scores](results/scores.csv)

| Paired change, in-mixture selection minus ordinary selection | Selection rows | Test rows |
|---|---:|---:|
| Raw candidate displacement from the prior | +0.05520 | +0.05618 |
| Brier of the deployed 25% mixture | −0.00348 | +0.00069 |
| Brier of the unmodified winning candidate | +0.02748 | +0.04490 |

On the test rows the raw displacement increased on 120 banks, decreased on two and tied on 70. The two reversals occurred on deadweight and needle. For the deployed mixtures, test Brier improved on 35 banks, worsened on 87 and tied on 70. The paired 95% interval for the mean test Brier change was −0.00132 to +0.00276. It resamples the eight banks within each fixed scenario, giving every scenario equal weight. Across scenario averages seven improved, fifteen worsened and two tied. Environment shortcut improved by 0.00980, polarity fault by 0.00742. Weakest link worsened by 0.00886 and two pathways by 0.00822. Environment shortcut uses its configured change in environment proportions on the standard test. The prior itself won the ordinary fit comparison on 71 banks and the in-mixture comparison on five. [All paired intervals and counts](results/contrasts.csv) · [Scenario-level changes](results/scenario-means.csv)

| Method | Mean test Brier |
|---|---:|
| Arithmetic mean | 0.37858 |
| Ordinary selection, raw candidate | 0.34777 |
| In-mixture selection, raw candidate | 0.39267 |
| Ordinary selection, 25% mixture | 0.36343 |
| In-mixture selection, 25% mixture | 0.36412 |
| Fit-ranked Top-N by Brier | 0.34353 |
| Probability logistic stacking, C=1 | 0.25911 |
| Logit logistic stacking, C=1 | 0.26415 |

These comparisons isolate selection over a finite menu of fixed predictions. They contain no genetic search or learned tree parameters. The intervals describe variation in the specified scenario draws. Transfer to another menu, a different deployment weight or a new real provider pool would require another measurement. [Mean losses, accuracies and displacements](results/means.csv)

A separate implementation regenerated all four tensor and label hashes for each bank. It reproduced the 92,512 candidate-domain rows, all 3,072 method-domain scores and both selections on all 192 banks exactly. The loss identity held for every candidate. The six bootstrap intervals and summaries matched to a maximum absolute difference of 6.7 × 10⁻¹⁶.

## Reproduction

From the FusionBench checkout, run the following command with its regular Python dependencies. The study uses one numerical thread and took approximately 13 seconds on this host. The output includes the registry order, seed schedule, software versions, candidate-level scores and per-bank fit/test tensor hashes. [Driver](run.py) · [Manifest](results/manifest.json)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python docs/ensemble-study/anchoring-geometry/menu/run.py --out results/anchor-menu
```
