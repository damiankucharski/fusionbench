# Improving a native OKAPI tree with FusionBench

Variance-scaled shrinkage during the existing derivative-free weight search improved the previous native OKAPI procedure across the fixed 19-scenario suite. The first attempt, increasing the availability of shallow pairwise subtrees, gave little benefit. A real ten-class digits experiment identifies a setting where probability stacking retains an advantage.

| Experiment | Paired banks | Previous OKAPI | Challenger | Change in full-class Brier (95% CI) |
| --- | ---: | ---: | ---: | --- |
| Add pairwise minima | 114 | 0.21975 | 0.21906 | −0.00069 [−0.00193, +0.00055] |
| Penalize logit weights | 114 | 0.21750 | 0.20856 | −0.00894 [−0.01189, −0.00599] |
| Confirm the same penalty | 190 | 0.21524 | 0.20673 | −0.00851 [−0.01079, −0.00624] |

Banks in the three rows are disjoint. The primary score averages clean test data (50%), independent logit jitter (25%) and a single provider supplying another row's predictions (25%). Each family has the same weight. Intervals use paired differences across the independent replicate macro-averages, conditioning on these 19 families. Each learner received the same 640 clean development rows. The four external comparisons were mean, development-ranked best Top-N prefix, and logistic regression on probabilities or logits (C = 1). Comparisons and decision rules were [published before the runs](protocol.md).

## What the new tree does

For a local native logit node, the search evaluates its usual scalar objective minus a penalty on the signed weight of each input. The penalty is proportional to the variance of that input's log evidence on the training rows. Holding the other terms fixed, doubling its weight quadruples its contribution to the penalty. The strength was fixed at 0.25 and divided by the fit sample count. For multiclass inputs the variance uses class-centered log evidence. Bias is unpenalized. Scalar search, held-out selection on raw fitness and deployment as one native tree are preserved. The change affects the flat and augmented heads during both discovery and final refit. The native GP trajectory is shared within a comparison. The two arms may subsequently select different subtree features and candidate families.

## What transfers

| Confirmation panel | Native L2 | Previous OKAPI | Simple mean | Top-N | LR on probabilities | LR on logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weighted primary, 19 scenarios | 0.20673 | 0.21524 | 0.38373 | 0.34259 | 0.24268 | 0.24778 |
| Clean, 19 scenarios | 0.17823 | 0.18433 | 0.37710 | 0.33246 | 0.21734 | 0.22072 |
| Coherent two-provider fault, 19 scenarios | 0.29914 | 0.31975 | 0.41432 | 0.39291 | 0.31568 | 0.31856 |
| Canonical shift, 9 scenarios | 0.30359 | 0.31692 | 0.38541 | 0.36288 | 0.30344 | 0.31430 |
| sklearn digits, 8 overlapping splits | 0.02718 | 0.02718 | 0.05456 | 0.02869 | 0.02285 | 0.03098 |

Lower is better. Each row has its own targets and averaging population. The native improvement was observed on 18/19 family means on the primary panel and 19/19 under the withheld two-provider fault. The nine canonical shifts gave no clear difference against probability stacking. On digits the regularized pipeline selected the ordinary GP tree in all eight splits and probability stacking won seven. Class-specific coefficients or cross-class probability information are possible explanations for the remaining multiclass gap. The present experiment does not distinguish them. Search-seed differences also persisted. A next bounded test could compare a shared-weight native multiclass head with one that allows class-specific weights, holding the same development split, fitness, search budget and three inexpensive baselines fixed. The digits results motivate this hypothesis and should be supplemented by new real prediction banks.

## Reproduction

Install the pinned optional OKAPI dependency from this FusionBench research checkout, for example with `uv sync --extra okapi`, and run the following commands at its root in the resulting environment. Use Python 3.12 and one BLAS/OpenMP thread per worker to reproduce the run conditions.

```sh
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
study=docs/ensemble-study/pattern-iterations
python "$study/run.py" --round 1 --workers 4 --out /tmp/native-round-1
python "$study/run.py" --round 2 --workers 4 --challenger-kwargs '{"l2_strength":0.25}' --out /tmp/native-round-2
python "$study/run.py" --round 3 --workers 4 --challenger-kwargs '{"l2_strength":0.25}' --out /tmp/native-round-3
for round in 1 2 3; do
  python "$study/analyze.py" "/tmp/native-round-$round"
  python "$study/verify.py" --round "$round" --out "/tmp/native-round-$round"
done
python "$study/digits.py" --challenger-kwargs '{"l2_strength":0.25}' --out /tmp/native-digits
```

Published artifacts include fit costs, the selection and test scores of all three candidate trees, stored tree samples, paired effects and per-family results. The replay verifier independently interprets archived trees, including their predictions on the held-out conditions. The public method ID `okapi_native_pattern_l2` exposes this fixed configuration in FusionBench.

- [Round 1 results and candidate diagnostics](round-1/README.md)
- [Round 2 results and regularization protocol](round-2/README.md)
- [Round 3 results, search-seed sensitivity and costs](round-3/README.md)
- [External digits results and split overlap](digits/README.md)
