# Crossing a neutral intermediate on complementary binary inputs

The four native providers return scores for A, its complement, B and its complement on an equally likely four-corner support. Their arithmetic mean is 0.5 on every corner. A positive case has probability 7/8 when A and B differ and 1/8 otherwise. We used all 32 predeclared seeds and charged twelve native offspring per generation. The loss below is full-class Brier, whose Bayes risk in this law is 0.21875. [Frozen protocol](protocol.md)

Two native conjunctions each identify exactly one of the two high-probability corners. Each conjunction's population risk is 0.5, identical to the incumbent's. Their native MAX gives exactly the posterior. This particular route has two neutral intermediates, verified with actual native tree predictions after each bank's provider permutation.

## Supplying the route

We offered the two intermediate programs in generation one, with ten copies of the incumbent. Open breeding kept both intermediates available. Strict breeding required a certified gain for each. When both were available, the second generation could offer their native MAX. Both policies used the identical KL test, actual independent labels, error allocation and twelve-slot multiplicity when deciding what to deploy. Later generations contained only copies of the incumbent. This replay isolates admission when the opportunity is supplied.

| Policy | Complete program offered in generation two | Complete program deployed |
|---|---:|---:|
| Open breeding, certified deployment | 32/32 | 30/32 |
| Strict breeding, certified deployment | 0/32 | 0/32 |

The two open rejections are possible because the final statistical check sees 128 noisy labels. The scheduled control supplies donors and a successful composition whenever the intermediates are available. It measures the consequences of admission on one known route. It does not measure its natural frequency. [All 384 decisions](supplied.csv)

## Searching freely with native variation

| Method | Exact excess Brier | Test Brier | Beat the starting mean on population risk |
|---|---:|---:|---:|
| Ordinary native GP | 0.27902 | 0.49829 | 5/32 |
| Open breeding, empirical replacement | 0.27550 | 0.49479 | 5/32 |
| Open breeding, certified replacement | 0.27301 | 0.49192 | 5/32 |
| Strict breeding and certified replacement | 0.28125 | 0.50000 | 0/32 |
| Arithmetic mean | 0.28125 | 0.50000 | 0/32 |
| Ranked-prefix Top-N, all development rows | 0.28125 | 0.50000 | 0/32 |
| Probability logistic stacking, all development rows | 0.28296 | 0.50163 | 0/32 |

No natural run produced the exact XOR posterior within 72 native offers, including exploratory offers never deployed. On the same open proposal path ordinary GP was worse than the mean in 14 banks, empirical replacement in eight, and certified replacement in zero. In five banks open certified search deployed a partial improvement whose exact excess was 0.228516. Every open certified deployment in this study referenced all four providers, including when it kept the original mean. [All native and baseline scores](live.csv)

Run from the pinned FusionBench checkout with the optional OKAPI dependency installed. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `BACKEND=numpy`, `PYTHONPATH=.` and `OKAPI_LOG_LEVEL=ERROR`, then run the following commands. Output includes a wall-clock file per component.

```sh
python docs/ensemble-study/replacement-experiment/barrier/run.py supplied
python docs/ensemble-study/replacement-experiment/barrier/run.py live
```
