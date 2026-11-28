# Sensitivity to numerical ties in raw replacement

The primary verifier found three raw promotions on controlled banks whose empirical gain was zero in exact arithmetic. We reran only open breeding with raw confirmation. A challenger had to exceed the preceding incumbent's gain threshold of **1e−12** in full-class Brier units, in place of the original threshold of zero. Only offers above the threshold qualified for deployment, and the offer with the highest qualifying gain won. All offers still competed for breeding. The script patched the selector inside the replay process. All 240 breeding-path hashes matched the primary records. [Primary verifier](../controlled/results/verification.json) · [Replay script](run.py) · [Paired bank results](banks.csv)

| Outcome | Primary raw | Raw with tolerance | Primary open KL |
|---|---:|---:|---:|
| Null banks ever deploying a function worse than the known mean | 41/48 | 41/48 | 0/48 |
| Null final mean exact Brier | .412832 | .412832 | .406250 |
| Signal banks ending better than the starting mean | 41/48 | 40/48 | 1/48 |
| Signal final mean exact Brier | .323576 | .323937 | .358073 |
| Signal final mean held-out Brier | .323316 | .323381 | .35825 |

On the controlled null, two promotions disappeared on bank 28 and all 48 final trees had the same birth IDs. On the signal, two banks ended on different trees. Exact Brier improved by .006944 on bank 29 and worsened by .024306 on bank 36, for a mean change of +.000362 over 48 banks. The one-bank change in the count above leaves raw replacement more likely than open KL to find a true improvement on this signal. [Controlled primary report](../controlled/README.md)

Across all 24 standard scenarios and six banks each, the largest absolute difference in final held-out Brier was below 1e−16. Both panel means were .362522. Seven banks recorded fewer promotions, reducing the total from 497 to 486, and five ended with a different program birth ID. Distinct IDs can represent equivalent predictions. [Panel primary report](../panel/README.md)

To reproduce, run from the FusionBench root with its experimental OKAPI revision installed. The runner compares the replay with the archived primary CSVs and requires a match on every breeding path before it writes the output.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python docs/ensemble-study/replacement-experiment/roundoff/run.py --workers 3
```
