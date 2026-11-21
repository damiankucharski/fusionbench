# Changing the starting trees changes measured optimization error

With 128 TwoPathways training labels, replacing four single providers by one randomly drawn monotone function of each arity reduced the genetic search's excess training errors from 12.17 to 8.42 per 128 labels after exactly 300 fitness evaluations. The paired change was −3.75 errors (descriptive 95% t interval −6.21 to −1.29). Exact IID population error decreased from 31.2% to 27.0%, a change of −4.17 percentage points (−6.36 to −1.97). Shifted population error decreased from 28.1% to 25.0%.

The intervention changes both the functions available at initialization and their tree structure. It uses the same objective, operators, initial population count, variation settings and fitness-call count. Choosing the best training score in the complete evaluated archive fixes the selection policy. Twelve new training draws were crossed with two prespecified GP seeds; intervals average the two searches per draw. Replaying the entire experiment reproduced every non-timing value in all 4,320 records.

| Initial population or search reference | Extra fit errors | Exact ERM hits | Distinct functions seen | IID population error |
|---|---:|---:|---:|---:|
| Four singleton providers | 12.17 | 0/24 | 12.3 | 31.2% |
| One starter per essential arity | 8.42 | 2/24 | 29.2 | 27.0% |
| Four uniformly sampled functions | 2.88 | 14/24 | 35.9 | 22.5% |
| Four random balanced binary trees | 5.58 | 11/24 | 21.4 | 23.5% |
| Uniform proposals without replacement, entire family | 0.00 | 24/24 | 166.0 | 19.0% |

The balanced seeds already hit an empirical optimum in 10 of 24 runs at the four-call checkpoint. The two initial draws were reused across datasets; the 24 runs do not represent 24 independent draws of the starter architectures. Uniform four-function initializations averaged 8.625 nodes per starter, compared with one node for singletons. Timings include the initialization and subsequent fitness-call prefix; canonical family enumeration is cached across runs. The direct finite-family reference terminates after 166 draws, so enumeration is a strong default for the measured four binary inputs. Counting all functions becomes impractical at higher dimension.

For additive labels, changing from singletons to one starter per arity reduced IID risk from .328 to .302; for parity the change was .410 to .377. On the 32-label TwoPathways control, exhausting the grammar reduced training error to .159 but IID error remained .252. These results identify the effect of the tested starter policy on this finite problem. A general-purpose recommendation needs more independent initial populations and cases where the finite family cannot be enumerated.

```bash
python -m fusionbench.eval.search_initialization --workers 2 --out initialization
```

[Frozen protocol](initialization-protocol.md) · [All method and budget means](initialization/means.csv) · [Raw runs](initialization/runs.csv.gz)
