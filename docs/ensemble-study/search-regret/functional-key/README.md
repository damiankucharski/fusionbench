# One representative per MIN/MAX function changes what GP explores

With 300 objective calls, keeping one tree per distinct prediction function lowered mean training regret from **0.12848 to 0.12103**. The 95% paired interval for functional minus structural deduplication was **−0.01612 to +0.00122**, so the primary quality endpoint was inconclusive. At the prespecified secondary budget of 1200 calls, GP found **113 versus 51 distinct functions** and fit-selected IID Brier improved from **0.21040 to 0.18678** (paired difference −0.02362, 95% interval −0.03517 to −0.01207). Logistic stacking still had lower IID Brier at 0.14065. [Prospective protocol](../functional-key-protocol.md)

The only optimizer change is the first-occurrence rule at the generation's population merge. The stock rule retains distinct tree structures. The experimental rule identifies a pure MIN/MAX function by its 256 binary outputs on the full cube of the eight providers. Each function is evaluated symbolically with bitwise AND/OR, preserving sequential operations on the parent value. Since MIN/MAX commute with thresholding, the signature distinguishes all distinct functions on real inputs as well. It requires no labels. A discarded structure may still have produced different descendants through future mutations. Related work used Boolean decision diagrams to check exact child–parent equivalence during crossover on multiplexer and parity problems. [Beadle and Johnson, 2008](https://doi.org/10.1109/CEC.2008.4630784)

## Quality at equal evaluation budgets

These are means across 12 independent TwoPathways draws and four GP seeds per draw. The fits use 128 labels. Validation has 128 fresh labels, and IID and shifted tests use 8192 examples each. Detector accuracy goes from 0.90 to 0.78 under shift. The three cheap baselines use all the same inputs and fit labels, then freeze. Top-N tried every prefix after ranking the eight providers by training Brier. It selected six to eight. The exact solution minimizes empirical positive-channel Brier over unrestricted-size MIN/MAX trees and picks the least tied optimizer. There were no exact hits by either GP policy at either budget.

| Method | Fit Brier | Validation | IID | Shift |
|---|---:|---:|---:|---:|
| Exact MIN/MAX | 0.05866 | 0.09726 | 0.09588 | 0.17616 |
| All-provider average | 0.16046 | 0.15408 | 0.15483 | 0.19289 |
| Ranked-prefix Top-N | 0.15749 | 0.15689 | 0.15791 | 0.19456 |
| Regularized logistic stack | 0.12476 | 0.14160 | 0.14065 | 0.18151 |
| Structural, 300, fit champion | 0.18714 | 0.23244 | 0.22475 | 0.23896 |
| Functional, 300, fit champion | 0.17968 | 0.21854 | 0.21544 | 0.23509 |
| Structural, 1200, fit champion | 0.17111 | 0.21952 | 0.21040 | 0.23076 |
| Functional, 1200, fit champion | 0.15224 | 0.19077 | 0.18678 | 0.22051 |
| Structural, 1200, validation champion | 0.20163 | 0.18096 | 0.20765 | 0.22980 |
| Functional, 1200, validation champion | 0.19028 | 0.15810 | 0.18876 | 0.22156 |

At 1200 calls the fit-regret reduction was 0.01887 [0.00698, 0.03075] and shifted Brier improved by 0.01025 [0.00360, 0.01690]. Fit regret improved in 33 of 48 seed pairs, worsened in 13, and tied in two. Training-selected predictions changed in 46 pairs. At 300 calls the corresponding fit outcomes were 26 wins, 14 losses and eight ties. Intervals are paired across datasets after averaging restarts within each dataset. The secondary intervals are descriptive and unadjusted for multiple endpoints. [All champions and all runs](gp.csv) · [All checkpoint means](means.csv) · [Paired intervals](paired.csv) · [Baselines with selected prefixes](baselines.csv)

## What changed inside the search?

| Matched checkpoint and policy | Objective calls | Joined batches | Distinct trees evaluated | Distinct functions evaluated | Mean essential providers per function | Fit regret | Fit champion IID |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 calls, structural | 300 | 21.58 | 85.98 | 28.25 | 2.05 | 0.12848 | 0.22475 |
| 300 calls, functional | 300 | 25.44 | 50.98 | 43.33 | 2.33 | 0.12103 | 0.21544 |
| 1200 calls, structural | 1200 | 72.81 | 290.75 | 50.65 | 2.57 | 0.11245 | 0.21040 |
| 1200 calls, functional | 1200 | 89.69 | 192.35 | 113.02 | 3.24 | 0.09358 | 0.18678 |
| 20 completed generations, structural | 289.40 | 20 | 86.77 | 28.42 | 2.05 | 0.13134 | 0.22832 |
| 20 completed generations, functional | 237.15 | 20 | 42.77 | 37.65 | 2.19 | 0.13037 | 0.22831 |

At the common 20-generation checkpoint functional pruning evaluated **52.25 fewer candidates** on average [35.35, 69.15], and still explored 9.23 more functions [4.87, 13.59]. The paired IID Brier difference for fit champions was −0.00001 [−0.01184, +0.01182]. The fixed-call contrast changes both population diversity and the number of rounds of variation that can be afforded. The early fixed-generation control distinguishes savings in objective calls from an immediate quality advantage at that point. A fixed-generation comparison later in evolution would be needed to separate those mechanisms at the 1200-call checkpoint.

At 1200 evaluations the two selected mask implementations averaged 0.0237 seconds per search for structural-on-structural trajectories and 0.0089 seconds for functional-on-functional trajectories in this run. The instrumented searches took 0.379 and 0.391 seconds, respectively. They compute both masks for verification, fingerprint structures and archive validation predictions. These timings describe this CPU experiment. The truth table grows exponentially with provider count, and agreement on the cube certifies equality only under the stated MIN/MAX grammar. [Bitset implementation](../../../../fusionbench/eval/lattice_signature.py) · [Tests against native predictions](../../../../tests/test_lattice_signature.py)

## Reproduction

Use the identified OKAPI revision with FusionBench's lattice extra. The bitset tests compare native outputs with symbolic signatures on every Boolean corner of 50 random sequentially composed trees, covering one to twelve providers. They also cover the previous natural collision witnesses and a known pair of genuinely distinct functions. The output records the structural and functional mask times on each visited candidate batch, with aggregate values at each checkpoint. A separate native engine replay reproduced all 96 structural-policy fit champions and losses. An independent tree evaluator reconstructed every one of the 576 recorded winners and matched all 2304 split-level Brier calculations, full prediction digests, exact fit regrets, hit flags and node counts. A second complete run matched every non-timing value. [Verification script](verify.py) · [Native records](native-structural.csv)

The complete study can be reproduced from the FusionBench checkout.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
export PYTHONPATH=".:/absolute/path/to/okapi-86f21cd"
python -m pytest -q tests/test_lattice_signature.py tests/test_structural_key_audit.py
python -m fusionbench.eval.functional_key_audit --out /tmp/functional-key --workers 2
python docs/ensemble-study/search-regret/functional-key/summarize.py /tmp/functional-key
python docs/ensemble-study/search-regret/structural-key/native_replay.py --policy structural --data-start 3600 --search-seeds 211 223 227 229 --out /tmp/functional-key/native.csv
python docs/ensemble-study/search-regret/functional-key/verify.py /tmp/functional-key /tmp/functional-key/native.csv
```
