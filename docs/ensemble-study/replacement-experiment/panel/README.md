# Does a certified replacement policy help on the standard scenarios?

## Is the earlier regularized OKAPI still ahead of probability stacking?

We replayed the earlier frozen regularized native method on the exact 144 banks in this panel, using the same 640 development and 1,024 test rows for both methods. All source/test input and label digests matched. Retraining probability logistic stacking with C=1 reproduced its archived predictions byte for byte on every bank. The earlier method retained its 600 GP discovery scoring visits, native regularized weight fitting, validation choice between three trees and full-development head refit. The replacement experiment instead used 72 offspring proposals to isolate birth and promotion. Scoring visits and offspring proposals measure different kinds of work.

| Method on the same banks | Mean test Brier |
|---|---:|
| Regularized native OKAPI with validation choice | 0.21456 |
| Logistic stacking on probabilities | 0.24431 |
| GP-only candidate from the longer regularized procedure | 0.28419 |
| Ordinary short native GP, all 640 rows used for discovery | 0.35892 |
| Short GP with open breeding and KL replacement | 0.44686 |

Regularized native minus probability LR was −0.02976. The paired 95% interval was −0.03165 to −0.02785, using the panel's within-scenario bootstrap. The regularized procedure had lower mean Brier on 18 of the 24 scenarios. Validation chose a weighted native tree combining provider and GP features on 77/144 banks, a weighted provider-only tree on 40 and the GP candidate on 27. Each deployment is one native tree. The GP-only row above fixes that candidate on all 144 banks and measures its test risk. These comparisons do not separately identify the effects of search budget and initialization. In the earlier panel of six different seeds per scenario, regularized native and probability stacking scored 0.21408 and 0.24545, respectively. [Earlier panel](../../validation-curse/panel/round2/README.md) · [Replay script](compare_variants.py) · [All paired results](results/matched-variants.csv)


On 144 fresh banks, an open breeding population with certified incumbent replacement scored **0.44686 Brier**, compared with **0.36252** for empirical replacement on the same native proposals. The paired difference was **+0.08434** (95% interval +0.06123 to +0.10481), with higher loss for certification on 96 banks, lower loss on 14 and a tie on 34. Nineteen of the 24 scenario means favored empirical replacement, three favored certification and two tied. The two open arms had identical proposal and survivor trajectories in every bank. [Frozen design](protocol.md) · [Paired contrasts and intervals](results/contrasts.csv) · [Per-scenario results](results/scenario-means.csv)

Allowing uncertified children to breed under the same certified deployment rule scored 0.00355 lower Brier than strict breeding (interval −0.00985 to +0.00177). It won three banks, lost three and tied on 138. The panel leaves the average advantage of open breeding unresolved. The deployed tree changed from the initial best singleton on only four of 144 banks in either certified arm. The low replacement rate restricts what this comparison can reveal about differences between their populations.

| Procedure | Discovery / gate labels | Mean test Brier | Accuracy |
|---|---:|---:|---:|
| Initial best provider | 256 / 0 | 0.45294 | 69.56% |
| Ordinary native, final discovery winner | 256 / 0 used in selection | 0.36053 | 74.32% |
| Open breeding, raw replacement | 256 / 384 | 0.36252 | 74.55% |
| Open breeding, certified replacement | 256 / 384 | 0.44686 | 69.81% |
| Strict breeding, certified replacement | 256 / 384 | 0.45041 | 69.57% |
| Ordinary native, full discovery budget | 640 / 0 | 0.35892 | 74.92% |
| Arithmetic mean | 0 / 0 | 0.37756 | 73.45% |
| Brier-ranked Top-N | 640 / 0 | 0.33838 | 75.57% |
| Probability logistic stacking, C=1 | 640 / 0 | 0.24431 | 81.14% |
| Logit logistic stacking, C=1 | 640 / 0 | 0.24744 | 80.69% |

The same three supervised controls on 256 rows scored 0.34074, 0.25440 and 0.25851 respectively. Ordinary native with all 640 rows improved by 0.00161 on average over ordinary native with 256 (interval −0.01443 to +0.01171 for full minus smaller). The ordinary procedure's gate outcomes at the 256-row budget were collected for accounting and did not influence either breeding or final selection. [All method means](results/means.csv) · [All 2,016 scores and prediction fingerprints](results/scores.csv)

## What was admitted and deployed?

Across 10,368 offered slots per arm, 2,364 had positive empirical gain in the raw open arm. Seven passed the simultaneous bound in the certified open arm, across only four of the 864 decisions. Four slots passed in strict breeding, again across four decisions. The raw arm made 497 changes, of which 313 helped on the fresh test sample, 173 hurt and 11 tied to tolerance 10⁻¹². Both certified arms made four changes each, all of which improved test Brier. A change is evaluated relative to the previous incumbent, so repeated changes over a path contribute separately. These are held-out sample outcomes. The gate's statistical statement concerns outcomes conditional on the gate inputs.

The four replacements in the open certified arm occurred twice on hardrescue, once on parity, and once on compositional_switch. On the two hardrescue banks the new trees lowered test Brier relative to the initial singleton by 0.28681 and 0.40240. Three of the four final replacements had a selected-parent lineage containing an unconfirmed tree. The ancestry records describe reproductive opportunity; crossover can select a parent without retaining a particular piece of its program. Across the six hardrescue banks, mean loss was 0.54543 for open certification, 0.66030 for strict certification and 0.23252 for raw replacement. [Per-bank paired outcomes](results/paired.csv) · [Generation records and retrospective change scores](results/generations.jsonl.gz)

## Resource accounting and inference

Every native search offered six generations of twelve proposals; duplicate and unchanged programs consumed a slot. The initial population had up to eight breeders, with a separate incumbent slot. The four matched arms averaged 77.83 native discovery evaluations, including initialization and all 72 proposal slots. The initial discovery-risk scan over all provider columns cost another 10.46 singleton scores on average. Each 256-row arm evaluated twelve incumbent–child contrasts on each fresh 64-row block. This required 78 model forwards per bank at the gate (six incumbent and 72 child forwards), or 4,992 model-row predictions. The 640-row ordinary arm performed no gate forwards; its discovery row-evaluation count including singleton scanning was 56,507 on average versus 22,603 for a 256-row arm. Post-search evaluation of the deployed and intermediate incumbents was separate from these budgets.

Average numbers of distinct genotypes visited were 20.44 for all three 256-row open policies and 13.23 for strict certification. Final trees used 1.94 distinct providers for empirical replacement, 1.03 for open certification and 1.03 for strict certification, compared with 1.99 for ordinary 256-row search. The two certified policies' smaller deployment footprint largely reflects keeping the initial single provider. The study counts required provider outputs; it does not train a network that maps raw inputs directly to predictions. [Resource table](results/mean-budgets.csv)

Each of the 24 fixed standard-v3 scenarios contributes six independent banks. Search and data have separate seeds. The test uses the generator's native test hook; for environment_shortcut this changes the environment mixture. Bootstrap intervals resample banks within fixed scenarios, preserving paired outputs. There were 720 native searches and 4,320 generation records. The two-worker panel took 14.81 seconds on CPU, with one numerical thread per worker. [Seeds and data hashes](results/banks.json) · [Runtime and software versions](results/run.json)

An independent single-thread replay refit all 720 native searches. All 2,016 held-out prediction hashes, scores and selections matched exactly, as did all 4,320 generation records. It also regenerated 2,304 source, gate and test fingerprints and reproduced every paired contrast, interval and sign count (maximum aggregate difference 1.12 × 10⁻¹⁶).

## Reproduction

Use the FusionBench research checkout containing the shared replacement runner with the corresponding OKAPI research dependency. The search and evaluation code does not send test labels into evolution or promotion. The native adapter can be replayed using the source and search seeds in the bank file. [Panel runner](run.py) · [Inputs and reference methods](inputs.py) · [Native search implementation](../../../../fusionbench/eval/replacement_search.py) · [OKAPI proposal and replacement API](https://github.com/damiankucharski/OKAPI/commit/1d7ad2d)

```sh
export BACKEND=numpy LOGURU_LEVEL=ERROR OKAPI_LOG_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python docs/ensemble-study/replacement-experiment/panel/run.py --workers 2 --out results/replacement-panel
```
