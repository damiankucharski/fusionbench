# The needed subtrees rarely became available for crossover

At generation 15, **none of 32 live populations** on the matched two-pathway task could produce the true rule in one native crossover from two parents that lacked the target. We checked every legally selectable parent, node type and pair of swap sites. All four individuals were target-free in every population. Giving every parent an equal selection probability did not create a route. Across the complete two-law panel, all **25,600 scored trees** computed functions in the 94-rule class that can be built by successively appending one raw input. The target is outside that class. [Preregistered protocol](protocol.md) · [All frozen-population opportunities](opportunities.csv) · [Independent ledger replay](ledger-verified.json)

This result concerns the next possible crossover from each measured state. Native GP can express the target as a seven-node tree and can create it on specially constructed parents. The same genuine native operation had positive probability in a separate test fixture. The empirical result identifies a shortage of useful intermediate programs in these small populations. [Native opportunity proof and positive fixture](https://github.com/damiankucharski/OKAPI/blob/ae020a4/experiments/donor_opportunity/README.md)

## What was present

The four providers were anonymous, with the true paired roles permuted independently for each of 16 banks. The target was max(min(A,B), min(C,D)). Two independently seeded GP runs per bank began with the four single providers. Sixteen Boolean corners distinguish every pure MIN/MAX function on these inputs, including when applied to all 256 four-level prediction vectors. Each offered swap considered the incoming parent value, its earlier and later updates, both resulting children, ties in the two-parent tournaments and self-mating. A separate hypothetical join places a new MIN or MAX above two ingredients. These are different proposal distributions.

| Matched law, 32 starts | Generation 1 | Generation 5 | Generation 10 | Generation 15 |
|---|---:|---:|---:|---:|
| Mean number of different functions among four live parents | 4.00 | 2.44 | 2.19 | 2.25 |
| Mean number of distinct previously scored functions | 4.00 | 6.63 | 8.25 | 9.09 |
| Starts with any productive one-step stock crossover | 0 | 0 | 0 | 0 |
| Starts with a productive new MIN/MAX joining two live roots | 0 | 0 | 0 | 0 |
| Starts with a productive new MIN/MAX joining two full Value-rooted subtrees of different live trees | 0 | 0 | 0 | 0 |
| Starts with two historically scored whole programs whose join gives the target | 0 | 1 | 1 | 1 |

At G15 the empirical mean probability and both ends of the preregistered t interval were zero because every measured probability was exactly zero. Sampling uncertainty remains for new populations. A supplementary exact one-sided 95% upper bound for the chance that a new independent bank has a positive G15 opportunity in at least one of its two frozen starts is **17.1%** (zero of 16 independent banks). It would be inappropriate to treat the two starts on one bank as separate data replicates. [Raw per-pair probability matrices and parent weights](focal.csv.gz)

By 400 evaluations, four of 32 matched archives had at some time scored two whole programs whose new MIN or MAX could express the target. An archive retaining every past root would still have lacked such a pair in 28 starts. Across 7,856 actual matched-law crossover draws and 8,852 additive-law draws, no productive swap yielded the respective population target. Neither subsequent mutation nor scoring produced either target in these samples. The frozen snapshots were exhaustive over swap sites at the four specified generations. The remaining generation traces record actual draws, whole-root joins and chronological membership. Full Value-rooted subtrees as ingredients exclude intermediate accumulated values inside a sequence of child operators. [Generation traces](traces.jsonl.gz) · [Actual stage events](stages.csv.gz) · [Per-generation functional diversity](shape-generations.csv.gz)

## Relation to training and test performance

| Positive-channel Brier on independent IID test labels | Mean | Top-N | Logistic stack | Stock GP at 400 calls | Exact training winner in MIN/MAX |
|---|---:|---:|---:|---:|---:|
| Two-pathway posterior | 0.21341 | 0.21610 | 0.21401 | 0.24719 | **0.18746** |
| Additive logit posterior | 0.19004 | 0.19250 | **0.17003** | 0.22974 | 0.20311 |

All methods learned from the same 256 hard labels. On the matched task the true population rule was among the exact empirical optima on 12 of 16 banks. The native archive contained no empirical optimum on any of the 32 starts, with mean fit Brier 0.04657 above the exact minimum. Its mean population excess was 0.06604, versus 0.00380 for the fit-selected exact winner. It visited just 13.44 distinct semantic rules on average (maximum 18). On the additive control it visited 12.19 (maximum 19), and the fitted logistic baseline beat the exact lattice ERM. A structural search improvement can only remove the optimization component within its repertoire. Ranked-prefix Top-N evaluated all four prefix lengths on its training bank. [Scores and spreads on IID and shift](model-scores.csv) · [Selected programs, ranks and per-run results](runs.csv.gz)

## Why joining branches matters

Start with one raw provider and repeatedly take MIN or MAX with another raw provider. At the final MIN, that last input is necessary on Boolean corners. At the final MAX, that input is sufficient. The true matched rule has no necessary input because either of its disjoint pathways can activate it, and no sufficient input because each pathway needs two. A tree can still compute a chain function even when its written structure branches; we used full input–output equality to classify every scored genotype. All native programs scored on both laws lay inside the chain class in this panel. [Functional classes reached in all 64 starts](shape-runs.csv)

Exhaustive enumeration gives **94** different raw-input chains on four inputs. **102** of the 166 nonconstant monotone rules have at least one necessary or sufficient input. One MIN or MAX joining two chain programs covers **all 166** rules. Both sets were checked by an independent enumeration over the entire four-input truth-table space. Expressibility requires appropriate operands, which were often absent in the actual GP populations. [Executable enumeration and bit sets](chain_reference.py) · [Counts and function signatures](chain-reference.json)

An exploratory extension to five inputs limits the scope of this result. A raw-input chain covers 667 of the 7,579 nonconstant monotone rules; one join of two chains covers 6,292, leaving 1,287 out. The majority-of-five rule is an explicit counterexample. Suppose it were the OR of two chains. Each component would be below majority and could have no sufficient singleton, since a single active input is insufficient for majority. Each such chain must therefore have a necessary input. Set those two inputs to zero and three others to one. Both components vanish while majority is one. Majority is self-dual, so the AND case follows by exchanging zero and one and swapping AND with OR. This is a representational result for a small grammar, independent of the empirical study.

## Reproduction and verification

The independent ledger interpreter reconstructed all 2,887 trace-local programs, 25,600 fitness evaluations, 4,177 generation records, 256 fit/validation-selected and baseline prediction vectors, and 2,048 scores. The largest scalar difference was 2.22×10⁻¹⁶. A bare native run reproduced 8 complete trajectories and all 3,200 ordered evaluations on the preselected replay banks. A separate physical-swap audit constructed 1,680 children across 840 site pairs in 16 real populations, then scored every child on 16 corners and 256 graded atoms. Its largest probability discrepancy was 2.22×10⁻¹⁶. It also tested a multi-target event and positive opportunity controls. OKAPI Lab independently used actual native crossover and both offspring on two disjoint G15 cases, confirming both complete parent-pair matrices and a constructed productive case with probability 1/24. [Ledger](ledger-verified.json) · [Bare search replay](native-verified.json) · [Physical swaps](swap-verification.json) · [Independent OKAPI fixture and real populations](https://github.com/damiankucharski/OKAPI/blob/ae020a4/experiments/donor_opportunity/README.md)

From the FusionBench checkout, put [OKAPI 3eae0e5](https://github.com/damiankucharski/OKAPI/commit/3eae0e5480cba0471cc5fda71b0c90f35b6a73f2) on PYTHONPATH alongside FusionBench and limit numerical libraries to one thread per worker.

```bash
python -m fusionbench.eval.semantic_donors --out /tmp/donors --workers 2
python docs/ensemble-study/semantic-donors/chain_reference.py /tmp/donors
python docs/ensemble-study/semantic-donors/verify_ledger.py /tmp/donors
python docs/ensemble-study/semantic-donors/summarize.py /tmp/donors
python docs/ensemble-study/semantic-donors/verify_swaps.py --folder /tmp/donors
```

Beadle and Johnson used Boolean decision diagrams to control semantic equality of genetic-programming offspring and their parents. Our earlier exact-key experiment removed whole-function duplicates. This study instead measures the full offspring distribution of unmodified crossover in fixed populations and checks whether usable ingredients were present. [Beadle and Johnson, 2008](https://doi.org/10.1109/CEC.2008.4630784) · [Earlier functional-key study](../search-regret/functional-key/README.md) · [Literature index](../literature/README.md) · [Dated source review](../literature/2026-11-21.md)
