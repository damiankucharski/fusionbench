# Changing an operator family and changing one input during evolution

At 1800 real training evaluations on a known compositional law, replacing the family of one existing operator per eligible generation gave mean excess Brier **0.05957**. Changing one direct provider ID gave **0.06259**. Their preregistered paired difference was **−0.00302** (95% interval −0.00923 to +0.00320). Neither the two-way interaction (−0.00276, −0.01331 to +0.00778) nor the comparisons with ordinary GP resolved an advantage. The fitted stacker scored 0.05587 even though an attainable seven-node MIN/MAX tree had zero excess. [Frozen protocol](../organic-composition-protocol.md) · [All paired estimates](paired.csv)

The outcome is compatible with useful individual edits. In one of 24 provider-only starts, ordinary survival and training selection recovered exactly the seven-node rule, also independently selected on validation. The other arms had no exact stationary test recovery at 1800 calls. This single event establishes reachability in this run; it does not establish a population advantage for changing provider IDs.

## Stationary performance and shifts

The two paths are hidden among six columns. Inputs are an equal mixture of 64 equiprobable corners and a grid of interior probabilities. The two fitted laws are max(min(A,B),min(C,D)) and a separate linear-in-log-odds rule. Each law uses the same predictions, randomized hidden roles and paired underlying label uniforms per bank. Scores below are conditional-excess Brier of the **fit-selected** frozen model. Lower is better. Twelve independent data banks contribute two GP starts each.

| Method | Composition IID | Composition interior only | Logistic IID | Logistic interior only |
|---|---:|---:|---:|---:|
| Ordinary GP | .06228 | .02109 | .00815 | .01085 |
| Replace operator family | .05957 | .01980 | .00629 | .01029 |
| Replace direct provider ID | .06259 | **.01945** | .00650 | .01095 |
| Replace both | .05711 | .01980 | .00696 | .01123 |
| Mean of all six | .09305 | .02943 | .07060 | .05813 |
| Ranked-prefix Top-N | .06878 | .02008 | .00626 | .01124 |
| Regularized logistic stacking | **.05587** | .02509 | **.00254** | **.00120** |

The arithmetic, Top-N and stacking methods yielded observed IID Brier .20118/.17675/.16355 under composition and .21771/.15314/.14962 under the logistic law. Type-only and provider-only gave .16756/.17054 under composition and .15313/.15346 under the logistic law. Top-N ranked the six providers once per fit and tested every prefix; it selected four for composition and two for the logistic control in all twelve banks.

At 600 evaluations the composition-law excess was .07609/.07451/.07490/.07321 for stock/type/provider/both. At 100 complete generations it was .06500/.06485/.06585/.05976. The 1800-call type-minus-provider interior-only contrast was +0.00035 (−0.00216 to +0.00287). Using independent validation to select the archive winner also left their IID ordering unresolved, −0.00135 (−0.00842 to +0.00572). [Fit and validation selections under both label laws, IID and input shift](means.csv) · [Programs and per-bank scores](runs.csv.gz)

## Why the search had room to improve

The three-operation witness computes the generating conditional probability on every input. Every single update in the six-family repertoire has at least **961/16384 ≈ .05865** excess on this mixed population. On the binary-coded cube, any nearest/farthest threshold reduces to MIN, MAX, or its first operand. Exhaustive enumeration over nonempty subsets gives best MIN and MAX bit-unit errors of 3/16 and 5/16. Every normalized weighted mean, including repeated provider IDs, is a simplex weight vector. With total signal mass s, symmetry minimizes corner error at equal weights within signals and within nuisance inputs, giving 1/4+s²/16+(1−s)²/8−3s/16 in bit units. Its minimum for 0≤s≤1 is 1/8 at s=1. Multiplying by (62/64)² and the 1/2 corner mixture yields the stated lower bound. The other half can add further error. This bounds the six stated families with zero or one update. It places no positive bound on two-update programs. [Independent OKAPI certificate and native witness](https://github.com/damiankucharski/OKAPI/blob/c91dd94/experiments/compositional_control/README.md)

The combined arm's point estimate fell below the bound by 0.00154, but its 95% interval relative to the bound extended from −0.01004 to +0.00696. None of the four mean GP results certified population performance beyond the whole one-update family at this budget. This coexists with the exact individual recovery. Average champion sizes at the 1800-call composition checkpoint ranged from 12.25 to 13.04 nodes and 4.63 to 5.29 operators. Adding nodes alone did not remove the search gap.

A supplementary algebraic check bounds any monotonically calibrated affine stacker on the same distribution by at least 961/49152 ≈ .01955 excess. Fix the two nuisance bits. The two positive complementary patterns 1100/0011 and each of the negative complementary pairs 1010/0101 and 1001/0110 have identical sums of input coordinates. Their respective pairs of affine scores therefore have equal sums. Let u be the lower of the two positive predicted probabilities. At least one prediction from each distinct negative pair is at least u (the ordering is preserved by monotone calibration). These three states alone incur total squared error of at least (2/3)(high−low)². They have mass 1/16 each within the corner cube for each nuisance setting, and the cube has mass 1/2. This lower bound is distinct from the preregistered single-native-operation bound, and the observed stack error exceeded it. This is a constructed challenge with a lower bound specific to its distribution.

## What happened to the chosen offspring

Each generation supplied at most one chance to edit a uniformly drawn eligible offspring, then a uniformly drawn operation and a direct child. Type replacement kept the ordered operands, while changing an input preserved that child's complete update subtree. Native deduplication and selection followed. The control made the same label-blind proposed choices on a separate stream and submitted the original child.

At 1800 training calls, composition runs had on average 121.7/124.9/119.3/122.4 eligible opportunities for stock/type/provider/both. At that same checkpoint 73.1/122.6/116.9/121.5 of the chosen children had received their real fitness evaluation. Only 7.7/7.2/7.7/6.3 of the submitted **objects** had survived at least their first natural selection. In the ordinary control a structurally equal program survived 48.2 times, much more often than the selected object. The edits frequently avoided immediate duplicate filtering, while their survival and final accuracy changed much less. The run records distinguish equality of prediction programs, identity of the submitted object, and a selection still pending at a mid-generation cutoff. No subsequent lineage is imputed from structural equality. [Event ledger](events.csv.gz) · [Checkpoint accounting, evaluations and timings](costs.csv.gz)

OKAPI's author had intentionally deferred operator-type mutation when finishing experiments for his dissertation. The optional extension here estimates its effect at matched cost. Langdon, Barrett and Buxton used same-arity point mutation in GP fusion of neural networks, alongside several different genetic moves. The literature precedent applies to preserving the argument subtrees. Our matched existing-node type/partner experiment and the certified input distribution are separate constructions. [Langdon et al., §5.5](https://doi.org/10.1007/978-1-4471-0123-9_51) · [Dated literature notes](../../literature/2026-11-21.md) · [Literature index and BibTeX](../../literature/README.md) · [Earlier singleton append pilot](../../operator-coverage/README.md) · [Pinned native type-replacement implementation](https://github.com/damiankucharski/OKAPI/commit/194e94a6c15994a29cdb509bd66c3e320a955c90)

## Reproduction

The independent evaluator recovered every one of the 7632 prediction vectors bit for bit, all 15,264 losses, and all 744 paired contrasts. It validated 25,532 recorded generation events, 576 checkpoints and 348 aggregate rows, including full enumeration of the corners. A separate native batched replay independently reproduced all 192 trajectories, 25,532 events, 576 checkpoints and 1152 selected programs. A second stock implementation without any edit hook reproduced all 48 stock trajectories, 144 checkpoint fingerprints and 288 winners. These checks used 432,000 native fitness calls. [Prediction and cost audit](verified.json) · [Independent native replays](trajectories-verified.json)

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-194e94a"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_compositional_switch.py tests/test_organic_gp.py
python -m fusionbench.eval.organic_composition --out /tmp/organic-composition --workers 2
python docs/ensemble-study/search-regret/organic-composition/verify.py /tmp/organic-composition
python docs/ensemble-study/search-regret/organic-composition/verify_trajectories.py /tmp/organic-composition --workers 2
```
