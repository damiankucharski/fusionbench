# A useful fitted operand and a neutral gate both change the search

On 16 independent FAR-switch data banks, the final fit-selected program had conditional-excess Brier **0.00267** when an admitted gate used the fitted second provider, versus **0.00555** when that provider was replaced by a second copy of the gate's anchor. The paired difference was **−0.00288** (95% interval −0.00557 to −0.00019). Both programs had the same shape, direction, fitted cutoff, breeding slot, five other parents and starting random-generator state. The actual training fitness differed because the duplicate is an identity operation at insertion. The comparison includes all 32 starts, including 16 in which no gate was admitted. [Frozen protocol](../matched-graft-protocol.md) · [Paired estimates](paired.csv)

A forced identity insertion also helped on FAR. With the fitted cutoff carried as inert data, its excess was 0.00555 versus 0.00889 after the same reordering and with all original parents present. The difference was −0.00334 (−0.00629 to −0.00040). Reordering the original parents compared with an untouched archive-only continuation gave +0.00045 (−0.00051 to +0.00142). These three measured differences sum to the full fitted-minus-archive difference of −0.00577. They describe one specified sequence of interventions. Subsequent ordinary mutation/crossover can expose previously inert material, and changed fitness can alter survival and future random draws. This experiment does not attribute the effects to any one operation after admission.

### Held-out results

The entries are mean squared error to the known conditional probability. Fit and independent validation determine separate archived champions. The table uses the fit champion and the corresponding stationary test law. Each bank supplied 512 fit, 256 validation and 8192 fresh test examples with six anonymous provider columns. Means weight independent banks equally and average two restarts within each bank. All methods were fitted without the hidden pair or generating parameters.

| Frozen choice | FAR | CLOSE | Logistic |
|---|---:|---:|---:|
| Full fitted pair and cutoff | .00267 | .01754 | .01221 |
| Fitted pair, cutoff 0.5 | .00594 | .02291 | .01310 |
| Duplicate anchor, fitted cutoff | .00555 | .02051 | .01295 |
| Duplicate anchor, cutoff 0.5 | .00633 | .02231 | .01290 |
| Original parents, matched reordering | .00889 | .02559 | .01357 |
| Untouched parents, same archive | .00844 | .02568 | .01334 |
| Ordinary GP at matched forward cost | .01099 | .02864 | .01304 |
| Ordinary GP at matched threshold-candidate cost | .01041 | .02204 | .01194 |
| Mean of all six providers | .09096 | .04377 | .05831 |
| Ranked-prefix Top-N | .04023 | .04170 | .01125 |
| Regularized logistic stacking | .02056 | .03399 | **.00280** |
| Exhaustive fitted single-gate reference | **.00181** | **.00201** | .02021 |

On CLOSE, the operand contrast was −0.00298 (−0.00683 to +0.00087). The fitted cutoff helped with the fitted operand retained, −0.00537 (−0.01063 to −0.00012). The FAR counterpart was −0.00327 (−0.00694 to +0.00040). With two identical operands at insertion, retaining the learned cutoff over 0.5 gave −0.00078 (−0.00296 to +0.00139) on FAR and −0.00179 (−0.00565 to +0.00206) on CLOSE. The positive-looking comparisons for the new nearest-provider law therefore depend on which intervention is evaluated. The designated operand contrast remains unresolved there. The exhaustive three-node fit reference also shows a larger remaining search gap on CLOSE.

On the logistic law the operand contrast was −0.00075 (−0.00189 to +0.00040), and regularized logistic stacking remained substantially better than any evolved arm. The full fitted procedure's **observed** stationary Brier was .13633/.22201/.18005 on FAR/CLOSE/logistic, versus .15398/.23846/.17082 for logistic stacking. A shift in the label law was consequential: FAR-trained fitted programs' excess rose from .00267 on FAR to .08131 on CLOSE and .02054 under the logistic law on the identical test features. Switching to independent validation selection yielded .00373/.01746/.01265 for the full treatment under the stationary laws. [Every selection and shift](means.csv)

### What entered the population

The fully fitted gate was naturally admitted in 16/32 FAR, 27/32 CLOSE, and 15/32 logistic starts. Had each same-shape alternative faced natural selection independently, the duplicate with fitted cutoff would have entered in **1, 2 and 3** starts respectively; the meaningful fitted pair at 0.5 would have entered in **5, 13 and 6**. Forcing the counterfactual into the same slot answers a different question from proposing the weaker program and letting selection decide. Among admitted cases the corresponding operand differences alone were −0.00536 (11 FAR banks), −0.00336 (16 CLOSE banks) and −0.00081 (12 logistic banks). These conditional estimates average bank means within the admitted subset.

All eight originally gate-free FAR champions admitted the full proposal. On those starts full, duplicated and ordinary candidate-cost-matched GP yielded .00331, .01249 and .03480 excess respectively. Among the 24 already containing a gate, those values were .00246, .00324 and .00228. On CLOSE the full and duplicated options gave .01503 and .01666 for the 12 initially gate-free cases, versus .01904 and .02283 for the other 20. These smaller, outcome-defined comparisons are descriptive.

For the fitted treatment, in 10/32 FAR, 19/32 CLOSE, and 13/32 logistic starts, the *first archived copy* of the final fit-winning program actually carried a descendant of the offered gate. Allowing a later marked copy of the same winning structure changed these counts to 16, 27 and 14. An independently rediscovered identical structure carries no lineage until a marked copy is observed. We froze a second prospective study to measure the functional effect of bypassing genuinely inherited occurrences on fresh examples. [Functional-ablation protocol](../gate-necessity-protocol.md)

### Costs, checks and reproduction

The shared generation boundary averaged 1205.00, 1207.03 and 1206.09 native fitness calls for FAR, CLOSE and logistic fits. Each arm saw the same candidate archive. Each fitted graft evaluated 10 possible gate-direction/partner combinations with 41 native fit forwards. Scoring the three actual alternate programs cost another three forwards on improved proposals, giving means of 1.97, 2.81 and 1.88 extra. The event sweep evaluated 2223.88, 2226.63 and 2225.88 scalar candidate thresholds on average. Each experimental arm continued for exactly 300 objective calls. The two plain search controls reached each start's actual total forward count or its forward-plus-threshold count. Top-N used all six ranked prefixes: it selected two in every FAR and logistic bank and two to five under CLOSE. Full fitted trees averaged 5.47, 8.78 and 12.28 nodes. [Raw programs, scores, parent order, RNG and lineage](runs.csv.gz) · [Native engine and origin markers](https://github.com/damiankucharski/OKAPI/blob/4c784f6/experiments/gate_lineage/README.md)

An independent interpreter reproduced all 12,720 prediction vectors bit for bit and all 25,440 losses exactly. An independent native batched replay used 535,173 objective evaluations to reproduce all 576 continuations, 960 checkpoint fingerprints and 1920 winner programs with their fit/validation scores. Running the whole study with the origin marker disabled reproduced all 12,720 rows across all 51 timing- and marker-independent columns. [Prediction and parent audit](verified.json) · [Independent trajectory audit](trajectories-verified.json) · [Marker-invariance audit](markers-verified.json)

Beadle and Johnson's Boolean crossover work compared offspring with parents by exact represented function. It motivated including identity programs as a control. Their task did not use probabilistic gates or a slot-matched population intervention. Kommenda et al.'s local constant fitting on symbolic-regression trees motivated holding the operand fixed while changing only the cutoff; a numerical cutoff at duplicated operands is a separate control introduced here. [Beadle and Johnson, 2008](https://doi.org/10.1109/CEC.2008.4630784) · [Kommenda et al., 2020](https://doi.org/10.1007/s10710-019-09371-3) · [Literature index](../../literature/README.md) · [Dated study notes](../../literature/2026-11-21.md)

Use OKAPI 4c784f6a217675d1305d378eaeb525738c37761d and the FusionBench checkout containing this report.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-4c784f6"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_matched_parent_gp.py tests/test_native_switch.py
python -m fusionbench.eval.matched_graft --out /tmp/matched-graft --workers 2
python -m fusionbench.eval.matched_graft --out /tmp/matched-unmarked --workers 2 --without-lineage
python docs/ensemble-study/search-regret/matched-graft/verify_markers.py /tmp/matched-graft /tmp/matched-unmarked
python docs/ensemble-study/search-regret/matched-graft/verify.py /tmp/matched-graft
python docs/ensemble-study/search-regret/matched-graft/verify_trajectories.py /tmp/matched-graft --workers 2
```
