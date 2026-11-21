# Native operator-type mutation on all 19 default FusionBench scenarios

At the matched budget of 1200 training fitness evaluations, enabling native operator-type mutation reduced equally weighted mean IID Brier from **0.26839 to 0.25947**. The point estimate improved on **15 of 19** scenarios. Five individual 95% intervals were entirely favorable and none was entirely unfavorable. These are 19 related synthetic scenarios with eight independent data draws each. The intervals are unadjusted; in a supplementary Holm correction across the 19 primary contrasts, only DeadWeight remained below 0.05 (adjusted p=0.0324). Regularized logistic stacking had the lowest overall mean, **0.22874**, and the best point estimate on 11 scenarios. [Frozen configuration](protocol.md) · [All paired intervals](paired.csv) · [Mean and spread of each method](spread.csv)

## The full regression panel

Entries use the public full-class Brier convention (sum across class channels before averaging examples). The model is selected only by fit Brier. Each independent bank supplies the same 384 fit, 256 validation and 2048 IID rows to every method; two ordinary GP restarts per bank are averaged before comparing methods. The columns labeled Δ give enabled minus disabled, so negative values favor the new mutation. Shift entries use frozen models on the nine predefined alternative distributions. A dash denotes no registered shift for this scenario in this battery.

| Scenario | Providers/classes | Mean | Top-N | Stack | Point off | Point on | Δ IID [95% CI] | Δ shift [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated_suppressor | 2/2 | 0.28812 | 0.22794 | 0.17068 | 0.22808 | 0.22808 | −0.000003 [−0.00033, +0.00032] | -0.00008 [-0.00047, +0.00031] |
| committee | 10/2 | 0.23733 | 0.24194 | 0.03098 | 0.22427 | 0.18575 | -0.03851 [-0.06211, -0.01491] | — |
| compositional_switch | 6/2 | 0.40430 | 0.35599 | 0.33389 | 0.34724 | 0.34013 | -0.00712 [-0.01710, +0.00287] | -0.00229 [-0.00593, +0.00134] |
| compound | 9/2 | 0.30668 | 0.24910 | 0.14720 | 0.18960 | 0.19175 | +0.00216 [-0.00319, +0.00751] | — |
| correlated | 10/2 | 0.28971 | 0.28292 | 0.17392 | 0.25104 | 0.22614 | -0.02489 [-0.03936, -0.01043] | -0.02912 [-0.04924, -0.00901] |
| deadweight | 10/2 | 0.32733 | 0.16020 | 0.06165 | 0.15134 | 0.11556 | -0.03578 [-0.05297, -0.01860] | — |
| gated_experts | 3/2 | 0.38746 | 0.32587 | 0.35571 | 0.31269 | 0.31600 | +0.00330 [-0.00428, +0.01088] | — |
| hardrescue | 6/2 | 0.42904 | 0.43267 | 0.10551 | 0.10406 | 0.10610 | +0.00204 [-0.00379, +0.00786] | — |
| hardveto | 6/2 | 0.30692 | 0.31295 | 0.09529 | 0.14950 | 0.14907 | -0.00042 [-0.01748, +0.01664] | — |
| miscalibration | 6/2 | 0.18937 | 0.19091 | 0.05318 | 0.17947 | 0.16993 | -0.00954 [-0.01805, -0.00102] | — |
| multiclass_committee | 8/5 | 0.70662 | 0.71040 | 0.72189 | 0.71003 | 0.70691 | -0.00312 [-0.00864, +0.00241] | — |
| native_close_switch | 6/2 | 0.49660 | 0.49348 | 0.47629 | 0.47132 | 0.46003 | -0.01129 [-0.03172, +0.00914] | +0.00795 [-0.00644, +0.02233] |
| native_switch | 6/2 | 0.44849 | 0.34554 | 0.30679 | 0.31504 | 0.30638 | -0.00866 [-0.02774, +0.01043] | +0.00029 [-0.01301, +0.01359] |
| needle | 100/2 | 0.48682 | 0.19647 | 0.21817 | 0.21423 | 0.22662 | +0.01239 [-0.01901, +0.04379] | — |
| parity | 4/2 | 0.66571 | 0.66571 | 0.50704 | 0.53261 | 0.50121 | -0.03139 [-0.06893, +0.00614] | -0.02002 [-0.05170, +0.01166] |
| polarity_fault | 7/2 | 0.22193 | 0.15247 | 0.10804 | 0.14938 | 0.14664 | -0.00274 [-0.00645, +0.00097] | -0.00252 [-0.00882, +0.00379] |
| saturated | 6/4 | 0.20332 | 0.20537 | 0.14994 | 0.18483 | 0.17730 | -0.00753 [-0.01176, -0.00329] | — |
| two_pathways | 12/2 | 0.30735 | 0.31117 | 0.25144 | 0.31086 | 0.30257 | -0.00829 [-0.02410, +0.00751] | -0.00483 [-0.01221, +0.00255] |
| weakest_link | 8/2 | 0.48236 | 0.48833 | 0.07847 | 0.07380 | 0.07369 | -0.00012 [-0.00306, +0.00282] | -0.00083 [-0.00430, +0.00264] |

With 600 calls the equally weighted IID difference was −0.00617; using independent validation selection at 1200 calls gave −0.00743 and favored enabled in 13 of 19. The five primary intervals favoring the enabled setting were on Committee, CorrelatedRedundancy, DeadWeight, Miscalibration and four-class Saturated. The 100-provider Needle had the largest adverse point estimate (+0.01239, interval −0.01901 to +0.04379); its three-provider Top-N fit was stronger than either GP setting. The five-class MulticlassCommittee difference was −0.00312 (−0.00864 to +0.00241). The simple all-provider mean scored 0.70662 on that case against 0.70691 with the mutation. Across the 304 paired IID restarts, enabled had the lower Brier in 179, the same value in 22 and a higher value in 103.

Across the nine predefined shifts, seven point estimates favored enabling the operator change and the equally weighted difference was −0.00572. Only the reduced-correlation version of CorrelatedRedundancy had its individual interval wholly below zero. Replacing a trained CLOSE law by the logistic law reversed the sign of its point estimate, from −0.01129 IID to +0.00795 under the shift; both intervals included zero. The study optimized probabilistic error. Mean macro-F1 was 0.79897 disabled and 0.79824 enabled, showing why the Brier result should not be treated as a classification-threshold improvement. [All score, selection and test choices](means.csv)

## What the comparison changes

The native switch adds arity-preserving replacement of one existing operator to the ordinary mutation repertoire. When eligible, this changes both the opportunity to replace a type and the relative frequency of competing native moves. The earlier organic study held the site fixed and randomized type versus provider identity. This panel exercises the deployable flag and the canonical default generator parameters; its comparison cannot decompose these two effects. A previous positive measurement or null result on a single target provides no substitute for the per-family table. [Native option and implementation](https://github.com/damiankucharski/OKAPI/commit/194e94a6c15994a29cdb509bd66c3e320a955c90) · [Mechanism study](../search-regret/organic-composition/README.md) · [Source precedent in neural-network fusion, Langdon et al., §5.5](https://doi.org/10.1007/978-1-4471-0123-9_51) · [Literature entry](../literature/2026-11-21.md) · [Source index](../literature/README.md)

The registry at the research revision contained 19 scenarios and the corresponding main-branch registry had 14. All canonical classes were instantiated with their default parameters, including 17 binary cases and the two multiclass cases with four and five classes. Each provider retained its full class vector for multiclass search, with simplex normalization inside each native operator. Binary trees evolved the positive channel and reconstructed its complement for full-vector Brier. Canonical constructor parameters apply to each generator; the finite evaluation budget and universal Brier objective are the prespecified experiment settings rather than the public command's ordinary 150-generation/macro-F1 budget. Arithmetic mean, one Brier ranking followed by every Top-N prefix, and regularized logistic stacking used the same labeled inputs as the two OKAPI arms. Rankings and scores of every prefix are archived for all 152 fits. [Raw programs, scores and Top-N paths](runs.csv.gz)

At 1200 calls the enabled and disabled champions averaged 17.81 and 15.78 nodes. Their archives contained 697 and 680 distinct evaluated programs on average; the budget fell in generation 10.97 and 11.14 on average. Search times were about 1.11 and 0.99 s per run on this CPU at the checkpoint. Time includes validation scoring of new archive members and reporting and does not define the comparison. We used the stock initialization, shallow random seed trees, parsimony and tournament rules from the existing FusionBench adapter. Equal training calls can imply different counts of generations and distinct structures.

The alternating-task preparation exposed a previously reproducible postprocessing error when an old model was reused after another task type had fitted. The adapter and native scope now restore the task's rule. Earlier published runners had scored before another task's fit or stayed in one task; the pinned historical audit identified no published score requiring correction. [Witness, scope and evidence](postprocessing-impact.md)

## Independent checks

A separate native batched replay reproduced all 608 trajectories and all 729,600 actual fitness evaluations. All 1216 checkpoint fingerprints, 2432 fit/validation winning programs, 8448 native prediction fingerprints and 42,240 scores agreed. Separately, an independent ordered-program interpreter, baseline implementation and paired-statistics script recovered all 10,032 published prediction vectors bit for bit, all 50,160 scores and all 1584 paired comparisons. [Independent native trajectories](trajectories-verified.json) · [Predictions, baselines and paired results](predictions-verified.json)

## Reproduction

Run from the research checkout with OKAPI db7b2b15c7064ad34f638e5e93a8cdb01e5271b8 and the regular FusionBench dependencies.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-db7b2b1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_suite_point_gp.py tests/test_okapi_postprocessing.py
python -m fusionbench.eval.full_suite_point --out /tmp/full-suite-point --workers 2
python docs/ensemble-study/full-suite-point/verify_predictions.py /tmp/full-suite-point
python docs/ensemble-study/full-suite-point/verify_trajectories.py /tmp/suite-native.jsonl --workers 2
python docs/ensemble-study/full-suite-point/verify_trajectories.py /tmp/suite-native.jsonl --compare /tmp/full-suite-point/runs.csv.gz
```
