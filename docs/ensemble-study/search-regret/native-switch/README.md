# Identifying a switch in anonymous provider probabilities

On a known FAR switch, admitting two threshold operators in native OKAPI improved mean test **excess Brier from 0.03674 to 0.03031** at 1200 fitness calls. The preregistered paired difference was −0.00643 (95% interval −0.01143 to −0.00143). A logistic stack scored 0.02089, and a single gate fitted by searching all providers, both directions and the threshold achieved **0.00191**. The exact search used the same noisy training labels as the other learners. On a separate logistic control the order reversed: stacking scored **0.00259** and the fitted gate **0.01864**. [Prospective protocol](../native-switch-protocol.md)

## What the switch does

Native OKAPI's FAR operation chooses whichever operand is farther from a threshold in probability space. CLOSE chooses the nearer one. The first operand wins ties. A third input acting as a selector in FusionBench's earlier GatedExperts scenario is a different mechanism. The new `native_switch` scenario hides two roles among six independent, anonymous scalar providers. Its conditional label probability is either the farther of A and B from 27/64, or sigmoid(0.2 + 0.7 logit(A) + 0.7 logit(B)). A weakened switch averages these two conditional probabilities. The score distribution remains fixed. Both laws govern Bernoulli labels, so training examples do not disclose the underlying probability. [Generator](../../../../fusionbench/scenarios/native_switch.py) · [Native operator](https://github.com/damiankucharski/OKAPI/blob/86f21cd/okapi/node.py#L857)

We drew twelve datasets, each with 512 fit, 256 independent validation and 8192 independent test cases. Validation and test use the same anonymous provider order as training. Three GP seeds share each data draw. Every learner was frozen before seeing the test outcomes. The table uses the training-selected GP champion and positive-class Brier. Excess is the mean squared difference between the fitted probability and the known conditional probability, which removes test-label noise from comparisons. The conditional law is used only for diagnosis.

| Fit and test on the same law | Switch Brier | Switch excess | Linear Brier | Linear excess |
|---|---:|---:|---:|---:|
| Known conditional probability | 0.13502 | 0 | 0.16885 | 0 |
| All-provider average | 0.22539 | 0.09117 | 0.22697 | 0.05827 |
| Ranked-prefix Top-N | 0.17485 | 0.04033 | 0.18005 | 0.01120 |
| Regularized logistic stack | 0.15575 | 0.02089 | 0.17158 | 0.00259 |
| Fit one gate, threshold fixed at 0.5 | 0.17550 | 0.04002 | 0.18768 | 0.01828 |
| Fit one gate including threshold | 0.13709 | 0.00191 | 0.18736 | 0.01864 |
| Native GP, four operators | 0.17129 | 0.03674 | 0.18052 | 0.01166 |
| Native GP, six operators | 0.16475 | 0.03031 | 0.18093 | 0.01214 |

The native four-operator palette contains MEAN, MIN, MAX and WEIGHTED_MEAN. The six-operator palette adds FAR and CLOSE, changing the probability of drawing other operators as well. At 300 calls the paired difference in switch excess was −0.00030 [−0.00703, +0.00642]. At 1200 calls only 12/36 six-operator training champions used a threshold node; 35/36 used only the correct two provider identities. None exactly recovered the teacher on the complete signal grid. The mean GP fit Brier exceeded the exact one-gate fit minimum by 0.02865. This is a signed comparison to an achievable one-gate program; a bigger tree can do better on these fit labels. Choosing champions from the archive by disjoint validation produced switch excesses of 0.03119 (six operators) and 0.03742 (four). Top-N tested every prefix and selected two providers on every dataset in either regime. [All raw runs, including programs, fit gaps and Top-N prefix scores](runs.csv.gz) · [All means](means.csv) · [Paired comparisons](paired.csv)

## Why fitting the threshold mattered

For the free-threshold gate, all twelve switch-training datasets selected FAR and the correct two provider identities. Six recovered the exact teacher over all 3969 input pairs, and their mean agreement across all twelve was 99.22%. Against the fitted gate with threshold fixed at 0.5, fitting the scalar threshold reduced switch excess by 0.03810 [0.03654, 0.03966]. The linear control also selected FAR and the same provider identities on all twelve datasets, even though its generator has no switch. Merely finding a named operator and the right inputs would misidentify that mechanism. The loss comparison against a frozen alternative and the known law distinguish the cases.

| Switch-trained procedure, held fixed | Full switch excess | Half switch excess | Switch abolished excess |
|---|---:|---:|---:|
| All-provider average | 0.09117 | 0.06947 | 0.05827 |
| Top-N | 0.04033 | 0.02052 | 0.01120 |
| Logistic stack | 0.02089 | 0.00698 | 0.00357 |
| Fitted one-gate threshold | 0.00191 | 0.00598 | 0.02055 |
| Native GP with six operators, 1200 calls | 0.03031 | 0.02039 | 0.02097 |

The gate's excess relative to the switch-trained stack changed from −0.01898 [−0.02083, −0.01713] with full switching to +0.01698 [+0.01569, +0.01827] after switching vanished. Test domains share feature rows and the uniforms used to draw labels, allowing paired contrasts. Intervals average restarts within each independent dataset before a two-sided t interval. The role of the single-gate family is a property of this synthetic generator. This experiment estimates no prevalence of such switches in real ensembles.

## Exact training reference and independent checks

The six scores take values from 1/64 to 63/64 in steps of 1/64. For each ordered pair, every threshold comparison can change only at its midpoint, a multiple of 1/128. A threshold grid at increments 1/256 represents the entire continuous-parameter family on this support, including exact ties. The solver accumulates integer loss changes at midpoint events and compares both directions, with no threshold picked using validation or test labels. It returns the exact numerator, denominator, number of attaining grid entries and first minimizer under the declared stable ordering. This proves optimality solely within the one-operator, two-distinct-input family. [Solver](../../../../fusionbench/eval/exact_switch.py)

A separate brute-force implementation checked all 48 fit certificates, including tie counts and selected programs. A NumPy interpreter reconstructed archived native programs and checked 6240 Brier and conditional-excess values. A complete rerun reproduced all selected programs and non-timing metrics within 10⁻¹⁵. Tests also verified genuine OKAPI outputs on all 3969 ordered signal pairs and the representable floating-point values immediately to either side of every possible threshold breakpoint. [Independent verification](verify.py) · [Tests](../../../../tests/test_native_switch.py)

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-86f21cd"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_native_switch.py
python -m fusionbench.eval.native_switch --out /tmp/native-switch --workers 2
python docs/ensemble-study/search-regret/native-switch/verify.py /tmp/native-switch
```
