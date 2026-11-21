# A local edit rarely closes the native switch search gap

Taking the best native GP tree after 1200 evaluations, changing one provider binding and then tuning one threshold gave switch-test excess Brier **0.023895**. Continuing the same GP trajectory at the matched candidate budget gave **0.024075**. The preregistered paired difference was **−0.000180** (95% interval −0.000558 to +0.000198). The result leaves the relative quality of the two procedures unresolved. The unchanged starting tree scored 0.024270, a logistic stack 0.021132 and a fitted single gate 0.001289. [Prospective protocol](../native-refinement-protocol.md) · [Preceding mechanism study](../native-switch/README.md)

We used twelve new anonymous provider panels, two native GP seeds per panel, 512 fit labels and 256 independent validation labels. The stationary switch chooses the farther of two signal operands from 27/64; the linear control is logistic in the same two providers' logits. Both generated Bernoulli labels. Test data have 8192 fresh examples per panel, with common feature draws for the full, half and abolished switches. Excess Brier is mean squared error to the generating probability. Average, fully ranked and fit-selected Top-N, logistic stacking and a global empirical one-gate optimizer use the same fit labels and inputs. Top-N selected two providers in every panel in both regimes.

| Fit and test on the same law | Switch Brier | Switch excess | Linear Brier | Linear excess |
|---|---:|---:|---:|---:|
| All-provider average | 0.225074 | 0.090751 | 0.226398 | 0.057887 |
| Ranked-prefix Top-N | 0.174990 | 0.040490 | 0.180239 | 0.011206 |
| Regularized logistic stacking | 0.156194 | 0.021132 | 0.172666 | 0.003262 |
| One gate, fit by exhaustive pair and threshold search | 0.136191 | 0.001289 | 0.188287 | 0.018760 |
| Native GP fit champion, 1200 calls | 0.159069 | 0.024270 | 0.181085 | 0.012009 |
| One provider reassignment | 0.159054 | 0.024254 | 0.181034 | 0.011994 |
| One threshold refit | 0.158679 | 0.023912 | 0.18091 | 0.01183 |
| Reassignment then threshold refit | 0.158664 | 0.023895 | 0.181048 | 0.011990 |
| Continue GP, matched full training forwards of both edits | 0.15900 | 0.02420 | 0.18101 | 0.01191 |
| Continue GP, also charge every threshold candidate | 0.15887 | 0.02408 | 0.18088 | 0.01179 |

The local procedure improved switch-test excess versus its starting tree by 0.000375 (95% interval 0.00004 to 0.00071). Its difference versus continuation with matched full training forwards was −0.00031 [−0.00070, +0.00009]. On the stationary linear control the combined update's difference versus candidate-matched continuation was +0.000197 [−0.000062, +0.000457]. Requiring a strict improvement on disjoint validation before deploying the candidate gave switch excess 0.024042 for the combined edit and 0.024121 for candidate-matched GP. Under switch disappearance, this fit-selected local-versus-continued excess difference was +0.000115 [−0.000191, +0.000420]. Intervals are paired t intervals over 12 data-level means after averaging restarts. [Per-split means](means.csv) · [Paired intervals](paired.csv)

## Where the edit could act

Only **11 of the 24** switch-trained champions contained any threshold node. In those 11, the starting trees had mean switch-test excess 0.01060 and the combined updates reached 0.00978. In the 13 without a threshold, both means were 0.03584. The subgroups were formed by search outcomes, so their difference cannot isolate the causal effect of adding a gate. Across all 24, threshold-only changed eight trees, operand-only changed three, and both changed nine. On the linear control six starting trees contained a threshold. These counts explain why a valid parameter solver can have little effect on the overall optimizer.

The ordered combined edit lowered training Brier from 0.157390 to 0.156909 on the switch. An independent set of validation labels would have accepted it in only 6/24 cases. A single gate searched across providers and thresholds still had far smaller stationary switch error on this new panel. Identifying a usable tree to tune remains a material constraint at this population size.

## How costs and predictions were checked

For the operand edit, native OKAPI scored the starting tree and every single-occurrence substitution. For the first threshold in native operator order, the wrapper ran the whole tree while forcing each operand in turn and supplied the resulting rowwise downstream losses to the event solver. It checked the proposed threshold in a normal unforced pass and kept the incumbent on a realized tie or worsening. Both modules preserve ordered operands, children and the sequentially accumulated parent signal. Of the 11 switch-trained threshold repairs, nine used the dyadic event sweep and two required direct scoring of registered binary64 candidates. General non-dyadic results are certified over the registered candidates. The separate one-gate fit has complete threshold coverage on the scenario's alphabet. [Native wrapper and integration tests](https://github.com/damiankucharski/OKAPI/blob/6daff8c/experiments/threshold_refinement/README.md) · [FusionBench solver](../../../../fusionbench/eval/gate_events.py)

On the switch panels the combined step averaged **27.42** extra full training forwards. It considered **170.38** threshold candidates on average, including runs without a threshold. Thus its two controls continued the original GP to averages of 1227.42 and 1397.79 objective calls. The second deliberately charges one entire evaluation for each scalar setting, including those scored by an incremental calculation. In this run the combined edits took 0.0104 seconds on average, alongside 0.0208 seconds of forward-matched or 0.1272 seconds of candidate-matched extra GP. These CPU measurements include archive monitoring on the GP path; the callback's time is excluded from both GP continuation clocks. Neither count is a universal compute equivalence. [Full records with programs, timing and per-tree budgets](runs.csv.gz)

Independent reconstruction matched all 10,800 Brier and conditional-excess values exactly, all 480 guarded choices and 288 matched continuation checkpoints. Exhaustive reassignment of 1180 single-provider candidates recovered all 48 operand winners. A separate native search reproduced all 48 initial training champions and their fit scores. A second full run with parameter updates reproduced all 5400 raw rows and all aggregate results exactly, excluding timings. [Independent checks](verified.json) · [Verification script](verify.py)

Reproduce from FusionBench with OKAPI commit 6daff8c41234fffecf604734b71caba2b25f0ac3 and the NumPy backend.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-6daff8c"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_gate_events.py tests/test_continued_gp.py tests/test_native_switch.py
python -m fusionbench.eval.native_refinement --out /tmp/native-refinement --workers 2
python docs/ensemble-study/search-regret/native-refinement/verify.py /tmp/native-refinement --native
```
