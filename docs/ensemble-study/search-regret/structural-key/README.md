# How often does fixing tree identity change the best ensemble?

The fix changed which programs OKAPI evaluated in **5 of 48** matched runs at 300 fitness calls. It changed **zero** training-selected or validation-selected trees. Both versions missed the exact training optimum in all 48 runs. The average fit Brier gap was **0.13401** (95% interval across independent dataset draws 0.12477 to 0.14324). At 1200 calls, seven paths had diverged and the winners still matched in every pair.

The experiment uses 12 freshly seeded TwoPathways datasets, four GP seeds per dataset, 128 fit labels and 128 independent validation labels. Each test domain has 8192 examples per dataset. There are eight continuous detectors. IID detector accuracy is 0.90 and the shifted test lowers accuracy to 0.78. MIN/MAX is the complete GP operator set in this panel. A minimum cut certifies the training optimum over unrestricted-size MIN/MAX expressions on the supplied scores. Each three-way baseline comparison uses exactly the same providers and fit labels and is frozen for both tests. [Frozen protocol](../structural-key-protocol.md) · [Exact solver](../continuous.md)

| Fit procedure | Fit Brier | Validation | IID | Shift |
|---|---:|---:|---:|---:|
| Exact MIN/MAX, least empirical minimizer | 0.05044 | 0.09681 | 0.09678 | 0.17760 |
| All-provider average | 0.15063 | 0.15737 | 0.15506 | 0.19314 |
| Top-N, training Brier ranking and best prefix | 0.14606 | 0.16062 | 0.16197 | 0.19742 |
| Regularized logistic stack | 0.10964 | 0.15021 | 0.14202 | 0.17969 |
| GP, 300 calls, best training Brier | 0.18445 | 0.22897 | 0.23349 | 0.24264 |
| GP, 300 calls, best validation Brier | 0.20271 | 0.20228 | 0.22388 | 0.23726 |
| GP, 1200 calls, best training Brier | 0.15339 | 0.19629 | 0.20204 | 0.22814 |
| GP, 1200 calls, best validation Brier | 0.17656 | 0.17300 | 0.19794 | 0.22563 |

The GP rows apply to both versions. The paired change in every displayed fit and test score was exactly zero. Averaging over the four restarts within each dataset before estimating two-sided Student-t intervals, quadrupling the budget improved the fit champion's IID Brier by 0.03145 [0.01661, 0.04629]. Even then its mean training regret was 0.10295. The exact training solution improved IID Brier relative to stacking by 0.04524 [0.04104, 0.04945]. The shifted improvement was 0.00209 [−0.00195, 0.00612]. This distribution shift leaves the relative test advantage unresolved. The fitted Top-N used 4 to 7 providers. Its complete ranking and all prefix losses are included in the raw baseline records.

## What the old key actually merged

In the legacy trajectories, nine structurally distinct candidates would be dropped across 28,218 population entries presented to the mask by the 300-call checkpoint. At 1200 calls, 26 were dropped across 94,154 entries. The latter events occurred in seven trajectories. Corresponding totals in the fixed trajectories were 9/28,218 and 25/94,099. The counts include repeat appearances in later generations and the full joined batch seen before the last objective call. No suppressible pair differed on its 128 training scores. Full path fingerprints still changed because retaining a second structure can change candidate order, available mutations and the point at which the evaluation budget ends.

After inspecting that result we checked every observed collision on the full Boolean input cube. **All 14 distinct ordered pairs across both policies agreed on all 256 binary inputs.** This establishes equality for every possible real input to these particular MIN/MAX programs. For any scalar threshold, thresholding a MIN/MAX expression gives its Boolean rule on the thresholded inputs. Identical Boolean truth tables therefore give identical threshold crossings and identical scalar predictions. A regression with two differently grouped trees sharing a printable representation gives different outputs on four of 16 Boolean inputs, including cases where both trees agree on every supplied training row. The broader OKAPI Lab panel found many prediction-changing collisions with its own operator repertoire. [Regression in the engine](https://github.com/damiankucharski/OKAPI/blob/86f21cd/test/test_structural_key.py) · [Broader OKAPI Lab collision study](https://github.com/damiankucharski/OKAPI/blob/21a8db3/experiments/structural_key/README.md)

The correction preserves distinct programs in the optimizer. In this panel, its measured effect on selected quality was zero and the large empirical search gap persisted. The conclusion applies to this problem, initializer, population size and pair of equal evaluation budgets. The measured CPU time includes audit instrumentation and validation archiving. Mean training-champion size increased from 6.35 to 11.12 nodes between budgets. Structural archives contained roughly 84 and 298 unique trees on average. [Per-run outcomes](gp.csv) · [Collision witnesses](collisions.csv) · [Exact minima and three baselines](baselines.csv)

## Independent native replay

The main audit ran both policies against the fixed branch with only the generation's deduplication mask switched. A separately written driver ran two unmodified and independently checked-out native engines, using the batched objective routine. All **192/192** ordered path digests, champion digests and training scores matched the instrumented benchmark. Repeating the full main panel with the subsequent Boolean-corner probe reproduced all non-timing columns. [Native replay records](replay.csv) · [Independent replay driver](native_replay.py) · [Comparison](compare.py)

To reproduce, install FusionBench dependencies including the lattice extra. Check out OKAPI in separate directories at the commits below and install its runtime dependencies. From the FusionBench checkout run the commands below, replacing the two absolute directory paths.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
out=/tmp/okapi-key-audit
fixed=/absolute/path/to/okapi-86f21cd
legacy=/absolute/path/to/okapi-ec1ba02
driver=docs/ensemble-study/search-regret/structural-key
PYTHONPATH=".:$fixed" python -m fusionbench.eval.structural_key_audit --out "$out" --workers 2
PYTHONPATH=".:$fixed" python "$driver/native_replay.py" --policy structural --out "$out/fixed.csv"
PYTHONPATH=".:$legacy" python "$driver/native_replay.py" --policy legacy --out "$out/legacy.csv"
head -n 1 "$out/fixed.csv" > "$out/replay.csv"
tail -n +2 "$out/fixed.csv" >> "$out/replay.csv"
tail -n +2 "$out/legacy.csv" >> "$out/replay.csv"
python "$driver/compare.py" "$out/replay.csv" "$out/gp.csv"
```
