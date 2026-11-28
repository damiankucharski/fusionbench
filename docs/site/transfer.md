# Measure validation-to-test transfer

A method can win on source validation and lose on future cases. FusionBench separates finite validation uncertainty, development annotation noise, damage to predictions at deployment and changes in the underlying scenario. In each data bank the methods see identical provider predictions and labels. The main comparison asks what happens to the **method chosen on source validation** when its deployment conditions change.

## Run a grid

```sh
fusionbench transfer-grid --preset smoke --out results/transfer-smoke
```

The smoke preset includes two scenarios and four lightweight methods. The standard preset (standard-v6) covers 30 scenarios. Presets `standard-v1` through `standard-v5` reproduce the earlier 19, 20, 24, 25 and 27 scenarios, respectively. All standard versions use eight replicates per scenario and retain the earlier seed slots. Both use 640 development rows, 192 independent source-validation rows and 2,048 test rows per bank. Larger runs add stability across seeds and mechanisms and consume more compute. For a targeted experiment, retain the scenario's fixed seed slot and choose the methods and perturbation levels explicitly.

```sh
fusionbench transfer-grid --preset standard --scenarios native_switch multiclass_committee \
  --methods simple_average top_n stacking_logit --repeats 2 \
  --dev-noise 0 0.30 --modes log_jitter --severities 0.25 1 \
  --out results/transfer-targeted
```

## Distinguish the three sources of labels

| Split | What happens there |
|---|---|
| Development | Each method is fitted once for the bank and development noise level. Methods may make their own internal split for tuning. At noise q, an annotation is replaced by a uniformly chosen different class with probability q. |
| Held-out source validation | Every frozen method predicts a third, independently generated source sample. Its annotations have the same noise probability q with an independent mask. Select the method with the lowest empirical validation Brier. Exact ties follow the order of methods in the resolved configuration. |
| Deployment test | Evaluate that same selected method, alongside all competitors, on a fresh test bank. Test labels retain their scenario meaning. A clean score and each stressed score use the same replicate. |

The ordinary IID suite's fit measurements and this independent source validation are distinct. Some algorithms additionally use an internal holdout within their 640 development rows; those rows do not double as the generic cross-method validation sample.

## Define what changes at deployment

| Condition | Perturbation |
|---|---|
| Clean | An independent draw from the original scenario. |
| Provider swap | One provider is fixed for the bank. On a fraction of rows selected with probability s, substitute its complete class vector from another row, with a derangement to avoid self-donors. Keep the original target. |
| Log jitter | Add Gaussian noise with standard deviation 1.5 × s to each centered log-probability coordinate, then renormalize each provider by softmax. Keep the original target. |
| Canonical shift | In supported scenarios an endpoint changes named generator parameters. With probability s, substitute the full row and its corresponding label from the endpoint bank. |

Masks and random disturbances are paired across severity levels. The clean posterior may cease to be the correct conditional posterior after these changes. Transfer uses Brier against the realized target. Each raw score ranges from 0 to 2 across any number of classes.

## Read and reproduce the result

`selection.csv` gives the validation winner, its test Brier, the retrospective best method among the submitted methods on that same test condition, and the difference between those two test losses. A hit includes an excess of at most 10⁻¹². A new candidate or a different menu can change this comparison. For individual rankings, join `fits.csv` and `scores.csv` on scenario, replicate, data seed, development noise and method.

`summary.csv` averages replicates within each scenario before weighting represented scenarios equally. Its coverage counts matter because canonical endpoints exist for only some scenarios. `overall.csv` first averages available deployment conditions within scenario, then weights scenarios equally. Conditions from one bank share rows. Paired differences within banks preserve that dependence.

The run's `resolved.yaml` records row counts, methods, constructor arguments and stable per-scenario seed slots. `manifest.json` includes dependency versions, source fingerprint, stream IDs and hashes of the tensors and labels. A bank seed is `base_seed + 1000 × seed_slot + replicate`. Replay a saved configuration or edit a copy to define your own panel.

```sh
fusionbench transfer-grid --config results/transfer-smoke/resolved.yaml --out results/transfer-replay
```

A full version-1 YAML file can be shared independently of the checkout. The following panel fixes the provider mapping and varies the scenario's `mixing` parameter at the canonical-shift endpoint. Save it as `panel.yaml` and run `fusionbench transfer-grid --config panel.yaml --out results/custom`.

```yaml
schema_version: 1
base_seed: 1700000
repeats: 2
fit_rows: 640
validation_rows: 192
test_rows: 2048
dev_noise: [0, 0.15]
modes: [provider_swap, canonical_shift]
severities: [0.5, 1.0]
methods: [simple_average, top_n, stacking_lr]
scenarios:
  native_switch:
    seed_slot: 14
    knobs: {mixing: 0.2}
    canonical_shift: {mixing: 0.8}
```

The literature on evaluation bias explains why reusing the winning selection score can be misleading. These measurements describe specified generators and stress levels; a claim about an actual deployment also requires evidence about its providers, labels and shift mechanisms. [Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) · [Full configuration schema and examples](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/transfer-benchmark.md)
