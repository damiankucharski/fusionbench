# Transfer benchmark

Given per-provider class probabilities, how does a fitted fusion rule perform when development annotations are noisy or the predictions available at deployment change? The unit of comparison is a simulated scenario, replicate and development noise level. Methods share the same tensors, labels and test conditions. The runner fits each method once per noisy development bank. A separate labeled source-validation sample selects among the frozen methods before any clean or stressed deployment score is computed.

## Run a published configuration

From a checkout, install the package with `python -m pip install .`. To install a built artifact use `python -m pip install /path/to/fusionbench-0.1.0-py3-none-any.whl`. Both commands below use packaged configuration resources and the base CPU dependencies.

```sh
fusionbench transfer-grid --preset smoke --out results/smoke
fusionbench transfer-grid --preset standard --out results/standard
```

The **smoke** preset uses native_switch (binary, six providers) and multiclass_committee, one replicate, 640 development rows, 192 independent source-validation rows and 2,048 test rows, annotation noise probabilities 0 and 0.15 and stress severity 0.5. The **standard** preset uses all 19 named scenario IDs in schema v1, eight replicates, the same row counts, noise 0/0.15/0.30, and severities 0.25/0.5/1. Both include arithmetic mean, ranked-prefix Top-N, probability-feature logistic stacking and logit-feature logistic stacking. Top-N ranks individual providers on the noisy fit labels, tests every unweighted prefix on those same rows, and chooses N by Brier.

```sh
fusionbench transfer --preset smoke --out results/smoke-alias
fusionbench transfer-grid --preset standard --scenarios native_switch multiclass_committee \
  --methods simple_average top_n stacking_lr --repeats 2 --modes log_jitter \
  --dev-noise 0 0.3 --severities 0.5 1 --fit-rows 640 --validation-rows 192 --test-rows 2048 --out results/subset
fusionbench transfer-grid --config results/smoke/resolved.yaml --out results/replay
```

Explicit `transfer --preset` or `transfer --config` selects this benchmark. The older `transfer --out DIR` invocation and its `--seeds`, `--rescue-rates` and `--followup` flags run the MLP mechanism study. Use `transfer-grid` for an unambiguous default smoke run. The `list` command displays method and scenario identifiers. Requesting an unavailable optional method produces an error. For OKAPI variants install the matching dependency using `python -m pip install '.[okapi]'` from a checkout. OKAPI comparisons automatically include the arithmetic mean, learned Top-N and at least one registered regularized stacker.

Suppress OKAPI's per-operation informational logging while retaining warnings:

```sh
LOGURU_LEVEL=WARNING fusionbench transfer-grid --preset smoke \
  --methods okapi_native_pattern stacking_logit --out results/native-smoke
```

## Define a panel in YAML

The schema version, stable slots and row counts below are explicit. A named scenario can override its registered constructor arguments using `knobs`. The canonical endpoint starts with those knobs and replaces the named fields. The runner checks that the endpoint retains provider IDs and class dimensions. Built-in endpoints are applied by default on the nine supported scenarios; set `canonical_shift: null` to omit one. A custom endpoint can supply new parameters for a scenario that has no built-in endpoint. Command-line switches override matching fields. A CLI scenario subset inherits its configuration and seed slots; a new registry entry needs an explicit YAML slot.

```yaml
schema_version: 1
base_seed: 1700000
fit_rows: 640
validation_rows: 192
test_rows: 2048
repeats: 2
dev_noise: [0, 0.15]
modes: [provider_swap, log_jitter, canonical_shift]
severities: [0.25, 1.0]
methods: [simple_average, top_n, stacking_logit]
scenarios:
  native_switch:
    seed_slot: 14
    knobs: {mixing: 0.2}
    canonical_shift: {mixing: 0.8}
  multiclass_committee: {seed_slot: 18}
```

Use an empty `modes: []` or CLI `--modes clean` for a clean-only check. Severity zero is always represented by the single clean condition. Scenario and method keys are checked against the registry before the run. Method-specific limitations still apply, for example a binary-only algorithm evaluated on multiclass inputs. A fit or prediction error identifies its method and scenario and aborts the run.

## What changes

| Condition | Mechanism |
|---|---|
| Development annotation noise q | Replace each label with a uniformly chosen different class with probability q. Each observation has a shared threshold and alternative class across q; the same noisy labels reach every fitted method and any internal selection. Provider probabilities are held fixed. |
| Clean | Draw fresh test examples independently from the source generator. Keep all original labels and probabilities. |
| Provider swap | Select one provider per bank and a row permutation without fixed points. On the fraction of rows selected at severity s, replace that provider's whole probability vector with its vector from the donor row. Keep the recipient labels. |
| Log jitter | Perturb every provider's centered log probabilities by independent Gaussian coordinate noise with standard deviation 1.5 × s, then apply softmax within each provider. The underlying random noise is paired across s. Keep the clean labels. |
| Canonical shift | Sample an independent test bank under the scenario's endpoint parameters. With probability s use the endpoint observation **and its label** at that position. The mask is nested across severities. Endpoints exist only for some scenarios. |

Every probability score is the mean, over test rows, of the sum of squared errors across classes. It ranges from 0 to 2. For two classes it is twice the positive-class mean squared error. Positive `delta_from_clean` is deterioration against that method's own clean score in the same bank and noise level. A source posterior conditional on clean predictions can change after an observation is swapped or jittered. This benchmark compares predictions with the realized test labels and reports no post-corruption oracle regret. A scenario's built-in posterior has its own stated input assumptions.

## Does source validation choose a method that transfers?

Each method receives only the development tensors and the development annotations for its noise level q. After fitting, its predictions are also scored on a fresh sample from the same source generator. That sample has `validation_rows` observations (default 192, minimum one), independent of both development and test. Validation annotations use the same flip probability q and an independent mask and wrong-class draw. Across noise levels the validation flips are nested on these same rows. Setting `--validation-rows` changes only this generic choice among methods. Existing version-1 YAML files that omit the field keep the 192-row default.

The optional native OKAPI methods can divide their **640 development rows** into 448 discovery and 192 internal selection rows to choose their own tree. The additional **192 source-validation rows** are shared by all submitted methods and are used to compare their frozen outputs. They are never passed to `fit`, and the chosen method is deployed without refitting. Submitted methods should implement deterministic prediction with frozen parameters.

For each bank and q, the method with the lowest computed validation Brier is selected once and its ID is retained across every test condition. Exactly equal validation scores are resolved by the first method in the resolved configuration; arbitrarily close but unequal validation scores still have an order. For a given deployment condition the hindsight reference is the lowest empirical test Brier from the same set of methods. Reported test excess is the selected method's test Brier minus that minimum. An excess at most 1e-12 counts as a hit, so several methods can qualify. The test-best ID itself also breaks exact ties by configuration order. This retrospective comparison depends on the submitted menu, finite holdout and corruption condition. Neither empirical winner establishes population optimality.

## Artifacts and aggregation

- `scores.csv` contains one row per scenario, data seed, development noise, method, mode and severity, including Brier and its paired change from clean.
- `fits.csv` contains one row per fitted method, with development-label Brier, independent source-validation Brier and row counts, fit time, trainer seed when exposed, and selected N where applicable. Both source losses use annotations at the configured q, from separate samples.
- `selection.csv` identifies the generic validation winner and its validation and test Brier in each deployment cell. It gives the hindsight test-best ID and loss, the selected method’s excess over that minimum, a tie-aware hit indicator and the number of methods compared. Join `fits.csv` with `scores.csv` on scenario, replicate, data seed, q (column `dev_noise`) and method to inspect all ranks.
- `summary.csv` averages replicates within each scenario, then weights the represented scenarios equally in each method/noise/mode/severity cell. It reports scenario and bank counts. Endpoint cells can have fewer scenarios than clean or jitter cells.
- `overall.csv` first averages the available condition means within each scenario, including the clean condition, then weights scenarios equally for each method and development noise. This defines a particular mixture of deployment conditions. Its value can change when the chosen stress grid changes.
- `resolved.yaml` is a runnable record of configuration decisions, including the actual method list and endpoint overrides. `manifest.json` records software versions, Python source fingerprint, available VCS package commit identifiers, random stream identifiers, per-bank seeds, provider IDs and SHA-256 fingerprints of development and validation tensors, both sets of annotations, and test tensors/labels. Timings are machine-dependent.

For each scenario the bank seed is `base_seed + 1000 * seed_slot + replicate`, with replicates starting at zero. Standard slots are fixed in the shipped YAML, so reordering and subsetting preserve the same data and stochastic method seed where exposed. Development, clean test and endpoint draws use streams 0, 1 and 2. Held-out source validation uses stream 3. Development and validation annotations use independent streams 10 and 14. Stress row thresholds, provider swapping and jitter use streams 11, 12 and 13. Distinct conditions within a bank share rows and randomness. For uncertainty calculations pair methods within the same banks; treating every corrupted copy or every method as a new independent sample understates uncertainty. Precision for comparisons across scenarios depends additionally on which scenarios the study aims to represent.

## Add a scenario or provider

Subclass `fusionbench.scenarios.base.Scenario`, define the task and class count, return ordered provider identifiers from `model_ids()`, and generate a probability tensor of shape `(N, providers, classes)` plus integer class labels and any oracle metadata from `_generate(n, rng)`. Use the supplied random generator. An analytic oracle is optional for transfer. Implement `bayes_oracle` when the law supplies one. To add a provider to an existing scenario, update its generation, identifiers and any oracle that depends on those inputs. IDs must keep their meaning across the endpoint. Register your class in process before calling the Python entry point, or add it to the package registry for CLI discovery.

```python
from fusionbench.scenarios.native_switch import NativeSwitch
from fusionbench.scenarios.registry import SCENARIOS
from fusionbench.transfer import load_preset, run_transfer

class PartialSwitch(NativeSwitch):
    def __init__(self, mixing=0.25, **kwargs):
        super().__init__(mixing=mixing, **kwargs)

SCENARIOS["partial_switch"] = PartialSwitch
config = load_preset("smoke")
config["scenarios"] = {"partial_switch": {"seed_slot": 100, "canonical_shift": {"mixing": 0.9}}}
run_transfer(config, "results/partial-switch")
```

An external method can implement the `Method.fit/predict` interface, then register a zero-argument factory in `fusionbench.methods.registry.FACTORIES` and include that key in `methods`. A fresh instance is constructed for each fitted bank. Python registrations persist for the process that created them.

## Interpretation and reporting

These are designed synthetic distributions with finite samples and particular corruption mechanisms. Results measure sensitivity under those settings. They cannot guarantee a ranking on a new real deployment, across untested noise processes, or even across finite IID samples. When reporting a result, provide the package/source revision, installed dependency versions, resolved YAML, raw per-bank scores, summary with coverage counts, data split and selection protocol, seed-level uncertainty, and compute environment/timings. Record the algorithm's own tuning budget and distinguish any exploratory selection from its frozen evaluation.

[Benchmark card and study protocol](ensemble-study/transfer-grid/benchmark-card.md)
