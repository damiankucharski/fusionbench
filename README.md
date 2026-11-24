# FusionBench

FusionBench is a Python library of controlled classification datasets for studying how to combine model predictions. Each dataset contains aligned class probabilities from several providers, labels, and independent fit and test examples. The 19 built-in scenarios cover correlated errors, weak and redundant models, calibration, noisy logical conditions and input-dependent expertise. Each scenario specifies how its predictions and labels arise. Researchers can vary its parameters, supply a fusion method or contribute a different generator.

## Load a dataset or generate an experiment

Install the research checkout with `python -m pip install .` (Python 3.11+). The base installation includes small reference datasets for all 19 built-ins. Generating data and running the standard baselines work on CPU.

```python
import fusionbench as fb

print(fb.list_scenarios())
reference = fb.load_reference_dataset("correlated")

# Change how strongly eight providers share their errors, keeping their marginal
# accuracy, the other two providers, the sample size and the seed fixed.
for corr in (0.1, 0.85):
    data = fb.generate_dataset("correlated", corr=corr, n_per_split=2048, seed=17)
    assert data.fit_preds.shape == (2048, 10, 2)
    print(corr, data.model_ids, data.fit_y.shape)
```

The bundled datasets have 128 examples per split. Use larger generated splits and independent seeds for an experiment. The Python API also supports saving and loading local datasets in the same per-model NumPy format. [Getting started](docs/site/generate.md) · [Scenario catalog and parameter defaults](docs/site/scenarios.md) · [Implement a scenario](docs/site/extend.md)

## Compare fusion methods

```sh
fusionbench list
fusionbench generate --scenarios correlated --param corr=0.1 --seed 17 --n 2048 --out data
fusionbench run --scenarios correlated --param corr=0.1 --seed 17 --n 2048 \
  --methods simple_average top_n stacking_lr stacking_logit --out results/correlated
```

Available comparisons include simple averaging, ranked-prefix Top-N, probability and logit stacking, tree operators, and optional learned fusion methods. Where the generating law supplies the conditional class probabilities, a clean-domain oracle can help distinguish limits of the data from limits of a method. For binary banks with up to 12 providers, the optional `lattice_brier_exact` comparator computes the lowest attainable empirical Brier for unrestricted MIN/MAX trees. [Evaluate a method](docs/site/evaluate.md)

## Use the benchmark to diagnose a method

A fixed scenario lets you ask whether a method can represent the target, whether its optimizer finds a suitable solution, and whether the chosen solution generalizes. The OKAPI case study uses known target trees, exact comparisons and independent evaluation to distinguish those questions. The transfer suite is another available protocol. Run `fusionbench transfer-grid --preset smoke --out results/transfer-smoke` to compare four baselines on paired disturbances. [OKAPI case study](docs/site/okapi.md) · [Transfer configurations](docs/transfer-benchmark.md) · [Earlier design and experiments](DESIGN.md)

## Evaluate predictions from your own models

Export probabilities on two disjoint sets. Each base model should have been trained on other examples, or its fit-set probabilities should be out-of-fold predictions. Keep all arrays in identical sample order within each split and identical class order across models. The CLI fits fusion methods on the fit split and scores them on the test split.

```text
my_data/
  meta.json
  gt/fit.npy       gt/test.npy
  fit/model1.npy   fit/model2.npy
  test/model1.npy  test/model2.npy
```

Each prediction file has shape `[N, C]`, finite probabilities and rows summing to one. Each ground truth file contains `N` integer labels from 0 to `C-1`. Example metadata follows.

```json
{"name":"my_data","task":"binary","n_classes":2,"model_ids":["model1","model2"],"knobs":{"primary_metric":"log_loss"}}
```

```sh
fusionbench evaluate --dataset my_data --methods simple_average stacking_lr --out results/my_data
```

The report ranks methods on the held-out primary metric. It includes regret when `oracle/test.npy` contains an analytic posterior for the test set. The default primary metric is macro-F1. For a reproducible multiclass example using the bundled scikit-learn digits data, run the following from the repository root.

```sh
python examples/digits_predictions.py --out data
fusionbench evaluate --dataset data/digits --methods best_single simple_average top_n stacking_lr okapi --out results/digits
```

The example needs the OKAPI extra for its last method. The 898 base-training, 449 fusion-fit and 450 test images come from a single stratified split.

## Research branch

Crossed-seed experiments measure when a reliable validation ranking breaks under a controlled distribution shift. Single-model experiments test which ensemble mechanisms can be recovered from raw inputs and labels. [Results, controls, data and reproduction commands](docs/ensemble-study/README.md)


### Ranked-prefix Top-N averaging

`top_n` scores each provider on the fusion-fit labels, orders the providers once, then scores each equal-weight prefix from one provider through the full pool. It chooses N with the best fit score and freezes both ranking and N for prediction. Tied individual scores preserve input order. Tied prefix scores choose smaller N. The default selection metric is the dataset's primary metric when using the benchmark runner, and macro-F1 for standalone fitting. Use `TopN(metric="brier")` to specify a metric. `average_precision` uses the positive class for binary tasks and macro one-vs-rest for multiclass tasks. As for all benchmark methods, Brier sums squared errors across classes before averaging examples.

`ranking_` holds zero-based provider indices, `prefix_scores_[n-1]` is the fit score of the first n providers, and `n_selected_` is the winning count. The `results.csv` row records `fit_metric` and `n_selected`. The fixed-size `top_5` comparator is available separately.

The `run`, `evaluate` and `sweep` commands automatically include arithmetic averaging, learned Top-N, and a regularized logistic stacker when explicitly requesting `okapi` or `okapi_shrunk`. The default stacker uses raw probabilities (`stacking_lr`). Explicitly requesting `stacking_logit` or `stacking_clr` selects that regularized variant. The method name in results identifies the feature transform. Fit all learned references on the same development examples and preserve their selection for IID and shifted tests.
