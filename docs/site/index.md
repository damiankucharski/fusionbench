# FusionBench

FusionBench compares methods for combining classification probabilities from several models. Each scenario specifies how the providers succeed or fail. You can change a mechanism, such as shared error or an input-dependent specialist, and give every fusion method the same predictions and labels.

## Compare two methods on the same examples

From a source checkout, install FusionBench with Python 3.11 or newer. The base package includes the generators, packaged reference data and ordinary statistical fusion methods.

```sh
python -m pip install .
```

Load one of the 24 ready-made datasets and compare an arithmetic mean with a trained logistic stacker. Both predict the same held-out examples. Lower test Brier means the probabilities were closer to the observed labels.

```python
from fusionbench import load_reference_dataset
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.stacking import StackingLogit

bank = load_reference_dataset("committee")
scores = evaluate_dataset(bank, [SimpleAverage(), StackingLogit()])
assert scores["error"].eq("").all()
print(scores[["method", "test_brier"]])
```

Now try the same comparison with five providers, a specified provider accuracy and a larger, freshly drawn dataset. This command fits both methods, evaluates them and writes their results.

```sh
fusionbench run --scenarios committee --param k=5 --param accuracy=0.8 \
  --methods simple_average stacking_logit --n 512 --seed 7 --out results/committee
```

[Generate your own data](generate.md) · [Add your own fusion method](evaluate.md)

## Choose an experiment

| Question | Where to start |
|---|---|
| Can I get a small dataset immediately, or generate larger fit and test sets with the same mechanism? | [Generate and save data](generate.md) |
| What exactly changes in the data, and what information is available to an ideal predictor? | [Scenario catalog](scenarios.md) |
| Does my method improve on averaging or a regularized stacker on the same test examples? | [Evaluate fusion methods](evaluate.md) |
| When does a validation winner lose its advantage under noisy labels or a deployment shift? | [Measure validation-to-test transfer](transfer.md) |
| How can I contribute a new probability generator? | [Add a scenario](extend.md) |

Some scenarios generate labels first and then simulate noisy model predictions. Others start with predictions and sample the label from a known conditional law. A clean-domain posterior, when available, is a reference for that scenario. Performance differences depend on the generator, training budget, held-out sample and metric. The OKAPI case study shows how these choices affect a comparison with learned trees. [OKAPI case study](okapi.md)

## Build this documentation

From the checkout, install the documentation extra and run the strict build. MkDocs writes the static site to `site/` with its theme, styles and search assets. To browse locally, run `python -m mkdocs serve` and open the address it prints.

```sh
python -m pip install ".[docs]"
python -m mkdocs build --strict
```
