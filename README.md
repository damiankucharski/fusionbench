# FusionBench

FusionBench is a synthetic benchmark for **combining classification predictions from several models**. A dataset contains the probability vector from each provider for each example, plus a class label. Its generator lets you change a known mechanism, such as error correlation, the frequency of a specialist's cases, or the reliability of a shortcut. Different fusion methods can therefore be compared on exactly the same inputs.

The `standard-v6` suite contains 30 scenarios, each with a small reference dataset included in the package. Use a reference bank to check an implementation, generate larger samples or fresh seeds to evaluate a method, and register your own scenarios through the same Python API.

## Try it from a checkout

This checkout contains the 1.0.0a1 prerelease. With Python 3.11 or newer, install the package from this directory.

```sh
python -m pip install .
```

This comparison fits a simple mean, a Brier-ranked prefix average (Top-N) and probability-space logistic stacking. It evaluates them on the held-out split of a bank with shared provider errors.

```python
from fusionbench import evaluate_dataset, load_reference_dataset
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

bank = load_reference_dataset("correlated")
scores = evaluate_dataset(bank, [SimpleAverage(), TopN(metric="brier"), StackingLR()])
print(scores.loc[scores.method != "oracle", ["method", "test_brier"]])
```

Run the same comparison on all 30 fixed banks and summarize it by mechanism or scientific purpose.

```sh
python examples/compare_groups.py
```

| What you want to do | Start here |
| --- | --- |
| Inspect the 30 standard scenarios and choose one to modify | [Scenario catalog](docs/site/scenarios.md) |
| Load a reference, generate seeded splits or save your own bank | [Datasets](docs/site/datasets.md) |
| Compare your method with the three baselines or run a transfer experiment | [Evaluation](docs/site/evaluation.md) |
| Score the standard suite, a mechanism group or an application profile | [Grouping and scoring](docs/site/scoring.md) |
| Implement and register a new scenario | [Extending the benchmark](docs/site/extend.md) |

The [design notes](DESIGN.md) describe what a score means and where the benchmark's assumptions enter. To build the documentation locally, run `python -m pip install '.[docs]'` followed by `python -m mkdocs serve`.
