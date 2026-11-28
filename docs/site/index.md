# FusionBench

Will a fusion method recognize that several providers repeat the same mistake? What if the only model that recognizes a rare event is unreliable on most rows? FusionBench generates labeled **prediction banks** for questions like these. You can evaluate competing methods, hold examples and seeds fixed across interventions, and examine where a method's advantage comes from.

Install FusionBench from a source checkout using Python 3.11 or newer.

```sh
python -m pip install .
```

Load a packaged bank with a region-dependent specialist. Each fit row contains every provider's probabilities for that example. The provider IDs give their order on the middle axis.

```python
from fusionbench import load_reference_dataset

bank = load_reference_dataset("rare_specialist")
print(bank.fit_preds.shape, bank.model_ids)
print(bank.meta["reference_sha256"])
```

The package includes a fixed 128-row fit split and 128-row test split for each of the 30 standard scenarios. The same generators also produce fresh, configurable draws. The core package runs the scenarios, ordinary fusion baselines and reports on CPU.

| Goal | Guide |
| --- | --- |
| Change how often a rare rule occurs, or reproduce a particular draw | [Generate and store datasets](datasets.md) |
| Find the mechanism behind a named dataset | [Scenario catalog](scenarios.md) |
| Compare a candidate with averaging, Top-N and logistic stacking | [Evaluate a fusion method](evaluation.md) |
| Focus an evaluation on shift, grouped sampling or representational capacity | [Group and score scenarios](scoring.md) |
| Add a failure mode with its own generator and parameters | [Add a scenario](extend.md) |
