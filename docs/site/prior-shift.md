# Change class prevalence between fit and test

Use a paired prior-shift bank to measure how a fusion method trained at one class frequency performs at another. The protocol changes the probability of drawing each class. For a given class, every provider retains the same emission law, including its signal strength and correlations with other providers. The result is a standard FusionBench `Dataset`.

## Compare deployment priors with one frozen fit

This example compares a balanced binary deployment to one with 20% positives. It fits each fusion method once per fold and evaluates those same fitted parameters in both deployments. Brier is summed over the two classes and averaged over test rows.

```python
import numpy as np
from fusionbench.eval.prior_shift import generate_prior_shift
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLogit

for fold in range(4):
    settings = dict(
        fit_prior=[0.5, 0.5], n_fit=512, n_test=2048,
        seed=17, fold=fold, k=4, accuracy=0.7,
    )
    source = generate_prior_shift("committee", test_prior=[0.5, 0.5], **settings)
    target = generate_prior_shift("committee", test_prior=[0.8, 0.2], **settings)
    for method in (SimpleAverage(), TopN("brier"), StackingLogit()):
        method.fit(source.fit_preds, source.fit_y, task=source.task, n_classes=2)
        losses = []
        for bank in (source, target):
            p = method.predict(bank.test_preds)
            labels = np.eye(2)[bank.test_y]
            losses.append(np.square(p - labels).sum(axis=1).mean())
        print(fold, method.name, losses[0], losses[1], losses[1] - losses[0])
```

A fold is an independently generated fit/test bank. Within a fold, keeping the seed, scenario parameters and split sizes fixed reuses the same row-level uniforms for selecting labels and the same Gaussian noise for all providers. Rows assigned the same class have exactly the same provider predictions under both test priors. A changed class uses the corresponding conditional emission with the paired noise draw. Changing the test prior leaves the fit arrays identical. Separate random streams also keep the test arrays fixed when you change only the fit size or fit prior. Average paired differences across folds to summarize variability across newly drawn training and evaluation data. [Evaluate methods](evaluate.md)

## Supported generators and references

| Scenario ID | Conditional predictions | Prior-aware posterior |
|---|---|---|
| `committee` | Independent, equally strong binary Gaussian signals | Combine the observed likelihood ratios with the split prior odds. At either clipping endpoint use the Gaussian tail probabilities for the two classes. |
| `multiclass_committee` | Class-dependent logit boost and independent Gaussian noise | Add each class's log prior to the strength-weighted sum of its provider log probabilities, then apply softmax. |
| `saturated` | Multiclass logit boost with shared provider noise | Use the same formula with the per-provider coefficient divided by `1 + (k - 1) * rho`, where rho is the shared-noise fraction. |

For a five-class committee set, for example, `fit_prior=[.2, .2, .2, .2, .2]` and `test_prior=[.6, .1, .1, .1, .1]`. The prior vector follows the generator's class-index order. Entries can be zero and the vector must sum to one. A zero-probability class has zero posterior in that split. Realized frequencies fluctuate around the requested prior; inspect `bank.meta["fit_class_counts"]` and `bank.meta["test_class_counts"]` before training a method that requires observations from several classes.

The clean `fit_oracle` and `test_oracle` condition on the observed provider probabilities and the known generating prior **for that split**. They give evaluation references for a method that knows the prior. The frozen methods in the example see only the fit predictions and labels. Predictions retain the original generator's calibration convention as prevalence changes. [Scenario definitions](scenarios.md) · [Dataset fields and storage](generate.md)
