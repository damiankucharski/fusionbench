"""Frozen production OKAPI reference for the binary checkpoint-reuse comparison."""

from sklearn.metrics import f1_score

from fusionbench.eval.metrics import brier_score
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI


def score_regularized(fit_x, fit_y, audit_x, audit_y, *, seed):
    """Fit on the checkpoint-selected fit bank, then score its deployed tree.

    Inputs have shape (rows, providers, 2). Use the same seed and fit-row order
    across checkpoint choices within each fit block to pair both GP randomness
    and the production discovery/validation split. On 192 rows the split has
    134 discovery and 58 validation rows. It samples individual rows, so rows
    from one scene may appear in both partitions. Audit rows are independent
    of these partitions and their labels are used only to compute metrics.
    """
    model = RegularizedNativePatternOKAPI(seed=seed, split_seed=seed)
    model.fit(fit_x, fit_y, task="binary", n_classes=2)
    result = {"family": model.chosen_}
    for split, inputs, labels in (("fit", fit_x, fit_y), ("audit", audit_x, audit_y)):
        probabilities = model.predict(inputs)
        result[f"{split}_selected"] = brier_score(labels, probabilities, n_classes=2)
        result[f"{split}_f1"] = float(
            f1_score(labels, probabilities[:, 1] > 0.5, pos_label=1, zero_division=0)
        )
    return result
