"""Fusion controls scored on the fit and independent audit partitions."""

from sklearn.metrics import f1_score

from fusionbench.eval.metrics import brier_score
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR


def score_controls(fit_x, fit_y, audit_x, audit_y):
    """Fit once on binary provider probabilities (N, M, 2) and score both splits.

    The selected losses are class-summed Brier. F1 is for class one with the
    strict rule p(class one) > 0.5. Only Top-N records a selected prefix.
    """
    models = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1),
    }
    results = {}
    for name, model in models.items():
        model.fit(fit_x, fit_y, task="binary", n_classes=2)
        row = {}
        for split, x, y in (("fit", fit_x, fit_y), ("audit", audit_x, audit_y)):
            p = model.predict(x)
            row[f"{split}_selected"] = brier_score(y, p, n_classes=2)
            row[f"{split}_f1"] = float(f1_score(y, p[:, 1] > 0.5, zero_division=0))
        if name == "top_n":
            row["n_selected"] = model.n_selected_
        results[name] = row
    return results
