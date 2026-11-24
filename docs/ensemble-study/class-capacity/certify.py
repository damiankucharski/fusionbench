"""Replay a fitted sklearn probability stacker as one native OKAPI tree."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.globals import use_postprocessing_function
from okapi.multiclass_affine import MulticlassAffineNode
from okapi.node import ValueNode
from okapi.tree import Tree

from fusionbench.methods.stacking import StackingLR
from run import REAL, SYNTHETIC, load_bank


def verify(ds):
    reference = StackingLR(c_value=1)
    reference.fit(ds.fit_preds, ds.fit_y, task="multiclass", n_classes=ds.n_classes)
    scaler, lr = reference.model.steps[0][1], reference.model.steps[1][1]
    np.testing.assert_array_equal(lr.classes_, np.arange(ds.n_classes))
    # Absorb the feature standardization into the input slopes and intercept.
    slopes = lr.coef_.T / scaler.scale_[:, None]
    bias = lr.intercept_ - scaler.mean_ @ slopes
    weights = slopes.reshape(ds.n_models, ds.n_classes, ds.n_classes)
    error = {}
    for split in ("fit", "test"):
        x = getattr(ds, split + "_preds")
        roots = [
            ValueNode(None, x[:, j].astype(np.float64), name)
            for j, name in enumerate(ds.model_ids)
        ]
        node = MulticlassAffineNode(roots[1:], weights, bias, mode="full")
        roots[0].add_child(node)
        with use_postprocessing_function(np.asarray):
            output = Tree(roots[0]).predict()
        expected = reference.predict(x)
        error[split + "_max_error"] = np.max(np.abs(output - expected))
        np.testing.assert_allclose(output, expected, atol=2e-6, rtol=0)
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for cohort in [*REAL, *SYNTHETIC]:
        for replicate in range(8 if cohort in REAL else 6):
            ds, seed = load_bank(cohort, replicate)
            rows.append(
                {"cohort": cohort, "replicate": replicate, "seed": seed, **verify(ds)}
            )
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
