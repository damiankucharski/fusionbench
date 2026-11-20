"""Check between-search variation on one practical held-out multiclass split."""
import argparse
from pathlib import Path

import pandas as pd

from digits_predictions import write_digits_predictions
from fusionbench.eval.metrics import score
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.shrinkage import ShrunkOKAPI
from fusionbench.store import load_dataset


def run(out):
    out = Path(out)
    ds = load_dataset(write_digits_predictions(out / "data"))
    rows = []
    for seed in range(8):
        candidate = ShrunkOKAPI(split_seed=0, seed=seed, fitness="brier",
                                generations=20, n_restarts=1, n_seeds=10,
                                max_nodes=11, population_multiplier=2)
        candidate.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
        for method in (candidate, candidate.tree, SimpleAverage()):
            heldout = score(ds.test_y, method.predict(ds.test_preds), ds.n_classes)
            rows.append(dict(search_seed=seed, method=method.name,
                             tree_weight=candidate.tree_weight, **heldout))
    pd.DataFrame(rows).to_csv(out / "shrinkage.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
