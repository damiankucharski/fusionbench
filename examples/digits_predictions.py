"""Write a three-way split of built-in sklearn digits for fusionbench evaluate."""
import argparse

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from fusionbench.store import save_dataset
from fusionbench.types import Dataset


def write_digits_predictions(root, seed=431):
    raw, labels = load_digits(return_X_y=True)
    raw = raw / 16.0
    base, rest = train_test_split(np.arange(len(labels)), stratify=labels,
                                  train_size=0.5, random_state=seed)
    fit, test = train_test_split(rest, stratify=labels[rest], test_size=0.5,
                                 random_state=seed + 1)
    names = ["lr", "neighbours", "forest"]
    experts = [LogisticRegression(max_iter=2000), KNeighborsClassifier(n_neighbors=5),
               RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)]
    for model in experts:
        model.fit(raw[base], labels[base])

    def predict(indices):
        return np.stack([model.predict_proba(raw[indices]) for model in experts], axis=1)

    ds = Dataset(name="digits", task="multiclass", n_classes=10, model_ids=names,
                 fit_preds=predict(fit), fit_y=labels[fit], test_preds=predict(test),
                 test_y=labels[test], fit_oracle=None, test_oracle=None,
                 meta={"primary_metric": "macro_f1", "base_train_n": len(base), "seed": seed})
    return save_dataset(ds, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=431)
    args = parser.parse_args()
    print(write_digits_predictions(args.out, args.seed))
