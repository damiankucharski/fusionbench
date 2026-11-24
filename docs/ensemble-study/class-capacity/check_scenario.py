"""Check the synthetic interventions on every provider row of the experiment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run import SYNTHETIC, load_bank


def compare(reference, candidate, cohort, replicate, split):
    x = getattr(reference, split + "_preds").astype(float)
    z = getattr(candidate, split + "_preds").astype(float)
    y = getattr(reference, split + "_y")
    np.testing.assert_array_equal(y, getattr(candidate, split + "_y"))
    np.testing.assert_array_equal(np.sort(x, axis=-1), np.sort(z, axis=-1))
    index = np.arange(len(y))
    np.testing.assert_array_equal(x[index, :, y], z[index, :, y])
    affected = np.isin(y, [3, 8])
    np.testing.assert_array_equal(x[~affected], z[~affected])
    onehot = np.eye(x.shape[-1])[y][:, None, :]
    drift = np.max(
        np.abs(np.square(x - onehot).sum(axis=-1) - np.square(z - onehot).sum(axis=-1))
    )
    if drift > 1e-12:
        raise AssertionError(f"Unexpected change in individual Brier: {drift}")

    wrong = z[affected].copy()
    selected = np.arange(affected.sum())
    labels = y[affected]
    wrong[selected, :, labels] = -1
    ties = wrong == wrong.max(axis=-1, keepdims=True)
    votes = ties / ties.sum(axis=-1, keepdims=True)
    common = votes[selected, :, np.where(labels == 3, 8, 3)].mean()
    private = np.mean([votes[:, j, j].mean() for j in range(z.shape[1])])
    return dict(
        cohort=cohort,
        replicate=replicate,
        split=split,
        provider_rows=len(y) * x.shape[1],
        affected_examples=int(affected.sum()),
        max_provider_brier_drift=drift,
        common_wrong_vote=common,
        private_wrong_vote=private,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    records = []
    for r in range(6):
        reference, _ = load_bank("original", r)
        for cohort in SYNTHETIC:
            candidate, _ = load_bank(cohort, r)
            for split in ["fit", "test"]:
                records.append(compare(reference, candidate, cohort, r, split))
    pd.DataFrame(records).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
