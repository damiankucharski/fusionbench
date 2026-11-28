"""Replay the earlier frozen 640-row method on the replacement panel's exact banks."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from inputs import fingerprint, load_bank, metrics
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent


def run(bank):
    law, seed, search, x, y, test, labels = load_bank(bank["scenario"], bank["rep"])
    assert (seed, search) == (bank["data_seed"], bank["search_seed"])
    for key, array in (
        ("source_inputs", x),
        ("source_labels", y),
        ("test_inputs", test),
        ("test_labels", labels),
    ):
        assert fingerprint(array) == bank[key + "_sha256"]
    with threadpool_limits(limits=1):
        native = RegularizedNativePatternOKAPI(seed=search, split_seed=seed)
        native.fit(x, y, task=law.task, n_classes=law.n_classes)
        lr = StackingLR(c_value=1).fit(x, y, task=law.task, n_classes=law.n_classes)
        score = metrics(native.predict(test), labels)
        lr_score = metrics(lr.predict(test), labels)
        row = {
            key: bank[key] for key in ("scenario", "rep", "data_seed", "search_seed")
        }
        row.update(
            chosen=native.chosen_,
            native=score["brier"],
            prob_lr=lr_score["brier"],
            lr_hash=lr_score["predictions_sha256"],
        )
        # Each component was fitted/selected without the test. These diagnostics
        # are never used to change the earlier method's validation-selected tree.
        for name, tree in native.trees_.items():
            row[name] = metrics(predict_tree(tree, test), labels)["brier"]
        return row


def main():
    results = HERE / "results"
    banks = json.loads((results / "banks.json").read_text())
    with ProcessPoolExecutor(max_workers=3) as pool:
        frame = pd.DataFrame(pool.map(run, banks))
    archived = pd.read_csv(results / "scores.csv")
    keys = ["scenario", "rep", "data_seed", "search_seed"]
    lr = archived.loc[archived.method == "prob_lr_640"].set_index(keys)
    frame = frame.set_index(keys)
    assert len(frame) == len(lr) == 144 and frame.index.is_unique
    assert (frame.lr_hash == lr.predictions_sha256.reindex(frame.index)).all()
    assert ((frame.prob_lr - lr.brier.reindex(frame.index)).abs() < 1e-12).all()
    for name in ("ordinary_full", "open_kl", "mean", "top_n_640"):
        frame[name] = archived[archived.method == name].set_index(keys).brier
    assert not frame.isna().any().any()
    frame.to_csv(results / "matched-variants.csv")
    print(frame.drop(columns=["chosen", "lr_hash"]).mean())
    print(frame.chosen.value_counts())


if __name__ == "__main__":
    main()
