"""Run the frozen label-only transfer study. See docs/ensemble-study/real-transfer."""
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.okapi import OKAPIMethod

SEED = 6211
EPOCHS = 50
HAR_SHA = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"


def populations(har_archive):
    X, y = load_breast_cancer(return_X_y=True)
    all_rows = np.arange(len(y))
    def stratify(rows, size):
        return train_test_split(rows, test_size=size, stratify=y[rows], random_state=SEED)
    dev, test = stratify(all_rows, 0.2)
    base, remainder = stratify(dev, 0.375)
    fit, selection = stratify(remainder, 0.5)
    yield "wisconsin", X, y, all_rows, (base, fit, selection, test)

    compressed = Path(har_archive).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != HAR_SHA:
        raise ValueError("Expected the UCI HAR archive identified in the study protocol")
    with zipfile.ZipFile(io.BytesIO(compressed)) as outer:
        zipped = outer.read("UCI HAR Dataset.zip")
    arrays = []
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        for split in ("train", "test"):
            def load(item, dtype):
                with archive.open(f"UCI HAR Dataset/{split}/{item}_{split}.txt") as file:
                    return np.loadtxt(file, dtype=dtype)
            arrays.append((load("X", np.float32), load("y", int) - 1, load("subject", int)))
    X, y, groups = [np.concatenate([train, test]) for train, test in zip(*arrays)]
    train_ids = np.unique(arrays[0][2])
    test_ids = np.unique(arrays[1][2])
    ids = np.random.default_rng(SEED).permutation(train_ids)
    assert len(ids) == 21 and len(test_ids) == 9 and not np.intersect1d(ids, test_ids).size
    indices = tuple(np.flatnonzero(np.isin(groups, which))
                    for which in (ids[:13], ids[13:17], ids[17:], test_ids))
    yield "har", X, y, groups, indices


def train_mlp(X, y, train, source_rows, width, seed):
    """Draw source_rows examples from the model's eligible training set per epoch."""
    start = time.perf_counter()
    scaler = StandardScaler().fit(X[train])
    inputs = torch.from_numpy(scaler.transform(X).astype(np.float32))
    target = torch.from_numpy(y.astype(np.int64))
    classes = int(y[train].max()) + 1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(),
                              torch.nn.Linear(width, classes))
    optimizer = torch.optim.Adam(net.parameters(), lr=.002)
    for _ in range(EPOCHS):
        order = rng.permutation(train)[:source_rows]
        for offset in range(0, source_rows, 128):
            rows = order[offset:offset + 128]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(inputs[rows]), target[rows])
            loss.backward()
            optimizer.step()
    seconds = time.perf_counter() - start
    with torch.no_grad():
        predictions = net(inputs).softmax(1).numpy()
    return predictions, dict(parameters=sum(p.numel() for p in net.parameters()),
                             linear_macs_per_example=width * (X.shape[1] + classes),
                             updates=EPOCHS * ((source_rows + 127) // 128),
                             row_exposures=EPOCHS * source_rows, train_seconds=seconds)


def group_means(values, groups):
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return np.bincount(inverse, weights=values, minlength=len(unique)) / counts


def losses(prob, truth):
    return np.square(prob - np.eye(prob.shape[1])[truth]).sum(axis=1)


def experiment(name, X, y, groups, splits):
    base, fit, select, test = splits
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(len(y)))
    assert len(np.unique(y[base])) == len(np.unique(y))
    costs, models = [], {}
    for model, width, train, seed in [
            *((f"member{s}", 24, base, s) for s in (11, 29, 47)),
            ("single72_pool", 72, np.r_[base, fit], 83),
            ("single72_base", 72, base, 83),
            ("single24_pool", 24, np.r_[base, fit], 83)]:
        models[model], cost = train_mlp(X, y, train, len(base), width, seed)
        costs.append(dict(dataset=name, model=model, **cost))
    members = [f"member{s}" for s in (11, 29, 47)]
    preds = np.stack([models[m] for m in members], axis=1)
    models["mean"] = SimpleAverage().predict(preds)
    start = time.perf_counter()
    gp = OKAPIMethod(seed=61, generations=20, n_restarts=1, n_seeds=10, max_nodes=11,
                     fitness="brier")
    classes = preds.shape[-1]
    gp.fit(preds[fit], y[fit], n_classes=classes,
           task="binary" if classes == 2 else "multiclass")
    gp_seconds = time.perf_counter() - start
    models["okapi"] = gp.predict(preds)
    score_select = {key: group_means(losses(p[select], y[select]), groups[select]).mean()
                    for key, p in models.items()}
    ref = min(members, key=score_select.get)
    fusion = min(("mean", "okapi"), key=score_select.get)
    student = "single72_pool"

    scores, test_risks = [], {}
    for key, p in models.items():
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-5)
        test_risks[key] = group_means(losses(p[test], y[test]), groups[test])
        acc = group_means(p[test].argmax(1) == y[test], groups[test]).mean()
        scores.append(dict(dataset=name, model=key, selection_brier=score_select[key],
                           test_brier=test_risks[key].mean(), test_accuracy=acc))
    rng = np.random.default_rng(8833)
    units = len(test_risks[ref])
    boot = rng.integers(units, size=(20000, units))
    contrasts = []
    for before, after, threshold in [(ref, fusion, 0), (ref, student, 0),
                                     (fusion, student, -.01)]:
        delta = test_risks[before] - test_risks[after]
        lo = np.quantile(delta[boot].mean(axis=1), .05 / 6)
        contrasts.append(dict(dataset=name, before=before, after=after, delta=delta.mean(),
                              lower_bound=lo, threshold=threshold, passes=bool(lo > threshold)))

    ref_ok, ens_ok, student_ok = [models[key][test].argmax(1) == y[test]
                                  for key in (ref, fusion, student)]
    repairs = ~ref_ok & ens_ok
    joint = ref_ok & ens_ok
    counts = dict(ensemble_repairs=int(repairs.sum()), retained=int((repairs & student_ok).sum()),
                  ensemble_harms=int((ref_ok & ~ens_ok).sum()), both_correct=int(joint.sum()),
                  student_new_errors=int((joint & ~student_ok).sum()))
    split_info = {key: dict(n=len(ix), unit_count=int(len(np.unique(groups[ix]))),
                            sha256_indices=hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest())
                  for key, ix in zip(("base", "fusion_fit", "selection", "test"), splits)}
    details = dict(selected_member=ref, chosen_fusion=fusion, split=split_info,
                   repair_row_counts=counts, gp_nodes=gp.best_nodes, gp_fit_seconds=gp_seconds,
                   retrospective_lowest_member_test_brier=min(test_risks[m].mean() for m in members))
    return scores, costs, contrasts, details


def run(archive, out):
    start, cpu_start = time.perf_counter(), time.process_time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    scores, costs, contrasts, details = [], [], [], {}
    with threadpool_limits(limits=1):
        for name, X, y, groups, splits in populations(archive):
            s, c, d, details[name] = experiment(name, X, y, groups, splits)
            scores.extend(s)
            costs.extend(c)
            contrasts.extend(d)
    for file, rows in [("scores", scores), ("fit_costs", costs), ("contrasts", contrasts)]:
        pd.DataFrame(rows).to_csv(out / f"{file}.csv", index=False)
    details.update(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu_start,
                   python=platform.python_version(), sklearn=sklearn.__version__,
                   torch=torch.__version__, numpy=np.__version__, har_sha256=HAR_SHA)
    (out / "details.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--har-archive", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.har_archive, args.out)
