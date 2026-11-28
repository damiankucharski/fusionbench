"""Frozen standard-v3 bank and reference settings for the replacement panel."""

import hashlib

import numpy as np

from fusionbench import get_scenario
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.blocks import draw_block
from fusionbench.scenarios.registry import STANDARD_V3_SCENARIOS

SCENARIOS = tuple(STANDARD_V3_SCENARIOS)
REPEATS = 6
FIT_ROWS = 640
TEST_ROWS = 1024
DATA_ROOT = 5_230_000
SEARCH_ROOT = 7_230_000


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_bank(scenario, replicate):
    if len(SCENARIOS) != 24 or replicate not in range(REPEATS):
        raise ValueError("Expected an index into the frozen 24 by 6 design")
    offset = 1000 * SCENARIOS.index(scenario) + replicate
    law = get_scenario(scenario)
    seed = DATA_ROOT + offset
    x, y, _ = draw_block(law, seed, 0, FIT_ROWS, split="fit")
    test, labels, _ = draw_block(law, seed, 1, TEST_ROWS, split="test")
    return law, seed, SEARCH_ROOT + offset, x, y, test, labels


def controls(x, y, task, n_classes):
    methods = {
        "mean": SimpleAverage(),
        "top_n": TopN("brier"),
        "prob_lr": StackingLR(c_value=1),
        "logit_lr": StackingLogit(c_value=1),
    }
    for model in methods.values():
        model.fit(x, y, task=task, n_classes=n_classes)
    return methods


def metrics(p, y):
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Expected a matrix of finite class probabilities")
    return {
        "brier": float(np.mean(np.square(p - np.eye(p.shape[1])[y]).sum(1))),
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "predictions_sha256": fingerprint(p),
    }
