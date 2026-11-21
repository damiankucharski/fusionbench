"""Exact Brier comparator for a small pool of binary positive-class predictors."""

from importlib.util import find_spec

import numpy as np

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier
from fusionbench.methods.base import Method


class ExactLatticeBrier(Method):
    name = "lattice_brier_exact"

    @classmethod
    def available(cls):
        return find_spec("networkx") is not None

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        if task != "binary" or n_classes != 2:
            raise ValueError(
                "lattice_brier_exact requires binary positive-class scores"
            )
        self.rule = fit_lattice_brier(fit_preds[:, :, 1], np.asarray(fit_y).ravel())
        return self

    def predict(self, preds):
        p = self.rule.predict(preds[:, :, 1])
        return np.column_stack((1 - p, p))
