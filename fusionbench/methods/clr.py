"""Linear stacking in centered log-probability coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusionbench.methods.base import Method


@dataclass
class AffineSoftmax:
    weights: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    n_classes: int

    def predict(self, features):
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self.weights.shape[1]:
            raise ValueError('expected one row per example with the fitted feature dimension')
        probability = softmax(features @ self.weights.T + self.intercept, axis=-1)
        result = np.zeros((len(probability), self.n_classes))
        result[:, self.classes] = probability
        return result


class StackingCLR(Method):
    """A multinomial ridge stacker using centered log probabilities per expert.

    Multiplicities give a standardized feature block the ridge geometry of
    exact duplicates without physically duplicating those features.
    """
    name = 'stacking_clr'

    def __init__(self, c_value=1., multiplicities=None, tol=1e-8):
        self.c_value = c_value
        self.multiplicities = multiplicities
        self.tol = tol

    @staticmethod
    def _features(preds):
        p = np.asarray(preds, dtype=float)
        if p.ndim != 3 or not np.isfinite(p).all() or (p<0).any() or (p.sum(axis=2)<=0).any():
            raise ValueError('expected finite nonnegative [row, expert, class] probabilities')
        lp = np.log(np.maximum(p, np.finfo(float).tiny))
        return (lp - lp.mean(axis=2, keepdims=True)).reshape(len(p), -1)

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds,fit_y,task=task,n_classes=n_classes)
        x = self._features(fit_preds)
        self.n_models = fit_preds.shape[1]
        if fit_preds.shape[2] != n_classes:
            raise ValueError('expert probabilities have an incompatible class axis')
        count = np.ones(self.n_models) if self.multiplicities is None else np.asarray(self.multiplicities)
        if count.shape != (self.n_models,) or not np.isfinite(count).all() or (count<=0).any():
            raise ValueError('positive multiplicity required per expert')
        scaler = StandardScaler().fit(x)
        scale = np.repeat(np.sqrt(count),n_classes) / scaler.scale_
        target = np.asarray(fit_y).ravel()
        if not np.issubdtype(target.dtype,np.integer) or target.min()<0 or target.max()>=n_classes:
            raise ValueError('expected integer class indices')
        logistic = LogisticRegression(C=self.c_value, max_iter=4000, tol=self.tol)
        logistic.fit((x-scaler.mean_) * scale, target)
        weights = logistic.coef_ * scale
        bias = logistic.intercept_ - weights @ scaler.mean_
        if len(logistic.classes_) == 2:
            # sklearn's binary sigmoid equals the second class of softmax(0,z).
            weights = np.vstack((np.zeros_like(weights), weights))
            bias = np.r_[0., bias]
        self.head = AffineSoftmax(weights, bias, logistic.classes_, n_classes)
        return self

    def predict(self, preds):
        if np.shape(preds)[1:] != (self.n_models,self.n_classes):
            raise ValueError('expert and class dimensions must match the fitted stacker')
        return self.head.predict(self._features(preds))

    def compile(self, expert_weights, expert_biases):
        """Compose affine expert logits and this head into one raw-input softmax.

        Expert arrays must align on the original feature and class axes and
        describe logits before softmax. Centering cancels each log normalizer.
        Exactness refers to unfloored probabilities of these specified logits.
        """
        w, b = np.asarray(expert_weights), np.asarray(expert_biases)
        if (w.ndim!=3 or w.shape[:2]!=(self.n_models,self.n_classes) or b.shape!=w.shape[:2]
                or not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError('expected finite expert logits [expert,class,feature] and aligned biases')
        centered_w = (w-w.mean(axis=1, keepdims=True)).reshape(-1,w.shape[-1])
        centered_b = (b-b.mean(axis=1, keepdims=True)).ravel()
        return AffineSoftmax(self.head.weights @ centered_w,
                             self.head.intercept + self.head.weights @ centered_b,
                             self.head.classes.copy(), self.head.n_classes)
