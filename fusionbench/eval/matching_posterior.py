"""Exact finite posteriors and compact fixed-weight networks for hidden disjoint pairs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from operator import index

import numpy as np
from scipy.special import logsumexp, xlogy


def _bits(values, n, low=0.0, high=1.0):
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != n:
        raise ValueError(f"Expected a binary row matrix with {n} columns")
    if np.all((x == 0) | (x == 1)):
        return x.astype(np.uint8, copy=False)
    if np.all((x == low) | (x == high)):
        return (x == high).astype(np.uint8)
    raise ValueError(
        "Expected active bits or entries equal to the two declared endpoints"
    )


def _limits(low, high):
    low, high = float(low), float(high)
    if not 0 <= low < high <= 1:
        raise ValueError("Endpoints must satisfy 0 <= low < high <= 1")
    return low, high


def _enumerate(vertices, edge_id):
    if not vertices:
        yield ()
        return
    first, *rest = vertices
    for j, partner in enumerate(rest):
        edge = edge_id[first, partner]
        for tail in _enumerate(rest[:j] + rest[j + 1 :], edge_id):
            yield (edge, *tail)


@dataclass(frozen=True)
class MatchingNetwork:
    """Deployable single-hidden-layer ReLU network on binary activity indicators.

    Each edge unit computes ReLU(z_i + z_j - 1). Optional quadruple units
    compute ReLU(sum(z_a,z_b,z_c,z_d) - 3). The output is a linear readout.
    Only endpoints, input coordinates and coefficients are stored. On a support
    exceeding its inclusion-exclusion order the linear readout can exceed [0,1].
    """

    n_inputs: int
    edges: np.ndarray
    edge_weights: np.ndarray
    quadruples: np.ndarray
    quad_weights: np.ndarray
    low: float
    high: float

    @property
    def n_units(self):
        return len(self.edge_weights) + len(self.quad_weights)

    def predict(self, z):
        x = _bits(z, self.n_inputs, self.low, self.high)
        # Reuse activations for repeated binary queries without building
        # an examples-by-hypotheses matrix.
        masks, take = np.unique(
            x @ (1 << np.arange(self.n_inputs)), return_inverse=True
        )
        patterns = ((masks[:, None] >> np.arange(self.n_inputs)) & 1).astype(np.int8)
        edge_units = np.maximum(
            patterns[:, self.edges[:, 0]] + patterns[:, self.edges[:, 1]] - 1, 0
        )
        score = edge_units @ self.edge_weights
        if len(self.quad_weights):
            quad_units = np.maximum(patterns[:, self.quadruples].sum(2) - 3, 0)
            score -= quad_units @ self.quad_weights
        return (self.low + (self.high - self.low) * score)[take.reshape(-1)]

    def to_torch(self):
        """Return a frozen float64 Linear–ReLU–Linear network taking raw low/high scores.

        The input shape is [batch,n_inputs] and the output shape is [batch,1].
        The affine first layer absorbs z=(p-low)/(high-low); the exported network
        carries no hypothesis table, requires no Python-side input decoder and
        has the same support guarantees as predict().
        """
        import torch
        from torch import nn

        delta = self.high - self.low
        # Linear constructors initialize temporary tensors. Preserve the caller's
        # CPU RNG; every trainable entry is then replaced by a closed-form value.
        with torch.random.fork_rng(devices=[]):
            hidden = nn.Linear(
                self.n_inputs, self.n_units, dtype=torch.float64, device="cpu"
            )
            readout = nn.Linear(self.n_units, 1, dtype=torch.float64, device="cpu")
        with torch.no_grad():
            hidden.weight.zero_()
            n_edges = len(self.edges)
            edge_rows = torch.arange(n_edges)[:, None]
            hidden.weight[edge_rows, torch.as_tensor(self.edges)] = 1 / delta
            hidden.bias[:n_edges].fill_(-1 - 2 * self.low / delta)
            if len(self.quadruples):
                quad_rows = torch.arange(n_edges, self.n_units)[:, None]
                hidden.weight[quad_rows, torch.as_tensor(self.quadruples)] = 1 / delta
                hidden.bias[n_edges:].fill_(-3 - 4 * self.low / delta)
            coefficients = np.concatenate((self.edge_weights, -self.quad_weights))
            readout.weight.copy_(torch.from_numpy(delta * coefficients[None, :]))
            readout.bias.fill_(self.low)
        return nn.Sequential(hidden, nn.ReLU(), readout).requires_grad_(False).eval()


class MatchingFamily:
    """Lexicographically ordered perfect matchings of 2K labeled vertices, K=2..6."""

    def __init__(self, k):
        self.k = index(k)
        if isinstance(k, (bool, np.bool_)) or not 2 <= self.k <= 6:
            raise ValueError("Use 2 through 6 disjoint pairs")
        self.n = 2 * self.k
        self.edges = np.asarray(list(combinations(range(self.n), 2)), dtype=np.intp)
        self.edge_id = {tuple(edge): i for i, edge in enumerate(self.edges)}
        self.matchings = np.asarray(
            list(_enumerate(range(self.n), self.edge_id)), dtype=np.int16
        )
        self.incidence = np.zeros((len(self.matchings), len(self.edges)), dtype=bool)
        self.incidence[np.arange(len(self.matchings))[:, None], self.matchings] = True
        self.quadruples = np.asarray(
            list(combinations(range(self.n), 4)), dtype=np.intp
        )
        quad_id = {tuple(quad): i for i, quad in enumerate(self.quadruples)}
        unions = np.full((len(self.edges), len(self.edges)), -1, dtype=np.int16)
        for i, j in combinations(range(len(self.edges)), 2):
            quad = tuple(sorted((*self.edges[i], *self.edges[j])))
            if len(set(quad)) == 4:
                unions[i, j] = unions[j, i] = quad_id[quad]
        self._model_quads = np.stack(
            [
                unions[self.matchings[:, a], self.matchings[:, b]]
                for a, b in combinations(range(self.k), 2)
            ],
            axis=1,
        )
        self._powers = 1 << np.arange(self.n)
        self._edge_masks = self._powers[self.edges].sum(1)
        self._mask_to_edge = np.full(1 << self.n, -1, dtype=np.intp)
        self._mask_to_edge[self._edge_masks] = np.arange(len(self.edges))
        self._truth_cache = {}

    def _truth(self, mask):
        """One Boolean target per perfect matching, for one active input subset."""
        mask = int(mask)
        if mask not in self._truth_cache:
            contained_edges = np.flatnonzero(
                (mask & self._edge_masks) == self._edge_masks
            )
            self._truth_cache[mask] = np.any(self.incidence[:, contained_edges], axis=1)
        return self._truth_cache[mask]

    def contains(self, z, indices=None, *, low=0.0, high=1.0):
        """Return [query, matching] edge-existence flags, or one vector for scalar index."""
        masks = _bits(z, self.n, low, high) @ self._powers
        unique, take = np.unique(masks, return_inverse=True)
        columns = slice(None) if indices is None else indices
        if not len(unique):
            return np.empty(
                (0, *np.asarray(np.arange(len(self.matchings))[columns]).shape),
                dtype=bool,
            )
        return np.stack([self._truth(m)[columns] for m in unique])[take.reshape(-1)]

    def _pair_rows(self, x, low, high):
        rows = self._mask_to_edge[_bits(x, self.n, low, high) @ self._powers]
        if np.any(rows < 0):
            raise ValueError(
                "Matching training inputs require exactly two active providers per row"
            )
        return rows

    def _label_counts(self, x, y, low, high):
        source = self._pair_rows(x, low, high)
        y = np.asarray(y)
        if y.shape != (len(source),) or not np.all((y == 0) | (y == 1)):
            raise ValueError("Expected one binary label per fit row")
        y = y.astype(bool)
        pos = np.bincount(source[y], minlength=len(self.edges))
        neg = np.bincount(source[~y], minlength=len(self.edges))
        return pos, neg

    def _erm(self, hit_pos, hit_neg, low, high):
        # Change in SSE from replacing the all-low predictor on internal edges:
        # (high-low)*((high+low-2)*hit_pos + (high+low)*hit_neg).
        s = Fraction.from_float(low) + Fraction.from_float(high)
        if s == 1:
            criterion = hit_neg - hit_pos
        else:
            criterion = np.asarray(hit_neg, dtype=object) * s.numerator + np.asarray(
                hit_pos, dtype=object
            ) * (s.numerator - 2 * s.denominator)
        best = np.flatnonzero(criterion == np.min(criterion))
        return int(best[0]), len(best)

    def fit_erm(self, x, y, *, low=1 / 64, high=63 / 64):
        """Lexicographically first matching minimizing fit Brier; equal scores tie exactly."""
        low, high = _limits(low, high)
        pos, neg = self._label_counts(x, y, low, high)
        return self._erm(
            pos[self.matchings].sum(1), neg[self.matchings].sum(1), low, high
        )[0]

    def fit(
        self,
        x_labeled,
        y,
        x_unlabeled=None,
        *,
        pi,
        low=1 / 64,
        high=63 / 64,
        mode="joint",
    ):
        """Uniform-prior posterior from conditional labels, input evidence, or both.

        X-only and joint use both labeled and unlabeled fit input rows. Conditional
        ignores input frequencies. X-only ignores y (which may be None) and
        returns no label-ERM diagnostics. The only likelihood inputs are aggregate edge
        counts, and each prediction uses the resulting *fixed* model weights.
        """
        if mode not in ("conditional", "x_only", "joint"):
            raise ValueError("Use conditional, x_only or joint")
        low, high = _limits(low, high)
        pi = float(pi)
        if not 0 <= pi <= 1:
            raise ValueError("pi must be in [0,1]")
        scores = np.zeros(len(self.matchings))
        if mode == "x_only":
            # This predictive policy never reads the label argument.
            total = np.bincount(
                self._pair_rows(x_labeled, low, high), minlength=len(self.edges)
            )
            erm = fit_ties = None
        else:
            pos, neg = self._label_counts(x_labeled, y, low, high)
            total = pos + neg
            matched_pos = pos[self.matchings].sum(1)
            matched_neg = neg[self.matchings].sum(1)
            erm, fit_ties = self._erm(matched_pos, matched_neg, low, high)
            # At symmetric endpoints the number of wrong binary assignments is
            # an integer sufficient statistic, preserving exact fit-score ties.
            if Fraction.from_float(low) + Fraction.from_float(high) == 1:
                wrong = pos.sum() - matched_pos + matched_neg
                scores += xlogy(wrong, low) + xlogy(pos.sum() + neg.sum() - wrong, high)
            else:
                scores += (
                    xlogy(matched_pos, high)
                    + xlogy(matched_neg, 1 - high)
                    + xlogy(pos.sum() - matched_pos, low)
                    + xlogy(neg.sum() - matched_neg, 1 - low)
                )
        if mode != "conditional":
            if x_unlabeled is not None:
                extra = self._pair_rows(x_unlabeled, low, high)
                total += np.bincount(extra, minlength=len(self.edges))
            if pi != 1 / (2 * self.k - 1):
                within = total[self.matchings].sum(1)
                scores += xlogy(within, pi / self.k) + xlogy(
                    total.sum() - within, (1 - pi) / (len(self.edges) - self.k)
                )
        norm = logsumexp(scores)
        if not np.isfinite(norm):
            raise ValueError(
                "The supplied observations have zero probability under every matching"
            )
        weights = np.exp(scores - norm)
        # The second normalization bounds numerical error for concentrated posteriors.
        weights /= weights.sum()
        return MatchingPosterior(
            self, weights, low, high, int(np.argmax(scores)), fit_ties, erm
        )


class MatchingPosterior:
    """Explicit reference mixture and compilable marginal readouts for a fixed posterior."""

    def __init__(self, family, weights, low, high, map_index, fit_tie_count, erm_index):
        self.family = family
        self.weights = weights
        self.low, self.high = low, high
        self.map_index = map_index
        self.fit_tie_count = fit_tie_count
        self.erm_index = erm_index
        self.edge_weights = np.bincount(
            family.matchings.ravel(),
            weights=np.repeat(weights, family.k),
            minlength=len(family.edges),
        )
        quads = family._model_quads
        self.quad_weights = np.bincount(
            quads.ravel(),
            weights=np.repeat(weights, quads.shape[1]),
            minlength=len(family.quadruples),
        )

    def compile(self, order=1):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        quad_count = len(self.quad_weights) if order == 2 else 0
        return MatchingNetwork(
            self.family.n,
            self.family.edges.copy(),
            self.edge_weights.copy(),
            self.family.quadruples[:quad_count].copy(),
            self.quad_weights[:quad_count].copy(),
            self.low,
            self.high,
        )

    def predict(self, z, order=1):
        return self.compile(order).predict(z)

    def explicit(self, z):
        x = _bits(z, self.family.n, self.low, self.high)
        masks, take = np.unique(x @ self.family._powers, return_inverse=True)
        mass = np.asarray(
            [np.sum(self.weights[self.family._truth(mask)]) for mask in masks]
        )
        return (self.low + (self.high - self.low) * mass)[take.reshape(-1)]


def four_active_counterexample(low=Fraction(0), high=Fraction(1)):
    """Four active of eight providers under the uniform matching prior, with exact fractions."""
    low, high = Fraction(low), Fraction(high)
    if not 0 <= low < high <= 1:
        raise ValueError("Use ordered probability endpoints")
    first = Fraction(6, 7)
    both = Fraction(3, 35)
    exact = first - both
    return {
        "first_event": first,
        "second_event": exact,
        "exact_event": exact,
        "event_gap": both,
        "first_probability": low + (high - low) * first,
        "exact_probability": low + (high - low) * exact,
        "probability_gap": (high - low) * both,
    }
