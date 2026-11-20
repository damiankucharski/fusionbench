"""Transfer an exchange symmetry of a fitted fusion rule into raw-input training.

Only the training distribution is augmented. The deployed student is a single
ordinary MLP. No expert targets, logits or initial weights enter its optimizer.
Sensor grouping and correspondence of within-view coordinates are supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from fusionbench.rules import eval_rule

VIEW_SIZE = 4
DIRECTION = np.array([0.6, -0.4, 0.3, 0.62])
DIRECTION /= np.linalg.norm(DIRECTION)


def make_split(seed, size, rescue_rate, polarity):
    rng = np.random.default_rng(seed)
    bits = rng.random((size, 3)) < [0.5, rescue_rate, 0.5]
    noise = rng.normal(0, 0.62, size=(size, 3, VIEW_SIZE))
    x = render(bits, noise, polarity)
    return x, target(bits), bits, noise


def target(bits):
    return (bits[:, 0] | bits[:, 1]) & bits[:, 2]


def render(bits, noise, polarity):
    means = (2 * bits.astype(float) - 1)[:, :, None] * DIRECTION
    means[:, 1] *= polarity
    return (means + noise).reshape(len(bits), -1)


class SmallMLP:
    """Binary one-hidden-layer tanh network with minibatch Adam."""

    def __init__(self, inputs, width, seed):
        rng = np.random.default_rng(seed)
        self.params = [
            rng.normal(0, 1 / np.sqrt(inputs), (inputs, width)), np.zeros(width),
            rng.normal(0, 1 / np.sqrt(width), width), np.zeros(1),
        ]

    def predict(self, x):
        w, b, v, c = self.params
        return expit(np.tanh(x @ w + b) @ v + c)

    def mean_logit_gradient(self, x):
        w, b, v, _ = self.params
        return w @ (v * (1 - np.tanh(x @ w + b) ** 2).mean(axis=0))

    def fit(self, x, y, seed, epochs, pair=None, sign=1, augment=None, diagonal_op=None,
            swap_control=False):
        batch_rng = np.random.default_rng(seed)
        swap_rng = np.random.default_rng(seed + 1)
        first = [np.zeros_like(p) for p in self.params]
        second = [np.zeros_like(p) for p in self.params]
        steps = 0
        for _ in range(epochs):
            order = batch_rng.permutation(len(y))
            for start in range(0, len(y), 128):
                idx = order[start:start + 128]
                batch = x[idx]
                if augment is not None:
                    batch = augment(batch, swap_rng)
                if pair is not None:
                    swap = swap_rng.random(len(idx)) < 0.5
                    views = batch.reshape(-1, 3, VIEW_SIZE)
                    left, right = pair
                    previous = views[swap, left].copy()
                    views[swap, left] = sign * views[swap, right]
                    views[swap, right] = sign * previous
                expanded = diagonal_op is not None or swap_control
                if expanded:
                    left, right = pair
                    first_view = batch.copy().reshape(-1, 3, VIEW_SIZE)
                    second_view = first_view.copy()
                    if swap_control:
                        # The same three forward passes with ordinary augmentation.
                        second_view[:, left] = sign * first_view[:, right]
                        second_view[:, right] = sign * first_view[:, left]
                    else:
                        first_view[:, right] = sign * first_view[:, left]
                        second_view[:, left] = sign * second_view[:, right]
                    batch = np.concatenate((batch, first_view.reshape(len(idx), -1),
                                            second_view.reshape(len(idx), -1)))
                w, b, v, c = self.params
                h = np.tanh(batch @ w + b)
                scores = h @ v + c
                if expanded:
                    original, first_score, second_score = np.split(scores, 3)
                    initial = 0.5 * (expit(original) - y[idx])
                    if swap_control:
                        first_error = 0.25 * (expit(first_score) - y[idx])
                        second_error = 0.25 * (expit(second_score) - y[idx])
                    elif diagonal_op == 'mean':
                        joined = 0.5 * (first_score + second_score)
                        first_error = second_error = 0.25 * (expit(joined) - y[idx])
                    else:
                        # The graph distributes over repeated sibling inputs.
                        choose_first = (first_score > second_score if diagonal_op == 'max'
                                        else first_score < second_score)
                        selected = np.where(choose_first, first_score, second_score)
                        error = 0.5 * (expit(selected) - y[idx])
                        first_error = error * choose_first
                        second_error = error * ~choose_first
                    residual = np.concatenate((initial, first_error, second_error)) / len(idx)
                else:
                    residual = (expit(scores) - y[idx]) / len(idx)
                delta = residual[:, None] * v * (1 - h ** 2)
                grads = (batch.T @ delta, delta.sum(axis=0), h.T @ residual, residual.sum())
                steps += 1
                rate = 0.007 * np.sqrt(1 - 0.999 ** steps) / (1 - 0.9 ** steps)
                for param, mu, nu, grad in zip(self.params, first, second, grads):
                    mu *= 0.9
                    mu += 0.1 * grad
                    nu *= 0.999
                    nu += 0.001 * grad ** 2
                    param -= rate * mu / (np.sqrt(nu) + 1e-8)
        return self


def discover(expert_probs, labels):
    """Exhaust the 6 alternating read-once min/max trees on three leaves."""
    candidates = []
    for lone in range(3):
        pair = tuple(i for i in range(3) if i != lone)
        for outer, inner in [('min', 'max'), ('max', 'min')]:
            rule = {'op': outer, 'args': [
                {'model': lone}, {'op': inner, 'args': [{'model': i} for i in pair]},
            ]}
            p = eval_rule(expert_probs[:, :, None], rule).ravel()
            loss = -np.mean(
                labels * np.log(p.clip(1e-9, 1)) + (1 - labels) * np.log((1 - p).clip(1e-9, 1))
            )
            candidates.append((float(loss), lone, outer, pair, rule))
    return min(candidates, key=lambda item: item[0])


def expert_scores(models, x):
    views = x.reshape(-1, 3, VIEW_SIZE)
    return np.column_stack([model.predict(views[:, i]) for i, model in enumerate(models)])


def fit_source(seed, polarity, epochs):
    raw, _, concepts, _ = make_split(seed + 1, 1536, 0.5, polarity)
    views = raw.reshape(-1, 3, VIEW_SIZE)
    experts = [SmallMLP(VIEW_SIZE, 8, seed + 20 + j).fit(
        views[:, j], concepts[:, j], seed + 30 + j, epochs) for j in range(3)]
    x_cal, y_cal, _, _ = make_split(seed + 2, 1536, 0.5, polarity)
    loss, lone, outer, pair, rule = discover(expert_scores(experts, x_cal), y_cal)
    gradients = [model.mean_logit_gradient(views[:, j]) for j, model in enumerate(experts)]
    g0, g1 = [gradients[i] for i in pair]
    cosine = (g0 @ g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-12)
    info = dict(lone=lone, outer=outer, pair=pair, calibration_loss=loss,
                gradient_cosine=float(cosine), sign=1 if cosine >= 0 else -1, rule=rule)
    return experts, info


def rate(a, mask):
    return float(a[mask].mean()) if mask.any() else np.nan


def evaluate(p, p_flip, test_y, bits, flip_y, reference, ensemble):
    pred = p >= 0.5
    repair = (reference != test_y) & (ensemble == test_y)
    damage = (reference == test_y) & (ensemble != test_y)
    rescue_only = ~bits[:, 0] & bits[:, 1] & bits[:, 2]
    should_change = flip_y != test_y
    predicted_flip = p_flip >= 0.5
    return dict(
        accuracy=np.mean(pred == test_y), ensemble_agreement=np.mean(pred == ensemble),
        rescue_recall=rate(pred, rescue_only), n_rescue=int(rescue_only.sum()),
        repair_retention=rate(pred == test_y, repair), n_repair=int(repair.sum()),
        error_inheritance=rate(pred != test_y, damage), n_damage=int(damage.sum()),
        # Correct BOTH decisions in counterfactual pairs with a changing label.
        causal_pair_accuracy=rate((pred == test_y) & (predicted_flip == flip_y), should_change),
        spurious_flip=rate(pred != predicted_flip, ~should_change),
        spurious_veto=rate(pred != predicted_flip, ~bits[:, 2]),
        spurious_core=rate(pred != predicted_flip, bits[:, 0] & bits[:, 2]),
        **{f'accuracy_{i}{j}{k}': rate(pred == test_y, (bits == [i, j, k]).all(axis=1))
           for i in range(2) for j in range(2) for k in range(2)},
    )


def run_study(output, seeds=range(8), n_train=1024, n_test=4096, epochs=60,
              rescue_rates=(0.0, 0.03125, 0.5), followup=False):
    """Write per-seed metrics, extracted rules and configuration. Return metrics."""
    seeds = list(seeds)
    if (not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds)
            or min(n_train, n_test, epochs) <= 0):
        raise ValueError('use unique nonnegative seeds and positive sizes/epochs')
    if not rescue_rates or any(not 0 <= q <= 1 for q in rescue_rates):
        raise ValueError('rescue_rates must be nonempty probabilities')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    with threadpool_limits(limits=1):
        for s in seeds:
            base = 1000 * s + 100
            for polarity in (1, -1):
                experts, info = fit_source(base, polarity, epochs)
                sources.append(dict(seed=s, polarity=polarity, **info))
                x, y, bits, noise = make_split(base + 3, n_test, 0.5, polarity)
                toggled = bits.copy()
                toggled[:, 1] = ~toggled[:, 1]
                x_cf = render(toggled, noise, polarity)
                y_cf = target(toggled)
                pred = expert_scores(experts, x)
                ensemble = eval_rule(pred[:, :, None], info['rule']).ravel()
                ensemble_cf = eval_rule(expert_scores(experts, x_cf)[:, :, None], info['rule']).ravel()
                reference = np.minimum(pred[:, 0], pred[:, 2]) >= 0.5
                # Alternative access regime: full labeled source examples can be reused.
                x_cal, y_cal, _, _ = make_split(base + 2, 1536, 0.5, polarity)
                source_mlp = SmallMLP(3 * VIEW_SIZE, 10, base + 5)
                source_mlp.fit(x_cal, y_cal, base + 6, epochs)
                source_metrics = evaluate(source_mlp.predict(x), source_mlp.predict(x_cf),
                                          y, bits, y_cf, reference, ensemble >= 0.5)
                for q in rescue_rates:
                    x_tr, y_tr, c_tr, _ = make_split(base + 4, n_train, q, polarity)
                    stats = dict(seed=s, polarity=polarity, rescue_rate=q,
                                 n_train_rescue_only=int((~c_tr[:, 0] & c_tr[:, 1] & c_tr[:, 2]).sum()))
                    variants = [('flat', None, 1), ('tree_swap', info['pair'], 1),
                                ('aligned_tree_swap', info['pair'], info['sign']),
                                ('wrong_pair_swap', (0, 2), 1)]
                    predictions = {}
                    for name, pair, sign in variants:
                        key = (pair, sign)
                        if key not in predictions:
                            net = SmallMLP(3 * VIEW_SIZE, 10, base + 5)
                            net.fit(x_tr, y_tr, base + 6, epochs, pair, sign)
                            predictions[key] = net.predict(x), net.predict(x_cf)
                        p, p_cf = predictions[key]
                        metrics = evaluate(p, p_cf, y, bits, y_cf, reference, ensemble >= 0.5)
                        records.append(dict(**stats, model=name, **metrics))
                    if followup:
                        pair, sign = info['pair'], info['sign']
                        inner = info['rule']['args'][1]['op']
                        opposite = 'min' if inner == 'max' else 'max'
                        for name, op, control in [('diagonal_transfer', inner, False),
                                                  ('wrong_reduction', opposite, False),
                                                  ('mean_reduction', 'mean', False),
                                                  ('three_view_control', None, True)]:
                            net = SmallMLP(3 * VIEW_SIZE, 10, base + 5)
                            net.fit(x_tr, y_tr, base + 6, epochs, pair, sign,
                                    diagonal_op=op, swap_control=control)
                            metrics = evaluate(net.predict(x), net.predict(x_cf), y, bits, y_cf,
                                               reference, ensemble >= 0.5)
                            records.append(dict(**stats, model=name, **metrics))
                    records.append(dict(**stats, model='source_flat', **source_metrics))
                    metrics = evaluate(ensemble, ensemble_cf, y, bits, y_cf, reference, ensemble >= 0.5)
                    records.append(dict(**stats, model='source_ensemble', **metrics))
            # Checkpoint each independent seed for long runs.
            pd.DataFrame(records).to_csv(output / 'metrics.csv', index=False)
            (output / 'sources.json').write_text(json.dumps(sources, indent=2))
    config = dict(seeds=seeds, n_train=n_train, n_test=n_test, epochs=epochs,
                  rescue_rates=list(rescue_rates), followup=followup, student_parameters=141, ensemble_parameters=147)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    return pd.DataFrame(records)
