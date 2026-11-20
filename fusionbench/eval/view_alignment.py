"""Test transport of sibling-view exchange symmetries across unknown sensor axes.

The three four-dimensional sensor groups are supplied. Each group measures a
single binary latent factor with antipodal Gaussian means and isotropic noise.
A source population contains all factors. The student has only task labels in
a rescue-poor training population. Swaps alter data only during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from fusionbench.eval.mechanism_transfer import (
    DIRECTION, VIEW_SIZE, SmallMLP, discover, evaluate, expert_scores, target,
)
from fusionbench.rules import eval_rule


def unit(v):
    length = np.linalg.norm(v)
    if length < 1e-10:
        raise ValueError("a directional estimate is required")
    return v / length


def reflection(a, b):
    """An orthogonal involution sending unit(a) to unit(b)."""
    diff = unit(a) - unit(b)
    H = np.eye(len(diff))
    if np.linalg.norm(diff) > 1e-10:
        u = unit(diff)
        H -= 2 * np.outer(u, u)
    return H


def render(bits, noise, axes):
    return ((2 * bits.astype(float) - 1)[:, :, None] * axes + noise).reshape(len(bits), -1)


def split(seed, n, q, axes):
    rng = np.random.default_rng(seed)
    bits = rng.random((n, 3)) < [0.5, q, 0.5]
    noise = rng.normal(0, 0.62, (n, 3, VIEW_SIZE))
    return render(bits, noise, axes), target(bits), bits, noise


def swap_transform(pair, H):
    def augment(batch, rng):
        views = batch.reshape(-1, 3, VIEW_SIZE)
        selected = rng.random(len(batch)) < 0.5
        i, j = pair
        a = views[selected, i].copy()
        views[selected, i] = views[selected, j] @ H.T
        views[selected, j] = a @ H
        return batch
    return augment


def noise_jitter(batch, rng):
    return batch + rng.normal(0, 0.1, batch.shape)


def estimate_views(views, labels):
    """Estimate each sensor's discriminating direction from source class means."""
    return np.stack([
        unit(v[y].mean(axis=0) - v[~y].mean(axis=0))
        for v, y in zip(views.transpose(1, 0, 2), labels.T)
    ])


def contrast_pair(views, x_cal, y_cal, directions):
    # Center and scale with the balanced source only. Final calibration labels
    # select between the same six fusion structures used by the expert route.
    projection = (views * directions).sum(-1)
    cal = (x_cal.reshape(-1, 3, VIEW_SIZE) * directions).sum(-1)
    probs = expit((cal - projection.mean(0)) / projection.std(0))
    _, _, outer, pair, _ = discover(probs, y_cal)
    return pair, ("max" if outer == "min" else "min")


def fit_aligned(train, labels, base, epochs, pair, transform, op=None, control=False, width=10):
    """Change basis during training, then fold it into the first layer."""
    j = pair[1]
    cols = slice(j * VIEW_SIZE, (j + 1) * VIEW_SIZE)
    x = train.copy()
    x[:, cols] = train[:, cols] @ transform.T
    model = SmallMLP(train.shape[1], width, base + 5)
    # Match the initial function to the other models in raw sensor coordinates.
    model.params[0][cols] = transform @ model.params[0][cols]
    model.fit(x, labels, base + 6, epochs, pair=pair, diagonal_op=op, swap_control=control)
    check = model.predict(x[:32])
    model.params[0][cols] = transform.T @ model.params[0][cols]
    np.testing.assert_allclose(model.predict(train[:32]), check, atol=1e-12)
    return model


def run_study(output, seeds=range(200, 212), n_train=1024, n_test=4096, epochs=60,
              rescue_rates=(0., 0.03125), student_width=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds, rescue_rates = list(seeds), list(rescue_rates)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct nonnegative integers")
    if min(n_train, n_test, epochs, student_width) < 1 or not rescue_rates or any(not 0 <= q <= 1 for q in rescue_rates):
        raise ValueError("positive sizes and valid rescue probabilities are required")
    rows, sources = [], []
    with threadpool_limits(limits=1):
        for seed in seeds:
            base = 1000 * seed + 100
            rot_rng = np.random.default_rng(base)
            axes = np.stack([DIRECTION, unit(rot_rng.normal(size=VIEW_SIZE)), DIRECTION])
            xs, ys, concepts, _ = split(base + 1, 1536, .5, axes)
            views = xs.reshape(-1, 3, VIEW_SIZE)
            xc, yc, _, _ = split(base + 2, 1536, .5, axes)
            experts = [SmallMLP(VIEW_SIZE, 8, base + 20 + j).fit(
                views[:, j], concepts[:, j], base + 30 + j, epochs) for j in range(3)]
            _, _, _, epair, erule = discover(expert_scores(experts, xc), yc)
            grads = np.stack([unit(m.mean_logit_gradient(views[:, j])) for j, m in enumerate(experts)])
            label_axes = estimate_views(views, concepts)
            task_axes = estimate_views(views, np.broadcast_to(ys[:, None], concepts.shape))
            cpair, _ = contrast_pair(views, xc, yc, label_axes)
            tpair, top = contrast_pair(views, xc, yc, task_axes)
            sign = 1 if grads[epair[0]] @ grads[epair[1]] >= 0 else -1
            variants = {'flat': None, 'jitter': noise_jitter,
                        'sign_swap': swap_transform(epair, sign * np.eye(VIEW_SIZE))}
            for name, pair, directions in (
                    ('expert_transport', epair, grads), ('concept_transport', cpair, label_axes),
                    ('task_transport', tpair, task_axes), ('oracle_transport', (0, 1), axes)):
                i, j = pair
                H = reflection(directions[i], directions[j])
                np.testing.assert_allclose(directions[i] @ H, directions[j], atol=1e-8)
                variants[name] = swap_transform(pair, H)
            eop = erule["args"][1]["op"]
            diagonal = {}
            for name, pair, directions, op, control in [
                ('expert_diagonal', epair, grads, eop, False),
                ('task_diagonal', tpair, task_axes, top, False),
                ('aligned_three_view_control', epair, grads, None, True),
                ('aligned_wrong_reduction', epair, grads, 'min' if eop == 'max' else 'max', False)
            ]:
                i, j = pair
                diagonal[name] = pair, reflection(directions[i], directions[j]), op, control
            sources.append(dict(seed=seed, rotation_cosine=float(axes[0] @ axes[1]),
                                expert_pair=list(epair), concept_pair=list(cpair), task_pair=list(tpair),
                                expert_axes=(axes * grads).sum(-1).tolist(),
                                concept_axes=(axes * label_axes).sum(-1).tolist(),
                                task_axes=(axes * task_axes).sum(-1).tolist()))
            tests = {}
            for drift in (False, True):
                test_axes = axes.copy()
                if drift:
                    test_axes[1] *= -1
                x, y, bits, noise = split(base + 3, n_test, .5, test_axes)
                changed = bits.copy()
                changed[:, 1] = ~changed[:, 1]
                xcf = render(changed, noise, test_axes)
                probs = expert_scores(experts, x)
                ens = eval_rule(probs[:, :, None], erule).ravel()
                ens_cf = eval_rule(expert_scores(experts, xcf)[:, :, None], erule).ravel()
                ref = np.minimum(probs[:, 0], probs[:, 2]) >= .5
                tests[drift] = x, xcf, y, bits, target(changed), ref, ens, ens_cf
            for q in rescue_rates:
                train, labels, _, _ = split(base + 4, n_train, q, axes)
                for name in [*variants, *diagonal, 'pooled_calibration_flat', 'source_ensemble']:
                    if name in variants:
                        model = SmallMLP(3 * VIEW_SIZE, student_width, base + 5)
                        model.fit(train, labels, base + 6, epochs, augment=variants[name])
                    elif name in diagonal:
                        model = fit_aligned(train, labels, base, epochs, *diagonal[name], width=student_width)
                    elif name == 'pooled_calibration_flat':
                        # Raw calibration examples already contributed final labels
                        # to rule selection. This control measures their direct value.
                        both = np.concatenate([train, xc])
                        pooled_y = np.concatenate([labels, yc])
                        target_steps = epochs * ((n_train + 127) // 128)
                        pooled_epochs = max(1, round(target_steps / ((len(both) + 127) // 128)))
                        model = SmallMLP(3 * VIEW_SIZE, student_width, base + 5)
                        model.fit(both, pooled_y, base + 6, pooled_epochs)
                    for drift, (x, xcf, y, bits, ycf, ref, ens, ens_cf) in tests.items():
                        p, pcf = (ens, ens_cf) if name == 'source_ensemble' else (model.predict(x), model.predict(xcf))
                        rows.append(dict(seed=seed, model=name, deployment_reversal=drift, rescue_rate=q,
                                         **evaluate(p, pcf, y, bits, ycf, ref, ens >= .5)))
            pd.DataFrame(rows).to_csv(output / 'alignment.csv', index=False)
            (output / 'sources.json').write_text(json.dumps(sources, indent=2))
    (output / 'config.json').write_text(json.dumps(dict(seeds=seeds, n_train=n_train, n_test=n_test,
                                                        epochs=epochs, rescue_rates=rescue_rates, student_width=student_width,
                                                        student_parameters=(3 * VIEW_SIZE + 2) * student_width + 1), indent=2))
    return pd.DataFrame(rows)


def run_linear_control(output, seeds=range(200, 212), n_test=4096):
    """Fit three logistic experts to the same source concepts as the neural ensemble."""
    from sklearn.linear_model import LogisticRegression

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        base = 1000 * seed + 100
        rescue_axis = unit(np.random.default_rng(base).normal(size=VIEW_SIZE))
        axes = np.stack([DIRECTION, rescue_axis, DIRECTION])
        xs, _, concepts, _ = split(base + 1, 1536, .5, axes)
        views = xs.reshape(-1, 3, VIEW_SIZE)
        experts = [
            LogisticRegression(C=1., max_iter=500, tol=1e-8).fit(views[:, j], concepts[:, j])
            for j in range(3)
        ]

        def predict(x):
            x_views = x.reshape(-1, 3, VIEW_SIZE)
            return np.column_stack([
                expert.predict_proba(x_views[:, j])[:, 1] for j, expert in enumerate(experts)
            ])

        xc, yc, _, _ = split(base + 2, 1536, .5, axes)
        _, _, outer, pair, rule = discover(predict(xc), yc)
        for drift in (False, True):
            test_axes = axes.copy()
            if drift:
                test_axes[1] *= -1
            x, y, bits, noise = split(base + 3, n_test, .5, test_axes)
            changed = bits.copy()
            changed[:, 1] = ~changed[:, 1]
            xcf = render(changed, noise, test_axes)
            probs = predict(x)
            ens = eval_rule(probs[:, :, None], rule).ravel()
            ens_cf = eval_rule(predict(xcf)[:, :, None], rule).ravel()
            reference = np.minimum(probs[:, 0], probs[:, 2]) >= .5
            rows.append(dict(seed=seed, deployment_reversal=drift, pair=str(pair),
                             outer=outer, parameters=3*(1+VIEW_SIZE),
                             **evaluate(ens, ens_cf, y, bits, target(changed),
                                        reference, ens >= .5)))
    df = pd.DataFrame(rows)
    df.to_csv(output / 'linear_ensemble.csv', index=False)
    return df
