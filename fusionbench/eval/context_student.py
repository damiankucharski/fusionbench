"""A raw-input classifier that monitors contemporaneous sensor reliability.

Run as a module. Torch is an optional experiment dependency. The support set
and query examples share a prevalence but never share individuals or labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

STABLE, VARIABLE = 0.65, 1.5


def generate(seed, prevalences, n_support, n_query, *, paired=True, gain=1.0):
    """Two sensors with balanced binary targets and a sign-flip mixture."""
    rng = np.random.default_rng(seed)
    p = np.asarray(prevalences).reshape(-1, 1)

    def sample(n, shared):
        shape = (len(p), n)
        y = rng.integers(0, 2, shape) * 2 - 1
        independent = rng.integers(0, 2, shape) * 2 - 1
        second_target = y if shared else independent
        flip = np.where(rng.random(shape) < p, -1., 1.)
        u = STABLE * y + rng.standard_normal(shape)
        v = gain * (VARIABLE * second_target * flip + rng.standard_normal(shape))
        return np.stack((u, v), axis=-1), (y + 1) / 2

    support, _ = sample(n_support, paired)
    query, labels = sample(n_query, True)
    return tuple(torch.as_tensor(a, dtype=torch.float32) for a in (support, query, labels, p))


class ContextMLP(nn.Module):
    """Pool a learned embedding over support, then condition one raw-input MLP."""
    def __init__(self, mode, normalize=False):
        super().__init__()
        self.mode, self.normalize = mode, normalize
        self.encoder = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 12), nn.Tanh())
        self.observer = nn.Linear(12, 1)
        self.classifier = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, support, query, p):
        if self.normalize:
            # Source RMS is also scaled to one. No deployment labels are used.
            scale = support.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
            support, query = support / scale, query / scale
        estimated = self.observer(self.encoder(support).mean(dim=1)).sigmoid()
        if self.mode == 'oracle':
            context = p
        elif self.mode == 'static':
            context = torch.full_like(p, .5)
        elif self.mode == 'observer':
            context = estimated.detach()
        else:
            context = estimated
        lift = (1 - 2 * context).expand_as(query[..., 1])
        features = torch.cat((query, lift[..., None], (query[..., 1] * lift)[..., None]), dim=-1)
        return self.classifier(features).squeeze(-1), estimated


def train(seed, steps, support_n, paired, modes):
    models, optimizers = {}, {}
    for name, mode, normalize in modes:
        torch.manual_seed(41 + seed)
        model = ContextMLP(mode, normalize)
        models[name] = model
        optimizers[name] = torch.optim.Adam(model.parameters(), lr=.006)
    rng = np.random.default_rng([seed, 1])
    for step in range(steps):
        p = rng.choice([0, .25, .5, .75, 1.], size=24)
        support, query, y, truth = generate([seed, 2, step], p, support_n, 16, paired=paired)
        for name, model in models.items():
            logits, estimated = model(support, query, truth)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if model.mode != 'task_only':
                loss = loss + 2 * F.mse_loss(estimated, truth)
            opt = optimizers[name]
            opt.zero_grad()
            loss.backward()
            opt.step()
    return models


def bayes_probability(query, p, gain):
    """Exact binary posterior conditional on the known sensor strengths."""
    u, v = query[..., 0], query[..., 1] / gain
    a, b = torch.log1p(-p), torch.log(p)
    strength = VARIABLE * v
    log_ratio = torch.logaddexp(a + strength, b - strength) - torch.logaddexp(a - strength, b + strength)
    return (2 * STABLE * u + log_ratio).sigmoid()


def study(output, seeds, steps=320, support_n=32, batches=200, query_n=64):
    from scipy.special import ndtr

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seeds = list(seeds)
    if not seeds or min(steps, support_n, batches, query_n) <= 0 or batches % 5:
        raise ValueError('use nonempty seeds, positive sizes and a batch count divisible by 5')
    config = dict(seeds=seeds, steps=steps, support_n=support_n, batches=batches, query_n=query_n,
                  train_batch=24, train_query=16, optimizer='Adam', lr=.006, context_loss_weight=2.,
                  stable=STABLE, variable=VARIABLE, test_p=[.05, .25, .5, .75, .95], gains=[1., .25, 4.])
    (output/'config.json').write_text(json.dumps(config, indent=2))
    observed, grouped, calibrations = [], [], []
    for seed in seeds:
        p = np.tile(config['test_p'], batches // 5)
        for paired in (True, False):
            modes = [('static', 'static', False), ('observer', 'observer', False),
                     ('joint', 'joint', False), ('task_only', 'task_only', False),
                     ('oracle_context', 'oracle', False)]
            if paired:
                modes += [('joint_rms', 'joint', True), ('oracle_context_rms', 'oracle', True),
                          ('observer_rms', 'observer', True), ('static_rms', 'static', True)]
            nets = train(seed, steps, support_n, paired, modes)
            # Freeze the prediction model. Estimate only an affine readout of its scalar
            # from nested, independent pools of annotated source environments.
            cal_rng = np.random.default_rng([seed, 5])
            cal_p = cal_rng.choice([0., .25, .5, .75, 1.], size=128)
            cal_support, cal_query, _, cal_truth = generate([seed, 6], cal_p, support_n, 1, paired=paired)
            with torch.no_grad():
                _, cal_latent = nets['task_only'](cal_support, cal_query, cal_truth)
            z = cal_latent.numpy().reshape(-1)
            mappings = {}
            for budget in (8, 128):
                design = np.column_stack((z[:budget], np.ones(budget)))
                slope, intercept = np.linalg.lstsq(design, cal_p[:budget], rcond=None)[0]
                mappings[budget] = slope, intercept
                calibrations.append(dict(seed=seed, paired=paired, budget=budget,
                                         slope=slope, intercept=intercept))
            for gain in config['gains'] if paired else [1.]:
                support, query, y, truth = generate([seed, 3], p, support_n, query_n,
                                                     paired=paired, gain=gain)
                opposite, _, _, _ = generate([seed, 4], 1-p, support_n, query_n, paired=paired, gain=gain)
                same, _, _, _ = generate([seed, 4], p, support_n, query_n, paired=paired, gain=gain)
                predictions = {}
                with torch.no_grad():
                    for name, net in nets.items():
                        logits, estimate = net(support, query, truth)
                        # Swap the context stream while freezing all query observations.
                        counter, _ = net(opposite, query, 1-truth)
                        control, _ = net(same, query, truth)
                        predictions[name] = (logits.sigmoid(), estimate, counter.sigmoid(), control.sigmoid())
                    predicted, latent, counter, control = predictions['task_only']
                    for n, (slope, intercept) in mappings.items():
                        predictions[f'posthoc_{n}'] = (predicted, (latent * slope + intercept).clamp(0, 1),
                                                        counter, control)
                    # Control with the data generator known to the estimator.
                    prob = bayes_probability(query, truth, gain)
                    predictions['bayes_known'] = (prob, truth, bayes_probability(query,1-truth,gain), prob)
                    sign_strength = (2 * ndtr(STABLE)-1) * (2 * ndtr(VARIABLE)-1)
                    def sign_est(s):
                        return ((1 - s.sign().prod(dim=-1).mean(dim=1, keepdim=True)/sign_strength)/2).clamp(0, 1)
                    estimate = sign_est(support)
                    predictions['sign_known_sensor'] = (
                        bayes_probability(query,estimate,gain),estimate,
                        bayes_probability(query,sign_est(opposite),gain),
                        bayes_probability(query,sign_est(same),gain))
                for name, (prob, estimate, counter, control) in predictions.items():
                    with torch.no_grad():
                        sq_err = (prob-y).square().mean(-1)
                        acc = ((prob>=.5) == y.bool()).float().mean(-1)
                        swap_brier = ((counter-y).square()-(control-y).square()).mean(-1)
                        sq_context = (estimate-truth).square().ravel()
                        row = dict(seed=seed, information='paired' if paired else 'independent',
                                   gain=gain, model=name)
                        for prevalence in config['test_p']:
                            sel = truth.ravel() == prevalence
                            grouped.append(dict(**row, prevalence=prevalence, brier=float(sq_err[sel].mean()),
                                                accuracy=float(acc[sel].mean()),
                                                p_estimate=float(estimate.ravel()[sel].mean())))
                        observed.append(dict(**row, accuracy=float(acc.mean()), brier=float(sq_err.mean()),
                                             context_rmse=float(sq_context.mean().sqrt()),
                                             wrong_support_delta=float(swap_brier.mean())))
        pd.DataFrame(observed).to_csv(output/'metrics.csv',index=False)
        pd.DataFrame(grouped).to_csv(output/'by_prevalence.csv',index=False)
        pd.DataFrame(calibrations).to_csv(output/'calibration.csv',index=False)
    return pd.DataFrame(observed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--steps', type=int, default=320)
    args = parser.parse_args()
    study(args.output, args.seeds, args.steps)
