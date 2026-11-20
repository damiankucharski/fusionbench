"""Test raw sensor fusion on held-out combinations of two reliability shifts."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (.25, .55, .85)
HELD_OUT = ((.25, .85), (.85, .25))
SOURCE = tuple(pair for pair in itertools.product(LEVELS, repeat=2) if pair not in HELD_OUT)
ANCHOR = .75


def examples(rng, strength, count, paired):
    """P(correct sensor sign | Y)=(1+r)/2 with conditionally independent noise."""
    shape = (len(strength), count)
    labels = 2*rng.integers(2, size=shape) - 1
    probabilities = (1 + np.column_stack((strength, np.full(len(strength), ANCHOR)))) / 2
    noise = 2*(rng.random(shape+(3,)) < probabilities[:, None, :]).astype(float) - 1
    observations = labels[..., None] * noise
    if not paired:
        observations[..., 2] = 2*rng.integers(2, size=shape) - 1
    return observations, labels


def as_float(a):
    return torch.as_tensor(a, dtype=torch.float32)


class FusionNet(nn.Module):
    """Estimate sensor weights from raw support tuples, with optional exchange symmetry."""
    def __init__(self, tied):
        super().__init__()
        self.tied = tied
        self.encoder = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 10), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(11, 12), nn.Tanh(), nn.Linear(12, 1 if tied else 2))

    def weights_for(self, support, available):
        def encode(x):
            pooled = self.encoder(x).mean(dim=1) * available
            return self.readout(torch.cat((pooled, available), dim=1))
        if self.tied:
            return torch.cat((encode(support), encode(support[..., [1, 0, 2]])), dim=1)
        return encode(support)

    def forward(self, support, available, query):
        return (self.weights_for(support, available)[:, None, :] * query).sum(-1)


def train(seed, paired, steps, unique_steps):
    models = []
    for index in range(4):
        torch.manual_seed(1000*seed + index)
        model = FusionNet(tied=index == 3)
        models.append((model, torch.optim.Adam(model.parameters(), lr=.005)))
    empirical = np.zeros(2)
    total = 0
    for step in range(steps):
        # Separate streams keep regimes, query labels and masks identical across controls.
        draw = step % unique_steps
        rng = np.random.default_rng([seed, draw, 1])
        strengths = np.array(SOURCE)[rng.integers(len(SOURCE), size=24)]
        available = as_float((rng.random((24, 1)) >= .2).astype(float))
        raw_support, _ = examples(np.random.default_rng([seed, draw, 2]), strengths, 32, paired)
        raw_query, y = examples(np.random.default_rng([seed, draw, 3]), strengths, 24, True)
        query = raw_query[..., :2]
        empirical += (query * y[..., None]).sum(axis=(0, 1))
        total += y.size
        x, xx, yy = as_float(raw_support), as_float(query), as_float((y+1)/2)
        for net, opt in models:
            loss = F.binary_cross_entropy_with_logits(net(x, available, xx), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return [net for net, _ in models], empirical / total


def conditional_risk(prob, x, strengths):
    """Integrate over the eight possible (Y, X1, X2) outcomes after prediction."""
    sq = np.zeros(len(strengths))
    accuracy = np.zeros_like(sq)
    for y in (-1, 1):
        weight = .5 * np.prod((1 + y * x[None, :, :] * strengths[:, None, :])/2, axis=-1)
        label = (1+y)/2
        sq += (weight*(prob-label)**2).sum(axis=1)
        accuracy += (weight*((prob>=.5)==bool(label))).sum(axis=1)
    return sq, accuracy


def evaluate(nets, single_strength, seed, paired, partitions, batches_per_cell):
    from scipy.special import expit
    rows = []
    queries = np.array(list(itertools.product((-1., 1.), repeat=2)))
    for partition, pairs in partitions.items():
        strength = np.repeat(pairs, batches_per_cell, axis=0)
        support, _ = examples(np.random.default_rng([seed, 900, int(partition == 'held')]),
                              strength, 128, paired)
        # Query distribution is integrated exactly, and is independent of the support draw.
        for support_n in (0, 8, 32, 128):
            raw = support[:, :max(1, support_n)]
            avail = as_float(np.full((len(strength), 1), int(support_n > 0)))
            with torch.no_grad():
                preds = [net(as_float(raw), avail, as_float(queries)).sigmoid().numpy() for net in nets]
            predictions = dict(single=preds[0], neural_ensemble=np.mean(preds[:3], axis=0), tied=preds[3])
            single = (1 + queries * single_strength)/2
            predictions['mean_experts'] = np.broadcast_to(single.mean(axis=1), preds[0].shape)
            if support_n:
                agreement = (raw[..., :2] * raw[..., 2:]).mean(axis=1)
                sign = np.sign(agreement[:, 0] - agreement[:, 1])
                coefficient = (1 + sign)/2
            else:
                coefficient = np.full(len(strength), .5)
            predictions['agreement_gate'] = coefficient[:,None]*single[:,0] + (1-coefficient[:,None])*single[:,1]
            # Privileged benchmark has the true deployed strengths.
            predictions['oracle'] = expit(2*np.arctanh(strength) @ queries.T)
            for model, prob in predictions.items():
                brier, acc = conditional_risk(prob, queries, strength)
                for a,b in pairs:
                    sel = np.all(strength == [a,b], axis=1)
                    rows.append(dict(seed=seed, information='paired' if paired else 'independent',
                                     partition=partition, r_a=a, r_b=b, support=support_n, model=model,
                                     brier=brier[sel].mean(), accuracy=acc[sel].mean()))
    return rows


def summarize(frame, output):
    from scipy.stats import t
    keys = ['seed','information','partition','support','model']
    per_seed = frame.groupby(keys)[['accuracy','brier']].mean().reset_index()
    per_seed.groupby(keys[1:])[['accuracy','brier']].mean().to_csv(output/'means.csv')
    contrasts = [('single',32,'single',0), ('tied',32,'tied',0),
                 ('tied',32,'neural_ensemble',32), ('single',32,'neural_ensemble',32),
                 ('single',32,'agreement_gate',32), ('tied',32,'agreement_gate',32),
                 ('tied',32,'single',32), ('tied',32,'oracle',32)]
    rows = []
    for (info,partition), group in per_seed.groupby(['information','partition']):
        for metric in ('accuracy','brier'):
            wide = group.pivot(index='seed', columns=['model','support'], values=metric)
            for a,na,b,nb in contrasts:
                d = wide[a,na] - wide[b,nb]
                half = t.ppf(.975,len(d)-1)*d.sem() if len(d)>1 else np.nan
                rows.append(dict(information=info, partition=partition, metric=metric, model=a,
                                 support=na, reference=b, ref_support=nb, seeds=len(d),
                                 delta=d.mean(), low=d.mean()-half, high=d.mean()+half))
    pd.DataFrame(rows).to_csv(output/'paired.csv', index=False)


def study(output, seeds, steps=360, batches_per_cell=256, unique_steps=None, paired_only=False):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds)<0 or min(steps,batches_per_cell)<=0:
        raise ValueError('use unique nonnegative seeds and positive budgets')
    unique_steps = steps if unique_steps is None else unique_steps
    if not 0 < unique_steps <= steps:
        raise ValueError("unique training steps must be within the step budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = dict(seeds=seeds, steps=steps, batches_per_cell=batches_per_cell,
                  source=SOURCE, held=HELD_OUT, anchor=ANCHOR, train_support=32,
                  unique_steps=unique_steps, paired_only=paired_only,
                  support_dropout=.2, training_batches_per_step=24, training_queries_per_batch=24)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for seed in seeds:
        for paired in ((True,) if paired_only else (True, False)):
            nets, empirical = train(seed,paired,steps,unique_steps)
            rows.extend(evaluate(nets, empirical, seed, paired, dict(held=HELD_OUT,source=SOURCE), batches_per_cell))
        pd.DataFrame(rows).to_csv(output/'cells.csv',index=False)
    summarize(pd.DataFrame(rows),output)


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--seeds',type=int,nargs='+',required=True)
    cli.add_argument('--steps',type=int,default=360)
    cli.add_argument('--unique-steps', type=int, default=None)
    cli.add_argument('--paired-only', action='store_true')
    args=cli.parse_args()
    study(args.output,args.seeds,args.steps,unique_steps=args.unique_steps,paired_only=args.paired_only)
