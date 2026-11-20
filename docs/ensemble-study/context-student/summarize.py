"""Create paired confidence intervals across independent training/test seeds."""
from pathlib import Path

import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).parent
METRICS = ['accuracy', 'brier', 'context_rmse', 'wrong_support_delta']
CONTRASTS = [
    ('joint', 'static'), ('joint', 'observer'), ('task_only', 'joint'),
    ('joint', 'oracle_context'), ('joint_rms', 'joint'),
    ('posthoc_8', 'task_only'), ('posthoc_128', 'joint'),
]
for series in ('pilot', 'confirmation'):
    frame = pd.read_csv(ROOT / series / 'metrics.csv')
    frame.groupby(['information','gain','model'])[METRICS].mean().to_csv(ROOT / series / 'means.csv')
    rows = []
    for (info, gain), group in frame.groupby(['information', 'gain']):
        for metric in METRICS:
            wide = group.pivot(index='seed', columns='model', values=metric)
            for a,b in CONTRASTS:
                if a not in wide or b not in wide:
                    continue
                difference = wide[a] - wide[b]
                margin = t.ppf(.975, len(difference)-1)*difference.sem()
                rows.append(dict(information=info, gain=gain, metric=metric, model=a, reference=b,
                                 seeds=len(difference), delta=difference.mean(),
                                 ci_low=difference.mean()-margin, ci_high=difference.mean()+margin))
    pd.DataFrame(rows).to_csv(ROOT / series / 'paired.csv', index=False)
