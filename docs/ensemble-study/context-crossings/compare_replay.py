"""Compare four training passes with one on the same independent seeds."""
from pathlib import Path

import pandas as pd
from scipy.stats import t

folder = Path(__file__).parent
def read_series(series):
    table = pd.read_csv(folder / series / 'cells.csv')
    trials = table.query('partition=="held" and information=="paired" and support==32')
    return trials.groupby(['seed', 'model'])[['accuracy', 'brier']].mean()

once = read_series('confirmation')
again = read_series('replay')
once = once.loc[again.index]
rows = []
for model in again.index.get_level_values('model').unique():
    for metric in ['accuracy', 'brier']:
        first = once.xs(model, level='model')[metric]
        repeated = again.xs(model, level='model')[metric]
        delta = repeated - first
        half = t.ppf(.975, len(delta) - 1) * delta.sem()
        rows.append(dict(model=model, metric=metric, once=first.mean(), four_passes=repeated.mean(),
                         delta=delta.mean(), low=delta.mean()-half, high=delta.mean()+half))
pd.DataFrame(rows).to_csv(folder/'replay'/'paired_to_one_pass.csv', index=False)
