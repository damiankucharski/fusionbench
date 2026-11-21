# Graded probabilities and Brier loss

The coverage intervention also improved stock search on a four-level probability grid. With 128 labels from a matched MIN/MAX rule, eight GP seeds, six independent data seeds and 300 fitness evaluations, one randomly constructed starter per arity reduced empirical Brier regret from .03944 to .03166. The paired reduction was .00778 (descriptive 95% t interval .00017 to .01538). Exact IID Brier risk improved from .24692 to .23580. Arithmetic averaging still scored .21375. Exhaustive MIN/MAX selection reached .18706, close to the Bayes floor of .18453.

All values below are expected positive-channel Brier loss; lower is better. Targets were specified before measurement. The family optimum is computed by evaluating every canonical lattice rule with the known conditional probability at all 256 grid cells. ERM uses fit labels only.

| Target, 128 labels | Bayes | Best possible MIN/MAX | Exact empirical MIN/MAX | Stock GP | Diverse starters | Mean | Logistic stack |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched | .18453 | .18453 | .18706 | .24692 | .23580 | .21375 | .21676 |
| Arithmetic mean | .22719 | .25641 | .25967 | .28991 | .28051 | .22719 | .23840 |
| Additive logit | .16530 | .19268 | .20304 | .23169 | .22288 | .18978 | .17137 |

With just 32 fit labels even the exact empirical rule scored .22646 IID on the matched target, worse than averaging's .21375. For the two misspecified controls, part of the gap remained at the population optimum. The actual GP also rounds population fitness to three decimal places; the reported archive regrets use independently evaluated unrounded Brier losses. Exact squared fit errors and tie handling use an integer scale of 400 on the chosen grid.

The starter intervention sampled one read-once rule per arity without enumerating the family. Initial widths and functional diversity both change. Fresh provider permutations and eight starter/search seeds reduce dependence on a lucky alignment. The benchmark remains a four-provider study with six independent data draws.

```bash
python -m fusionbench.eval.search_graded --workers 2 --out graded
```

[Frozen protocol](graded-protocol.md) · [Method means](graded/means.csv) · [Raw runs](graded/runs.csv.gz)
