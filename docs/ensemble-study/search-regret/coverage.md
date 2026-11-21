# Which functions did genetic programming actually try?

In 28 of 36 fresh TwoPathways runs, 48 generations of MIN/MAX GP failed to evaluate even one decision function depending on all four providers. Every empirical optimum on all 12 datasets depended on four providers. This mismatch accounts for most of the empirical search gap relative to uniformly drawing the same number of distinct functions.

A provider counts as essential if flipping its bit can change the output on any of the 16 rows. This measures the function, so operators that reuse a provider without changing a decision cannot increase the count. For each realized GP budget we calculated the exact distribution of the lowest fit loss under uniform finite sampling, then conditioned that reference on the observed number of distinct functions at each arity. Each cell averages three GP restarts within each of 12 independent datasets.

| Target, 128 fit labels | Excess fit errors, GP | Uniform unique-function reference | Uniform within matched arities | GP hit empirical optimum | Four-provider functions visited per run |
|---|---:|---:|---:|---:|---:|
| TwoPathways | 13.67 | 7.52 | 13.78 | 0/36 | 0.33 |
| Additive | 7.89 | 2.73 | 7.68 | 1/36 | 0.44 |
| Parity | 4.00 | 9.66 | 7.96 | 22/36 | 0.03 |

On TwoPathways, the paired difference in excess fit errors between GP and the unconditional reference was +6.15 (descriptive 95% t interval over datasets +4.44 to +7.85). After matching arity counts the difference was −0.12 (−1.22 to +0.98). On parity, GP found better functions within those counts, with a difference of −3.96 (−7.58 to −0.34). Whether underexploration of complex functions hurts depends on the task. The parity control also has a 16-percentage-point irreducible gap between the monotone family and the Bayes classifier.

This comparison conditions on the realized search trajectory. A controlled change to function proposals is needed to estimate the effect of increasing coverage. All results concern compact four-provider input distributions and the specific four-individual population.

```bash
python -m fusionbench.eval.search_coverage --workers 2 --out coverage
```

[Frozen protocol](coverage-protocol.md) · [Summary](coverage/means.csv) · [All 324 checkpoints](coverage/runs.csv.gz)
