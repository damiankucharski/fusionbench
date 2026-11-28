# Series 1 · Eight pair proposals after search

On the task with a true two-provider minimum, eight distinct pairs produced
mean audit excess Brier **0.03007**, compared with **0.02942** for eight IID
draws. The primary paired difference was **+0.00065** (95% bootstrap interval
−0.01691 to +0.01777). Unique improved on seven banks, worsened on seven,
and tied within $10^{-12}$ on 18. These 32 banks do not resolve a deployment
advantage for either schedule. [Protocol](r1-protocol.md) · [Paired contrasts](results/r1-summary.csv)

Both schedules helped compared with the same frozen GP finalist. Unique
reduced excess Brier by 0.02222 (0.01019 to 0.03578), and IID by 0.02287
(0.01025 to 0.03675). These secondary comparisons add eight fit-scored MIN
programs to the finalist. They combine an extra search budget with knowledge
of the correct operator family. Only the unique–IID comparison holds both
of those factors fixed. [Bank-level scores](results/r1-scores.csv)

## Pair coverage, availability and deployment

In a catalogue of 15 unordered pairs, the probability of drawing a given
pair in eight proposals is exactly **53.33%** without replacement and
**42.42%** with replacement. That 10.92 percentage point improvement in
coverage holds for the proposal process before evaluating any fit labels.
Actual schedules need not be nested within a bank. Here five banks received
the true pair in both schedules, ten in unique only, eight in IID only,
and nine in neither.

| MIN law, 32 banks | Frozen GP finalist | Finalist + IID | Finalist + unique |
| --- | ---: | ---: | ---: |
| Distinct extra pairs per bank, mean | 0 | 6.34 | 8.00 |
| True pair explicitly proposed | 0 | 13 | 15 |
| Exact function in deployment menu | 5 | 17 | 17 |
| Exact function deployed | 5 | 16 | 16 |
| Exact function deployed when the true pair was offered | — | 12/13 | 14/15 |

The original GP evaluated an exact function in eight banks and retained one
among its final survivors in six. Its fit-ranked finalist was exact in five.
Only this frozen winner entered the two augmented menus. Some pair proposals
therefore duplicated a solution GP already deployed. Each augmentation added
11 exact deployments and preserved all five existing ones. The 16 exact
deployments under each schedule overlapped on nine banks. Nine banks
were exact under neither, seven under unique alone, and seven under IID alone.
We checked GP program agreement with the truth on every fit and audit
row. An offered native MIN on the hidden pair equals the rule on every input
by construction. [GP audit](results/r1-banks.csv) · [Ordered probes](results/r1-probes.csv)

## References and singleton control

| Method | MIN excess Brier | Singleton excess Brier |
| --- | ---: | ---: |
| Pooled GP finalist | .052293 | .001997 |
| GP + eight IID pairs | .029421 | .001997 |
| GP + eight distinct pairs | .030074 | .001997 |
| Arithmetic mean | .134320 | .133972 |
| Brier-ranked best-prefix Top-N | .085776 | 0 |
| Probability logistic stacking | .039012 | .016255 |
| Regularized native-pattern OKAPI | .035329 | .026377 |

Under the singleton law both augmented policies retained the original finalist
in all 32 banks. GP began with the correct singleton in every bank, retained
an equivalent function in 31 final populations, and selected it in 23.
Top-N achieved zero excess on every bank. Both laws used the same provider
scores, paired Bernoulli uniforms, role assignments and menus. Lower scores
are better. Excess Brier integrates out audit-label noise conditional on
the 4,096 fresh inputs per bank. [All fit and audit scores](results/r1-scores.csv)

## Reproduction

The run used 64 native searches, which made 11,559 objective evaluations.
The final menus added 1,024 native evaluations on fit rows, including
duplicates. An independent enumeration recreated each pair, its fit loss
and all 128 terminal choices, then checked the recorded audit risks. The
final replay took 29.0 seconds on one worker with one numerical thread.
[Runtime and source hashes](results/r1-manifest.json)

```sh
PYTHONPATH=.:/path/to/OKAPI OKAPI_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python docs/ensemble-study/pair-coverage/round1.py
```
