# Round 2 · Finding and choosing a two-provider minimum

On the primary two-input MIN task with 192 fit rows, the **joint and fold
terminal rules chose the same program in all 32 banks**. Their mean difference
in conditional excess Brier was exactly 0, with a paired 95% percentile
interval of [0, 0]. Its upper end satisfies the prespecified noninferiority
margin of +0.002. The data show no superiority. Every resampled bank difference
was zero, so this interval gives little information about rare disagreements
on future banks. [Protocol](protocol.md) · [Paired estimates](results/r2-summary.csv)

The harder step was getting useful proposals. The live search found an exact
MIN on 12 banks, retained one in all 12 final populations, and deployed it in
six. In a separate two-choice diagnostic, the pooled, fold and joint rules
each picked the correct MIN over the actual fit-selected founder on **32 of
32** banks. A candidate that clears this comparison can still lose to another
finalist on empirical fold gain. [Search diagnostics](results/r2-diagnostics.csv)
· [Two-choice audit](results/r2-witness.csv)

## When was the correct rule available?

Counts are out of 32 independent banks per row. A live program counts as
exact when its predicted probabilities agree with the known law within
$10^{-12}$ on every fit row and every fresh audit row. This checks the realized
support. The separate native witness has a global proof on the score grid.

| Law | Fit rows | Exact initial anchor | Exact ever evaluated | Exact in final population | Chosen by fold | Chosen by joint | Chosen by pooled | Chosen by veto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Singleton | 192 | 32 | 32 | 32 | 26 | 27 | 24 | 28 |
| Singleton | 1,536 | 32 | 32 | 32 | 29 | 30 | 30 | 30 |
| MIN | 192 | 0 | 12 | 12 | 6 | 6 | 9 | 6 |
| MIN | 1,536 | 0 | 6 | 6 | 5 | 5 | 5 | 5 |

With 1,536 fit rows on the MIN task, every two-choice rule still accepted
the known witness in 32/32 banks. The search visited an exact rule in six
banks and selected one in five. Additional labels changed native breeding
decisions under the same generation budget. The two sample sizes produced
different candidate pools. On the singleton control, the fit-selected founder
was already correct in all 64 fits, so every two-choice gate correctly kept
the anchor on a tie. [Decisions and data identities](results/r2-diagnostics.csv)

## How much excess Brier remained?

Full-class conditional excess Brier is $2\,\mathrm{mean}((p-q)^2)$ on
4,096 audit observations. Zero is attainable under both laws. Each column
averages 32 banks. The first four rows share the same evolving population and
the same final menu in each cell.

| Frozen prediction rule | Singleton, 192 | Singleton, 1,536 | MIN, 192 | MIN, 1,536 |
| --- | ---: | ---: | ---: | ---: |
| Fold terminal | .001730 | .000013 | .047959 | .048718 |
| Joint terminal | .001656 | .000004 | .047959 | .048718 |
| Veto terminal | .001556 | .000004 | .047959 | .048718 |
| Pooled terminal | .001707 | .000011 | .045466 | .048652 |
| Fit-selected initial anchor | 0 | 0 | .146855 | .135031 |
| Arithmetic mean | .134466 | .134466 | .134312 | .134312 |
| Best ranked prefix (Top-N) | 0 | 0 | .081312 | .080420 |
| Probability logistic stacking | .015514 | .003530 | .036226 | .023289 |
| Regularized native-pattern OKAPI | .021067 | .001258 | .029316 | .000668 |

On MIN with 192 rows the joint program's .047959 splits into .039207 for
the best visited candidate measured on audit, .003919 from narrowing to
final survivors plus the anchor, and .004833 from choosing among that menu
using fit labels. With 1,536 rows these terms were .048376, .000182 and
.000160. Audit-ranked alternatives are retrospective bounds and were never
available for fit selection. The regularized native reference came closer to
MIN than ordinary fold-trained GP at both sample sizes. Top-N recovered the
singleton control in every bank. [Full results](results/r2-scores.csv)

## Data, implementation and checks

Six anonymous predictors are independent uniform draws from the 63 positive
grid scores $1/64,\ldots,63/64$. Two provider roles are permuted once per bank.
The two laws use $q=p_A$ and $q=\min(p_A,p_B)$ with the same score matrix
and uniform random draws for Bernoulli labels. The smaller fit reuses the first
192 of 1,536 rows. Plain GP receives only anonymous predictions and labels
for those rows, with one group per row. Four terminal decisions use the same
fold-trained trajectory, six singleton founders, and one arithmetic mean.
References receive identical predictions and labels. The counterfactual
witness enters only a separate fit-only comparison after choices are frozen.

The witness has a native value node for $A$, one MIN update, and one value
child for $B$. The binary operator returns the smaller positive-class score
of its two operands. Its predictions matched $q$ for **all 3,969 ordered
operand pairs** and on each fit and audit bank. The initial all-provider mean
has population excess 155/1152 (.13455) under MIN, compared with 31/192
(.16146) for either true operand alone. The empirical anchor can differ from
the best of these fixed founders. [Exact population calculation](results/r2-population-founders.csv)

The preregistered 128 fold-trained searches took 93.3 s with two single-threaded
workers. They scored 22,692 native program visits. An independent reconstruction
from the seed streams reproduced anchor risks, all three witness acceptance
indicators, four-fold gain summaries, and held-out anchor and mean risks for
128/128 fits. The bootstrap samples banks, keeps both methods paired and uses
10,000 draws with seed 97827. [Runtime manifest](results/r2-manifest.json)
· [Reproduction script](round2.py)

```sh
PYTHONPATH=.:/path/to/OKAPI OKAPI_LOG_LEVEL=WARNING python docs/ensemble-study/joint-gain/round2.py --workers 2
```
