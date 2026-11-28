# Series 2. Eight terminal probes when the right operator is MAX

On the hidden MAX law, allocating four probes to MIN and four to MAX lowered conditional audit excess Brier by **0.016201** relative to eight MIN probes. The paired 95% interval for balanced minus MIN-only was **−0.028557 to −0.005296**. Each of 24 independent banks contributed one difference averaged over its six- and twelve-provider versions. Eight banks improved, 14 tied and two worsened. The same comparison on sampled audit labels gave −0.016484 (−0.028708 to −0.005686). [Frozen protocol](r2-protocol.md) · [Paired contrasts](results/r2-contrasts.csv) · [Bank differences](results/r2-primary-by-bank.csv)

| Public providers | Possible unordered pairs | Banks offered the true MAX by balanced probes | Exact GP incumbent | Exact deployed with balanced probes | Balanced minus MIN-only excess Brier, 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 15 | 8/24 | 4/24 | 8/24 | −0.015941 (−0.034271 to −0.001224) |
| 12 | 66 | 5/24 | 3/24 | 8/24 | −0.016460 (−0.032347 to −0.002373) |

**Coverage becomes harder as the provider pool grows.** Four distinct MAX pairs include the planted pair with probability 4/15 (26.7%) or 4/66 (6.1%). This sample happened to cover five twelve-provider banks, versus 1.45 expected out of 24. The similar observed risk reductions at the two sizes therefore offer limited evidence about how the gain will scale. The MIN-only proposals never replaced the native GP finalist on this law. [Coverage counts](results/r2-coverage.csv)

Ordinary GP evaluated an exact prediction in six banks at each size. An exact prediction remained among its final survivors in six six-provider and four twelve-provider banks. The ordinary fit winner deployed it in four and three, respectively. With balanced probes, an exact prediction was available in ten and eight incumbent-plus-probe menus and deployed in eight and eight. At six providers the true MAX was offered in eight banks and the final choice matched it in six, including two ties retained by an already exact incumbent. The two observed regressions in audit excess were +0.036686 on six-provider bank 242123 and +0.044433 on twelve-provider bank 242102. On the matched singleton law all three probe schedules retained the GP incumbent on all 48 size-specific cases. [Ordered candidates](results/r2-candidates.csv.gz) · [Deployed scores](results/r2-scores.csv)

| Method | Mean hidden-MAX audit excess, sizes weighted equally | Mean required providers |
| --- | ---: | ---: |
| Frozen pooled GP | 0.059357 | 3.90 |
| GP plus eight MIN | 0.059357 | 3.90 |
| GP plus four MIN and four MAX | 0.043156 | 3.19 |
| GP plus eight MAX | 0.040837 | 2.79 |
| Arithmetic mean | 0.141045 | 9.00 |
| Brier-ranked Top-N | 0.089745 | 2.38 |
| Probability logistic stacking | 0.043766 | 9.00 |
| Regularized native-pattern OKAPI | 0.028225 | 4.06 |

The MAX-only comparator drew eight chances to offer the correct operation and found its pair in 14 six-provider and five twelve-provider banks. Its difference from the balanced menu was small relative to between-bank variation. Balanced minus MAX-only was +0.002319 (−0.005447 to +0.010978). The primary result compares two ways of spending the same eight terminal probes in a MAX-generating task. [Method levels](results/r2-summary.csv) · [Paired uncertainty](results/r2-contrasts.csv)

## Computation and reproducibility

Each bank had 192 fit and 4,096 independent audit observations. Within each bank the two sizes shared the latent signal, outcomes and first six inputs. Provider identities were independently permuted at each size. Only the fit scores and labels went into evolution, probe ranking and reference fitting. Audit excess integrates binary label uncertainty as twice the mean squared difference from the planted probability. Exactness requires agreement within 1e−12 on every fit and audit input.

The frozen protocol describes a target population of eight. The referenced ordinary native search applies a ceiling of eight, giving actual survivor targets of six for six providers and eight for twelve providers. Each expanded policy shared the same already-scored incumbent and paid for eight native fit predictions in its own menu, including candidates shared with another policy. Across 96 GP searches the native engine made 21,735 fit objective evaluations. The three probe policies added 2,304 real fit evaluations. The experiment took 57.26 seconds on one CPU worker using one numerical thread. [Runner](r2.py) · [Search and data fingerprints](results/r2-banks.csv) · [Timing and source hashes](results/r2-run.json)
