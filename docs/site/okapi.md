# Diagnosing a fusion tree with FusionBench

OKAPI uses genetic programming to build a tree that combines provider predictions. If its test loss is high, three questions lead to different experiments. Can the allowed operations express the generating rule? Does a finite search visit a tree that computes it? Does validation choose a good tree from those visited? FusionBench can answer each question separately when the conditional probability of the label is known.

## What can one operation express?

In a binary test with six anonymous providers, four contain the signals `A`, `B`, `C` and `D`. The true probability is `max(min(A, B), min(C, D))`. A pair supplies strong positive evidence only when both of its scores are high. A tree with two minimum operations and one maximum reproduces this probability on every input.

There is also a **proved limit** for a shallower tree. The generator draws half its examples uniformly from 64 corners where each of the six scores is either 1/64 or 63/64. On these corners one threshold operation reduces to a minimum, a maximum or its first input. A normalized weighted mean reduces to a convex average. Enumerating the subsets and optimizing the convex weights gives a minimum squared error of 1/8 on the rescaled zero–one corners. The corner component alone therefore contributes at least **½ × (62/64)² × ⅛ = 961/16384 ≈ 0.05865** to the population squared error relative to the true probability. This bound covers a lone provider or one application of minimum, maximum, mean, normalized weighted mean, nearest-to-threshold or farthest-from-threshold. The three-operation witness achieves zero. The status of two-operation trees is open in this comparison. [Proof and native witness](https://github.com/damiankucharski/OKAPI/blob/c91dd94ffc432b08a882fd64f281966bc47a5348/experiments/compositional_control/README.md) · [Independent certificate](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/search-regret/organic-composition/verify.py)

## Did search find it, and did validation return it?

A separate experiment hid the same two-pair rule among eight providers. The evaluator could compare a candidate's function with the target on the entire finite input grid. Each run used 1,200 training-score evaluations and a separate set of 256 validation labels. The two deep initializations used the same provider occurrences, operators and node counts, arranged as two branches or as a chain.

| Starting trees | Exact rule present at initialization | Exact rule visited by 1,200 evaluations | Exact rule returned by validation |
| --- | ---: | ---: | ---: |
| Two branches | 1/24 | 21/24 | 14/24 |
| Matched chains | 0/24 | 11/24 | 8/24 |
| Ordinary shallow trees | 0/24 | 2/24 | 2/24 |

With branched starts, three runs never visited the rule and seven visited it but returned something else. Increasing the search budget addresses a different failure from improving how the final tree is selected. The 24 runs used two search seeds on each of 12 sampled data sets. The counts describe this target, initialization and budget. [Experiment and protocol](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/seed-geometry/README.md) · [Per-run results](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/seed-geometry/focal-runs.csv.gz)

## Does the validation choice transfer?

Keep the candidates fixed and change the test inputs. Across 19 scenarios and eight data sets per scenario, four native trees competed on 192 clean validation labels in each run. The validation winner tied for the lowest test Brier among the four in **65.1%** of clean test sets and **27.6%** under the strongest tested log-probability noise. Its average excess over the best of those four rose from **0.00338** to **0.06271** in full-class Brier. These comparisons use the same trees before refitting. Test labels identify the best available candidate retrospectively. [Transfer results and definitions](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/transfer-grid/README.md)

You can apply that last diagnostic to other registered fusion methods. Choose on source validation labels, freeze the candidates and compare their rankings under paired test disturbances. [Run the transfer benchmark](transfer.md)
