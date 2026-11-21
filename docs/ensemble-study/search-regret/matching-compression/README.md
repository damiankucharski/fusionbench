# A 66-unit network exactly represents a mixture of 10,395 pairing trees on sparse inputs

With 12 provider scores and exactly two or three high values, **one hidden layer of 66 ReLUs reproduces the fixed predictive average over all 10,395 possible perfect-matching MIN/MAX trees**. Each unit detects one pair of high providers. Its output coefficient is the posterior probability that the pair belongs to the hidden matching. No teacher predictions or distillation are used. The largest observed discrepancy was below 3.0×10⁻¹⁴. Each provider's probability still has to be supplied as an input. [Frozen experiment](../matching-compression-protocol.md) · [Runnable PyTorch networks](example/README.md) · [Exact inference and compilation](../../../../fusionbench/eval/matching_posterior.py)

With **four** high inputs, a matching can contain two active pairs. Simply adding their marginal probabilities can count the same hypothesis twice. Adding one conjunction per input quadruple and subtracting its expected number of two-pair matchings gives exact predictions through five active inputs. In this experiment the 12-input, 561-ReLU network reproduced the full mixture on every four-active state within 3.6×10⁻¹⁴. For eight inputs and a uniform pairing prior, four active inputs give a concrete failure of the pair-only formula: it predicts active-path probability 6/7 while the exact mixture gives 27/35. Both lie inside [0,1]. The difference in the actual probability output is (63/64−1/64)·3/35. [Pointwise compilation tests across all cases](compilation.csv.gz) · [Exact example and exported weights](example/export.json)

## What revealed the hidden groups

The six-pair experiment distinguishes two sources of information. At fit time each row highlights two anonymous providers. In the balanced design half of those rows show a true pair, although true pairs make up only 6 of the 66 possible edges. Their frequency therefore reveals the pairs without labels. In the uniform design every edge is equally likely, so the input distribution carries zero information about the pairing. With 16 fit labels and 128 extra unlabeled inputs per bank, the combined likelihood reduced excess Brier on balanced held-out pairs by **0.02559** compared with conditional labels alone (95% interval 0.00742 to 0.04376) after balanced training. With uniform inputs, the difference was **exactly zero** in every bank. The joint and input-only fits used all 144 input rows. No held-out query or label updated the frozen weights. [Paired effects and controls](paired.csv)

| Method, 12 providers and 16 fit labels | Balanced training, matching law | Uniform training, matching law | Balanced training, logistic law |
|---|---:|---:|---:|
| Mean across providers | .33889 | .33889 | .15639 |
| Fit-ranked Top-N | .38024 | .37990 | .17427 |
| Regularized logistic stacking | .31949 | .38300 | **.11872** |
| Ordinary OKAPI, 600 evaluations | .39384 | .40787 | .19044 |
| Single fit-Brier-optimal pairing tree | .05377 | .31722 | .27704 |
| Average using conditional labels | .02559 | .25992 | .22278 |
| Average using input frequencies alone | <.00001 | .39168 | .26281 |
| Average using inputs and labels | **<.00001** | **.25992** | .26281 |

Entries are mean squared error to the generating conditional probability on the exact balanced distribution of two-active test states, across 16 independent banks. On the primary matched design, joint and label-only **observed** Brier were .01578 and .04149. The joint model's excess was 3.94×10⁻²⁸. The matching prior and probability endpoints were given as modeling assumptions. On the mismatched logistic label law, the same strong input evidence identified a pairing that failed to describe the labels, and logistic stacking won. On uniformly generated training pairs, the conditional-label model's error on a *uniform* two-pair test reached .05622 versus .25992 on the balanced test that upweights the rare true pairs. For four latent pairs, adding input evidence to 16 labels in the balanced design changed excess by only −0.000021 (−0.000051 to +0.000008). With 64 labels the matched-law changes at either K were below the displayed precision. [All sample sizes, laws and support shifts](means.csv)

## The two identities

Let zᵢ be the indicator that provider i is high and let θᵢⱼ be the probability, under the *fixed trained weights*, that i and j form a true pair. A tree outputs high precisely when the set of active providers contains at least one of its matched pairs. Up to three active providers admit at most one such pair, so the exact model average is

$$l+(h-l)\sum_{i<j}\theta_{ij}\,\mathrm{ReLU}(z_i+z_j-1).$$

For four or five active providers, subtract one term per input quadruple. The coefficient is the sum of the three joint probabilities that its four vertices form two disjoint matched edges. Multiply that coefficient by ReLU(zᵢ+zⱼ+zₖ+zₗ−3) and by h−l. This is the second term of inclusion–exclusion. With six active inputs, three disjoint pairs can coexist and the second-order formula can fail. A test constructs that counterexample. Pointwise agreement does not imply that the learned mixture is close to the unknown target; the table separates those properties.

The two exported float64 PyTorch models take the original 12 endpoint probabilities directly. For the prespecified example, the pair-only network stores 265 nonzero parameters (925 in a dense representation). The pair-plus-quadruple network stores 3235 nonzero parameters (7855 dense). They use 66 and 561 hidden units respectively. The deployed models do not store or evaluate the 10,395 hypotheses, although the learning procedure enumerates them to compute the coefficients. Encoding a single matching requires much less storage when the data determine it. [Export script and instructions](example/README.md)

## A lower bound for flat stacking

On balanced matching queries each individual coordinate has identical activation probability in the within-pair and cross-pair groups. A logistic learner minimizing population cross-entropy with an intercept therefore attains its optimum with constant prediction 1/2. The cross-entropy gradient vanishes there; regularization chooses the zero-weight representative. Finite-sample fits can deviate.

An independent argument bounds the Brier error of *any* nondecreasing transformation of an affine score. Sort the K positive matched-pair states by their affine scores. For each lower positive state and each of its m higher partners, form two complementary partitions of their four coordinates into cross pairs. In both partitions, the sum of the two negative states' affine scores equals the sum of the two positive scores. At least one negative probability in each partition therefore equals or exceeds the lower positive probability u. States assigned to different unordered pairs of positive states are distinct. Each positive has weight a=1/(2K), each negative weight b=1/[4K(K−1)]. Their errors sum to at least the minimum of a(h−u)²+2mb·max(0,u−l)², which is a(h−l)²·m/(K−1+m). Summing gives a lower bound

$$\frac{(h-l)^2}{2K}\sum_{m=0}^{K-1}\frac{m}{K-1+m},$$

whose limit is (h−l)²(1−ln 2)/2. This is a lower bound, not a computation of the Brier-optimal flat score. The 2/3/4-active support formulas, the stack lower bound and the use of informative versus uninformative input frequencies are mathematical consequences or tests of this constructed problem, not extrapolations to naturally occurring ensembles.

## Reproduction and checks

The generator used K=4 and 6, 16 independent data banks, nested training sizes of 16 and 64 labels, two input distributions and two label laws. Every experimental fit used the same observations for all supervised baselines, both native restarts and the conditional matching learner. The strong joint model also received the designated sampling prior and 128 unlabeled inputs. Exhaustive lexicographic empirical Brier selection over each finite family agreed with an independent maximum-weight perfect-matching dynamic program and its compiled native MAX/MIN program on all held-out supports. The distribution of a training label is Bernoulli with endpoint high or low, so the likelihood maximum and Brier minimum coincide for conditional-label-only inference here. [Independent native exact-matching implementation](https://github.com/damiankucharski/OKAPI/blob/76f96bc/experiments/matching_oracle/README.md)

An independent integer-likelihood evaluator reconstructed all 16,896 non-GP prediction fingerprints, 33,792 recorded losses and all 1152 paired intervals, and checked all 256 optimal native matching programs and their integer tie counts. It separately verified the 10,240 compilation checks. The largest discrepancy from an exact-integer posterior weight was below 1.8×10⁻¹⁵. A separate native replay recovered all 512 GP trajectories, all 307,200 fitness calls, all 1024 fit/validation winners and 7680 GP or oracle prediction fingerprints. The 15,360 corresponding metric values agreed exactly. [Native replay](stock-verified.json) · [Exact reconstruction](posterior-verified.json) · [Raw scores, models, rankings and checksums](runs.csv.gz) · [Native search costs](gp.csv.gz)

Use OKAPI 76f96bc64b8bd2295aef783c47153f72678501e9 as a checkout on PYTHONPATH, including its experiments directory.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-76f96bc"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_matching_panel.py tests/test_matching_posterior.py
python -m fusionbench.eval.matching_compression --out /tmp/matching-compression --workers 2
python docs/ensemble-study/search-regret/matching-compression/verify_stock.py /tmp/matching-compression --workers 2
python docs/ensemble-study/search-regret/matching-compression/verify_posterior.py /tmp/matching-compression
```
