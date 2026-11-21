# Correct probability references for the two multiclass generators

The research branch now uses the exact posterior for the generated multiclass logit-normal observations in Saturated and MulticlassCommittee. In the opportunity-matched study's fresh IID cases, this reduced the **reference's own full Brier** from 0.19698 to 0.13590 on Saturated and from 0.67695 to 0.59569 on MulticlassCommittee. Archived competing methods produced exactly the same predictions, true-label Brier and decisions in all 2,460 affected rows. Some secondary distances to the former reference had a different interpretation. [Production fix and fixtures](https://github.com/damiankucharski/fusionbench/commit/a3db9f13b1a68ebb9bc96f03110c64b5a25f14a8) · [Per-bank scores of the probability reference](oracle-scores.csv) · [Verified row counts](verification.json)

## From class-conditional logits to probabilities

With a uniform class prior, each provider m observes logits equal to s times the one-hot true class plus Gaussian noise of variance one. The generator sets s through its function mc_strength. Noise is independent between class coordinates, with equicorrelation ρ across the K providers within each coordinate. MulticlassCommittee uses ρ=0. Saturated uses its configured shared-noise mixing coefficient, clipped to [0,1] by the generator.

Subtracting an arbitrary constant from all class logits of a provider preserves its softmax. The observable ratios therefore satisfy log(p_mc/p_mC)=z_mc−z_mC. For class c the row-mean logit has the same distribution for every class, and its Gaussian noise is independent of the centered contrasts. Gaussian discriminant analysis of the latter gives

$$
P(Y=c\mid p) = \operatorname{softmax}_c\!\left(
\frac{s}{1+(K-1)\rho}\sum_{m=1}^K\log p_{mc}\right).
$$

Indeed the K-by-K covariance times the all-ones vector is [1+(K−1)ρ] times that vector. All class means have the same Mahalanobis norm by class symmetry; only the linear discriminant remains. At ρ=1 every generated provider is identical, so the coefficient s/K recovers the single independent observation. The formula applies to this generator's common signal strengths, uniform prior and covariance. The helper retains positive softmax probabilities below 10⁻¹² when evaluating their log ratios.

For the default four-class Saturated the coefficient is **0.4248594060**. The previous code used 1/6 as a heuristic for six largely redundant providers. For the five-class MulticlassCommittee it is **0.3960583546**. The old unit coefficient implicitly assumed the class-dependent logit boost was one. All coefficients are positive, so rescaling the aggregate preserves the predicted class on these banks. The observed minimum input probability over all archived examples was 0.00010969. An independent evaluation of the multivariate normal densities on observable log ratios recovered the corrected probabilities within 8.6×10⁻¹⁵ on fresh default cases. [Independent density and native representation check](https://github.com/damiankucharski/OKAPI/blob/a3735e3/experiments/standard_point_control/multiclass_posterior.md)

Kahn's shared-covariance Gaussian derivation for two-class log odds motivates checking the role of signal strength and correlation. The displayed softmax formula follows from this generator's multiclass distribution by the calculation above. [Kahn, equations 5, 10 and 24](https://arxiv.org/pdf/1207.4144) · [Dated source review](../literature/2026-11-21.md) · [Literature index](../literature/README.md)

## Effects on reported studies

| Fixed report and fresh IID banks | MulticlassCommittee, former → corrected oracle Brier | Saturated, former → corrected oracle Brier |
|---|---:|---:|
| Native mutation on/off, eight banks per family | 0.69083 → 0.60421 | 0.20122 → 0.14103 |
| Opportunity-matched mutation, six banks per family | 0.67695 → 0.59569 | 0.19698 → 0.13590 |

The public reports on these experiments choose search winners with fit labels and compare IID Brier with actual test labels. Neither training nor testing in their primary comparisons uses q. For Saturated in the first report, however, the secondary point-on minus point-off IID squared distance to q changes from **+0.00273 to −0.00754** [−0.01168, −0.00339] at 1,200 calls. For the second report the corresponding additional suite-wide retype-minus-identity distance changes from −0.01159 to −0.01217 [−0.01774, −0.00661]. Its primary true-label Brier remains −0.01202. The revised squared distances and underlying reference fingerprints are supplied for every affected method, bank, split and selection rule. [All corrected raw diagnostics](corrected-rows.csv.gz) · [Corrected suite point-mutation contrasts](suite-pairs.csv) · [Corrected opportunity-matched aggregate](opportunity-suite-contrasts.csv)

The original output files remain readable at their historical commits. For the first report use [corrected method means](suite-methods.csv), [method spreads](suite-spread.csv) and [paired effects](suite-pairs.csv) for these two families' conditional distances. In the opportunity-matched report use [corrected means and spreads](opportunity-methods.csv), [paired effects](opportunity-paired.csv) and [fixed-suite contrasts](opportunity-suite-contrasts.csv). Per-row prediction hashes, scores against labels, fit/validation winners, oracle-derived class decisions on these archived rows and all other scenarios were preserved. The earlier sample report uses macro-F1 as its primary metric for both cases; its regret and winner tables concern hard decisions. Future evaluations of the oracle itself under proper probability scoring should use the corrected formula. [Sample report](../../sample_run/report.md) · [Archive impact in OKAPI](https://github.com/damiankucharski/OKAPI/blob/a3735e3/experiments/standard_point_control/multiclass_posterior.md)

Run the audit from the research checkout with one BLAS/OpenMP thread. It regenerates the 84 affected split groups, separately reconstructs baselines and all stored programs, checks the original q hash and squared distance before computing the corrected values. The simulation and all native search trajectories are reused.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/ensemble-study/multiclass-oracle/recompute.py /tmp/multiclass-oracle
```
