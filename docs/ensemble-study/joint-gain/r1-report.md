# Round 1. Blocking a fold winner that loses on pooled fit data

On 48 fresh banks with a perfect provider and 24 fit scenes, joint terminal selection lowered conditional audit excess Brier by **0.002250** relative to ordinary fold selection on the same evolved trees. The signed difference was **−0.002250** (95% paired bank interval **−0.004590 to −0.000459**). Seven banks improved and 41 tied. With 192 fit scenes, the difference was −0.000011 (−0.000094 to +0.000057), with one improvement and two regressions. Each scene contained eight observations. [Protocol](protocol.md) · [Paired comparisons](results/r1-contrasts.csv)

## What changed on the seven banks

The initial anchor was the best singleton or all-provider mean on pooled fit Brier. Ordinary fold selection ranked the final survivors and anchor by their improvement over the anchor on at least three of four scene folds. In seven banks its winner lost to the anchor on pooled fit Brier. A veto sent all seven back to the anchor, which happened to be the perfect provider in each case. Joint reranking chose the anchor in three of those banks and another qualifying survivor in four. All four alternatives improved pooled and fold fit scores over the anchor, and all four increased audit loss relative to it. [Decisions by bank](results/r1-events.csv)

| Fold-trained terminal policy | Mean fit Brier | Mean audit excess | Exact predictions / 48 | Mean providers used |
| --- | ---: | ---: | ---: | ---: |
| Anchor alone | 0.218448 | 0.001952 | 46 | 1.15 |
| Ordinary fold winner | 0.216439 | 0.009853 | 15 | 2.56 |
| Veto with anchor fallback | 0.215952 | 0.007407 | 22 | 2.29 |
| Rerank eligible finalists | 0.215751 | 0.007603 | 18 | 2.46 |

Veto alone changed excess Brier by **−0.002446** against ordinary fold selection (−0.004872 to −0.000569). Reranking changed it by **+0.000196** against veto (+0.000002 to +0.000505). On bank 288106 the alternative improved pooled fit Brier over the anchor by 0.000109 and the three-fold gain by 0.000269. Deploying it cost 0.005846 audit excess relative to fallback. On bank 288116 with 192 scenes, reranking also increased excess by 0.000914 relative to the original fold winner. [Fit and audit scores](results/r1-scores.csv) · [Bank events](results/r1-events.csv)

## Other searches and references

With equally competent providers, reranking the fold-trained finalists changed audit excess relative to ordinary fold selection by −0.000555 (−0.001211 to −0.000077) at 24 scenes and −0.000269 (−0.000653 to −0.000005) at 192 scenes. On pooled-trained paths, the joint and fold decisions agreed on all 48 banks for both equal-provider sizes and the smaller perfect-provider condition. At 192 scenes with a perfect provider two pooled-path banks improved and 46 tied. Across both paths, laws and sizes, 20 ordinary winners were blocked. A qualifying alternative replaced the veto fallback in 13 of these cases. All 13 had higher audit excess than fallback. These are paired decisions across 48 data banks. [Paired comparisons](results/r1-contrasts.csv) · [Bank events](results/r1-events.csv)

In the primary condition, ordinary unprotected GP attained mean excess 0.007669, ranked-prefix Top-N 0.007484, the arithmetic mean 0.019685, regularized native-pattern OKAPI 0.024892 and probability logistic stacking 0.030486. Joint deployment on the fold-trained path differed from ordinary GP by −0.000066 (−0.002693 to +0.002763). The result supports a change to the fold-trained terminal decision in this strong-anchor condition. Its measured advantage came from blocking replacements, and the initially selected anchor had the lowest average risk. [Method levels](results/r1-summary.csv) · [Paired comparisons](results/r1-contrasts.csv)

## Computation

Each seed used checkpoint choices from an independent historical calibration block and fresh audit data from 512 eight-row scenes. Each fit had two actual 16-generation searches. The pooled path used 51,120 native objective evaluations across 192 fits, and the fold path used 51,089. Terminal policies reused the same scored menu within a path and added no model evaluations during selection. The complete run took 159.26 seconds on two single-threaded CPU workers. All 2,688 decisions and 6,528 generation records are archived. The intervals resample the 48 paired bank differences 10,000 times with seed 97826. [Runner](round1.py) · [Manifest](results/r1-run.json) · [Generation records](results/r1-trajectories.csv.gz)
