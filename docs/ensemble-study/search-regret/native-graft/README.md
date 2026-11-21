# Giving evolution a fitted threshold gate as a parent

On fresh data with a real FAR switch, offering a fitted gate-bearing tree to the breeding population and running 300 more ordinary evaluations gave **0.00539 excess Brier**. Ordinary native GP reached **0.01545** when it received the same budget, with a full fitness call charged for every threshold setting the proposal examined. The preregistered difference was **−0.01006** (paired 95% interval −0.01772 to −0.00241). Keeping the identical proposed tree available only for final selection produced **0.01791**. Thus the benefit in this panel came principally from what GP could find after admission. [Prospective protocol and no-op rule](../native-graft-protocol.md)

The fitted single-gate family reached 0.00159 and logistic stacking reached 0.02085. Relative to the empirical single-gate reference, the active proposal closed about **73%** of the test gap left by cost-matched native GP, using the ratio of differences in these means. The paired advantage over ordinary GP varied by seed: 11 improved, six tied and seven worsened. Eight of twelve dataset-level means improved. Excess Brier is the mean squared difference from the known label probability. Observed Brier additionally reflects Bernoulli outcome noise.

| Trained and tested on the same law | Switch Brier | Switch excess | Logistic-control Brier | Logistic-control excess |
|---|---:|---:|---:|---:|
| Known conditional probability | 0.13355 | 0 | 0.16780 | 0 |
| All-provider average | 0.22472 | 0.09053 | 0.22678 | 0.05786 |
| Ranked-prefix Top-N | 0.17416 | 0.04029 | 0.17939 | 0.01118 |
| Regularized logistic stack | 0.15441 | 0.02085 | 0.16977 | 0.00199 |
| Best fitted one-gate family member | 0.13507 | 0.00159 | 0.18603 | 0.01926 |
| Original GP fit champion | 0.15415 | 0.02050 | 0.18090 | 0.01308 |
| Full graft, immediate fit selection | 0.15158 | 0.01791 | 0.18028 | 0.01241 |
| Full graft in selection archive only, 300 ordinary calls | 0.15158 | 0.01791 | 0.17953 | 0.01162 |
| Full graft offered to active population, 300 ordinary calls | 0.13886 | 0.00539 | 0.17740 | 0.00967 |
| Plain GP matched to full native forwards plus 300 | 0.15407 | 0.02041 | 0.17959 | 0.01173 |
| Plain GP also charged every threshold candidate, plus 300 | 0.14911 | 0.01545 | 0.17854 | 0.01060 |

Each regime has twelve new independent datasets and two native GP seeds per dataset. Fit, validation and test sizes are 512, 256 and 8192. All three inexpensive references saw the same six anonymous provider scores and the same fit labels. Top-N fit one Brier ranking and evaluated every prefix, selecting two providers on every bank in both regimes. Pairwise intervals average restarts within datasets before a two-sided t interval. The favorable generator deliberately contains a threshold switch. On the logistic control, the primary contrast was −0.00094 [−0.00253, +0.00065], and a logistic stack achieved the smallest fitted loss.

## Where the improvement appeared

The original native fit champion was frozen at the first complete generation after 1200 objective calls. A seed-selected terminal kept its provider and received one new CLOSE or FAR operator and one new provider leaf. Existing assignments and parameters stayed fixed during the proposal. The generator's roles, cutoff and test outcomes were unavailable for learning. The full arm tried all five other providers, both directions and fit the cutoff against the squared error of the whole tree.

| Immediate proposal on the switch | Full native training forwards | Threshold candidates, mean | Switch-test excess |
|---|---:|---:|---:|
| First shuffled provider, fixed 0.5 | 3 | 0 | 0.02059 |
| All five, fixed 0.5 | 11 | 0 | 0.01965 |
| First shuffled provider, fitted cutoff | 9 | 444 | 0.02073 |
| All five, fitted cutoff | 41 | 2216 | 0.01791 |

With a fitted cutoff, trying all five providers improved immediate excess relative to trying just the first by 0.00282 [0.00093, 0.00471]. Against the immediate cost-matched GP, the full graft was **worse** by 0.00247 [−0.00221, +0.00715]. After 300 more calls with the proposed tree eligible to breed, the excess difference versus the archive-only arm was −0.01252 [−0.01696, −0.00808]. An archive champion independently selected by validation also favored the active arm over candidate-matched native GP, by 0.01125 [0.00302, 0.01949].

The full proposal improved training loss in 18/24 switch-trained pairs. Native survivor selection admitted 15 and rejected three. No new tree was submitted in the remaining six. All 15 admissions changed the subsequent ordered path of objective evaluations. None of the nine rejected or unchanged interventions changed that path relative to the archive-only control.

- In **eight** pairs the starting champion contained **no** threshold. Seven of the eight offers were admitted. The original, immediately proposed, archive-only, active and candidate-matched champions had mean switch excesses of 0.03374, 0.02755, 0.02755, **0.00678** and **0.02070**, respectively.
- In the other **16** pairs, which started with a threshold, the corresponding means were 0.01387, 0.01309, 0.01309, **0.00469** and **0.01283**. The randomly selected terminal happened to be one of the signal inputs in 22/24 pairs overall because it was sampled from an already trained tree. These groups depend on prior search outcomes.

A post hoc inspection of the saved trees found that 14/15 final winners after admission contained the exact direction and cutoff of the offered gate. That combination was absent from all 15 starting winners. Only two final programs were the unchanged offer. Across all 24 runs, the final active champion was a simple two-provider gate in 13 cases, compared with five for archive-only and seven for candidate-matched GP. In one example (data seed 3901, GP seed 293) a 34-node starting tree was ultimately replaced by a 3-node tree with the offered cutoff. Its switch excess fell from 0.0299 to 0.0014. These observations are consistent with GP reusing the fitted operation in smaller trees. The recorded programs establish parameter retention, while full crossover ancestry was unobserved. Admission can also change parent ordering and subsequent random draws. [Raw trees, costs and fingerprints](runs.csv.gz)

## When the switch weakens

| Switch-trained and then frozen | Full switch excess | Half switch excess | Switch abolished excess |
|---|---:|---:|---:|
| All-provider average | 0.09053 | 0.06893 | 0.05786 |
| Ranked-prefix Top-N | 0.04029 | 0.02047 | 0.01118 |
| Logistic stack | 0.02085 | 0.00735 | 0.00438 |
| Fitted single gate | 0.00159 | 0.00604 | 0.02101 |
| Active graft and 300 ordinary calls | 0.00539 | 0.00814 | 0.02141 |
| Same graft in the archive only | 0.01791 | 0.01384 | 0.02030 |
| Candidate-matched plain GP | 0.01545 | 0.01278 | 0.02064 |

The primary comparison's active-minus-plain difference became −0.00464 [−0.00815, −0.00114] at half strength and +0.00077 [−0.00137, +0.00291] with switching abolished. The frozen logistic stack performed best among these fitted models after the switch vanished. All shifted labels use the same feature rows and paired uniform draws. [Means](means.csv) · [All prespecified paired contrasts](paired.csv)

## Design rationale and reproduction

Kommenda and colleagues found that local numerical constant fitting improved many of 94 symbolic-regression problems. Their study motivated separating the effect of fitting a numerical threshold from selecting a structure capable of using it. Our earlier experiment found an actual structural obstruction: 13/24 training champions contained no threshold to refit. The targeted graft and archive-only control are our hypotheses about this search process. The cited regression experiments did not evaluate classification ensembles or this proposal. [Kommenda et al., 2020](https://doi.org/10.1007/s10710-019-09371-3) · [Literature index and BibTeX key kommenda2020parameter](../../literature/README.md) · [Research entry, 2026-11-21](../../literature/2026-11-21.md) · [Preceding coordinate experiment](../native-refinement/README.md)

On the switch banks GP reached the shared generation boundary after an average 1207.71 native evaluations. The full proposal cost another 41 forwards and examined an average 2216.17 threshold settings. After 300 more ordinary calls, the two corresponding costs were **1548.71** actual training forwards and **3764.88** full-candidate credits. Ordinary GP was run to both exact limits, including partial generations. The larger accounting charges a complete fitness call even when the threshold sweep incrementally updated its objective. The active and archive-only arms used the same proposed tree and exactly 300 subsequent ordinary evaluations. [Native proposal and tests](https://github.com/damiankucharski/OKAPI/blob/5062ee4/experiments/threshold_graft/README.md)

An independent interpreter reconstructed all 18,480 observed and conditional-excess Brier scores exactly. All 192 local proposal neighborhoods agreed with exhaustive evaluation over the dyadic threshold support, and all 24 single-gate fit certificates agreed with direct enumeration. An independent native batched replay, using 253,152 real objective calls, matched all 672 recorded path fingerprints and archive sizes, as well as all 1344 fit and validation champion programs and their training scores. A second full experimental run reproduced all 9240 raw rows and all aggregate estimates exactly apart from measured times. [Independent prediction and graft audit](verified.json) · [Ordinary native trajectory verifier](verify_trajectories.py)

Run the study from the FusionBench checkout with OKAPI commit 5062ee43b773ef4d6e7c8d0286d7d9be3b03beda.

```bash
export PYTHONPATH=".:/absolute/path/to/okapi-5062ee4"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q tests/test_gate_proposal_gp.py tests/test_gate_events.py
python -m fusionbench.eval.native_graft --out /tmp/native-graft --workers 2
python docs/ensemble-study/search-regret/native-graft/verify.py /tmp/native-graft
python docs/ensemble-study/search-regret/native-graft/verify_trajectories.py /tmp/native-graft
```
