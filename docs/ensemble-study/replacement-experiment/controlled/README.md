# Can a useful nonlinear correction replace a known incumbent?

The KL rule kept the optimal arithmetic mean on all 48 null banks. On 48 matched signal banks with the same provider inputs, it also kept the mean in 47/48 open-breeding searches and all 48 strict-breeding searches. Handing the exact nonlinear solution directly to the test exposed a separate limitation: raw confirmation accepted it on **48/48** fresh blocks, while KL accepted it on **0/48**. Each block contained 128 labels. On these particular blocks, with twelve proposal slots and the frozen error allocation, even the most favorable possible labels could not certify this solution. [Frozen protocol](protocol.md) · [Direct comparisons](results/power.csv)

The input is three anonymous provider scores, each either 1/8 or 7/8. All eight combinations are equally likely. The null conditional probability is the arithmetic mean. The signal probability is min(C, max(A, B)), with the role of C randomly reassigned per bank. Each bank shares the provider matrices, underlying label uniforms and native search seed across the two laws. Every run starts with the mean and three native provider trees. Six generations offer twelve children apiece, using 128 discovery labels and six fresh blocks of 128 independent confirmation labels. The ordinary reference selects on discovery labels alone. Each policy makes 76 discovery evaluations including the founders. Fresh labeled tests use 2,048 additional examples; exact population Brier averages over the eight corners and all possible test labels. [Shared engine and gate](../protocol.md) · [Input generation](inputs.py)

## Deployment and safety

| Policy | Null exact Brier | Null banks with a harmful move | Signal exact Brier | Signal test Brier | Signal banks finishing better than the mean |
|---|---:|---:|---:|---:|---:|
| Ordinary discovery | .41336 | 24/48 | .32197 | .32106 | 43/48 |
| Open breeding, raw confirmation | .41283 | 41/48 | .32358 | .32332 | 41/48 |
| Open breeding, KL confirmation | .40625 | 0/48 | .35807 | .35825 | 1/48 |
| Strict breeding, KL confirmation | .40625 | 0/48 | .35938 | .35965 | 0/48 |
| Starting mean | .40625 | — | .35938 | .35965 | — |
| Generating probability | .40625 | — | .21875 | — | — |

The table counts **ever** adopting a worse function under the null. Raw open search sometimes repaired that error, ending above optimal risk in 22/48 banks. Holding the open proposal path fixed, KL improved final expected null Brier by .00658 relative to raw confirmation (paired 95% t interval .00316–.01000). On the nonlinear signal the same switch worsened expected Brier by .03450 (.02667–.04232). The primary strict-minus-open KL contrast on the signal was **+.00130** (−.00132 to +.00392). The policies deployed equivalent functions in 47 banks and the open policy performed better in one. [All 384 deployments](results/search.csv) · [Paired contrasts](results/paired.csv) · [Counts](results/counts.csv)

With 128 discovery labels, arithmetic averaging, Brier-ranked Top-N and probability logistic stacking had signal Brier **.35938, .35938 and .26917**. With access to all 896 development labels, the same three scored **.35938, .35938 and .26091**. The logistic stack outperformed all four GP policies on this law. On the null, the discovery-only controls scored .40625, .40918 and .41992; the full-development controls scored .40625, .40625 and .40906. Top-N considered all three ranked prefixes separately in every bank. [Held-out and population baselines](results/reference-means.csv) · [Rankings and selected prefixes](results/baselines.csv.gz)

## Creation, breeding and promotion

A genuine native tree computes the signal with two sequential updates, MAX(A,B) followed by MIN with C. A single native update has exact minimum excess full Brier **3/32**, achieved by weighted averaging with weights (1/6,1/6,2/3) on (A,B,C). On the two-level inputs, every MIN/MAX/threshold alternative with any repeated operands is a min, max or projection. Independent corner enumeration and native execution verify both the numerical bound and the exact two-update program for all six role permutations. [Proof and native checker](verify_laws.py) · [Certificate](law-certificate.json)

- **Discovery.** Across 72 realized proposal slots per run, no arm generated the exact signal rule. The three open policies shared the exact same proposal and survival trajectory in every bank. Each offered some improvement over the starting mean in 47/48 signal banks; strict breeding did so in 48/48.
- **Functional complexity.** Open breeding produced a program with exact risk below the best possible one-update program in 10/48 banks; strict breeding did so in 1/48. Comparing each run's best actually offered program, strict-minus-open excess was +.00983 (.00402–.01565). This is an archive diagnostic that uses the true conditional law after fitting. Ordinary, raw open, certified open and certified strict deployed a program below the one-update bound in 10, 8, 1 and 0 banks respectively. [One-update comparison](results/one-update.csv)
- **Ancestry.** In all ten open-KL banks with a proposal below the one-update bound, such a proposal had a parent or earlier ancestor that failed confirmation. In six, one of its recorded ancestors computed MAX(A,B), MIN(A,C) or MIN(B,C). MIN(A,C) and MIN(B,C) individually tie the starting mean in population risk; MAX(A,B) is worse. The recorded edges establish that a parent was selected for variation. A two-parent edge cannot determine which parent supplied a particular operation. [Ancestry by bank](results/ancestry.csv) · [All birth records with native programs and corner predictions](results/births.jsonl.gz)
- **Direct promotion.** The planted tree improves true Brier by exactly 9/64 (.140625). In a menu with this tree and eleven copies of the mean, its average observed gain was .13700 on the signal. The average KL lower bound was −.14373. The highest possible lower bound across these 48 input blocks, even assigning the best possible label on every row, was −.00596. A one-offer menu relaxed the cap, yet still accepted on 0/48 blocks. An independent calculation gives an average single-offer acceptance probability of .0692% under the signal, conditional on these specific inputs. [All direct witness decisions](results/direct-witness.csv) · [Independent conditional power calculation](verify_results.py#L177) · [All-block maxima and conditional probabilities](results/verification.json)

Under the stated signal law, 128 discovery labels also contain enough information to identify the role C if a learner is given the correct three-template family. Each incorrect choice of C differs from the true template on two of the eight corners. On these rows the two conditional Bernoulli laws have probabilities 1/8 and 7/8, so their Bhattacharyya coefficient (the sum of the square roots of the two probability products) is √7/4. For one observation the coefficient including the other six corners is 3/4 + √7/16. Markov's inequality on the square root of the alternative-to-true likelihood ratio bounds the probability of preferring an incorrect template. A union bound over the two alternatives gives at most 2(3/4 + √7/16)^128 = 0.0000243. This supplemental calculation assumes knowledge of the grammar and endpoints; native search was given neither role labels nor this restricted candidate menu.

The eight-corner population permits exact judgments of predictive improvement; the live gate sees only observed labels. Its formal simultaneous guarantee concerns expected differences on the fresh block's fixed inputs. The trial describes this binary law, this error allocation and this budget. All inference still requires the providers referenced by the deployed fusion tree.

## Reproduction

With FusionBench and the matching native OKAPI revision on the Python import path, run from the FusionBench repository root:

```bash
python docs/ensemble-study/replacement-experiment/controlled/verify_laws.py
python docs/ensemble-study/replacement-experiment/controlled/references.py
python docs/ensemble-study/replacement-experiment/controlled/run.py --workers 2
python docs/ensemble-study/replacement-experiment/controlled/summarize.py
python docs/ensemble-study/replacement-experiment/controlled/verify_results.py
```

The output includes all 29,184 native birth records and all 2,304 generation snapshots, including rejected and duplicate proposals. A replay removing an unused observation-only confirmation calculation from the ordinary arm reproduced all 384 trajectory digests and final risks. [Generation decisions](results/generations.jsonl.gz) · [Numerical means and costs](results/means.csv)

An independent evaluator reconstructed all 29,184 native programs from their genotypes, checked 20,736 paired confirmation bounds, and recalculated all 384 final risks, 18 paired intervals, 576 baseline fits and 384 direct-witness decisions. Three raw promotions had exact empirical gain zero when evaluated with rational arithmetic: floating-point gains of 4×10⁻¹⁸ to 8×10⁻¹⁷ made them pass the implemented strict-positive comparison. The frozen results retain them. None of the KL promotions had this discrepancy. The final 384-search replay ran in **9.4 s** on two CPU workers. The interval from protocol commit (10:15:00 UTC) to the completed independent result (10:20:55 UTC) was **5 min 55 s**, including implementation, earlier runs and verification. [Verifier](verify_results.py) · [Numerical results including the three ties](results/verification.json)
