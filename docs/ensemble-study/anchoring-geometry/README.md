# Why anchoring during tree selection can favor larger corrections

## The empirical question

Kucharski's dissertation used the prediction $(1-t)b+tT$, where $b$ is the prior, $T$ the tree and $t=1-\alpha$ the tree's share. At a prior weight of 0.85, STARCOP F1 was 0.7537 on validation and 0.6996 on test when evolution scored the mixture. Mixing the already selected tree gave 0.7497 and 0.7020. On MedMNISTv2 the corresponding test PR-AUC values were 0.8054 and 0.8056, against 0.8047 for the mean of all 30 providers. During anchored evolution the mean number of providers referenced by the correction fell from 3.0 to 2.2 as the prior weight rose from 0.50 to 0.85. The deployed mixture still required all 30. STARCOP instead used the six models selected by a validation-ranked prefix procedure. These numbers come from the published tables. The original prediction arrays were unavailable for individual-case analysis. [Dissertation, sections 5.7 and 6.3](https://bip.polsl.pl/wp-content/uploads/sites/4/2026/11/Damian-Kucharski_rozprawa-doktorska.pdf#page=124)

Our previous study traced the loss of an optimal, representable program during evolution. A one-standard-error penalty in search worsened suite-wide Brier. Those results motivated asking what property of a *candidate prediction* the training objective actually rewards. [Population study](../validation-curse/README.md)

## A finite-menu result for full-class Brier

Write $d=T-b$ and let $Y$ denote the one-hot label. On any one sample, define $H(T)=\operatorname{mean}\langle Y-b,d\rangle$ and $D(T)=\operatorname{mean}\|d\|^2$. The same algebra holds for population expectations. Expanding the square gives the reduction in loss from deploying a mixture.

$$
R(b)-R(b+t d)=2tH(T)-t^2D(T).
$$

For a common positive tree share, the winning tree maximizes $H(T)-tD(T)/2$. Let $T_s,T_t$ maximize this score on the same menu and sample for $0<s<t$. Write the winning inequality once for each share and add them. The alignment terms cancel, leaving

$$
(t-s)\big(D(T_s)-D(T_t)\big)\ge 0.
$$

**Reducing the tree share can only increase or preserve the squared distance of the selected raw tree from the prior on the selection sample.** The result allows ties and heterogeneous model quality. It applies to exact selection on a common candidate set. Independent GP runs can visit different programs. Their final raw trees need not obey the ordering. The theorem orders raw displacements. It leaves open the deployed displacement, the test loss and the optimal deployment strategy. Number of nodes or providers measures a different property.

Pointwise, if the true posterior is $q$ and the class can realize $T=b+(q-b)/t$ within the probability simplex, a population optimizer can exactly undo the mixture. If the direction reflects genuine signal, that compensation can improve test loss. If it reflects label noise, the same objective can reward a spurious correction. FusionBench already estimates the coefficient of a frozen direction with held-out Brier. The result above concerns how the direction itself changes when training with a fixed coefficient. [Existing implementation](../../../fusionbench/methods/shrinkage.py)

### Exact counterexample with five single-provider programs

Take constant binary providers predicting 0.1, 0.4, 0.5, 0.6 and 0.9. Their average is 0.5. With 12 positives among 20 validation cases, raw Brier selects 0.6. Applying a tree share of 0.25 afterwards deploys 0.525. Selection inside the same mixture picks 0.9 and deploys 0.6. If the true prevalence is 0.5, the respective excess full-class Brier losses are 0.00125 and 0.02000. If prevalence is 0.6 they are 0.01125 and zero. All candidates read one provider, so structural parsimony alone cannot distinguish these cases. Summing over every binomial validation count preserves the reversal: at prevalence 0.5 the expected excess is 0.00133 after selection and 0.01047 inside, whereas at prevalence 0.6 it is 0.01573 after selection and 0.01088 inside. [Exact enumeration](check_geometry.py) · [Fixed-menu experiment on the 24 standard scenarios](menu/README.md)

On 192 paired banks from all 24 standard FusionBench scenarios, switching selection to the mixture changed 122 winners. Every changed winner moved farther from the prior on the selection data. Both deployed mixtures used tree share 0.25. Their paired Brier difference was −0.00348 on selection and +0.00069 on the suite test (95% within-scenario resampling interval −0.00132 to +0.00276). The ordinary and anchored rules selected from the same single-provider programs, pair means and full mean. This diagnoses the objective even without repeated adaptive GP mutations. [Protocol, baselines and bank-level scores](menu/README.md)

## Predictions on the dissertation's metrics

Brier does not imply an F1 or average-precision ranking. These metrics yield their own testable predictions. For an individual positive-class score and a fixed prior weight of 0.85, every feasible mixture lies in the interval $[0.85b,\,0.85b+0.15]$.

* Under the strict STARCOP threshold of 0.5, two candidate trees can give different decisions only if $7/17<b\le 10/17$, approximately 0.412 to 0.588. Gains and losses in F1 must come from changes in TP, FP and FN within this band. The number of independent scenes contributing such changes matters even when the count of ambiguous pixels is large.
* For per-class PR-AUC based on average precision, examples whose prior scores differ by more than $3/17$, approximately 0.176, retain their strict order under every tree. A change in AP requires a changed ordering or tie involving opposite labels. Record the affected positive cases and negative cases in fit and test, preserving the original split and aggregation units.

For binary F1 there is also an exact comparison for two decisions at the **same** operating threshold. Write $P$ for the number of positive labels, $Q_S$ for the number of predicted positives and $A_S$ for the true positives under the incumbent. Then $F_S=2A_S/(P+Q_S)$. When the denominators are positive, the sign of the change in F1 is the sign of

$$
(A_T-A_S)-\tfrac12 F_S(Q_T-Q_S).
$$

This follows by cross-multiplying the two F1 scores. It applies to a pooled confusion table or separately to each unit whose F1 is reported. With multiple separately averaged F1 scores, aggregate the actual per-unit differences using their denominators. This identity distinguishes corrections that find more positives from changes that simply predict positive more often. Under a fixed feature design, $Q_T-Q_S$ is label-free, whereas the change in true positives needs labels only at changed decisions. Inferring population F1 would also require accounting for the uncertain baseline ratio and dependence between units.

These are necessary conditions. Raw predictions would allow a direct audit of whether selection exploited unstable exceptions or found corrections shared by independent units. No conclusion about that distinction follows from the aggregate F1 and PR-AUC tables.

## A candidate selection rule that deploys one tree

A proposed intervention compares each proposed child $T$ directly to an executable incumbent $S$ using current training labels. Its incentive is to find an improvement that survives uncertainty in the labels where the two trees disagree. The deployed child reads only its own leaves. A safety guarantee relative to an all-model mean cannot by itself supply a cheap fallback. An implementable comparison can start from a prespecified single-provider program and account for all other possible incumbents.

Here is a limited, falsifiable mathematical target. Condition on the base predictions at $n$ independent labeled units with one categorical outcome each. Let $d_i=T_i-S_i$, $r_i=\max_c d_{ic}-\min_c d_{ic}$. Let $\hat G$ be the empirical full-class Brier gain of $T$ over $S$, and $G_X$ its conditional expected gain on those same features. Assign weights $\pi_{S,T}$ summing to at most one over all candidate pairs before seeing their labels. A one-sided Hoeffding bound and a union bound give, simultaneously with probability at least $1-\delta$,

$$
G_X(S,T)\ \ge\ \hat G(S,T)
 - \frac{\sqrt{2\log(1/(\delta\pi_{S,T}))\,\sum_i r_i^2}}{n}.
$$

Only $2d_{i,Y_i}$ in the observed gain varies with the label at fixed features, and its range has width $2r_i$. This proves the displayed form. The cost of searching many pairs and the size of the prediction change both enter. With a fixed code for the grammar one may use a pair weight of $2^{-L(S,T)}$, where $L$ is the bit length of a prefix code for an executable pair. Mutation may use the labels adaptively because coverage is simultaneous over the declared programs. A fitted real-valued literal, selected provider subset, chosen reference or fitted postprocessor must also be covered. Paying only for the pairs actually visited after seeing labels breaks the argument. With repeated measurements from one independent group, each group's entire loss difference must be treated as a single bounded contribution with an appropriate range and weight. Treating its rows as independent can give false certainty. [Hoeffding, 1963](https://doi.org/10.1080/01621459.1963.10500830)

This result controls label noise at the observed features. Sampling new features, changing environments, F1 and average precision require additional analysis. The bound may be too conservative for real grammars even when its coverage is correct. A first implementable experiment is to apply the comparison to *replacement* of an incumbent, while allowing mutations with uncertain gain to continue exploring. Compare against raw fitness, plain size parsimony, and equal-budget survival of simple incumbents. Track false replacement when the incumbent is optimal, power with a real nonlinear advantage, test Brier and the number of providers actually invoked. Reject the intervention if it mostly preserves the starting tree and loses genuine improvements. [Exact size and power check on a finite family](certificate/README.md)

### A small-sample failure and a sharper calculation

At 16 labels the simultaneous Hoeffding lower bound selected the baseline on **every possible sample**, even when an alternative was the true posterior. We therefore assessed a sharper bound exploiting the exact two-point gain on this fixed binary grid. On each changed decision, a direction that agrees with the label gains 0.48 full Brier, while a disagreement costs 1.12. A positive true mean gain requires an agreement probability above 0.70. Conditional on these inputs, a Bernoulli relative-entropy bound on that probability can be inverted and corrected for searching all eight challengers. Independence suffices even when agreement probabilities differ across examples: the product of their moment generating functions is bounded by the corresponding binomial moment generating function at their average probability.

We summed over every labeled count pair. No sampling error is involved in the reported powers. With a true 90% agreement rate the sharper rule selected a beneficial candidate with probability **18.53%, 83.16% and >99.99%** at 16, 64 and 256 labels. The Hoeffding rule had power 0%, 54.15% and >99.99%. Under the null at 16 labels the sharper rule falsely promoted a program with probability 0.0061%, compared with 49.46% for raw empirical selection. The method was specified after the first rule's failure. At 16 labels it still missed over four fifths of the strong improvements. Extending this particular calculation to variable correction magnitudes, a larger encoded grammar, metrics such as AP, and independent deployment inputs remains open. [Enumeration, assumptions and full results](certificate/kl/README.md)

## How much predictive information can a sparse tree retain?

When the full posterior $q(P)=E[Y\mid P]$ is known, even an unrestricted predictor restricted to provider subset $A$ pays at least

$$
E\big\|q(P)-E[q(P)\mid P_A]\big\|^2
$$

in excess Brier over the full posterior. Conditional expectation is the least-squares projection onto predictions using those providers. In a generator with two independent fair bits as providers and their mean as the positive-class posterior, observing only one provider costs exactly 0.125 full-class Brier. If the second is a duplicate, that cost is zero. This is a population statement about a fixed subset. It distinguishes unavailable information from failure to optimize a representable sparse rule. The full mean is a useful diagnostic whenever the simulator makes it optimal. A universal requirement to match it using a strict subset is impossible. [Exact calculation](check_geometry.py)

## Relation to LeJEPA

Balestriero and LeCun motivate a target distribution of representations and regularize projected distributions while learning agreement across views. For tree fusion the transferable idea is to identify a measurable failure, derive a target from it, and state when that target must fail. Matching a marginal distribution of tree outputs to the distribution of the prior leaves the assignment of predictions to individual cases undetermined. Permuting all predictions across cases preserves the marginal while changing predictive accuracy. Equal-weight shrinkage for combined forecasts has a separate history. [LeJEPA](https://arxiv.org/abs/2511.08544) · [Diebold and Pauly, 1990](https://doi.org/10.1016/0169-2070(90)90028-A)
