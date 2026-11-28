# How much endpoint regret is unavoidable?

Consider one frozen prediction for each of the 16 possible patterns of four binary provider votes. Regional prevalence is 50% in both target environments. In the first environment the shortcut has 50% accuracy and in the second it has 90% accuracy. A forecaster that knows both laws but must deploy the same rule in both incurs at least **0.03568 full-class Brier** above the relevant Bayes score in one of them. The optimal rule attains this excess in both environments.

| Population criterion | Brier, inactive shortcut | Brier, active shortcut | Worst absolute Brier | Worst excess above each endpoint's Bayes score |
| --- | ---: | ---: | ---: | ---: |
| Separate Bayes forecasts with the environment known | 0.23217 | 0.10895 | 0.23217 | 0.00000 |
| One forecast minimizing worst absolute Brier | 0.23217 | 0.23217 | 0.23217 | 0.12322 |
| One forecast minimizing worst excess Brier | 0.26785 | 0.14462 | 0.26785 | 0.03568 |

Under the absolute criterion, the inactive world's Bayes forecast already attains the best possible worst-case score. It ignores the shortcut. The other three predictions and their relationship with the label have the same distribution in both worlds, so ignoring the shortcut gives identical risk in both. This leaves substantial information unused when the shortcut is active. The regret criterion credits how much **could** have been gained in that easier environment. [Exact risks and forecasts](results/endpoint-regret-oracle.json)

## Derivation

Let $m_j(x)$ be the probability of vote pattern $x$ in endpoint $j$, and let $q_j(x)$ be the corresponding probability of the positive class. For any shared probability forecast $p(x)$, the binary full-class Brier risk exceeds its endpoint-specific Bayes risk by

$$
E_j(p)=2\sum_x m_j(x)\big[p(x)-q_j(x)\big]^2.
$$

Assign weight $t$ to the active endpoint and $1-t$ to the inactive one. The weighted excess $(1-t)E_0(p)+tE_1(p)$ is minimized separately at each pattern by the posterior of the **joint mixture** of outcomes and reports,

$$
p_t(x)=\frac{(1-t)m_0(x)q_0(x)+t\,m_1(x)q_1(x)}
{(1-t)m_0(x)+t\,m_1(x)}.
$$

The pattern frequencies matter. A fixed scalar applied directly to the two endpoint posteriors would omit the evidence in the observed pattern about which endpoint generated it. In this generator, $p_t$ is also the Bayes forecast for activity $t$, where shortcut accuracy is $0.5+0.4t$.

As $t$ increases from zero to one, each pattern's forecast moves monotonically from $q_0(x)$ to $q_1(x)$. Hence $E_0(p_t)$ increases and $E_1(p_t)$ decreases. Bisection finds the equalizing value $t^*=0.54889774$, equivalent here to shortcut accuracy 71.96%. Any forecast's maximum excess is at least its weighted average excess at $t^*$. The forecast $p_{t^*}$ minimizes that average, and its two excesses are equal. It therefore attains the lower bound. This makes $t^*$ the least favorable prior for the regret criterion. Its numerical value comes from these two joint laws and the loss, independently of the frequency of the target environments in practice.

The same bound covers every intermediate activity in this generator at the same regional prevalence. For fixed $p$, risk is linear in the mixture weight. Bayes risk is the infimum over forecasts of those linear risks, hence concave. Their difference, excess risk, is convex in the weight and is bounded above by its worse endpoint. Changes in regional prevalence require a different calculation.

## Reproduction

Run this command from the FusionBench checkout.

~~~sh
PYTHONPATH=. python docs/ensemble-study/provider-use-ten/regret_oracle.py
~~~

The calculation enumerates the public generator's exact support. It verifies the mixture posterior against the generator at $t^*$ and directly scores both possible outcomes at each pattern. A second calculation completes the square in the weighted excess and gives the lower bound $2t^*(1-t^*)\sum_x m_0(x)m_1(x)[q_1(x)-q_0(x)]^2/[(1-t^*)m_0(x)+t^*m_1(x)]$. All checks agree within $2\times 10^{-15}$. The output includes both endpoint risks at 21 equally spaced weights, the equalizing rule, and its predictions on all 16 patterns. [Reproduction code](regret_oracle.py)
