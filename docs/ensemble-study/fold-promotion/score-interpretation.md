# What the three-of-four gain measures

For a candidate and anchor fixed independently of scoring data, suppose the
four equal-sized folds give independent gain estimates
$G_j=\mu+\sigma Z_j$, where the $Z_j$ are standard normal. Positive gain means
lower Brier than the anchor. Pooled gain is the arithmetic mean, with expectation
$\mu$ and population standard error $\sigma/2$. The fold score is $G_{(2)}$,
the second smallest gain.

Write $\Phi$ and $\phi$ for the standard normal distribution and density. For
$Z_{(2)}$ to lie near $z$, one other draw must lie below it and two above it.
There are four choices for the draw near $z$ and three for the draw below.
Integrating $12z\phi(z)\Phi(z)[1-\Phi(z)]^2$ gives $E[Z_{(2)}]=-0.29701138$.
Thus the expected fold score is **0.59402276 pooled standard errors below the
true mean gain**. This is an average over repeated independent draws. With gains
$(-.08,.01,.02,.03)$ the fold score is $.01$ and the pooled score is $-.005$.
With $(-.03,-.02,-.01,.08)$ the fold score is $-.02$ and the pooled score is
$.005$. Three modest gains can outweigh one large loss in the fold decision.

Under any independent, continuous, symmetric null centered at zero, a fixed
candidate strictly beats its anchor on the fold score with probability
$[\binom{4}{3}+\binom{4}{4}]/16=5/16$ (**31.25%**). A positive score requires
at least three positive gains. Independent fair signs suffice for this count
even when the fold variances differ. Ties or dependent signs can change it.
In the [study](protocol.md), the search picks
an anchor on the fit labels and repeatedly evaluates and selects trees on the
same folds. The 31.25% and the expected offset describe a fixed comparison.
The false-promotion frequency of the full adaptive search requires separate
measurement.

```python
from math import comb, exp, inf, pi, sqrt
from scipy.integrate import quad
from scipy.special import ndtr

def integrand(z):
    phi = exp(-z * z / 2) / sqrt(2 * pi)
    return 12 * z * phi * ndtr(z) * ndtr(-z) ** 2

d = quad(integrand, -inf, inf, epsabs=1e-12, epsrel=1e-12)[0]
print(d, 2 * d, (comb(4, 3) + comb(4, 4)) / 16)
```
