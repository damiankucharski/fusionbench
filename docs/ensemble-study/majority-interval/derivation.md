# A continuum of compatible third votes

Let the unknown angle t range from 0 to A<π/2, with two other vote axes and the plane of t specified. Each labeled row with disagreeing shared votes imposes a signed linear inequality on tan(t). Intersecting these inequalities with the allowed range gives an interval; endpoints below refer to its closure. For a continuous Gaussian target the distinction at a single boundary direction has probability zero.

Define F(t) as the source probability that angle t and 0 produce different final labels. From the chi-tilted source model F(t) = 0.5 F_Student,k(−√k cot t). On any row the outcome can change at most once in the candidate interval. The source disagreement probability between any t and s therefore equals |F(t)−F(s)|. Conditional on true t, another angle s is compatible after n independent labels with probability (1−|F(t)−F(s)|)ⁿ. Integrating this probability gives

```math
E[U-L\mid t]=\int_0^A(1-|F(s)-F(t)|)^n\,ds.
```

For an isotropic target, D = (U−L)/(2π) is exactly the probability that some admissible rules disagree. The factor 1/2 comes from requiring the two fixed votes to disagree. On the remaining target inputs every compatible model gives the same answer. No additional labels are needed to certify those pointwise predictions if the assumed family is correct.

### Two forms of optimal prediction

The midpoint rule has worst compatible classification error D/2. No predictor can obtain less, since its errors against both endpoints must sum to at least their disagreement D. This is a single ordinary majority network using the midpoint direction for its third vote.

With binary squared error, one half on disputed rows and the common label elsewhere has risk D/4 against **every** consistent direction. For an arbitrary prediction p at a disputed row, the average squared loss against the two endpoints equals (p−0.5)²+0.25. Hence the minimax risk is exactly D/4 and the optimal probabilities can be implemented by averaging the two endpoints. A continuum of rules reduces to two endpoint votes plus the two fixed shared votes. This is an exact equality for the specified loss and target, without fitting to a teacher's outputs.

If the true angle has a uniform prior, the noiseless data leave a uniform posterior on the feasible interval because all compatible angles give the same likelihood to the observations. Given an ambiguous isotropic query, its switching angle is uniform in [L,U]. The posterior positive probability on such rows is uniform on [0,1]. Its posterior-averaged Brier is D/6. Its Brier at either endpoint is D/3, so minimizing posterior average loss and minimizing worst compatible loss have different solutions.

### How fast the interval shrinks

Let aₖ = Γ((k+1)/2) / (2√π Γ(k/2)). Differentiating the source CDF gives F′(t) = aₖ sin(t)^(k−1). At t = 0, F(s) is asymptotic to (aₖ/k)sᵏ, and consequently

```math
E[U-L\mid t=0]\sim\Gamma(1+1/k)\big(k/(a_k n)\big)^{1/k}.
```

For fixed interior t the leading asymptote is 2/(nF′(t)), one inter-observation spacing on either side. This pointwise result is not uniform in t. Near zero, F′ is tiny for large k, and an interior angle can remain in the slow finite-sample regime for a long time. The calculation assumes known plane and two known common votes, noiseless labels and the specified source/target laws. Reducing arbitrary unknown ensembles to this family is outside its scope.
