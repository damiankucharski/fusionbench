# Conditional moments when a covariance spike is a true vote

Assume the oriented leading direction a is one of three orthogonal vote axes and its sign is independent of two remaining Gaussian coordinates Z₂, Z₃ and standard Gaussian nuisance. The leading direction may have a different marginal variance. The final signed label is Y = majority(sign(a·X), sign Z₂, sign Z₃). With three fair, independent signs, the leading vote agrees with Y with probability 3/4. A pure independent nuisance direction agrees with Y with probability 1/2 before any sample-based orientation.

Let M denote a contradiction between the oriented leading vote and Y. On M both remaining signs must equal Y. The event only filters signs. The independent magnitudes |Z₂| and |Z₃| retain their original half-normal laws. Project off a and let X⊥ denote the residual vector. Writing c = √(2/π) gives

```math
E[Y X_\perp\mid M] = c(u_2+u_3).
```

The raw conditional residual mean is zero by sign symmetry. Each residual axis has conditional variance one. The conditional cross-moment E[Z₂Z₃ | M] equals E[|Z₂|]E[|Z₃|] = c². Nuisance coordinates retain covariance I and zero cross-covariance with the two votes. The residual covariance therefore has eigenvalues 1+c² on e = (u₂+u₃)/√2, 1−c² on v = (u₂−u₃)/√2, and 1 in unrelated directions. After finding the orientation of e from the signed mean, the least-variance direction perpendicular to e yields v up to sign. The pair (e+v)/√2 and (e−v)/√2 recovers both axes up to permutation.

When the variance spike is instead an independent nuisance, projecting it out leaves a Gaussian problem with three orthogonal votes. A tilted direction containing both signal and nuisance does not satisfy either structural assumption. The empirical switch uses a variance ratio of 2 and agreement fraction 0.625, fixed before the experiment. These cutoffs are merely the tested finite-sample rule.
