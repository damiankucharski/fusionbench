# Recovery under a standard Gaussian input

Let the rows of U be three orthonormal vectors u₁, u₂, u₃ in dimension d ≥ 3. Write Zᵢ = uᵢ·X for X distributed as N(0,I), sᵢ = sign Zᵢ and c = √(2/π). The signed target is

```math
Y=(s_1+s_2+s_3-s_1s_2s_3)/2.
```

For standard Gaussian Z, E[sign(Z)Z] = c and E[sign(Z)(Z³−3Z)] = −c. Independence gives m = E[YX] = (c/2)(u₁+u₂+u₃). Define the tensor T by Tᵢⱼₖ = E[Y(XᵢXⱼXₖ − Xᵢδⱼₖ − Xⱼδᵢₖ − Xₖδᵢⱼ)]. In the orthonormal coordinates containing the true axes, the only nonzero entries are

- Tᵢᵢᵢ = −c/2 for the three signal axes.
- Tᵢⱼₖ = −c³/2 when the indices are a permutation of (1,2,3).

Flattening T into a d by d² matrix gives three mutually orthogonal nonzero rows in these coordinates. The left singular space therefore recovers span(U). Its nonzero squared singular values all equal c²/4 + c⁶/2, so singular vectors alone have no preferred orientation inside this space.

Normalize m to e. Within the recovered subspace take any unit vector v perpendicular to e and put aᵢ = v·uᵢ. Their sum vanishes and the sum of their squares is one. Since Σaᵢ³ = 3a₁a₂a₃, contraction gives

```math
T[v,v,v] = -3c(1/2+c^2)a_1a_2a_3.
```

The product is maximal for the three permutations of (2,−1,−1)/√6. The corresponding unit vectors v are separated by 120 degrees in the plane perpendicular to e. Each hidden axis equals e/√3 + √(2/3)v. This identifies all oriented axes up to permutation using only the joint distribution of X and the final signed label.

For empirical m and T, first project the left side of the tensor matricization off the estimated e. Use the top two left singular vectors p,q as a basis of the residual plane. The cubic on v(θ) = p cos θ + q sin θ contains first and third harmonics. Finite samples introduce a first harmonic which vanishes at population. Extract the third-harmonic coefficients C = (T[p,p,p] − 3T[p,q,q])/4 and D = (3T[p,p,q] − T[q,q,q])/4. Three minima of this component occur at θ = (atan2(D,C)+π)/3 + 2πk/3. The rotation phase, not an arbitrary orientation of the singular vectors, carries the remaining information.

The model assumes centered isotropic Gaussian inputs, orthonormal axes and zero thresholds. Independent symmetric label noise with flip probability below one half scales m and T by the same positive factor at population. Finite-sample uncertainty can still increase substantially.
