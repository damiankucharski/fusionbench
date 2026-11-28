# What the configuration rule can observe

Each fitted rule receives the source provider predictions, their labels, and source scene IDs when a generator exposes genuine groups. At prediction time it receives the new provider predictions. The runner writes generator settings, target labels and target event probabilities to its scoring records. These fields never enter a choice on a new bank. The separate retrospective portfolio oracle reads target labels and serves as a comparison.

| Bank | Independent draws per setting in R1 / R2 / R3 | What changes |
|---|---:|---|
| All 30 `standard-v6` scenarios | 4 / 3 / 6 | Registered scenarios and parameters, with 512 source and 512 test examples each |
| Eight paired rare-event settings | 8 / 8 / 16 | Toggle provider heterogeneity (0 or .65), shared residual error (0 or .65) and local specialist strength (0 or .85) independently |
| Two scarce-scene controls | 8 / 8 / 16 | Increase source coverage from 16 to 128 scenes under the earlier clustered law |
| Two stratified controls | 8 / 8 / 16 | Weak or strong specialization, 128 balanced source draws and exact test risk on a finite support |
| Two indistinguishable source laws | 0 / 0 / 16 | Fit data and test predictions identical, test label probability different |

Each round uses `610000 + 1000 * round` as its base seed. The offsets are 0 for the rare-event factorial, 100 for the two scarce-scene controls, 200 for the stratified controls, 400 for the standard suite and 800 for the indistinguishable pair. This gives 216, 186 and 404 conditions. The eight rare-event settings at the same seed share both splits' labels, latent events and specialist regions. Treating the settings as eight independent repeats would understate uncertainty.

In the factorial, each fit contains 16 scenes of eight examples and each test contains 256 new scenes. The specialist region covers 20% of scenes in expectation. The latent event logit and providers use the same slope of 1.8, which removes the previous deliberately unequal slopes. Monte Carlo variation, additive errors and shrinkage can still affect calibration. Provider columns have anonymous names. An independent seeded permutation assigns structural roles to columns once per draw and holds them fixed across splits and across the eight settings. For example, switching on the specialist with the other knobs fixed changes only that provider, and only in the specialist region. The registered suite and four historical controls retain their original laws.

## A target change that training diagnostics cannot identify

At each row draw three independent positive-class probabilities uniformly from .05 to .95. Both worlds draw their source labels with probability equal to the first provider, and expose the exact same source examples and labels. They also share every test prediction. In world A the test labels have probability $q_A=p_0$. In world B the probability is $q_B=(p_0+p_1+p_2)/3$. Using a shared uniform draw for the two test labels reduces the variance of comparisons while preserving each world's law.

For a shared forecast $p$, its expected **full-class** Brier loss above the Bayes forecast on a binary row is $2(p-q)^2$. Completing the square gives

$$
\frac{2(p-q_A)^2+2(p-q_B)^2}{2}
= 2\left(p-\frac{q_A+q_B}{2}\right)^2+\frac{(q_A-q_B)^2}{2}.
$$

The last term averaged over test rows is an exact lower bound for any rule given the same information in these two worlds. The midpoint attains it. Under the specified uniform generator its population value is 0.0225. This diagnoses a target-law ambiguity even for a rule given unlimited source data. It does not explain losses within a stationary population with few independent training scenes. Observed losses fluctuate with the test labels, whereas this identity applies to risks averaged over those labels.

After round 3, run `python docs/ensemble-study/autoconfig-three/witness_auto.py` from the checkout. It verifies matching source data, actions, and fitted and test forecast hashes across worlds, then writes per-seed and per-method observed and expected excess losses to `results/r3-witness-*.csv`. The retrospective portfolio oracle is excluded.
