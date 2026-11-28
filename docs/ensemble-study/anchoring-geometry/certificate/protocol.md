# Fixed-design certificate protocol

Freeze the following calculations before enumerating the binomial counts. There are two equally frequent contexts, with exactly n/2 independent labels from each. Analyze n = 16, 64 and 256, at Bernoulli probabilities (0.5, 0.5), (0.1, 0.9), (0.4, 0.6) and (0.3, 0.7). These represent the null, a strong representable alternative, overshooting the target and the boundary where a directional challenger ties the mean.

The anchor predicts 0.5 in both contexts. Freeze the eight other ordered probability pairs in the Cartesian grid {0.1, 0.5, 0.9}². Give the eight challengers weight 1/8 each and use a one-sided family error target of 0.05. Evaluate each challenger at its stated probabilities, with full binary Brier loss summed across both class coordinates.

The raw rule maximizes empirical gain over the anchor. The certified rule maximizes empirical gain minus its simultaneous Hoeffding charge. Both rules include the anchor with score zero, require a strictly positive winning challenger score and resolve challenger ties by lexicographic probability-pair order. Treat the fixed anchor as the reference for all comparisons. Enumerate every pair of positive-label counts, using its exact product-binomial mass.

Report the expected true gain of the selected prediction, the probability of promoting a challenger with zero or negative true gain, the probability of strictly harmful promotion, and the probability of promoting any strictly beneficial challenger. The last quantity measures power only when a beneficial challenger exists. Measure all gains against the anchor using the same fixed two-context proportions. Use the results to assess certification under these specified designs and outcomes.
