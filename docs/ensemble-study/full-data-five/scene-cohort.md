# How much evidence is in 128 fusion examples?

The scene generator represents a minority class observed repeatedly within independent groups. A scene has a shared latent signal. Individual rows add variation to it. Positive outcomes have a 9.90% marginal probability in the population. Eight base providers observe noisy versions of the same signal and have equal population competence. For each provider, 85% of its residual logit variance is shared by rows from the same scene. This percentage describes provider noise; label outcomes have a separate dependence on the shared signal. We fixed checkpoint zero for every provider before drawing any labels. [Generator and parameters](../../../fusionbench/scenarios/checkpoint_bank.py) · [Bank construction](banks_full.py)

Every training bank contains 128 labeled rows. One allocation takes eight observations from each of 16 scenes. The other takes one observation from each of 128 scenes. The first row of each of the first 16 scenes occurs in both training designs. Both use exactly the same 256 new test scenes, each contributing eight rows. The source and test use the same generating parameters and independent random streams. The effective amount of information depends on which quantity is being estimated and how strongly it varies between scenes. There is no single effective sample size for Brier, macro-F1 and average precision.

A scene is a synthetic independent group. Persistent errors within a scene and low event frequency approximate two plausible challenges in remote-sensing segmentation. They do not reproduce a STARCOP sensor, methane-plume geometry, MedMNIST images or either original pool of base models. Only the separate stratified-pool experiment enforces identical class and amplitude proportions. The class balance of a finite set of 16 or 128 sampled scenes fluctuates even with a fixed underlying law.

## Event counts in the 32 confirmation draws

The following counts use the draws assigned to round five. They were summarized after the policy was locked. Each common audit appears once. A scene is counted as having zero positives if all of its sampled labels are negative. For the one-row allocation this is the outcome of observing just one row of a scene, even if an unobserved row would have been positive.

| Sample | Positive rows across draws | Positive rows per draw, min–max | Sampled scenes with zero positives | Zero-positive scenes per draw, min–max |
| --- | ---: | ---: | ---: | ---: |
| Fit, 16 scenes × 8 rows | 388 / 4,096 (9.47%) | 6–21 / 128 | 289 / 512 (56.45%) | 5–12 / 16 |
| Fit, 128 scenes × 1 row | 438 / 4,096 (10.69%) | 6–23 / 128 | 3,658 / 4,096 (89.31%) | 105–122 / 128 |
| Shared audit, 256 scenes × 8 rows | 6,609 / 65,536 (10.08%) | 179–256 / 2,048 | 4,642 / 8,192 (56.67%) | 122–159 / 256 |

In the 16-scene fits, positive labels came from as few as four and as many as eleven scenes. Sampling eight rows from each of those scenes exposes variation within the scene, but also repeats a large component of a provider's error. Sampling one observation per scene changes how many independent provider-error patterns enter evolution at the same label budget. [Per-draw counts](results/round5-scene-counts.csv) · [Reproduction script](scene_counts.py) · [Policy lock](https://github.com/damiankucharski/fusionbench/commit/229131d972f5f2e0bdf69f3ab45ffb95059ddbba)
