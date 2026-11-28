# Independent confirmation on all 24 standard scenarios

Penalizing uncertain improvements during evolution increased held-out Brier on this fixed suite. The incumbent's mean Brier was 0.19603 on the independent source audit, and the challenger scored 0.19782. Their paired difference was **+0.00179**, with a within-scenario bank-resampling interval of **+0.00028 to +0.00330**. The penalty changed the GP traversal on 140 of 144 banks and changed the chosen family on 27. On the audit the challenger won 50 banks, tied nine and lost 85. It improved seven scenario means; its largest improvement was on native_switch (−0.00625) and its largest deterioration was on parity (+0.02532). These estimates describe repeated draws of these 24 generators.

| Method | Same-law source audit | Suite test | Half-row provider swap |
|---|---:|---:|---:|
| Arithmetic mean | 0.36167 | 0.37853 | 0.39680 |
| Brier-ranked Top-N | 0.32273 | 0.33962 | 0.37215 |
| Probability logistic stacking, C=1 | 0.22859 | 0.24545 | 0.28200 |
| Logit logistic stacking, C=1 | 0.23151 | 0.24836 | 0.28953 |
| Historical native, 640 development rows | **0.19603** | 0.21408 | 0.26659 |
| Paired-penalty native, 640 development rows | 0.19782 | 0.21606 | 0.26963 |
| Historical native, 448 development rows | 0.20189 | 0.21974 | 0.26791 |
| Raw outer selection, 448 fit + 192 selection | 0.19771 | **0.21322** | **0.26178** |
| Guarded outer selection, same four trees | 0.19975 | 0.21485 | 0.26293 |

Scores are full-class Brier. Each scenario contributes its average across six fresh banks with identical draws and search seeds for paired arms. The suite-test hook changes environment proportions for environment_shortcut. The provider-swap test adds a controlled corruption to those same suite-test rows. Compared with historical native, the training penalty raised suite-test loss by 0.00198 (interval 0.00042 to 0.00354) and swap-test loss by 0.00305 (0.00102 to 0.00513). [All method means and accuracies](method-means.csv) · [Paired differences and intervals](summary.csv) · [Scenario-level differences](scenario-summary.csv)

The same frozen discovery winner showed mean validation optimism relative to averaging of 0.00482 for the incumbent and 0.00417 for the challenger. The paired change was −0.00065 (interval −0.00177 to +0.00039), leaving a reduction uncertain. All 144 winners in each arm appeared better than averaging on the internal validation rows. Four in each arm scored worse than averaging on the independent source audit. The deployed procedures subsequently refit their heads; the optimism comparison uses the frozen trees on both samples. [Bank-level differences](bank-summary.csv)

The secondary matched outer comparison held 192 rows back from discovery, feature screening, GP nomination and all head fits. Both rules could deploy the same four unchanged trees. The raw validation winner was the mean on four banks. The simultaneous guard retained the mean on those four and reverted 15 further winners to the mean. Every one of those 15 raw winners had lower independent source-audit loss than the mean. Consequently the guard increased audit loss by 0.00203 (interval 0.00123 to 0.00292). On the provider-swap target, four reversions improved loss and eleven worsened it. Limiting the ordinary incumbent to the 448 inner rows cost 0.00586 audit Brier before any outer rule was applied. The matched comparison measures the protection afforded by the final guard, while the full-budget comparison also includes the opportunity cost of the 192 withheld rows.

The full-budget comparison uses the same 600 native GP scoring visits per search and the same 448/192 internal split. History-dependent feature screening and head coordinate fits led to averages of 5,255 total objective calls for the incumbent and 5,308 for the challenger. Zero penalty reproduced the historical GP path, validation choice, objective counts and test predictions exactly on binary and multiclass integration tests. The complete three-worker confirmation took 94.55 seconds on CPU. [Frozen design](../round2-protocol.md) · [Bank and selection metadata](banks.json) · [Per-bank raw scores](scores.csv) · [Timing](run.json)
