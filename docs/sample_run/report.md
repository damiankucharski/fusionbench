# fusionbench characterization report

## Regret to oracle (lower is better)

| scenario                                                       |   best_single |   op_close |   op_far |   op_max |   op_min |   simple_average |   stacking_logit |   stacking_lr |   top_5 |   val_weighted |
|:---------------------------------------------------------------|--------------:|-----------:|---------:|---------:|---------:|-----------------:|-----------------:|--------------:|--------:|---------------:|
| committee_k10_accuracy0.75                                     |        0.2354 |     0.4348 |   0.0417 |   0.0417 |   0.0417 |           0.0013 |          -0      |        0.0013 |  0.0517 |         0.0011 |
| compound_acc_core0.68_acc_rescue0.9_acc_safe0.9                |        0.0568 |     0.3426 |   0.1241 |   0.1241 |   0.1241 |           0.096  |           0.0383 |        0.0078 |  0.096  |         0.0496 |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 |        0.1373 |     0.2713 |   0.0241 |   0.0241 |   0.0241 |           0.0828 |          -0.0001 |        0.0007 |  0.0873 |         0.0826 |
| deadweight_n_good4_n_bad6_accuracy0.82                         |        0.1471 |     0.4679 |   0.0133 |   0.0133 |   0.0133 |           0.0056 |           0.0004 |        0.0029 |  0.0035 |         0.0031 |
| gatedexperts_acc_select0.92_acc_expert0.9                      |        0.1684 |     0.2582 |   0.2296 |   0.2296 |   0.2296 |           0.1661 |           0.1378 |        0.136  |  0.1661 |         0.1396 |
| hardrescue_n_detect3_accuracy0.85                              |        0.2427 |     0.2784 |   0.2116 |   0.2116 |   0.2116 |           0.2108 |           0.0581 |       -0.0003 |  0.2108 |         0.2109 |
| hardveto_acc_detect0.8_acc_veto0.88                            |        0.1817 |     0.2806 |   0.1484 |   0.1484 |   0.1484 |           0.1479 |           0.0432 |        0.0063 |  0.1479 |         0.1476 |
| miscalibration_k6_accuracy0.78                                 |        0.4798 |     0.5709 |   0.2971 |   0.4504 |   0.1381 |           0.2732 |           0.0001 |        0.0147 |  0.2873 |         0.2732 |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  |        0.2266 |     0.2485 |   0.1052 |   0.1441 |   0.1052 |           0.0225 |           0.0036 |        0.0232 |  0.0908 |         0.0221 |
| needle_n_good3_n_bad97_accuracy0.8                             |        0.1258 |     0.4321 |   0.0088 |   0.0088 |   0.0088 |           0.0823 |           0.0002 |        0.0036 |  0.0056 |         0.0045 |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  |        0.0077 |     0.0044 |   0.0015 |   0.0009 |   0.0015 |          -0      |           0.0004 |        0.0003 |  0.0009 |        -0      |
| twopathways_accuracy0.9                                        |        0.2167 |     0.2716 |   0.1827 |   0.1827 |   0.1827 |           0.0882 |           0.0964 |        0.0743 |  0.0882 |         0.0885 |
| weakestlink_m4_accuracy0.82                                    |        0.2167 |     0.2404 |   0.1973 |   0.1973 |   0.1973 |           0.1953 |           0.0485 |        0.0069 |  0.1953 |         0.1951 |

## Per-scenario winners

| scenario                                                       | primary_metric   | best_method    |   best_regret | expected_winners               | as_expected   |
|:---------------------------------------------------------------|:-----------------|:---------------|--------------:|:-------------------------------|:--------------|
| committee_k10_accuracy0.75                                     | macro_f1         | stacking_logit |       -0      | simple_average, stacking_logit | True          |
| deadweight_n_good4_n_bad6_accuracy0.82                         | macro_f1         | stacking_logit |        0.0004 | stacking_logit, top_5          | True          |
| needle_n_good3_n_bad97_accuracy0.8                             | macro_f1         | stacking_logit |        0.0002 | stacking_logit, top_5          | True          |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  | macro_f1         | val_weighted   |       -0      | best_single, simple_average    | False         |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 | macro_f1         | stacking_logit |       -0.0001 | stacking_logit                 | True          |
| hardveto_acc_detect0.8_acc_veto0.88                            | macro_f1         | stacking_lr    |        0.0063 | op_min                         | False         |
| hardrescue_n_detect3_accuracy0.85                              | macro_f1         | stacking_lr    |       -0.0003 | op_max                         | False         |
| weakestlink_m4_accuracy0.82                                    | macro_f1         | stacking_lr    |        0.0069 | op_min                         | False         |
| twopathways_accuracy0.9                                        | macro_f1         | stacking_lr    |        0.0743 |                                |               |
| compound_acc_core0.68_acc_rescue0.9_acc_safe0.9                | macro_f1         | stacking_lr    |        0.0078 |                                |               |
| gatedexperts_acc_select0.92_acc_expert0.9                      | macro_f1         | stacking_lr    |        0.136  |                                |               |
| miscalibration_k6_accuracy0.78                                 | log_loss         | stacking_logit |        0.0001 | stacking_logit, stacking_lr    | True          |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  | macro_f1         | stacking_logit |        0.0036 | simple_average, stacking_lr    | False         |
