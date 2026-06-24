# fusionbench characterization report

## Regret to oracle (lower is better)

| scenario                                                       |   best_single |   okapi |   op_close |   op_far |   op_max |   op_min |   simple_average |   stacking_logit |   stacking_lr |   top_5 |   val_weighted |
|:---------------------------------------------------------------|--------------:|--------:|-----------:|---------:|---------:|---------:|-----------------:|-----------------:|--------------:|--------:|---------------:|
| committee_k10_accuracy0.75                                     |        0.2448 |  0.0035 |     0.4298 |   0.0456 |   0.5955 |   0.5948 |           0.0016 |           0.0009 |        0.0026 |  0.0548 |         0.0016 |
| compound_acc_core0.78_acc_rescue0.9_acc_safe0.9_rep3           |        0.1749 |  0.0571 |     0.4248 |   0.2208 |   0.6313 |   0.4938 |           0.1407 |           0.0687 |        0.0534 |  0.1487 |         0.1034 |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 |        0.1331 |  0.0044 |     0.2635 |   0.0202 |   0.2819 |   0.2766 |           0.0807 |           0      |        0.003  |  0.0408 |         0.0791 |
| deadweight_n_good4_n_bad6_accuracy0.82                         |        0.1555 |  0.0016 |     0.4593 |   0.0136 |   0.6295 |   0.6255 |           0.0044 |          -0.0002 |        0.0014 |  0.0016 |         0.0026 |
| gatedexperts_acc_select0.92_acc_expert0.9                      |        0.1719 |  0.0916 |     0.2608 |   0.225  |   0.3319 |   0.3403 |           0.1678 |           0.1352 |        0.1388 |  0.1678 |         0.1388 |
| hardrescue_n_detect3_accuracy0.88_rep2                         |        0.3966 |  0.0378 |     0.4541 |   0.3613 |   0.1309 |   0.7418 |           0.358  |           0.0879 |        0.0426 |  0.3613 |         0.3574 |
| hardveto_acc_detect0.85_acc_veto0.88_rep3                      |        0.297  |  0.0394 |     0.4207 |   0.2388 |   0.6296 |   0.2259 |           0.2319 |           0.0523 |        0.0321 |  0.2398 |         0.2365 |
| miscalibration_k6_accuracy0.78                                 |        0.5308 |  0.299  |     0.5672 |   0.3034 |   0.7829 |   0.7818 |           0.2701 |          -0.0001 |        0.0131 |  0.2502 |         0.2688 |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  |        0.2215 |  0.0213 |     0.2465 |   0.0961 |   0.1429 |   0.0961 |           0.0214 |           0.0041 |        0.0258 |  0.0883 |         0.0228 |
| needle_n_good3_n_bad97_accuracy0.8                             |        0.1274 |  0.0061 |     0.424  |   0.009  |   0.5913 |   0.5934 |           0.0827 |           0.0046 |        0.0059 |  0.0065 |         0.0046 |
| parity_accuracy0.92_rep2                                       |        0.4514 |  0.254  |     0.4497 |   0.4574 |   0.316  |   0.7007 |           0.4546 |           0.4527 |        0.4806 |  0.4546 |         0.453  |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  |        0.0075 |  0.001  |     0.0044 |   0.0025 |   0.0007 |   0.0025 |          -0      |           0.0022 |        0.0025 |  0.0012 |        -0      |
| twopathways_accuracy0.9_rep3                                   |        0.3343 |  0.1782 |     0.4361 |   0.298  |   0.6417 |   0.5793 |           0.1755 |           0.1653 |        0.1528 |  0.2264 |         0.1744 |
| weakestlink_m4_accuracy0.85_rep2                               |        0.4227 |  0.0453 |     0.4572 |   0.4    |   0.7745 |   0.1507 |           0.3974 |           0.1076 |        0.054  |  0.3968 |         0.3965 |

## Per-scenario winners

| scenario                                                       | primary_metric   | best_method    |   best_regret | expected_winners                   | as_expected   |
|:---------------------------------------------------------------|:-----------------|:---------------|--------------:|:-----------------------------------|:--------------|
| committee_k10_accuracy0.75                                     | macro_f1         | stacking_logit |        0.0009 | simple_average, stacking_logit     | True          |
| deadweight_n_good4_n_bad6_accuracy0.82                         | macro_f1         | stacking_logit |       -0.0002 | okapi, stacking_logit, top_5       | True          |
| needle_n_good3_n_bad97_accuracy0.8                             | macro_f1         | val_weighted   |        0.0046 | okapi, stacking_logit, top_5       | False         |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  | macro_f1         | val_weighted   |       -0      | best_single, okapi, simple_average | False         |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 | macro_f1         | stacking_logit |        0      | okapi, stacking_logit              | True          |
| hardveto_acc_detect0.85_acc_veto0.88_rep3                      | macro_f1         | stacking_lr    |        0.0321 | okapi                              | False         |
| hardrescue_n_detect3_accuracy0.88_rep2                         | macro_f1         | okapi          |        0.0378 | okapi                              | True          |
| weakestlink_m4_accuracy0.85_rep2                               | macro_f1         | okapi          |        0.0453 | okapi                              | True          |
| twopathways_accuracy0.9_rep3                                   | macro_f1         | stacking_lr    |        0.1528 | okapi                              | False         |
| parity_accuracy0.92_rep2                                       | macro_f1         | okapi          |        0.254  |                                    |               |
| compound_acc_core0.78_acc_rescue0.9_acc_safe0.9_rep3           | macro_f1         | stacking_lr    |        0.0534 | okapi                              | False         |
| gatedexperts_acc_select0.92_acc_expert0.9                      | macro_f1         | okapi          |        0.0916 |                                    |               |
| miscalibration_k6_accuracy0.78                                 | log_loss         | stacking_logit |       -0.0001 | stacking_logit, stacking_lr        | True          |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  | macro_f1         | stacking_logit |        0.0041 | simple_average, stacking_lr        | False         |
