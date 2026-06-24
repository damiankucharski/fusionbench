# fusionbench characterization report

## Regret to oracle (lower is better)

| scenario                                                       |   best_single |   okapi |   op_close |   op_far |   op_max |   op_min |   simple_average |   stacking_logit |   stacking_lr |   top_5 |   val_weighted |
|:---------------------------------------------------------------|--------------:|--------:|-----------:|---------:|---------:|---------:|-----------------:|-----------------:|--------------:|--------:|---------------:|
| committee_k10_accuracy0.75                                     |        0.2313 |  0.1214 |     0.4401 |   0.0442 |   0.5877 |   0.5849 |           0.0017 |           0.0002 |        0.0018 |  0.0524 |         0.0015 |
| compound_acc_core0.74_acc_rescue0.92_acc_safe0.92              |        0.0767 |  0.0128 |     0.3625 |   0.1561 |   0.5274 |   0.3827 |           0.1306 |           0.0548 |        0.0086 |  0.1306 |         0.0721 |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 |        0.1353 |  0.0045 |     0.2721 |   0.0208 |   0.2803 |   0.2778 |           0.079  |          -0.0006 |        0.001  |  0.0845 |         0.0789 |
| deadweight_n_good4_n_bad6_accuracy0.82                         |        0.1499 |  0.0039 |     0.4705 |   0.0144 |   0.624  |   0.6258 |           0.0073 |           0.0003 |        0.0032 |  0.0047 |         0.0037 |
| gatedexperts_acc_select0.92_acc_expert0.9                      |        0.1666 |  0.1299 |     0.2644 |   0.2257 |   0.3352 |   0.3358 |           0.1688 |           0.1321 |        0.1338 |  0.1688 |         0.1383 |
| hardrescue_n_detect3_accuracy0.9                               |        0.3087 | -0.0003 |     0.3439 |   0.2818 |  -0.0003 |   0.5957 |           0.2831 |           0.0823 |        0.0016 |  0.2831 |         0.2828 |
| hardveto_acc_detect0.9_acc_veto0.93                            |        0.2293 |  0.0047 |     0.2975 |   0.2013 |   0.4685 |   0.0469 |           0.2011 |           0.0501 |        0.0043 |  0.2011 |         0.2011 |
| miscalibration_k6_accuracy0.78                                 |        0.3893 |  0.289  |     0.5711 |   0.2938 |   0.7715 |   0.7786 |           0.2731 |           0.0002 |        0.0109 |  0.2532 |         0.2726 |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  |        0.23   |  0.1122 |     0.2563 |   0.1107 |   0.1465 |   0.1107 |           0.0235 |           0.0045 |        0.0222 |  0.0905 |         0.022  |
| needle_n_good3_n_bad97_accuracy0.8                             |        0.1317 |  0.0078 |     0.4225 |   0.0083 |   0.5931 |   0.5956 |           0.0847 |           0.001  |        0.0056 |  0.0063 |         0.0053 |
| parity_accuracy0.92                                            |        0.3531 |  0.1962 |     0.3546 |   0.3525 |   0.1979 |   0.5755 |           0.3525 |           0.3706 |        0.2717 |  0.3525 |         0.3517 |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  |        0.0062 |  0.001  |     0.0036 |   0.0009 |   0.0013 |   0.0009 |           0.0001 |           0      |        0.0012 |  0.0003 |         0.0001 |
| twopathways_accuracy0.9                                        |        0.217  |  0.0915 |     0.2658 |   0.183  |   0.4422 |   0.368  |           0.0865 |           0.0967 |        0.0728 |  0.0865 |         0.0875 |
| weakestlink_m4_accuracy0.9                                     |        0.362  |  0.0889 |     0.3808 |   0.3464 |   0.6762 |  -0.0115 |           0.3465 |           0.0958 |        0.0023 |  0.3465 |         0.3463 |

## Per-scenario winners

| scenario                                                       | primary_metric   | best_method    |   best_regret | expected_winners                   | as_expected   |
|:---------------------------------------------------------------|:-----------------|:---------------|--------------:|:-----------------------------------|:--------------|
| committee_k10_accuracy0.75                                     | macro_f1         | stacking_logit |        0.0002 | simple_average, stacking_logit     | True          |
| deadweight_n_good4_n_bad6_accuracy0.82                         | macro_f1         | stacking_logit |        0.0003 | okapi, stacking_logit, top_5       | True          |
| needle_n_good3_n_bad97_accuracy0.8                             | macro_f1         | stacking_logit |        0.001  | okapi, stacking_logit, top_5       | True          |
| saturated_k6_n_classes4_accuracy0.95_corr0.95                  | macro_f1         | stacking_logit |        0      | best_single, okapi, simple_average | False         |
| correlatedredundancy_n_block8_n_diverse2_accuracy0.75_corr0.85 | macro_f1         | stacking_logit |       -0.0006 | okapi, stacking_logit              | True          |
| hardveto_acc_detect0.9_acc_veto0.93                            | macro_f1         | stacking_lr    |        0.0043 | okapi, op_min                      | False         |
| hardrescue_n_detect3_accuracy0.9                               | macro_f1         | op_max         |       -0.0003 | okapi, op_max                      | True          |
| weakestlink_m4_accuracy0.9                                     | macro_f1         | op_min         |       -0.0115 | okapi, op_min                      | True          |
| twopathways_accuracy0.9                                        | macro_f1         | stacking_lr    |        0.0728 | okapi                              | False         |
| parity_accuracy0.92                                            | macro_f1         | okapi          |        0.1962 |                                    |               |
| compound_acc_core0.74_acc_rescue0.92_acc_safe0.92              | macro_f1         | stacking_lr    |        0.0086 | okapi                              | False         |
| gatedexperts_acc_select0.92_acc_expert0.9                      | macro_f1         | okapi          |        0.1299 |                                    |               |
| miscalibration_k6_accuracy0.78                                 | log_loss         | stacking_logit |        0.0002 | stacking_logit, stacking_lr        | True          |
| multiclasscommittee_k8_n_classes5_accuracy0.6                  | macro_f1         | stacking_logit |        0.0045 | simple_average, stacking_lr        | False         |
