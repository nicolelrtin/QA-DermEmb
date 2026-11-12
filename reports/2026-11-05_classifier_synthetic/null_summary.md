# Classifier results

**Synthetic answers** (effect = 0.0): the labels are the real manifest labels, the answers are simulated with planted informative questions (qaemb/synthetic.py). Numbers here test the code, not the method.

1000 images, 446 questions (artefact questions left out). Each model was scored by 5 repeats of 5-fold cross-validation, stratified on the 10 subtypes; hyperparameters were chosen by 3-fold cross-validation inside each training set, so no score comes from an image the model or its tuning saw.

How to read the scores:
- **AUROC** (2-class): the chance that a random malignant image gets a higher P(malignant) than a random benign one. 0.5 is chance, 1 is perfect.
- **Specificity at 95% sensitivity** (2-class): the share of benign images correctly called benign when the threshold is set to catch 95% of malignant ones.
- **Balanced accuracy** (3- and 10-class): the average, over classes, of the share of that class predicted correctly. Chance is 1 / number of classes (0.33 for 3 classes, 0.10 for 10).
- **Macro-F1**: the average over classes of the F1 score, which balances missed and wrongly assigned images.
- Each cell is the mean ± standard deviation over the 25 test folds.

## Task sizes

| task | images fitted | classes (images per class) |
|---|---|---|
| s1_2c | 800 | benign (500); malignant (300) |
| s1_3c | 1000 | benign (500); indeterminate (200); malignant (300) |
| flat_10c | 1000 | benign fibro-histiocytic (100); benign melanocytic (100); benign vascular (100); flat melanotic pigmentation (100); inflammatory or infectious (100); indeterminate epidermal (100); indeterminate melanocytic (100); malignant adnexal follicular (100); malignant epidermal (100); malignant melanocytic (100) |
| s2_benign | 500 | benign fibro-histiocytic (100); benign melanocytic (100); benign vascular (100); flat melanotic pigmentation (100); inflammatory or infectious (100) |
| s2_malignant | 300 | malignant adnexal follicular (100); malignant epidermal (100); malignant melanocytic (100) |
| s2_indeterminate | 200 | indeterminate epidermal (100); indeterminate melanocytic (100) |

## Validity check: no signal

No question was made informative (effect = 0), so every score should sit at chance: AUROC 0.5, balanced accuracy 1 / number of classes. Anything clearly above chance would mean information leaking from test images into training or tuning.

## Results: P(yes) (`orig`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.517 ± 0.033 | 0.537 ± 0.043 | 0.527 ± 0.031 | 0.532 ± 0.043   | 0.529 ± 0.035       | 0.533 ± 0.040 |
| s1_3c            | balanced accuracy              | 0.328 ± 0.021 | 0.343 ± 0.034 | 0.324 ± 0.022 | 0.336 ± 0.027   | 0.336 ± 0.025       | 0.324 ± 0.028 |
| flat_10c         | balanced accuracy              | 0.102 ± 0.009 | 0.102 ± 0.016 | 0.096 ± 0.011 | 0.098 ± 0.017   | 0.115 ± 0.022       | 0.095 ± 0.016 |
| s2_benign        | balanced accuracy              | 0.201 ± 0.004 | 0.198 ± 0.038 | 0.192 ± 0.027 | 0.220 ± 0.042   | 0.215 ± 0.036       | 0.195 ± 0.024 |
| s2_malignant     | balanced accuracy              | 0.333 ± 0.000 | 0.306 ± 0.064 | 0.324 ± 0.034 | 0.324 ± 0.050   | 0.293 ± 0.055       | 0.298 ± 0.059 |
| s2_indeterminate | balanced accuracy              | 0.518 ± 0.041 | 0.593 ± 0.069 | 0.536 ± 0.058 | 0.575 ± 0.068   | 0.527 ± 0.066       | 0.585 ± 0.067 |
| s1_2c            | specificity at 95% sensitivity | 0.028 ± 0.039 | 0.078 ± 0.046 | 0.055 ± 0.052 | 0.094 ± 0.050   | 0.086 ± 0.046       | 0.075 ± 0.038 |
| s1_2c            | macro F1                       | 0.397 ± 0.110 | 0.521 ± 0.037 | 0.463 ± 0.110 | 0.515 ± 0.034   | 0.515 ± 0.036       | 0.510 ± 0.036 |
| s1_3c            | macro F1                       | 0.212 ± 0.073 | 0.340 ± 0.032 | 0.231 ± 0.060 | 0.326 ± 0.024   | 0.334 ± 0.026       | 0.320 ± 0.027 |
| flat_10c         | macro F1                       | 0.028 ± 0.014 | 0.098 ± 0.016 | 0.045 ± 0.033 | 0.096 ± 0.017   | 0.113 ± 0.022       | 0.093 ± 0.016 |
| s2_benign        | macro F1                       | 0.068 ± 0.008 | 0.195 ± 0.038 | 0.145 ± 0.043 | 0.218 ± 0.041   | 0.210 ± 0.037       | 0.193 ± 0.025 |
| s2_malignant     | macro F1                       | 0.167 ± 0.000 | 0.304 ± 0.063 | 0.235 ± 0.074 | 0.321 ± 0.049   | 0.290 ± 0.054       | 0.296 ± 0.059 |
| s2_indeterminate | macro F1                       | 0.423 ± 0.109 | 0.591 ± 0.070 | 0.488 ± 0.112 | 0.576 ± 0.068   | 0.522 ± 0.066       | 0.582 ± 0.068 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.096 |                    0.093 |                    0.105 |           0.087 |                  0.617 |                                    0.242 |                          0.68  |
| gradient boosting |                        0.115 |                    0.101 |                    0.118 |           0.101 |                  0.608 |                                    0.257 |                          0.676 |
| lasso (L1)        |                        0.102 |                    0.097 |                    0.103 |           0.087 |                  0.641 |                                    0.272 |                          0.71  |
| ridge (L2)        |                        0.102 |                    0.109 |                    0.119 |           0.107 |                  0.63  |                                    0.294 |                          0.707 |
| MLP               |                        0.095 |                    0.094 |                    0.109 |           0.093 |                  0.656 |                                    0.272 |                          0.723 |
| random forest     |                        0.098 |                    0.108 |                    0.116 |           0.104 |                  0.659 |                                    0.318 |                          0.739 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.492 |                       0.501 |
| gradient boosting |                     0.479 |                       0.483 |
| lasso (L1)        |                     0.499 |                       0.501 |
| ridge (L2)        |                     0.486 |                       0.51  |
| MLP               |                     0.489 |                       0.524 |
| random forest     |                     0.494 |                       0.504 |

## Results: Directional Score (`dir`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.512 ± 0.039 | 0.545 ± 0.041 | 0.520 ± 0.030 | 0.527 ± 0.036   | 0.512 ± 0.041       | 0.543 ± 0.041 |
| s1_3c            | balanced accuracy              | 0.329 ± 0.018 | 0.342 ± 0.030 | 0.328 ± 0.010 | 0.355 ± 0.035   | 0.342 ± 0.031       | 0.332 ± 0.027 |
| flat_10c         | balanced accuracy              | 0.099 ± 0.006 | 0.111 ± 0.017 | 0.096 ± 0.015 | 0.101 ± 0.020   | 0.109 ± 0.026       | 0.107 ± 0.018 |
| s2_benign        | balanced accuracy              | 0.200 ± 0.000 | 0.205 ± 0.041 | 0.183 ± 0.020 | 0.214 ± 0.033   | 0.202 ± 0.038       | 0.198 ± 0.035 |
| s2_malignant     | balanced accuracy              | 0.333 ± 0.000 | 0.327 ± 0.055 | 0.308 ± 0.045 | 0.307 ± 0.053   | 0.311 ± 0.058       | 0.317 ± 0.050 |
| s2_indeterminate | balanced accuracy              | 0.499 ± 0.033 | 0.586 ± 0.063 | 0.527 ± 0.048 | 0.561 ± 0.084   | 0.540 ± 0.066       | 0.572 ± 0.059 |
| s1_2c            | specificity at 95% sensitivity | 0.032 ± 0.051 | 0.080 ± 0.039 | 0.047 ± 0.053 | 0.079 ± 0.037   | 0.074 ± 0.044       | 0.076 ± 0.049 |
| s1_2c            | macro F1                       | 0.402 ± 0.112 | 0.521 ± 0.036 | 0.449 ± 0.106 | 0.499 ± 0.054   | 0.506 ± 0.036       | 0.518 ± 0.032 |
| s1_3c            | macro F1                       | 0.205 ± 0.072 | 0.339 ± 0.029 | 0.232 ± 0.068 | 0.343 ± 0.033   | 0.339 ± 0.033       | 0.328 ± 0.025 |
| flat_10c         | macro F1                       | 0.022 ± 0.007 | 0.108 ± 0.017 | 0.050 ± 0.029 | 0.099 ± 0.020   | 0.107 ± 0.027       | 0.106 ± 0.019 |
| s2_benign        | macro F1                       | 0.067 ± 0.000 | 0.202 ± 0.042 | 0.110 ± 0.038 | 0.212 ± 0.033   | 0.200 ± 0.040       | 0.196 ± 0.034 |
| s2_malignant     | macro F1                       | 0.167 ± 0.000 | 0.326 ± 0.055 | 0.223 ± 0.069 | 0.304 ± 0.054   | 0.307 ± 0.059       | 0.315 ± 0.050 |
| s2_indeterminate | macro F1                       | 0.398 ± 0.088 | 0.584 ± 0.063 | 0.478 ± 0.103 | 0.557 ± 0.086   | 0.535 ± 0.067       | 0.569 ± 0.060 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.096 |                    0.096 |                    0.105 |           0.092 |                  0.622 |                                    0.256 |                          0.688 |
| gradient boosting |                        0.109 |                    0.106 |                    0.127 |           0.105 |                  0.606 |                                    0.268 |                          0.677 |
| lasso (L1)        |                        0.099 |                    0.101 |                    0.099 |           0.088 |                  0.644 |                                    0.285 |                          0.716 |
| ridge (L2)        |                        0.111 |                    0.105 |                    0.117 |           0.103 |                  0.631 |                                    0.284 |                          0.705 |
| MLP               |                        0.107 |                    0.091 |                    0.103 |           0.09  |                  0.648 |                                    0.259 |                          0.713 |
| random forest     |                        0.101 |                    0.109 |                    0.114 |           0.104 |                  0.643 |                                    0.305 |                          0.721 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.487 |                       0.501 |
| gradient boosting |                     0.477 |                       0.478 |
| lasso (L1)        |                     0.498 |                       0.499 |
| ridge (L2)        |                     0.479 |                       0.504 |
| MLP               |                     0.478 |                       0.521 |
| random forest     |                     0.493 |                       0.505 |

## Results: Coherence Score (`coh`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.523 ± 0.041 | 0.550 ± 0.042 | 0.534 ± 0.035 | 0.523 ± 0.029   | 0.507 ± 0.031       | 0.552 ± 0.044 |
| s1_3c            | balanced accuracy              | 0.328 ± 0.016 | 0.338 ± 0.030 | 0.328 ± 0.017 | 0.334 ± 0.039   | 0.339 ± 0.025       | 0.329 ± 0.022 |
| flat_10c         | balanced accuracy              | 0.097 ± 0.008 | 0.107 ± 0.015 | 0.098 ± 0.010 | 0.112 ± 0.021   | 0.113 ± 0.020       | 0.101 ± 0.017 |
| s2_benign        | balanced accuracy              | 0.200 ± 0.000 | 0.200 ± 0.033 | 0.186 ± 0.026 | 0.218 ± 0.039   | 0.229 ± 0.048       | 0.192 ± 0.024 |
| s2_malignant     | balanced accuracy              | 0.332 ± 0.007 | 0.316 ± 0.068 | 0.315 ± 0.050 | 0.309 ± 0.052   | 0.348 ± 0.057       | 0.315 ± 0.057 |
| s2_indeterminate | balanced accuracy              | 0.520 ± 0.053 | 0.596 ± 0.073 | 0.528 ± 0.056 | 0.564 ± 0.063   | 0.531 ± 0.083       | 0.570 ± 0.072 |
| s1_2c            | specificity at 95% sensitivity | 0.043 ± 0.061 | 0.087 ± 0.047 | 0.067 ± 0.054 | 0.082 ± 0.044   | 0.066 ± 0.032       | 0.082 ± 0.047 |
| s1_2c            | macro F1                       | 0.421 ± 0.116 | 0.524 ± 0.041 | 0.465 ± 0.110 | 0.509 ± 0.037   | 0.509 ± 0.035       | 0.524 ± 0.038 |
| s1_3c            | macro F1                       | 0.216 ± 0.072 | 0.335 ± 0.028 | 0.246 ± 0.067 | 0.323 ± 0.036   | 0.338 ± 0.026       | 0.325 ± 0.019 |
| flat_10c         | macro F1                       | 0.022 ± 0.008 | 0.103 ± 0.014 | 0.048 ± 0.036 | 0.111 ± 0.020   | 0.112 ± 0.022       | 0.100 ± 0.017 |
| s2_benign        | macro F1                       | 0.067 ± 0.000 | 0.198 ± 0.034 | 0.109 ± 0.038 | 0.214 ± 0.038   | 0.227 ± 0.049       | 0.190 ± 0.024 |
| s2_malignant     | macro F1                       | 0.171 ± 0.024 | 0.314 ± 0.069 | 0.276 ± 0.072 | 0.307 ± 0.051   | 0.345 ± 0.058       | 0.312 ± 0.056 |
| s2_indeterminate | macro F1                       | 0.418 ± 0.116 | 0.592 ± 0.074 | 0.491 ± 0.097 | 0.562 ± 0.061   | 0.526 ± 0.080       | 0.566 ± 0.072 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.098 |                    0.094 |                    0.103 |           0.089 |                  0.627 |                                    0.254 |                          0.692 |
| gradient boosting |                        0.113 |                    0.113 |                    0.128 |           0.113 |                  0.608 |                                    0.29  |                          0.686 |
| lasso (L1)        |                        0.097 |                    0.099 |                    0.104 |           0.087 |                  0.646 |                                    0.283 |                          0.717 |
| ridge (L2)        |                        0.107 |                    0.103 |                    0.118 |           0.1   |                  0.634 |                                    0.281 |                          0.706 |
| MLP               |                        0.101 |                    0.09  |                    0.104 |           0.088 |                  0.648 |                                    0.256 |                          0.712 |
| random forest     |                        0.112 |                    0.098 |                    0.115 |           0.094 |                  0.662 |                                    0.291 |                          0.734 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.49  |                       0.504 |
| gradient boosting |                     0.476 |                       0.48  |
| lasso (L1)        |                     0.497 |                       0.501 |
| ridge (L2)        |                     0.478 |                       0.501 |
| MLP               |                     0.476 |                       0.514 |
| random forest     |                     0.489 |                       0.499 |

## Most important questions (P(yes))

Ranked by the mean SHAP rank over all models (1 = most important). *Ridge coef* is the standardised weight (positive = pushes towards the class named in the task; for multiclass tasks, the largest |weight| over classes). *Lasso kept* is the number of the outer fits in which the lasso kept the question.

**s1_2c**

| question_id   | question                                                                                                        |   ridge coef |   lasso kept |   mean SHAP rank | planted as    |
|:--------------|:----------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------|
| DQ423         | Is the lesion border low in contrast with the surrounding skin?                                                 |       -0.022 |           10 |            1.167 | uninformative |
| DQ449         | Can the lesion be described by plain clinical concepts such as asymmetry, irregular border or multiple colours? |        0.018 |            5 |            2.833 | uninformative |
| DQ218         | Does the visible skin appear to be Fitzpatrick type III–IV?                                                     |        0.02  |            9 |            4.5   | uninformative |
| DQ064         | Are vessels arranged in clusters?                                                                               |       -0.018 |            6 |            4.667 | indeterminate |
| DQ335         | Does a haemangioma follow a facial segment?                                                                     |        0.017 |            4 |            6.167 | uninformative |
| DQ398         | Is colour variance inside the lesion high?                                                                      |       -0.015 |            5 |            8.667 | uninformative |
| DQ304         | Is a vascular blush (pink veil) present?                                                                        |        0.012 |            0 |            8.667 | uninformative |
| DQ195         | Is a diffuse grey dotted pattern present?                                                                       |       -0.018 |            5 |           10.333 | uninformative |
| DQ165         | Does the lesion have a crater-like keratin core?                                                                |       -0.014 |            1 |           12.167 | uninformative |
| DQ136         | Is perifollicular tubular scaling present?                                                                      |        0.013 |            2 |           13     | uninformative |

**s1_3c**

| question_id   | question                                                                                                        |   ridge coef |   lasso kept |   mean SHAP rank | planted as                       |
|:--------------|:----------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:---------------------------------|
| DQ423         | Is the lesion border low in contrast with the surrounding skin?                                                 |        0.009 |            6 |            2.667 | uninformative                    |
| DQ376         | Do the naevi in the image show an eclipse pattern (tan centre with a brown rim)?                                |        0.008 |            3 |            4.167 | malignancy                       |
| DQ333         | Is a large congenital melanocytic naevus visible?                                                               |        0.01  |            7 |            4.833 | subtype: indeterminate epidermal |
| DQ166         | Is an ivory-white peripheral zone present?                                                                      |        0.008 |            5 |            8.667 | uninformative                    |
| DQ421         | Is the melanocytic lesion flat and subtle, without obvious melanoma features?                                   |        0.009 |            7 |            9.833 | uninformative                    |
| DQ449         | Can the lesion be described by plain clinical concepts such as asymmetry, irregular border or multiple colours? |        0.007 |            1 |           10.833 | uninformative                    |
| DQ304         | Is a vascular blush (pink veil) present?                                                                        |        0.007 |            2 |           10.833 | uninformative                    |
| DQ064         | Are vessels arranged in clusters?                                                                               |        0.007 |            2 |           11.5   | indeterminate                    |
| DQ401         | Is the lesion border perimeter long relative to the lesion area?                                                |        0.007 |            2 |           13.167 | uninformative                    |
| DQ033         | Are dotted vessels visible within a pigment network?                                                            |        0.009 |            5 |           15.333 | uninformative                    |

**flat_10c**

| question_id   | question                                                                    |   ridge coef |   lasso kept |   mean SHAP rank | planted as                          |
|:--------------|:----------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:------------------------------------|
| DQ423         | Is the lesion border low in contrast with the surrounding skin?             |        0.004 |            0 |            5     | uninformative                       |
| DQ401         | Is the lesion border perimeter long relative to the lesion area?            |        0.005 |            1 |            7     | uninformative                       |
| DQ206         | Does a perivascular white halo stand out against dark background skin?      |        0.005 |            4 |            7.167 | subtype: inflammatory or infectious |
| DQ197         | Are fine parallel lines present in a flat brown lesion?                     |        0.004 |            0 |           10.333 | subtype: inflammatory or infectious |
| DQ274         | Are nail pits visible?                                                      |        0.004 |            0 |           11.333 | uninformative                       |
| DQ156         | Are maple-leaf-like structureless black areas present on a pink background? |        0.005 |            0 |           14.167 | malignancy                          |
| DQ064         | Are vessels arranged in clusters?                                           |        0.004 |            0 |           14.5   | indeterminate                       |
| DQ103         | Is lichenification (thickened skin with accentuated markings) present?      |        0.004 |            0 |           17.333 | uninformative                       |
| DQ218         | Does the visible skin appear to be Fitzpatrick type III–IV?                 |        0.005 |            3 |           18     | uninformative                       |
| DQ096         | Are chalky-white macules or patches present?                                |        0.005 |            0 |           18.833 | uninformative                       |

**s2_benign**

| question_id   | question                                                                   |   ridge coef |   lasso kept |   mean SHAP rank | planted as                          |
|:--------------|:---------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:------------------------------------|
| DQ197         | Are fine parallel lines present in a flat brown lesion?                    |        0.004 |            0 |            5.167 | subtype: inflammatory or infectious |
| DQ096         | Are chalky-white macules or patches present?                               |        0.004 |            0 |           11.333 | uninformative                       |
| DQ059         | Are dots (structures no wider than a terminal hair) present in the lesion? |        0.003 |            0 |           12     | uninformative                       |
| DQ372         | Is fair or red hair visible in the image?                                  |        0.004 |            0 |           12.167 | uninformative                       |
| DQ339         | Does a facial haemangioma involve the mandibular (S3) segment?             |        0.004 |            0 |           12.667 | uninformative                       |
| DQ146         | Are short white streaks (chrysalis structures) present?                    |        0.004 |            0 |           13.667 | subtype: malignant epidermal        |
| DQ486         | Are wrinkles closely spaced (short wrinkle pitch)?                         |        0.004 |            0 |           14.667 | uninformative                       |
| DQ270         | Are irregular vessels visible in the nail unit?                            |        0.004 |            0 |           15.5   | uninformative                       |
| DQ174         | Is a double-edge sign present?                                             |        0.004 |            0 |           16.167 | uninformative                       |
| DQ087         | Are crusts or scabs present?                                               |        0.004 |            0 |           22.833 | uninformative                       |

**s2_malignant**

| question_id   | question                                                                               |   ridge coef |   lasso kept |   mean SHAP rank | planted as    |
|:--------------|:---------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------|
| DQ489         | Is facial skin texture uneven?                                                         |        0.002 |            0 |            4.5   | uninformative |
| DQ012         | Is a blue-white veil present?                                                          |        0.002 |            0 |            4.5   | uninformative |
| DQ066         | Are vessels distributed randomly without a specific arrangement?                       |        0.002 |            0 |            7.167 | uninformative |
| DQ062         | Are vessels arranged radially around the lesion centre?                                |        0.002 |            0 |            7.667 | uninformative |
| DQ471         | Is the lesion redder than the surrounding skin, suggesting higher haemoglobin content? |        0.002 |            0 |            9.333 | uninformative |
| DQ370         | Does the image show any naevus of at least 5 mm?                                       |        0.002 |            0 |            9.833 | malignancy    |
| DQ488         | Are red areas (diffuse vascular redness) visible on the facial skin?                   |        0.002 |            0 |           11.167 | uninformative |
| DQ067         | Are curved vessels present in the lesion?                                              |        0.002 |            0 |           11.167 | uninformative |
| DQ364         | Does the image show more than 50 common naevi?                                         |        0.002 |            0 |           11.667 | uninformative |
| DQ258         | Is a longitudinal nail band heterogeneous in colour?                                   |        0.002 |            0 |           13.167 | uninformative |

**s2_indeterminate**

| question_id   | question                                                                                                     |   ridge coef |   lasso kept |   mean SHAP rank | planted as    |
|:--------------|:-------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------|
| DQ156         | Are maple-leaf-like structureless black areas present on a pink background?                                  |       -0.062 |           11 |            1.167 | malignancy    |
| DQ308         | Are hairpin vessels present?                                                                                 |       -0.054 |            9 |            2.333 | uninformative |
| DQ218         | Does the visible skin appear to be Fitzpatrick type III–IV?                                                  |        0.064 |           11 |            4.5   | uninformative |
| DQ134         | Is yellow exudate present?                                                                                   |       -0.047 |           10 |            6.5   | uninformative |
| DQ418         | Is the lesion border impossible to delineate (not segmentable)?                                              |       -0.046 |            8 |            8.833 | uninformative |
| DQ412         | Is the skin in the image dark (a group where diagnostic accuracy is lower)?                                  |       -0.044 |            7 |           11     | uninformative |
| DQ199         | Is the lesion a yellow, exophytic or polypoid nodule?                                                        |        0.043 |            5 |           15.833 | uninformative |
| DQ410         | Does the image contain visible demographic cues (e.g., age-related skin changes) that a model could exploit? |       -0.045 |            6 |           16.833 | uninformative |
| DQ401         | Is the lesion border perimeter long relative to the lesion area?                                             |        0.04  |            5 |           17.833 | uninformative |
| DQ311         | Does a pink nodule lack any pigmented structures?                                                            |       -0.033 |            6 |           21.333 | uninformative |

## Figures

- [calibration_flat_10c.png](../../outputs/qaemb/classify/636270338498/figures/calibration_flat_10c.png)
- [calibration_s1_2c.png](../../outputs/qaemb/classify/636270338498/figures/calibration_s1_2c.png)
- [calibration_s1_3c.png](../../outputs/qaemb/classify/636270338498/figures/calibration_s1_3c.png)
- [calibration_s2_benign.png](../../outputs/qaemb/classify/636270338498/figures/calibration_s2_benign.png)
- [calibration_s2_indeterminate.png](../../outputs/qaemb/classify/636270338498/figures/calibration_s2_indeterminate.png)
- [calibration_s2_malignant.png](../../outputs/qaemb/classify/636270338498/figures/calibration_s2_malignant.png)
- [confusion_cascade.png](../../outputs/qaemb/classify/636270338498/figures/confusion_cascade.png)
- [confusion_flat_10c.png](../../outputs/qaemb/classify/636270338498/figures/confusion_flat_10c.png)
- [confusion_s1_3c.png](../../outputs/qaemb/classify/636270338498/figures/confusion_s1_3c.png)
- [confusion_s2_benign.png](../../outputs/qaemb/classify/636270338498/figures/confusion_s2_benign.png)
- [confusion_s2_malignant.png](../../outputs/qaemb/classify/636270338498/figures/confusion_s2_malignant.png)
- [importance_ranks_coh.png](../../outputs/qaemb/classify/636270338498/figures/importance_ranks_coh.png)
- [importance_ranks_dir.png](../../outputs/qaemb/classify/636270338498/figures/importance_ranks_dir.png)
- [importance_ranks_orig.png](../../outputs/qaemb/classify/636270338498/figures/importance_ranks_orig.png)
- [indeterminate_2c_coh.png](../../outputs/qaemb/classify/636270338498/figures/indeterminate_2c_coh.png)
- [indeterminate_2c_dir.png](../../outputs/qaemb/classify/636270338498/figures/indeterminate_2c_dir.png)
- [indeterminate_2c_orig.png](../../outputs/qaemb/classify/636270338498/figures/indeterminate_2c_orig.png)
- [model_vs_ridge_coh.png](../../outputs/qaemb/classify/636270338498/figures/model_vs_ridge_coh.png)
- [model_vs_ridge_dir.png](../../outputs/qaemb/classify/636270338498/figures/model_vs_ridge_dir.png)
- [model_vs_ridge_orig.png](../../outputs/qaemb/classify/636270338498/figures/model_vs_ridge_orig.png)

## Files

- metrics_summary.csv: every metric, mean and SD over test folds, and the pooled out-of-fold value
- metrics_folds.csv, metrics_pooled_by_repeat.csv, best_params.csv: per fold / per repeat detail
- oof_predictions.parquet: out-of-fold class probabilities per image, model, source and repeat
- importance.csv: per question and output class: ridge/lasso/elastic-net weights, lasso selection count, mean |SHAP|
- question_ranking.csv: one row per question and task, all importance measures side by side
- cascade.csv, indeterminate_2c.csv, planted.csv, recovery.csv
