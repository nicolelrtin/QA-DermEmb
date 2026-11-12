# Classifier results

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

## Benchmark: the answerer asked directly

P(yes) to the control question, used as P(malignant) without any fitting, scored on the same 2-class test folds as the classifiers. A classifier on the other questions has to beat this to be worth its cost.

| question | AUROC | specificity at 95% sensitivity | balanced accuracy (P(yes) ≥ 0.5) |
|---|---|---|---|
| DQC01 Is this a malignant lesion? | 0.786 ± 0.036 | 0.258 ± 0.093 | 0.695 ± 0.039 |

Median P(yes) it gives the indeterminate images:

| subtype                   |   DQC01 |
|:--------------------------|--------:|
| indeterminate epidermal   |   0.325 |
| indeterminate melanocytic |   0.225 |

## Results: P(yes) (`orig`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.810 ± 0.032 | 0.819 ± 0.034 | 0.813 ± 0.031 | 0.829 ± 0.029   | 0.808 ± 0.027       | 0.826 ± 0.033 |
| s1_3c            | balanced accuracy              | 0.543 ± 0.031 | 0.545 ± 0.025 | 0.540 ± 0.028 | 0.598 ± 0.025   | 0.573 ± 0.032       | 0.563 ± 0.030 |
| flat_10c         | balanced accuracy              | 0.451 ± 0.032 | 0.488 ± 0.025 | 0.462 ± 0.028 | 0.482 ± 0.028   | 0.456 ± 0.036       | 0.475 ± 0.024 |
| s2_benign        | balanced accuracy              | 0.708 ± 0.041 | 0.725 ± 0.051 | 0.712 ± 0.046 | 0.702 ± 0.042   | 0.705 ± 0.048       | 0.727 ± 0.049 |
| s2_malignant     | balanced accuracy              | 0.722 ± 0.047 | 0.750 ± 0.053 | 0.733 ± 0.050 | 0.733 ± 0.049   | 0.713 ± 0.048       | 0.747 ± 0.054 |
| s2_indeterminate | balanced accuracy              | 0.855 ± 0.048 | 0.871 ± 0.042 | 0.862 ± 0.047 | 0.876 ± 0.050   | 0.856 ± 0.043       | 0.868 ± 0.044 |
| s1_2c            | specificity at 95% sensitivity | 0.424 ± 0.100 | 0.443 ± 0.122 | 0.428 ± 0.093 | 0.414 ± 0.102   | 0.368 ± 0.097       | 0.441 ± 0.114 |
| s1_2c            | macro F1                       | 0.713 ± 0.028 | 0.719 ± 0.037 | 0.716 ± 0.029 | 0.730 ± 0.032   | 0.718 ± 0.029       | 0.728 ± 0.036 |
| s1_3c            | macro F1                       | 0.525 ± 0.029 | 0.525 ± 0.025 | 0.524 ± 0.026 | 0.565 ± 0.024   | 0.566 ± 0.033       | 0.541 ± 0.028 |
| flat_10c         | macro F1                       | 0.442 ± 0.033 | 0.480 ± 0.025 | 0.453 ± 0.029 | 0.474 ± 0.026   | 0.452 ± 0.038       | 0.466 ± 0.024 |
| s2_benign        | macro F1                       | 0.705 ± 0.041 | 0.721 ± 0.051 | 0.708 ± 0.045 | 0.698 ± 0.043   | 0.702 ± 0.048       | 0.724 ± 0.050 |
| s2_malignant     | macro F1                       | 0.721 ± 0.049 | 0.749 ± 0.055 | 0.733 ± 0.052 | 0.733 ± 0.050   | 0.712 ± 0.048       | 0.746 ± 0.055 |
| s2_indeterminate | macro F1                       | 0.854 ± 0.049 | 0.870 ± 0.043 | 0.861 ± 0.048 | 0.875 ± 0.051   | 0.855 ± 0.044       | 0.867 ± 0.045 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.462 |                    0.413 |                    0.42  |           0.406 |                  0.457 |                                    0.761 |                          0.778 |
| gradient boosting |                        0.456 |                    0.444 |                    0.453 |           0.437 |                  0.402 |                                    0.743 |                          0.724 |
| lasso (L1)        |                        0.451 |                    0.408 |                    0.421 |           0.402 |                  0.458 |                                    0.753 |                          0.773 |
| ridge (L2)        |                        0.488 |                    0.418 |                    0.433 |           0.408 |                  0.46  |                                    0.774 |                          0.79  |
| MLP               |                        0.475 |                    0.434 |                    0.439 |           0.425 |                  0.444 |                                    0.781 |                          0.785 |
| random forest     |                        0.482 |                    0.446 |                    0.431 |           0.428 |                  0.426 |                                    0.777 |                          0.769 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.608 |                       0.249 |
| gradient boosting |                     0.577 |                       0.222 |
| lasso (L1)        |                     0.615 |                       0.266 |
| ridge (L2)        |                     0.593 |                       0.247 |
| MLP               |                     0.611 |                       0.245 |
| random forest     |                     0.591 |                       0.282 |

## Results: Directional Score (`dir`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.833 ± 0.032 | 0.839 ± 0.030 | 0.838 ± 0.030 | 0.823 ± 0.027   | 0.811 ± 0.035       | 0.834 ± 0.029 |
| s1_3c            | balanced accuracy              | 0.548 ± 0.026 | 0.562 ± 0.028 | 0.554 ± 0.031 | 0.584 ± 0.024   | 0.554 ± 0.031       | 0.566 ± 0.032 |
| flat_10c         | balanced accuracy              | 0.472 ± 0.027 | 0.487 ± 0.026 | 0.482 ± 0.030 | 0.466 ± 0.027   | 0.464 ± 0.030       | 0.499 ± 0.029 |
| s2_benign        | balanced accuracy              | 0.716 ± 0.055 | 0.719 ± 0.056 | 0.722 ± 0.056 | 0.687 ± 0.046   | 0.694 ± 0.049       | 0.731 ± 0.053 |
| s2_malignant     | balanced accuracy              | 0.725 ± 0.053 | 0.751 ± 0.049 | 0.735 ± 0.047 | 0.723 ± 0.045   | 0.701 ± 0.050       | 0.742 ± 0.041 |
| s2_indeterminate | balanced accuracy              | 0.850 ± 0.045 | 0.867 ± 0.047 | 0.850 ± 0.049 | 0.873 ± 0.049   | 0.862 ± 0.042       | 0.874 ± 0.042 |
| s1_2c            | specificity at 95% sensitivity | 0.423 ± 0.107 | 0.444 ± 0.102 | 0.434 ± 0.106 | 0.421 ± 0.106   | 0.384 ± 0.099       | 0.449 ± 0.114 |
| s1_2c            | macro F1                       | 0.745 ± 0.038 | 0.748 ± 0.027 | 0.749 ± 0.031 | 0.727 ± 0.034   | 0.723 ± 0.041       | 0.733 ± 0.032 |
| s1_3c            | macro F1                       | 0.530 ± 0.022 | 0.548 ± 0.026 | 0.538 ± 0.027 | 0.553 ± 0.021   | 0.549 ± 0.031       | 0.545 ± 0.028 |
| flat_10c         | macro F1                       | 0.467 ± 0.026 | 0.479 ± 0.026 | 0.475 ± 0.030 | 0.457 ± 0.027   | 0.459 ± 0.032       | 0.494 ± 0.028 |
| s2_benign        | macro F1                       | 0.713 ± 0.057 | 0.714 ± 0.058 | 0.718 ± 0.057 | 0.684 ± 0.045   | 0.691 ± 0.050       | 0.728 ± 0.054 |
| s2_malignant     | macro F1                       | 0.723 ± 0.054 | 0.750 ± 0.051 | 0.734 ± 0.049 | 0.722 ± 0.046   | 0.699 ± 0.050       | 0.741 ± 0.042 |
| s2_indeterminate | macro F1                       | 0.849 ± 0.046 | 0.866 ± 0.047 | 0.849 ± 0.050 | 0.875 ± 0.049   | 0.861 ± 0.042       | 0.873 ± 0.042 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.482 |                    0.423 |                    0.428 |           0.416 |                  0.44  |                                    0.756 |                          0.763 |
| gradient boosting |                        0.464 |                    0.426 |                    0.442 |           0.42  |                  0.419 |                                    0.733 |                          0.729 |
| lasso (L1)        |                        0.472 |                    0.412 |                    0.417 |           0.403 |                  0.451 |                                    0.749 |                          0.766 |
| ridge (L2)        |                        0.487 |                    0.444 |                    0.45  |           0.438 |                  0.428 |                                    0.776 |                          0.77  |
| MLP               |                        0.499 |                    0.439 |                    0.447 |           0.43  |                  0.438 |                                    0.78  |                          0.78  |
| random forest     |                        0.466 |                    0.427 |                    0.423 |           0.407 |                  0.437 |                                    0.758 |                          0.762 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.672 |                       0.198 |
| gradient boosting |                     0.615 |                       0.2   |
| lasso (L1)        |                     0.669 |                       0.201 |
| ridge (L2)        |                     0.659 |                       0.2   |
| MLP               |                     0.673 |                       0.231 |
| random forest     |                     0.615 |                       0.26  |

## Results: Coherence Score (`coh`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.824 ± 0.029 | 0.830 ± 0.029 | 0.825 ± 0.030 | 0.818 ± 0.030   | 0.812 ± 0.032       | 0.829 ± 0.030 |
| s1_3c            | balanced accuracy              | 0.545 ± 0.036 | 0.554 ± 0.026 | 0.545 ± 0.033 | 0.589 ± 0.028   | 0.559 ± 0.029       | 0.559 ± 0.034 |
| flat_10c         | balanced accuracy              | 0.463 ± 0.027 | 0.477 ± 0.028 | 0.468 ± 0.029 | 0.456 ± 0.031   | 0.452 ± 0.028       | 0.480 ± 0.028 |
| s2_benign        | balanced accuracy              | 0.700 ± 0.056 | 0.715 ± 0.055 | 0.700 ± 0.051 | 0.687 ± 0.042   | 0.704 ± 0.044       | 0.720 ± 0.056 |
| s2_malignant     | balanced accuracy              | 0.731 ± 0.047 | 0.747 ± 0.052 | 0.735 ± 0.043 | 0.710 ± 0.053   | 0.697 ± 0.040       | 0.741 ± 0.038 |
| s2_indeterminate | balanced accuracy              | 0.862 ± 0.040 | 0.869 ± 0.045 | 0.858 ± 0.038 | 0.870 ± 0.052   | 0.844 ± 0.053       | 0.860 ± 0.050 |
| s1_2c            | specificity at 95% sensitivity | 0.422 ± 0.120 | 0.425 ± 0.100 | 0.410 ± 0.107 | 0.419 ± 0.094   | 0.390 ± 0.083       | 0.444 ± 0.100 |
| s1_2c            | macro F1                       | 0.733 ± 0.029 | 0.734 ± 0.033 | 0.738 ± 0.027 | 0.723 ± 0.032   | 0.723 ± 0.035       | 0.728 ± 0.030 |
| s1_3c            | macro F1                       | 0.527 ± 0.033 | 0.538 ± 0.025 | 0.528 ± 0.029 | 0.558 ± 0.024   | 0.553 ± 0.029       | 0.539 ± 0.030 |
| flat_10c         | macro F1                       | 0.455 ± 0.028 | 0.468 ± 0.028 | 0.461 ± 0.030 | 0.447 ± 0.031   | 0.446 ± 0.028       | 0.473 ± 0.028 |
| s2_benign        | macro F1                       | 0.697 ± 0.058 | 0.710 ± 0.057 | 0.698 ± 0.051 | 0.683 ± 0.040   | 0.700 ± 0.044       | 0.717 ± 0.057 |
| s2_malignant     | macro F1                       | 0.729 ± 0.048 | 0.746 ± 0.053 | 0.734 ± 0.045 | 0.708 ± 0.055   | 0.695 ± 0.041       | 0.740 ± 0.039 |
| s2_indeterminate | macro F1                       | 0.861 ± 0.041 | 0.868 ± 0.045 | 0.857 ± 0.039 | 0.870 ± 0.053   | 0.843 ± 0.053       | 0.859 ± 0.050 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.468 |                    0.41  |                    0.42  |           0.403 |                  0.452 |                                    0.75  |                          0.767 |
| gradient boosting |                        0.452 |                    0.433 |                    0.446 |           0.426 |                  0.416 |                                    0.741 |                          0.733 |
| lasso (L1)        |                        0.463 |                    0.408 |                    0.417 |           0.399 |                  0.455 |                                    0.748 |                          0.768 |
| ridge (L2)        |                        0.477 |                    0.428 |                    0.433 |           0.42  |                  0.441 |                                    0.765 |                          0.77  |
| MLP               |                        0.48  |                    0.428 |                    0.441 |           0.418 |                  0.444 |                                    0.77  |                          0.777 |
| random forest     |                        0.456 |                    0.429 |                    0.413 |           0.41  |                  0.432 |                                    0.754 |                          0.755 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.689 |                       0.208 |
| gradient boosting |                     0.621 |                       0.214 |
| lasso (L1)        |                     0.695 |                       0.236 |
| ridge (L2)        |                     0.704 |                       0.211 |
| MLP               |                     0.695 |                       0.241 |
| random forest     |                     0.631 |                       0.281 |

## Most important questions (P(yes))

Ranked by the mean SHAP rank over all models (1 = most important). *Ridge coef* is the standardised weight (positive = pushes towards the class named in the task; for multiclass tasks, the largest |weight| over classes). *Lasso kept* is the number of the outer fits in which the lasso kept the question.

**s1_2c**

| question_id   | question                                                                      |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ359         | Are signs of chronic sun damage visible around the lesions?                   |        0.084 |           25 |            1.833 |
| DQ020         | Does the lesion show a pigmented homogeneous global pattern?                  |       -0.091 |           25 |            3.167 |
| DQ484         | Does the skin surface appear rough?                                           |        0.098 |           25 |            5     |
| DQ421         | Is the melanocytic lesion flat and subtle, without obvious melanoma features? |       -0.075 |           25 |            9.333 |
| DQ220         | Is the individual typology angle (ITA) of the visible normal skin above 28°?  |       -0.067 |           23 |           16.333 |
| DQ399         | Is colour asymmetry within the lesion high?                                   |        0.069 |           25 |           19     |
| DQ472         | Does the lesion look like a vascular neoplasm (uniform red to purple colour)? |       -0.072 |           25 |           22.333 |
| DQ383         | Is dyspigmentation visible?                                                   |        0.076 |           24 |           22.833 |
| DQ038         | Is colour variegation present across the lesion?                              |        0.056 |           23 |           24     |
| DQ454         | Is the lesion's darkest region located off-centre?                            |        0.076 |           20 |           24.667 |

**s1_3c**

| question_id   | question                                                                                                     |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:-------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ359         | Are signs of chronic sun damage visible around the lesions?                                                  |        0.046 |           25 |            2     |
| DQ020         | Does the lesion show a pigmented homogeneous global pattern?                                                 |        0.05  |           25 |            4.667 |
| DQ421         | Is the melanocytic lesion flat and subtle, without obvious melanoma features?                                |        0.04  |           25 |           11.333 |
| DQ301         | Does the lesion resemble a vascular lesion?                                                                  |        0.044 |           25 |           12.833 |
| DQ410         | Does the image contain visible demographic cues (e.g., age-related skin changes) that a model could exploit? |        0.041 |           23 |           16     |
| DQ497         | Does lesion texture differ markedly between the red and blue colour channels?                                |        0.039 |           23 |           22.167 |
| DQ337         | Does a haemangioma show a C-shaped pattern on the scalp?                                                     |        0.041 |           25 |           24.833 |
| DQ472         | Does the lesion look like a vascular neoplasm (uniform red to purple colour)?                                |        0.043 |           25 |           27.833 |
| DQ220         | Is the individual typology angle (ITA) of the visible normal skin above 28°?                                 |        0.036 |           25 |           30.333 |
| DQ304         | Is a vascular blush (pink veil) present?                                                                     |        0.051 |           25 |           30.333 |

**flat_10c**

| question_id   | question                                                                                                     |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:-------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ359         | Are signs of chronic sun damage visible around the lesions?                                                  |        0.068 |           25 |            8.5   |
| DQ301         | Does the lesion resemble a vascular lesion?                                                                  |        0.152 |           25 |           12.167 |
| DQ471         | Is the lesion redder than the surrounding skin, suggesting higher haemoglobin content?                       |        0.068 |           25 |           16.333 |
| DQ363         | Is the skin immediately surrounding the lesion erythematous?                                                 |        0.084 |           25 |           20.333 |
| DQ421         | Is the melanocytic lesion flat and subtle, without obvious melanoma features?                                |        0.077 |           25 |           20.333 |
| DQ484         | Does the skin surface appear rough?                                                                          |        0.073 |           25 |           20.667 |
| DQ480         | Is epidermal pigmentation visibly heterogeneous across the normal skin?                                      |        0.089 |           25 |           21.667 |
| DQ410         | Does the image contain visible demographic cues (e.g., age-related skin changes) that a model could exploit? |        0.069 |           25 |           24.833 |
| DQ390         | Are solar lentigines visible?                                                                                |        0.067 |           25 |           24.833 |
| DQ374         | Do the naevi visible in the image share a predominantly reticular pattern?                                   |        0.092 |           25 |           24.833 |

**s2_benign**

| question_id   | question                                                                      |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ301         | Does the lesion resemble a vascular lesion?                                   |        0.107 |           25 |            3.833 |
| DQ023         | Is light brown colour present in the lesion?                                  |        0.07  |           25 |            8     |
| DQ472         | Does the lesion look like a vascular neoplasm (uniform red to purple colour)? |        0.104 |           25 |            9.333 |
| DQ480         | Is epidermal pigmentation visibly heterogeneous across the normal skin?       |        0.078 |           25 |            9.5   |
| DQ390         | Are solar lentigines visible?                                                 |        0.078 |           25 |           11.5   |
| DQ080         | Are white structureless areas present (a correlate of dermal fibrosis)?       |        0.079 |           25 |           13.333 |
| DQ454         | Is the lesion's darkest region located off-centre?                            |        0.131 |           25 |           14.167 |
| DQ043         | Are white dots or clods clustered or disseminated across the lesion?          |        0.082 |           25 |           19.167 |
| DQ489         | Is facial skin texture uneven?                                                |        0.049 |           23 |           27.833 |
| DQ072         | Is a brown structureless area present in the lesion?                          |        0.053 |           25 |           28.833 |

**s2_malignant**

| question_id   | question                                                                  |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:--------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ400         | Is the lesion's minor-axis diameter greater than 6 mm?                    |        0.046 |           22 |           10.167 |
| DQ180         | Is pink-red the predominant colour of a keratinizing lesion?              |        0.033 |           23 |           12.167 |
| DQ193         | Does the lesion have a scalloped border?                                  |        0.043 |           25 |           13.333 |
| DQ363         | Is the skin immediately surrounding the lesion erythematous?              |        0.045 |           18 |           16.167 |
| DQ397         | Is the blue-yellow (b*) contrast between lesion and surrounding skin low? |        0.038 |           25 |           17.333 |
| DQ222         | Does the visible skin have a yellow (rather than red) undertone?          |        0.048 |           25 |           19.167 |
| DQ001         | Is the lesion diameter larger than 6 mm?                                  |        0.041 |           23 |           20.333 |
| DQ110         | Are round white structures (corn pearls) present?                         |        0.051 |           25 |           21.667 |
| DQ353         | Are scaly plaques present on sun-exposed skin?                            |        0.03  |           22 |           22.167 |
| DQ306         | Are dotted vessels present?                                               |        0.043 |           25 |           24     |

**s2_indeterminate**

| question_id   | question                                                                           |   ridge coef |   lasso kept |   mean SHAP rank |
|:--------------|:-----------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|
| DQ363         | Is the skin immediately surrounding the lesion erythematous?                       |       -0.067 |           25 |            1.167 |
| DQ373         | Is there a single darker or larger 'ugly duckling' lesion among the naevi visible? |        0.072 |           23 |            8.333 |
| DQ196         | Is homogeneous brown pigmentation present?                                         |        0.058 |           23 |           14.167 |
| DQ155         | Does the lesion appear as an erythematous plaque or patch?                         |       -0.037 |            5 |           18.5   |
| DQ154         | Does the lesion appear scar-like?                                                  |       -0.052 |           17 |           19.167 |
| DQ108         | Are white striae arranged in an annular or circular pattern?                       |       -0.051 |           21 |           19.333 |
| DQ024         | Is dark brown or black colour present in the lesion?                               |        0.046 |           18 |           21     |
| DQ176         | Are whitish structureless areas present?                                           |       -0.039 |            9 |           22.5   |
| DQ194         | Is fine scale present on the lesion surface?                                       |       -0.041 |           19 |           25.5   |
| DQ418         | Is the lesion border impossible to delineate (not segmentable)?                    |       -0.055 |           21 |           26.667 |

## Figures

- [calibration_flat_10c.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_flat_10c.png)
- [calibration_s1_2c.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_s1_2c.png)
- [calibration_s1_3c.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_s1_3c.png)
- [calibration_s2_benign.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_s2_benign.png)
- [calibration_s2_indeterminate.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_s2_indeterminate.png)
- [calibration_s2_malignant.png](../../outputs/qaemb/classify/670746415c57/figures/calibration_s2_malignant.png)
- [confusion_cascade.png](../../outputs/qaemb/classify/670746415c57/figures/confusion_cascade.png)
- [confusion_flat_10c.png](../../outputs/qaemb/classify/670746415c57/figures/confusion_flat_10c.png)
- [confusion_s1_3c.png](../../outputs/qaemb/classify/670746415c57/figures/confusion_s1_3c.png)
- [confusion_s2_benign.png](../../outputs/qaemb/classify/670746415c57/figures/confusion_s2_benign.png)
- [confusion_s2_malignant.png](../../outputs/qaemb/classify/670746415c57/figures/confusion_s2_malignant.png)
- [importance_ranks_coh.png](../../outputs/qaemb/classify/670746415c57/figures/importance_ranks_coh.png)
- [importance_ranks_dir.png](../../outputs/qaemb/classify/670746415c57/figures/importance_ranks_dir.png)
- [importance_ranks_orig.png](../../outputs/qaemb/classify/670746415c57/figures/importance_ranks_orig.png)
- [indeterminate_2c_coh.png](../../outputs/qaemb/classify/670746415c57/figures/indeterminate_2c_coh.png)
- [indeterminate_2c_dir.png](../../outputs/qaemb/classify/670746415c57/figures/indeterminate_2c_dir.png)
- [indeterminate_2c_orig.png](../../outputs/qaemb/classify/670746415c57/figures/indeterminate_2c_orig.png)
- [model_vs_ridge_coh.png](../../outputs/qaemb/classify/670746415c57/figures/model_vs_ridge_coh.png)
- [model_vs_ridge_dir.png](../../outputs/qaemb/classify/670746415c57/figures/model_vs_ridge_dir.png)
- [model_vs_ridge_orig.png](../../outputs/qaemb/classify/670746415c57/figures/model_vs_ridge_orig.png)

## Files

- metrics_summary.csv: every metric, mean and SD over test folds, and the pooled out-of-fold value
- metrics_folds.csv, metrics_pooled_by_repeat.csv, best_params.csv: per fold / per repeat detail
- oof_predictions.parquet: out-of-fold class probabilities per image, model, source and repeat
- importance.csv: per question and output class: ridge/lasso/elastic-net weights, lasso selection count, mean |SHAP|
- question_ranking.csv: one row per question and task, all importance measures side by side
- cascade.csv, indeterminate_2c.csv, benchmark_folds.csv, benchmark_summary.csv
