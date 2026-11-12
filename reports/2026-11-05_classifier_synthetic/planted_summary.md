# Classifier results

**Synthetic answers** (effect = 0.75): the labels are the real manifest labels, the answers are simulated with planted informative questions (qaemb/synthetic.py). Numbers here test the code, not the method.

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

## Validity check: are the planted questions found?

For each task, a question counts as relevant if the simulation made its answers differ between the task's classes. *AUROC relevant vs not* is the chance that a random relevant question ranks above a random uninformative one under that importance measure (0.5 = no better than chance); *recall at k* is the share of the k relevant questions found among the top k.

AUROC relevant vs not (P(yes)):

| task             |   n_relevant |   elastic net: abs_coef |   elastic net: n_selected |   elastic net: shap |   gradient boosting: shap |   lasso (L1): abs_coef |   lasso (L1): n_selected |   lasso (L1): shap |   ridge (L2): abs_coef |   ridge (L2): shap |   MLP: shap |   random forest: shap |
|:-----------------|-------------:|------------------------:|--------------------------:|--------------------:|--------------------------:|-----------------------:|-------------------------:|-------------------:|-----------------------:|-------------------:|------------:|----------------------:|
| flat_10c         |           68 |                    0.99 |                      0.87 |                0.98 |                      0.97 |                   0.99 |                     0.93 |               0.99 |                   1    |               0.93 |        0.94 |                  0.96 |
| s1_2c            |           52 |                    0.78 |                      0.77 |                0.79 |                      0.82 |                   0.77 |                     0.8  |               0.77 |                   0.79 |               0.77 |        0.77 |                  0.83 |
| s1_3c            |           68 |                    0.8  |                      0.79 |                0.81 |                      0.82 |                   0.8  |                     0.79 |               0.81 |                   0.8  |               0.79 |        0.8  |                  0.84 |
| s2_benign        |           20 |                    1    |                      0.99 |                1    |                      1    |                   1    |                     0.98 |               1    |                   1    |               0.99 |        0.99 |                  1    |
| s2_indeterminate |            8 |                    0.94 |                      0.94 |                0.94 |                      0.97 |                   0.94 |                     0.95 |               0.91 |                   0.95 |               0.94 |        0.93 |                  0.99 |
| s2_malignant     |           12 |                    0.99 |                      0.99 |                0.98 |                      0.92 |                   0.99 |                     0.99 |               0.99 |                   1    |               0.94 |        0.94 |                  0.91 |

## Results: P(yes) (`orig`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.926 ± 0.024 | 0.891 ± 0.020 | 0.924 ± 0.021 | 0.929 ± 0.017   | 0.918 ± 0.023       | 0.892 ± 0.020 |
| s1_3c            | balanced accuracy              | 0.710 ± 0.032 | 0.632 ± 0.041 | 0.699 ± 0.032 | 0.693 ± 0.034   | 0.674 ± 0.039       | 0.642 ± 0.039 |
| flat_10c         | balanced accuracy              | 0.359 ± 0.031 | 0.269 ± 0.034 | 0.349 ± 0.026 | 0.279 ± 0.038   | 0.304 ± 0.023       | 0.280 ± 0.031 |
| s2_benign        | balanced accuracy              | 0.405 ± 0.036 | 0.303 ± 0.035 | 0.394 ± 0.047 | 0.382 ± 0.033   | 0.352 ± 0.048       | 0.288 ± 0.043 |
| s2_malignant     | balanced accuracy              | 0.548 ± 0.057 | 0.413 ± 0.066 | 0.539 ± 0.058 | 0.484 ± 0.048   | 0.487 ± 0.055       | 0.415 ± 0.061 |
| s2_indeterminate | balanced accuracy              | 0.684 ± 0.066 | 0.666 ± 0.067 | 0.677 ± 0.067 | 0.662 ± 0.069   | 0.655 ± 0.060       | 0.661 ± 0.063 |
| s1_2c            | specificity at 95% sensitivity | 0.661 ± 0.110 | 0.528 ± 0.095 | 0.639 ± 0.115 | 0.656 ± 0.113   | 0.639 ± 0.107       | 0.546 ± 0.101 |
| s1_2c            | macro F1                       | 0.847 ± 0.032 | 0.815 ± 0.026 | 0.845 ± 0.030 | 0.802 ± 0.046   | 0.836 ± 0.035       | 0.810 ± 0.023 |
| s1_3c            | macro F1                       | 0.702 ± 0.032 | 0.629 ± 0.040 | 0.692 ± 0.030 | 0.666 ± 0.034   | 0.680 ± 0.039       | 0.633 ± 0.036 |
| flat_10c         | macro F1                       | 0.357 ± 0.031 | 0.269 ± 0.033 | 0.347 ± 0.026 | 0.278 ± 0.038   | 0.302 ± 0.022       | 0.279 ± 0.030 |
| s2_benign        | macro F1                       | 0.404 ± 0.038 | 0.301 ± 0.036 | 0.393 ± 0.049 | 0.380 ± 0.032   | 0.351 ± 0.048       | 0.286 ± 0.041 |
| s2_malignant     | macro F1                       | 0.546 ± 0.058 | 0.411 ± 0.067 | 0.538 ± 0.058 | 0.483 ± 0.048   | 0.484 ± 0.056       | 0.413 ± 0.061 |
| s2_indeterminate | macro F1                       | 0.675 ± 0.089 | 0.663 ± 0.069 | 0.674 ± 0.067 | 0.660 ± 0.067   | 0.652 ± 0.061       | 0.658 ± 0.065 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.349 |                    0.358 |                    0.362 |           0.356 |                  0.279 |                                    0.497 |                          0.434 |
| gradient boosting |                        0.304 |                    0.312 |                    0.325 |           0.313 |                  0.274 |                                    0.429 |                          0.398 |
| lasso (L1)        |                        0.359 |                    0.371 |                    0.373 |           0.369 |                  0.27  |                                    0.508 |                          0.429 |
| ridge (L2)        |                        0.269 |                    0.27  |                    0.269 |           0.269 |                  0.331 |                                    0.403 |                          0.453 |
| MLP               |                        0.28  |                    0.266 |                    0.266 |           0.263 |                  0.336 |                                    0.4   |                          0.457 |
| random forest     |                        0.279 |                    0.331 |                    0.201 |           0.325 |                  0.319 |                                    0.486 |                          0.477 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.423 |                       0.438 |
| gradient boosting |                     0.489 |                       0.413 |
| lasso (L1)        |                     0.454 |                       0.451 |
| ridge (L2)        |                     0.389 |                       0.387 |
| MLP               |                     0.435 |                       0.439 |
| random forest     |                     0.543 |                       0.525 |

## Results: Directional Score (`dir`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.925 ± 0.022 | 0.898 ± 0.023 | 0.925 ± 0.020 | 0.908 ± 0.020   | 0.888 ± 0.024       | 0.900 ± 0.024 |
| s1_3c            | balanced accuracy              | 0.708 ± 0.031 | 0.646 ± 0.037 | 0.702 ± 0.033 | 0.659 ± 0.035   | 0.647 ± 0.037       | 0.656 ± 0.034 |
| flat_10c         | balanced accuracy              | 0.377 ± 0.034 | 0.274 ± 0.024 | 0.365 ± 0.037 | 0.260 ± 0.036   | 0.272 ± 0.027       | 0.289 ± 0.027 |
| s2_benign        | balanced accuracy              | 0.387 ± 0.042 | 0.306 ± 0.031 | 0.384 ± 0.045 | 0.338 ± 0.039   | 0.350 ± 0.051       | 0.298 ± 0.038 |
| s2_malignant     | balanced accuracy              | 0.553 ± 0.054 | 0.420 ± 0.059 | 0.537 ± 0.054 | 0.476 ± 0.055   | 0.463 ± 0.052       | 0.419 ± 0.059 |
| s2_indeterminate | balanced accuracy              | 0.708 ± 0.079 | 0.654 ± 0.054 | 0.680 ± 0.074 | 0.678 ± 0.076   | 0.632 ± 0.077       | 0.651 ± 0.055 |
| s1_2c            | specificity at 95% sensitivity | 0.662 ± 0.128 | 0.551 ± 0.119 | 0.670 ± 0.104 | 0.575 ± 0.135   | 0.545 ± 0.130       | 0.558 ± 0.148 |
| s1_2c            | macro F1                       | 0.849 ± 0.028 | 0.818 ± 0.025 | 0.848 ± 0.030 | 0.773 ± 0.038   | 0.795 ± 0.031       | 0.814 ± 0.029 |
| s1_3c            | macro F1                       | 0.699 ± 0.029 | 0.643 ± 0.036 | 0.694 ± 0.031 | 0.633 ± 0.035   | 0.655 ± 0.039       | 0.646 ± 0.033 |
| flat_10c         | macro F1                       | 0.375 ± 0.034 | 0.275 ± 0.023 | 0.363 ± 0.036 | 0.257 ± 0.037   | 0.268 ± 0.028       | 0.289 ± 0.027 |
| s2_benign        | macro F1                       | 0.385 ± 0.042 | 0.303 ± 0.030 | 0.383 ± 0.044 | 0.335 ± 0.038   | 0.348 ± 0.050       | 0.296 ± 0.037 |
| s2_malignant     | macro F1                       | 0.552 ± 0.055 | 0.419 ± 0.059 | 0.536 ± 0.055 | 0.475 ± 0.056   | 0.460 ± 0.052       | 0.418 ± 0.059 |
| s2_indeterminate | macro F1                       | 0.699 ± 0.101 | 0.652 ± 0.055 | 0.677 ± 0.075 | 0.674 ± 0.076   | 0.630 ± 0.078       | 0.649 ± 0.056 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.365 |                    0.357 |                    0.365 |           0.355 |                  0.279 |                                    0.495 |                          0.434 |
| gradient boosting |                        0.272 |                    0.297 |                    0.317 |           0.298 |                  0.297 |                                    0.423 |                          0.423 |
| lasso (L1)        |                        0.377 |                    0.37  |                    0.374 |           0.367 |                  0.274 |                                    0.509 |                          0.435 |
| ridge (L2)        |                        0.274 |                    0.272 |                    0.273 |           0.272 |                  0.319 |                                    0.4   |                          0.439 |
| MLP               |                        0.289 |                    0.272 |                    0.276 |           0.27  |                  0.326 |                                    0.404 |                          0.447 |
| random forest     |                        0.26  |                    0.301 |                    0.187 |           0.293 |                  0.353 |                                    0.465 |                          0.505 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.425 |                       0.419 |
| gradient boosting |                     0.466 |                       0.42  |
| lasso (L1)        |                     0.44  |                       0.425 |
| ridge (L2)        |                     0.356 |                       0.395 |
| MLP               |                     0.399 |                       0.424 |
| random forest     |                     0.531 |                       0.523 |

## Results: Coherence Score (`coh`)

| task             | metric                         | lasso (L1)    | ridge (L2)    | elastic net   | random forest   | gradient boosting   | MLP           |
|:-----------------|:-------------------------------|:--------------|:--------------|:--------------|:----------------|:--------------------|:--------------|
| s1_2c            | auroc                          | 0.931 ± 0.021 | 0.897 ± 0.022 | 0.927 ± 0.020 | 0.909 ± 0.026   | 0.890 ± 0.027       | 0.898 ± 0.023 |
| s1_3c            | balanced accuracy              | 0.708 ± 0.031 | 0.642 ± 0.044 | 0.697 ± 0.035 | 0.650 ± 0.036   | 0.634 ± 0.043       | 0.653 ± 0.041 |
| flat_10c         | balanced accuracy              | 0.368 ± 0.030 | 0.281 ± 0.027 | 0.355 ± 0.035 | 0.274 ± 0.031   | 0.282 ± 0.032       | 0.293 ± 0.028 |
| s2_benign        | balanced accuracy              | 0.396 ± 0.042 | 0.304 ± 0.036 | 0.387 ± 0.053 | 0.372 ± 0.040   | 0.348 ± 0.042       | 0.304 ± 0.043 |
| s2_malignant     | balanced accuracy              | 0.569 ± 0.062 | 0.418 ± 0.072 | 0.540 ± 0.052 | 0.484 ± 0.059   | 0.501 ± 0.050       | 0.431 ± 0.062 |
| s2_indeterminate | balanced accuracy              | 0.685 ± 0.076 | 0.639 ± 0.065 | 0.659 ± 0.062 | 0.649 ± 0.064   | 0.594 ± 0.077       | 0.630 ± 0.060 |
| s1_2c            | specificity at 95% sensitivity | 0.685 ± 0.100 | 0.547 ± 0.104 | 0.651 ± 0.080 | 0.582 ± 0.140   | 0.555 ± 0.117       | 0.554 ± 0.115 |
| s1_2c            | macro F1                       | 0.846 ± 0.033 | 0.811 ± 0.027 | 0.843 ± 0.032 | 0.808 ± 0.041   | 0.800 ± 0.032       | 0.811 ± 0.024 |
| s1_3c            | macro F1                       | 0.698 ± 0.031 | 0.638 ± 0.042 | 0.689 ± 0.034 | 0.628 ± 0.035   | 0.641 ± 0.046       | 0.643 ± 0.038 |
| flat_10c         | macro F1                       | 0.366 ± 0.030 | 0.282 ± 0.026 | 0.353 ± 0.034 | 0.272 ± 0.030   | 0.279 ± 0.032       | 0.292 ± 0.027 |
| s2_benign        | macro F1                       | 0.394 ± 0.041 | 0.302 ± 0.035 | 0.385 ± 0.053 | 0.370 ± 0.039   | 0.347 ± 0.042       | 0.302 ± 0.042 |
| s2_malignant     | macro F1                       | 0.568 ± 0.062 | 0.415 ± 0.073 | 0.539 ± 0.051 | 0.482 ± 0.059   | 0.500 ± 0.050       | 0.428 ± 0.064 |
| s2_indeterminate | macro F1                       | 0.677 ± 0.098 | 0.637 ± 0.065 | 0.657 ± 0.063 | 0.651 ± 0.066   | 0.591 ± 0.078       | 0.627 ± 0.060 |

Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls that were already misrouted at stage 1.

|                   |   flat_10c_balanced_accuracy |   hard_balanced_accuracy |   soft_balanced_accuracy |   hard_macro_f1 |   stage1_routing_error |   subtype_accuracy_when_routed_correctly |   share_of_errors_from_routing |
|:------------------|-----------------------------:|-------------------------:|-------------------------:|----------------:|-----------------------:|-----------------------------------------:|-------------------------------:|
| elastic net       |                        0.355 |                    0.349 |                    0.354 |           0.346 |                  0.283 |                                    0.487 |                          0.435 |
| gradient boosting |                        0.282 |                    0.3   |                    0.321 |           0.3   |                  0.307 |                                    0.433 |                          0.439 |
| lasso (L1)        |                        0.368 |                    0.371 |                    0.37  |           0.368 |                  0.276 |                                    0.512 |                          0.439 |
| ridge (L2)        |                        0.281 |                    0.266 |                    0.268 |           0.265 |                  0.325 |                                    0.394 |                          0.442 |
| MLP               |                        0.293 |                    0.272 |                    0.269 |           0.269 |                  0.328 |                                    0.404 |                          0.45  |
| random forest     |                        0.274 |                    0.306 |                    0.186 |           0.302 |                  0.355 |                                    0.475 |                          0.512 |

Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):

|                   |   indeterminate epidermal |   indeterminate melanocytic |
|:------------------|--------------------------:|----------------------------:|
| elastic net       |                     0.45  |                       0.432 |
| gradient boosting |                     0.473 |                       0.413 |
| lasso (L1)        |                     0.463 |                       0.459 |
| ridge (L2)        |                     0.379 |                       0.42  |
| MLP               |                     0.42  |                       0.463 |
| random forest     |                     0.522 |                       0.516 |

## Most important questions (P(yes))

Ranked by the mean SHAP rank over all models (1 = most important). *Ridge coef* is the standardised weight (positive = pushes towards the class named in the task; for multiclass tasks, the largest |weight| over classes). *Lasso kept* is the number of the outer fits in which the lasso kept the question.

**s1_2c**

| question_id   | question                                                                     |   ridge coef |   lasso kept |   mean SHAP rank | planted as   |
|:--------------|:-----------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:-------------|
| DQ040         | Is the overall global dermoscopic pattern of the lesion globular?            |        0.415 |           25 |            1     | malignancy   |
| DQ163         | Are glomerular vessels arranged in clusters?                                 |       -0.353 |           25 |            3.167 | malignancy   |
| DQ292         | Is pink to red the predominant colour of a small elevated lesion?            |        0.336 |           25 |            4     | malignancy   |
| DQ072         | Is a brown structureless area present in the lesion?                         |       -0.327 |           25 |            5.667 | malignancy   |
| DQ370         | Does the image show any naevus of at least 5 mm?                             |        0.319 |           25 |            5.833 | malignancy   |
| DQ276         | Is hyperpigmentation of the proximal nail fold visible?                      |        0.297 |           25 |            7.333 | malignancy   |
| DQ342         | Do the lesions follow a dermatomal pattern?                                  |        0.286 |           25 |            8     | malignancy   |
| DQ168         | Is a faint whitish halo present at the lesion periphery?                     |        0.334 |           25 |            8     | malignancy   |
| DQ347         | Are scaly papules and plaques arranged in a continuous or intermittent line? |        0.318 |           25 |            8.833 | malignancy   |
| DQ090         | Is the lesion a plaque?                                                      |        0.307 |           25 |            9.833 | malignancy   |

**s1_3c**

| question_id   | question                                                          |   ridge coef |   lasso kept |   mean SHAP rank | planted as    |
|:--------------|:------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------|
| DQ040         | Is the overall global dermoscopic pattern of the lesion globular? |        0.189 |           25 |            2     | malignancy    |
| DQ064         | Are vessels arranged in clusters?                                 |        0.205 |           25 |            4.5   | indeterminate |
| DQ072         | Is a brown structureless area present in the lesion?              |        0.182 |           25 |            5.667 | malignancy    |
| DQ163         | Are glomerular vessels arranged in clusters?                      |        0.189 |           25 |            6.167 | malignancy    |
| DQ484         | Does the skin surface appear rough?                               |        0.22  |           25 |            6.5   | indeterminate |
| DQ292         | Is pink to red the predominant colour of a small elevated lesion? |        0.17  |           25 |            6.833 | malignancy    |
| DQ246         | Does the acral lesion resemble a non-healing ulcer?               |        0.217 |           25 |            8.667 | indeterminate |
| DQ378         | Are fine wrinkles visible?                                        |        0.218 |           25 |           10     | indeterminate |
| DQ370         | Does the image show any naevus of at least 5 mm?                  |        0.153 |           25 |           10.167 | malignancy    |
| DQ168         | Is a faint whitish halo present at the lesion periphery?          |        0.17  |           25 |           10.833 | malignancy    |

**flat_10c**

| question_id   | question                                                                     |   ridge coef |   lasso kept |   mean SHAP rank | planted as    |
|:--------------|:-----------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------|
| DQ040         | Is the overall global dermoscopic pattern of the lesion globular?            |        0.128 |           25 |            2.333 | malignancy    |
| DQ370         | Does the image show any naevus of at least 5 mm?                             |        0.151 |           25 |            3.333 | malignancy    |
| DQ072         | Is a brown structureless area present in the lesion?                         |        0.122 |           25 |            6.167 | malignancy    |
| DQ292         | Is pink to red the predominant colour of a small elevated lesion?            |        0.126 |           25 |            6.667 | malignancy    |
| DQ163         | Are glomerular vessels arranged in clusters?                                 |        0.13  |           25 |            6.667 | malignancy    |
| DQ064         | Are vessels arranged in clusters?                                            |        0.17  |           25 |            8.5   | indeterminate |
| DQ090         | Is the lesion a plaque?                                                      |        0.125 |           25 |            8.5   | malignancy    |
| DQ168         | Is a faint whitish halo present at the lesion periphery?                     |        0.138 |           25 |           10     | malignancy    |
| DQ484         | Does the skin surface appear rough?                                          |        0.172 |           25 |           10.5   | indeterminate |
| DQ347         | Are scaly papules and plaques arranged in a continuous or intermittent line? |        0.128 |           25 |           12.667 | malignancy    |

**s2_benign**

| question_id   | question                                                                  |   ridge coef |   lasso kept |   mean SHAP rank | planted as                           |
|:--------------|:--------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:-------------------------------------|
| DQ425         | Is the lesion colour distinct from the surrounding skin colour?           |        0.084 |           25 |            1.667 | subtype: benign vascular             |
| DQ197         | Are fine parallel lines present in a flat brown lesion?                   |        0.052 |           25 |            3     | subtype: inflammatory or infectious  |
| DQ200         | Is the lesion an erythematous crateriform nodule with central ulceration? |        0.056 |           25 |            3.333 | subtype: flat melanotic pigmentation |
| DQ283         | Does the nail plate appear wavy?                                          |        0.066 |           25 |            3.667 | subtype: benign fibro-histiocytic    |
| DQ206         | Does a perivascular white halo stand out against dark background skin?    |        0.064 |           25 |            5.5   | subtype: inflammatory or infectious  |
| DQ138         | Are pinpoint bleeding points visible where scale has been removed?        |        0.044 |           25 |            6.5   | subtype: benign melanocytic          |
| DQ091         | Is the lesion a flat macule?                                              |        0.06  |           25 |            7.333 | subtype: inflammatory or infectious  |
| DQ055         | Is a structureless pink zone (milky-red area) present in the lesion?      |        0.055 |           25 |            8.333 | subtype: benign melanocytic          |
| DQ340         | Are the lesions confined to one side of the body?                         |        0.051 |           25 |           10.5   | subtype: benign fibro-histiocytic    |
| DQ149         | Is annular hypopigmentation present around an ulcer?                      |        0.05  |           25 |           10.833 | subtype: benign melanocytic          |

**s2_malignant**

| question_id   | question                                                                      |   ridge coef |   lasso kept |   mean SHAP rank | planted as                            |
|:--------------|:------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:--------------------------------------|
| DQ209         | Does the lesion appear as a pigmented patch on dark skin?                     |        0.054 |           25 |            1.333 | subtype: malignant melanocytic        |
| DQ472         | Does the lesion look like a vascular neoplasm (uniform red to purple colour)? |        0.049 |           25 |            1.667 | subtype: malignant adnexal follicular |
| DQ499         | Does the lesion show a periodic, repeating texture?                           |        0.044 |           25 |            4     | subtype: malignant epidermal          |
| DQ146         | Are short white streaks (chrysalis structures) present?                       |        0.041 |           25 |            4.333 | subtype: malignant epidermal          |
| DQ020         | Does the lesion show a pigmented homogeneous global pattern?                  |        0.037 |           25 |            4.5   | subtype: malignant melanocytic        |
| DQ399         | Is colour asymmetry within the lesion high?                                   |        0.031 |           25 |            6     | subtype: malignant epidermal          |
| DQ203         | Is the lesion a firm-appearing pink to red papule?                            |        0.037 |           25 |            8.5   | subtype: malignant melanocytic        |
| DQ099         | Is the lesion umbilicated (centrally depressed)?                              |        0.029 |           24 |            9     | subtype: malignant adnexal follicular |
| DQ224         | Are asymmetrical pigmented follicular openings present?                       |        0.03  |           25 |            9.667 | subtype: malignant adnexal follicular |
| DQ488         | Are red areas (diffuse vascular redness) visible on the facial skin?          |        0.022 |           18 |           16.5   | uninformative                         |

**s2_indeterminate**

| question_id   | question                                                                              |   ridge coef |   lasso kept |   mean SHAP rank | planted as                         |
|:--------------|:--------------------------------------------------------------------------------------|-------------:|-------------:|-----------------:|:-----------------------------------|
| DQ190         | Does the lesion have a sharply demarcated border?                                     |       -0.161 |           24 |            1.667 | subtype: indeterminate melanocytic |
| DQ205         | Is the contrast between epidermal structures and background low because of dark skin? |       -0.168 |           24 |            1.833 | subtype: indeterminate epidermal   |
| DQ207         | Are well-circumscribed pigmented ovoid areas present on dark skin?                    |        0.119 |           24 |            3.833 | subtype: indeterminate epidermal   |
| DQ288         | Is a starburst pattern present?                                                       |       -0.123 |           24 |            4.667 | subtype: indeterminate melanocytic |
| DQ156         | Are maple-leaf-like structureless black areas present on a pink background?           |       -0.112 |           24 |            6.167 | malignancy                         |
| DQ218         | Does the visible skin appear to be Fitzpatrick type III–IV?                           |        0.12  |           24 |            7     | uninformative                      |
| DQ019         | Does the lesion show a multicomponent global pattern?                                 |        0.094 |           23 |            7     | subtype: indeterminate melanocytic |
| DQ308         | Are hairpin vessels present?                                                          |       -0.094 |           19 |            7.333 | uninformative                      |
| DQ134         | Is yellow exudate present?                                                            |       -0.084 |           23 |           11.5   | uninformative                      |
| DQ333         | Is a large congenital melanocytic naevus visible?                                     |       -0.084 |           20 |           13     | subtype: indeterminate epidermal   |

## Figures

- [calibration_flat_10c.png](figures/planted_calibration_flat_10c.png)
- [calibration_s1_2c.png](figures/planted_calibration_s1_2c.png)
- [calibration_s1_3c.png](figures/planted_calibration_s1_3c.png)
- [calibration_s2_benign.png](figures/planted_calibration_s2_benign.png)
- [calibration_s2_indeterminate.png](figures/planted_calibration_s2_indeterminate.png)
- [calibration_s2_malignant.png](figures/planted_calibration_s2_malignant.png)
- [confusion_cascade.png](figures/planted_confusion_cascade.png)
- [confusion_flat_10c.png](figures/planted_confusion_flat_10c.png)
- [confusion_s1_3c.png](figures/planted_confusion_s1_3c.png)
- [confusion_s2_benign.png](figures/planted_confusion_s2_benign.png)
- [confusion_s2_malignant.png](figures/planted_confusion_s2_malignant.png)
- [importance_ranks_coh.png](figures/planted_importance_ranks_coh.png)
- [importance_ranks_dir.png](figures/planted_importance_ranks_dir.png)
- [importance_ranks_orig.png](figures/planted_importance_ranks_orig.png)
- [indeterminate_2c_coh.png](figures/planted_indeterminate_2c_coh.png)
- [indeterminate_2c_dir.png](figures/planted_indeterminate_2c_dir.png)
- [indeterminate_2c_orig.png](figures/planted_indeterminate_2c_orig.png)
- [model_vs_ridge_coh.png](figures/planted_model_vs_ridge_coh.png)
- [model_vs_ridge_dir.png](figures/planted_model_vs_ridge_dir.png)
- [model_vs_ridge_orig.png](figures/planted_model_vs_ridge_orig.png)

## Files

- metrics_summary.csv: every metric, mean and SD over test folds, and the pooled out-of-fold value
- metrics_folds.csv, metrics_pooled_by_repeat.csv, best_params.csv: per fold / per repeat detail
- oof_predictions.parquet: out-of-fold class probabilities per image, model, source and repeat
- importance.csv: per question and output class: ridge/lasso/elastic-net weights, lasso selection count, mean |SHAP|
- question_ranking.csv: one row per question and task, all importance measures side by side
- cascade.csv, indeterminate_2c.csv, planted.csv, recovery.csv
