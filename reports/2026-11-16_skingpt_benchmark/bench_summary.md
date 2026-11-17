# S13 bench: ridge on question answers against SkinGPT-R1, same folds

1000 images (840 dermoscopic); outer folds exactly as in S10 (5 repeats of 5-fold, stratified on the 10 subtypes, grouped by patient with near-duplicate pictures merged). Questions: ridge on the Directional Score of 446 questions (S10 'all', reproduced fold by fold, see check_s10.csv). SkinGPT-R1 embeddings (3584 numbers per image): ridge with scaling and lambda refitted in every outer training split; main embedding fixed in advance: img_L28 (SkinGPT-R1 layer 28 (image tokens), ridge). SkinGPT-R1 diagnosis: its probability over 16 candidate labels, without reasoning.

## s1_2c (auroc)

| method | images | fold mean ± sd | pooled per repeat: mean (min to max) | specificity at 95% sensitivity | lambda (median) |
|---|---|---|---|---|---|
| ridge on 446 question answers (Sonnet) | all images | 0.839 ± 0.034 | 0.837 (0.828 to 0.843) | 0.45 | 50 |
| ridge on 446 question answers (Sonnet) | dermoscopic only | 0.824 ± 0.039 | 0.823 (0.811 to 0.828) | 0.43 | 50 |
| SkinGPT-R1 layer 28 (image tokens), ridge | all images | 0.788 ± 0.030 | 0.786 (0.776 to 0.794) | 0.36 | 500 |
| SkinGPT-R1 layer 28 (image tokens), ridge | dermoscopic only | 0.775 ± 0.033 | 0.773 (0.761 to 0.782) | 0.33 | 500 |
| SkinGPT-R1 layer 21 (image tokens), ridge | all images | 0.787 ± 0.038 | 0.784 (0.778 to 0.793) | 0.32 | 500 |
| SkinGPT-R1 layer 21 (image tokens), ridge | dermoscopic only | 0.772 ± 0.037 | 0.769 (0.760 to 0.779) | 0.29 | 500 |
| SkinGPT-R1 layer 14 (image tokens), ridge | all images | 0.792 ± 0.038 | 0.791 (0.780 to 0.799) | 0.35 | 500 |
| SkinGPT-R1 layer 14 (image tokens), ridge | dermoscopic only | 0.776 ± 0.037 | 0.775 (0.762 to 0.787) | 0.28 | 500 |
| SkinGPT-R1 vision encoder, ridge | all images | 0.814 ± 0.033 | 0.812 (0.800 to 0.820) | 0.40 | 500 |
| SkinGPT-R1 vision encoder, ridge | dermoscopic only | 0.794 ± 0.033 | 0.792 (0.778 to 0.802) | 0.37 | 500 |
| SkinGPT-R1 layer 28 (last token), ridge | all images | 0.766 ± 0.035 | 0.763 (0.755 to 0.773) | 0.33 | 500 |
| SkinGPT-R1 layer 28 (last token), ridge | dermoscopic only | 0.753 ± 0.035 | 0.750 (0.741 to 0.762) | 0.33 | 500 |
| SkinGPT-R1 diagnosis, ridge readout | all images | 0.698 ± 0.033 | 0.696 (0.690 to 0.702) | 0.24 | 5 |
| SkinGPT-R1 diagnosis, ridge readout | dermoscopic only | 0.680 ± 0.037 | 0.678 (0.674 to 0.686) | 0.23 | 5 |
| SkinGPT-R1 diagnosis, zero-shot | all images | 0.599 ± 0.032 | 0.599 (0.599 to 0.599) | 0.16 | n/a |
| SkinGPT-R1 diagnosis, zero-shot | dermoscopic only | 0.619 ± 0.035 | 0.618 (0.618 to 0.618) | 0.16 | n/a |
| Sonnet asked "malignant?" | all images | 0.786 ± 0.038 | 0.786 (0.786 to 0.786) | 0.27 | n/a |
| Sonnet asked "malignant?" | dermoscopic only | 0.785 ± 0.042 | 0.785 (0.785 to 0.785) | 0.30 | n/a |

Against the questions on the same folds (positive = the method scored higher):

| method | images | mean difference | folds higher |
|---|---|---|---|
| SkinGPT-R1 layer 28 (image tokens), ridge | all images | -0.051 ± 0.039 | 2 of 25 |
| SkinGPT-R1 layer 28 (image tokens), ridge | dermoscopic only | -0.049 ± 0.046 | 4 of 25 |
| SkinGPT-R1 layer 21 (image tokens), ridge | all images | -0.051 ± 0.038 | 2 of 25 |
| SkinGPT-R1 layer 21 (image tokens), ridge | dermoscopic only | -0.052 ± 0.041 | 5 of 25 |
| SkinGPT-R1 layer 14 (image tokens), ridge | all images | -0.046 ± 0.036 | 3 of 25 |
| SkinGPT-R1 layer 14 (image tokens), ridge | dermoscopic only | -0.047 ± 0.039 | 5 of 25 |
| SkinGPT-R1 vision encoder, ridge | all images | -0.024 ± 0.032 | 6 of 25 |
| SkinGPT-R1 vision encoder, ridge | dermoscopic only | -0.030 ± 0.037 | 6 of 25 |
| SkinGPT-R1 layer 28 (last token), ridge | all images | -0.073 ± 0.039 | 2 of 25 |
| SkinGPT-R1 layer 28 (last token), ridge | dermoscopic only | -0.070 ± 0.043 | 3 of 25 |
| SkinGPT-R1 diagnosis, ridge readout | all images | -0.141 ± 0.035 | 0 of 25 |
| SkinGPT-R1 diagnosis, ridge readout | dermoscopic only | -0.144 ± 0.039 | 0 of 25 |
| SkinGPT-R1 diagnosis, zero-shot | all images | -0.240 ± 0.035 | 0 of 25 |
| SkinGPT-R1 diagnosis, zero-shot | dermoscopic only | -0.205 ± 0.041 | 0 of 25 |
| Sonnet asked "malignant?" | all images | -0.053 ± 0.032 | 1 of 25 |
| Sonnet asked "malignant?" | dermoscopic only | -0.039 ± 0.036 | 4 of 25 |

## flat_10c (balanced accuracy)

| method | images | fold mean ± sd | pooled per repeat: mean (min to max) | macro-F1 | lambda (median) |
|---|---|---|---|---|---|
| ridge on 446 question answers (Sonnet) | all images | 0.479 ± 0.034 | 0.479 (0.469 to 0.487) | 0.47 | 100 |
| ridge on 446 question answers (Sonnet) | dermoscopic only | 0.472 ± 0.042 | 0.470 (0.461 to 0.476) | 0.46 | 100 |
| SkinGPT-R1 layer 28 (image tokens), ridge | all images | 0.473 ± 0.038 | 0.473 (0.462 to 0.487) | 0.47 | 500 |
| SkinGPT-R1 layer 28 (image tokens), ridge | dermoscopic only | 0.447 ± 0.042 | 0.445 (0.437 to 0.453) | 0.44 | 500 |
| SkinGPT-R1 layer 21 (image tokens), ridge | all images | 0.475 ± 0.034 | 0.475 (0.465 to 0.485) | 0.47 | 500 |
| SkinGPT-R1 layer 21 (image tokens), ridge | dermoscopic only | 0.444 ± 0.037 | 0.443 (0.431 to 0.454) | 0.44 | 500 |
| SkinGPT-R1 layer 14 (image tokens), ridge | all images | 0.485 ± 0.030 | 0.485 (0.471 to 0.501) | 0.48 | 500 |
| SkinGPT-R1 layer 14 (image tokens), ridge | dermoscopic only | 0.454 ± 0.036 | 0.452 (0.437 to 0.463) | 0.45 | 500 |
| SkinGPT-R1 vision encoder, ridge | all images | 0.484 ± 0.025 | 0.484 (0.470 to 0.495) | 0.48 | 500 |
| SkinGPT-R1 vision encoder, ridge | dermoscopic only | 0.459 ± 0.032 | 0.458 (0.444 to 0.469) | 0.45 | 500 |
| SkinGPT-R1 layer 28 (last token), ridge | all images | 0.438 ± 0.038 | 0.438 (0.431 to 0.444) | 0.43 | 500 |
| SkinGPT-R1 layer 28 (last token), ridge | dermoscopic only | 0.415 ± 0.048 | 0.413 (0.407 to 0.419) | 0.41 | 500 |
| SkinGPT-R1 diagnosis, ridge readout | all images | 0.367 ± 0.038 | 0.367 (0.357 to 0.373) | 0.36 | 5 |
| SkinGPT-R1 diagnosis, ridge readout | dermoscopic only | 0.348 ± 0.050 | 0.346 (0.334 to 0.353) | 0.34 | 5 |
| SkinGPT-R1 diagnosis, zero-shot | all images | 0.241 ± 0.029 | 0.241 (0.241 to 0.241) | 0.17 | n/a |
| SkinGPT-R1 diagnosis, zero-shot | dermoscopic only | 0.231 ± 0.028 | 0.231 (0.231 to 0.231) | 0.17 | n/a |

Against the questions on the same folds (positive = the method scored higher):

| method | images | mean difference | folds higher |
|---|---|---|---|
| SkinGPT-R1 layer 28 (image tokens), ridge | all images | -0.006 ± 0.038 | 10 of 25 |
| SkinGPT-R1 layer 28 (image tokens), ridge | dermoscopic only | -0.025 ± 0.042 | 8 of 25 |
| SkinGPT-R1 layer 21 (image tokens), ridge | all images | -0.003 ± 0.039 | 10 of 25 |
| SkinGPT-R1 layer 21 (image tokens), ridge | dermoscopic only | -0.027 ± 0.040 | 6 of 25 |
| SkinGPT-R1 layer 14 (image tokens), ridge | all images | +0.006 ± 0.032 | 14 of 25 |
| SkinGPT-R1 layer 14 (image tokens), ridge | dermoscopic only | -0.017 ± 0.035 | 10 of 25 |
| SkinGPT-R1 vision encoder, ridge | all images | +0.006 ± 0.035 | 13 of 25 |
| SkinGPT-R1 vision encoder, ridge | dermoscopic only | -0.013 ± 0.046 | 11 of 25 |
| SkinGPT-R1 layer 28 (last token), ridge | all images | -0.041 ± 0.034 | 3 of 25 |
| SkinGPT-R1 layer 28 (last token), ridge | dermoscopic only | -0.057 ± 0.037 | 2 of 25 |
| SkinGPT-R1 diagnosis, ridge readout | all images | -0.112 ± 0.046 | 0 of 25 |
| SkinGPT-R1 diagnosis, ridge readout | dermoscopic only | -0.124 ± 0.054 | 0 of 25 |
| SkinGPT-R1 diagnosis, zero-shot | all images | -0.238 ± 0.042 | 0 of 25 |
| SkinGPT-R1 diagnosis, zero-shot | dermoscopic only | -0.241 ± 0.042 | 0 of 25 |

## Lambda

| task | method | lambda median (range) | folds at the grid edge |
|---|---|---|---|
| flat_10c | ridge on 446 question answers (Sonnet) | 100 (50 to 100) | 0 of 25 |
| flat_10c | SkinGPT-R1 diagnosis, ridge readout | 5 (2 to 10) | 0 of 25 |
| flat_10c | SkinGPT-R1 vision encoder, ridge | 500 (500 to 1000) | 0 of 25 |
| flat_10c | SkinGPT-R1 layer 14 (image tokens), ridge | 500 (500 to 500) | 0 of 25 |
| flat_10c | SkinGPT-R1 layer 21 (image tokens), ridge | 500 (500 to 1000) | 0 of 25 |
| flat_10c | SkinGPT-R1 layer 28 (image tokens), ridge | 500 (500 to 1000) | 0 of 25 |
| flat_10c | SkinGPT-R1 layer 28 (last token), ridge | 500 (333 to 1000) | 0 of 25 |
| s1_2c | ridge on 446 question answers (Sonnet) | 50 (33 to 100) | 0 of 25 |
| s1_2c | SkinGPT-R1 diagnosis, ridge readout | 5 (1 to 20) | 1 of 25 |
| s1_2c | SkinGPT-R1 vision encoder, ridge | 500 (500 to 1000) | 0 of 25 |
| s1_2c | SkinGPT-R1 layer 14 (image tokens), ridge | 500 (333 to 1000) | 0 of 25 |
| s1_2c | SkinGPT-R1 layer 21 (image tokens), ridge | 500 (333 to 2000) | 0 of 25 |
| s1_2c | SkinGPT-R1 layer 28 (image tokens), ridge | 500 (333 to 1000) | 0 of 25 |
| s1_2c | SkinGPT-R1 layer 28 (last token), ridge | 500 (333 to 2000) | 0 of 25 |

## SkinGPT-R1's first-choice label by subtype (all images)

| row_0                        |   Actinic keratosis |   Angiokeratoma |   Atypical melanocytic proliferation |   Basal cell carcinoma |   Dermatofibroma |   Fibroepithelial polyp |   Melanocytic nevus |   Melanoma |   Pyogenic granuloma |   Scar |   Seborrheic keratosis |   Solar lentigo |   Verruca |
|:-----------------------------|--------------------:|----------------:|-------------------------------------:|-----------------------:|-----------------:|------------------------:|--------------------:|-----------:|---------------------:|-------:|-----------------------:|----------------:|----------:|
| benign fibro-histiocytic     |                  14 |               0 |                                    0 |                      0 |                0 |                       0 |                   9 |         25 |                    7 |      0 |                      1 |              44 |         0 |
| benign melanocytic           |                   0 |               0 |                                    0 |                      0 |                0 |                       0 |                  40 |         26 |                    1 |      0 |                      0 |              33 |         0 |
| benign vascular              |                   3 |               0 |                                    0 |                      1 |                0 |                       0 |                   2 |         44 |                   48 |      0 |                      0 |               2 |         0 |
| flat melanotic pigmentation  |                   3 |               0 |                                    0 |                      0 |                0 |                       0 |                   3 |         22 |                    0 |      1 |                      2 |              69 |         0 |
| indeterminate epidermal      |                  23 |               0 |                                    1 |                      2 |                0 |                       0 |                   3 |         17 |                    8 |      0 |                      9 |              37 |         0 |
| indeterminate melanocytic    |                   0 |               0 |                                    0 |                      0 |                0 |                       0 |                  23 |         18 |                    1 |      0 |                      0 |              58 |         0 |
| inflammatory or infectious   |                  14 |               1 |                                    0 |                      8 |                1 |                       1 |                   2 |         36 |                   17 |      1 |                      8 |              11 |         0 |
| malignant adnexal follicular |                  26 |               0 |                                    0 |                      1 |                0 |                       0 |                   5 |         29 |                   18 |      1 |                      3 |              16 |         1 |
| malignant epidermal          |                  23 |               0 |                                    0 |                      6 |                0 |                       0 |                   1 |         37 |                   14 |      2 |                      9 |               7 |         1 |
| malignant melanocytic        |                   3 |               0 |                                    0 |                      2 |                0 |                       0 |                   9 |         51 |                    4 |      0 |                      1 |              30 |         0 |
