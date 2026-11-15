# S10 ridge: main model with leakage controls, trimmed question sets, final models

Ridge (L2 logistic regression) on the Directional Score, 446 clinical questions, 1000 images (840 dermoscopic). Nested CV: 5 repeats of 5-fold, stratified on the 10 subtypes, grouped by patient (ISIC patient where recorded, else lesion; near-duplicate pictures merged); lambda from 21 values by 3-fold grouped inner CV on log loss. Question selection and principal components are refitted inside every training split.

Groups: 943 groups for 1000 images; 43 groups hold more than one image (largest 5). Near-duplicate image pairs (difference hash within 12 of 256 bits): 1.

## s1_2c (auroc)

| question set | grouping | images | fold mean ± sd | pooled per repeat: mean (min to max) | specificity at 95% sensitivity | lambda (median) |
|---|---|---|---|---|---|---|
| all | lesion | all images | 0.840 ± 0.030 | 0.839 (0.832 to 0.842) | 0.44 | 33 |
| all | lesion | dermoscopic only | 0.824 ± 0.037 | 0.824 (0.816 to 0.828) | 0.41 | 33 |
| all | patient | all images | 0.839 ± 0.034 | 0.837 (0.828 to 0.843) | 0.45 | 50 |
| all | patient | dermoscopic only | 0.824 ± 0.039 | 0.823 (0.811 to 0.828) | 0.43 | 50 |
| lasso27 | patient | all images | 0.822 ± 0.034 | 0.821 (0.815 to 0.830) | 0.43 | 33 |
| lasso27 | patient | dermoscopic only | 0.810 ± 0.039 | 0.809 (0.802 to 0.822) | 0.41 | 33 |
| lasso60 | patient | all images | 0.832 ± 0.034 | 0.830 (0.818 to 0.839) | 0.43 | 33 |
| lasso60 | patient | dermoscopic only | 0.819 ± 0.040 | 0.817 (0.805 to 0.829) | 0.41 | 33 |
| lasso120 | patient | all images | 0.838 ± 0.034 | 0.837 (0.828 to 0.844) | 0.46 | 33 |
| lasso120 | patient | dermoscopic only | 0.825 ± 0.040 | 0.824 (0.813 to 0.834) | 0.43 | 33 |
| top27 | patient | all images | 0.770 ± 0.035 | 0.769 (0.760 to 0.776) | 0.35 | 10 |
| top27 | patient | dermoscopic only | 0.748 ± 0.040 | 0.746 (0.740 to 0.750) | 0.31 | 10 |
| top60 | patient | all images | 0.789 ± 0.040 | 0.790 (0.784 to 0.796) | 0.38 | 10 |
| top60 | patient | dermoscopic only | 0.770 ± 0.046 | 0.770 (0.763 to 0.777) | 0.31 | 10 |
| reps27 | patient | all images | 0.774 ± 0.034 | 0.772 (0.768 to 0.782) | 0.36 | 20 |
| reps27 | patient | dermoscopic only | 0.764 ± 0.036 | 0.761 (0.754 to 0.772) | 0.34 | 20 |
| reps60 | patient | all images | 0.792 ± 0.036 | 0.790 (0.784 to 0.798) | 0.37 | 33 |
| reps60 | patient | dermoscopic only | 0.778 ± 0.040 | 0.776 (0.768 to 0.783) | 0.34 | 33 |
| pca27 | patient | all images | 0.788 ± 0.039 | 0.786 (0.782 to 0.793) | 0.37 | 20 |
| pca27 | patient | dermoscopic only | 0.777 ± 0.042 | 0.774 (0.769 to 0.784) | 0.33 | 20 |
| pca60 | patient | all images | 0.818 ± 0.038 | 0.816 (0.810 to 0.819) | 0.42 | 20 |
| pca60 | patient | dermoscopic only | 0.802 ± 0.045 | 0.800 (0.792 to 0.804) | 0.39 | 20 |
| rand27 | patient | all images | 0.773 ± 0.035 (draws 0.757 to 0.783) | 0.771 (0.743 to 0.791) | 0.35 | 20 |
| rand27 | patient | dermoscopic only | 0.761 ± 0.037 (draws 0.751 to 0.766) | 0.758 (0.734 to 0.772) | 0.32 | 20 |
| rand60 | patient | all images | 0.795 ± 0.035 (draws 0.778 to 0.809) | 0.793 (0.773 to 0.817) | 0.38 | 20 |
| rand60 | patient | dermoscopic only | 0.785 ± 0.039 (draws 0.763 to 0.798) | 0.783 (0.757 to 0.806) | 0.36 | 20 |
| rand120 | patient | all images | 0.807 ± 0.037 (draws 0.795 to 0.818) | 0.806 (0.790 to 0.821) | 0.39 | 33 |
| rand120 | patient | dermoscopic only | 0.791 ± 0.041 (draws 0.779 to 0.805) | 0.789 (0.770 to 0.811) | 0.35 | 33 |
| control question | patient | all images | 0.786 ± 0.038 | n/a (n/a to n/a) | 0.27 | n/a |
| control question | patient | dermoscopic only | 0.785 ± 0.042 | n/a (n/a to n/a) | 0.30 | n/a |

Against all questions on the same folds (patient grouping):

| set | images | mean difference | folds higher |
|---|---|---|---|
| control question | all images | -0.053 ± 0.032 | 1 of 25 |
| lasso120 | all images | -0.001 ± 0.008 | 15 of 25 |
| lasso27 | all images | -0.017 ± 0.013 | 1 of 25 |
| lasso60 | all images | -0.007 ± 0.008 | 7 of 25 |
| pca27 | all images | -0.051 ± 0.018 | 0 of 25 |
| pca60 | all images | -0.021 ± 0.013 | 1 of 25 |
| rand120 | all images | -0.032 ± 0.010 | 0 of 25 |
| rand27 | all images | -0.066 ± 0.015 | 0 of 25 |
| rand60 | all images | -0.044 ± 0.012 | 0 of 25 |
| reps27 | all images | -0.065 ± 0.018 | 0 of 25 |
| reps60 | all images | -0.047 ± 0.019 | 1 of 25 |
| top27 | all images | -0.069 ± 0.024 | 0 of 25 |
| top60 | all images | -0.050 ± 0.024 | 1 of 25 |
| control question | dermoscopic only | -0.039 ± 0.036 | 4 of 25 |
| lasso120 | dermoscopic only | +0.001 ± 0.010 | 15 of 25 |
| lasso27 | dermoscopic only | -0.014 ± 0.016 | 5 of 25 |
| lasso60 | dermoscopic only | -0.005 ± 0.008 | 7 of 25 |
| pca27 | dermoscopic only | -0.047 ± 0.022 | 1 of 25 |
| pca60 | dermoscopic only | -0.022 ± 0.014 | 1 of 25 |
| rand120 | dermoscopic only | -0.033 ± 0.012 | 0 of 25 |
| rand27 | dermoscopic only | -0.063 ± 0.015 | 0 of 25 |
| rand60 | dermoscopic only | -0.039 ± 0.013 | 0 of 25 |
| reps27 | dermoscopic only | -0.060 ± 0.023 | 0 of 25 |
| reps60 | dermoscopic only | -0.046 ± 0.026 | 2 of 25 |
| top27 | dermoscopic only | -0.076 ± 0.028 | 0 of 25 |
| top60 | dermoscopic only | -0.054 ± 0.030 | 1 of 25 |

## flat_10c (balanced accuracy)

| question set | grouping | images | fold mean ± sd | pooled per repeat: mean (min to max) | macro-F1 | lambda (median) |
|---|---|---|---|---|---|---|
| all | lesion | all images | 0.490 ± 0.026 | 0.490 (0.486 to 0.497) | 0.48 | 100 |
| all | lesion | dermoscopic only | 0.488 ± 0.027 | 0.485 (0.480 to 0.492) | 0.48 | 100 |
| all | patient | all images | 0.479 ± 0.034 | 0.479 (0.469 to 0.487) | 0.47 | 100 |
| all | patient | dermoscopic only | 0.472 ± 0.042 | 0.470 (0.461 to 0.476) | 0.46 | 100 |
| lasso27 | patient | all images | 0.442 ± 0.042 | 0.442 (0.428 to 0.458) | 0.43 | 20 |
| lasso27 | patient | dermoscopic only | 0.437 ± 0.052 | 0.436 (0.418 to 0.451) | 0.43 | 20 |
| lasso60 | patient | all images | 0.456 ± 0.028 | 0.456 (0.447 to 0.466) | 0.45 | 33 |
| lasso60 | patient | dermoscopic only | 0.442 ± 0.038 | 0.442 (0.436 to 0.447) | 0.44 | 33 |
| lasso120 | patient | all images | 0.472 ± 0.033 | 0.472 (0.460 to 0.489) | 0.47 | 50 |
| lasso120 | patient | dermoscopic only | 0.458 ± 0.039 | 0.457 (0.449 to 0.473) | 0.45 | 50 |
| top27 | patient | all images | 0.411 ± 0.033 | 0.411 (0.396 to 0.416) | 0.40 | 10 |
| top27 | patient | dermoscopic only | 0.399 ± 0.035 | 0.398 (0.382 to 0.411) | 0.39 | 10 |
| top60 | patient | all images | 0.448 ± 0.032 | 0.448 (0.438 to 0.463) | 0.44 | 10 |
| top60 | patient | dermoscopic only | 0.437 ± 0.037 | 0.436 (0.421 to 0.455) | 0.43 | 10 |
| reps27 | patient | all images | 0.422 ± 0.028 | 0.422 (0.413 to 0.429) | 0.41 | 20 |
| reps27 | patient | dermoscopic only | 0.413 ± 0.035 | 0.411 (0.403 to 0.420) | 0.40 | 20 |
| reps60 | patient | all images | 0.447 ± 0.030 | 0.447 (0.442 to 0.451) | 0.44 | 33 |
| reps60 | patient | dermoscopic only | 0.436 ± 0.040 | 0.435 (0.432 to 0.438) | 0.43 | 33 |
| pca27 | patient | all images | 0.453 ± 0.026 | 0.453 (0.447 to 0.461) | 0.45 | 10 |
| pca27 | patient | dermoscopic only | 0.450 ± 0.039 | 0.448 (0.439 to 0.455) | 0.44 | 10 |
| pca60 | patient | all images | 0.455 ± 0.031 | 0.455 (0.447 to 0.464) | 0.45 | 20 |
| pca60 | patient | dermoscopic only | 0.453 ± 0.037 | 0.451 (0.446 to 0.459) | 0.45 | 20 |
| rand27 | patient | all images | 0.415 ± 0.025 (draws 0.403 to 0.423) | 0.415 (0.398 to 0.434) | 0.41 | 20 |
| rand27 | patient | dermoscopic only | 0.404 ± 0.030 (draws 0.383 to 0.414) | 0.403 (0.377 to 0.425) | 0.40 | 20 |
| rand60 | patient | all images | 0.438 ± 0.027 (draws 0.428 to 0.453) | 0.438 (0.425 to 0.465) | 0.43 | 20 |
| rand60 | patient | dermoscopic only | 0.428 ± 0.036 (draws 0.417 to 0.439) | 0.426 (0.410 to 0.444) | 0.42 | 20 |
| rand120 | patient | all images | 0.467 ± 0.029 (draws 0.464 to 0.474) | 0.467 (0.453 to 0.478) | 0.46 | 33 |
| rand120 | patient | dermoscopic only | 0.459 ± 0.037 (draws 0.454 to 0.462) | 0.457 (0.445 to 0.470) | 0.45 | 33 |

Against all questions on the same folds (patient grouping):

| set | images | mean difference | folds higher |
|---|---|---|---|
| lasso120 | all images | -0.006 ± 0.024 | 12 of 25 |
| lasso27 | all images | -0.036 ± 0.036 | 2 of 25 |
| lasso60 | all images | -0.022 ± 0.025 | 4 of 25 |
| pca27 | all images | -0.025 ± 0.026 | 3 of 25 |
| pca60 | all images | -0.024 ± 0.022 | 2 of 25 |
| rand120 | all images | -0.012 ± 0.020 | 8 of 25 |
| rand27 | all images | -0.064 ± 0.028 | 0 of 25 |
| rand60 | all images | -0.041 ± 0.021 | 0 of 25 |
| reps27 | all images | -0.057 ± 0.036 | 1 of 25 |
| reps60 | all images | -0.032 ± 0.028 | 3 of 25 |
| top27 | all images | -0.067 ± 0.039 | 0 of 25 |
| top60 | all images | -0.030 ± 0.029 | 2 of 25 |
| lasso120 | dermoscopic only | -0.013 ± 0.027 | 9 of 25 |
| lasso27 | dermoscopic only | -0.034 ± 0.044 | 7 of 25 |
| lasso60 | dermoscopic only | -0.029 ± 0.027 | 6 of 25 |
| pca27 | dermoscopic only | -0.022 ± 0.029 | 5 of 25 |
| pca60 | dermoscopic only | -0.019 ± 0.028 | 4 of 25 |
| rand120 | dermoscopic only | -0.013 ± 0.022 | 8 of 25 |
| rand27 | dermoscopic only | -0.068 ± 0.030 | 0 of 25 |
| rand60 | dermoscopic only | -0.044 ± 0.021 | 1 of 25 |
| reps27 | dermoscopic only | -0.059 ± 0.039 | 0 of 25 |
| reps60 | dermoscopic only | -0.035 ± 0.035 | 2 of 25 |
| top27 | dermoscopic only | -0.073 ± 0.045 | 0 of 25 |
| top60 | dermoscopic only | -0.034 ± 0.033 | 2 of 25 |

## Stability of the chosen questions (refitted in each of the 25 outer training sets)

- flat_10c, lasso120: 68 of 120 questions chosen in at least 20 of 25 outer folds; 270 different questions chosen overall.
- flat_10c, lasso27: 14 of 27 questions chosen in at least 20 of 25 outer folds; 63 different questions chosen overall.
- flat_10c, lasso60: 36 of 60 questions chosen in at least 20 of 25 outer folds; 138 different questions chosen overall.
- flat_10c, reps27: 13 of 27 questions chosen in at least 20 of 25 outer folds; 89 different questions chosen overall.
- flat_10c, reps60: 33 of 60 questions chosen in at least 20 of 25 outer folds; 139 different questions chosen overall.
- flat_10c, top27: 24 of 27 questions chosen in at least 20 of 25 outer folds; 39 different questions chosen overall.
- flat_10c, top60: 53 of 60 questions chosen in at least 20 of 25 outer folds; 83 different questions chosen overall.
- s1_2c, lasso120: 57 of 120 questions chosen in at least 20 of 25 outer folds; 304 different questions chosen overall.
- s1_2c, lasso27: 14 of 27 questions chosen in at least 20 of 25 outer folds; 68 different questions chosen overall.
- s1_2c, lasso60: 27 of 60 questions chosen in at least 20 of 25 outer folds; 177 different questions chosen overall.
- s1_2c, reps27: 12 of 27 questions chosen in at least 20 of 25 outer folds; 90 different questions chosen overall.
- s1_2c, reps60: 32 of 60 questions chosen in at least 20 of 25 outer folds; 149 different questions chosen overall.
- s1_2c, top27: 19 of 27 questions chosen in at least 20 of 25 outer folds; 47 different questions chosen overall.
- s1_2c, top60: 43 of 60 questions chosen in at least 20 of 25 outer folds; 99 different questions chosen overall.

## Final models (fitted on all labelled images; not scored)

| task | set | questions | images | lambda |
|---|---|---|---|---|
| s1_2c | all | 446 | 800 | 50 |
| s1_2c | lasso27 | 27 | 800 | 33 |
| s1_2c | lasso60 | 60 | 800 | 33 |
| s1_2c | lasso120 | 120 | 800 | 50 |
| s1_2c | reps27 | 27 | 800 | 33 |
| s1_2c | reps60 | 60 | 800 | 20 |
| flat_10c | all | 446 | 1000 | 100 |
| flat_10c | lasso27 | 27 | 1000 | 20 |
| flat_10c | lasso60 | 60 | 1000 | 33 |
| flat_10c | lasso120 | 120 | 1000 | 50 |
| flat_10c | reps27 | 27 | 1000 | 10 |
| flat_10c | reps60 | 60 | 1000 | 20 |
