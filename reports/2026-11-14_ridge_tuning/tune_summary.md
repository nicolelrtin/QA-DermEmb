# S8 tune: choosing the ridge (L2) penalty

1000 images, 446 questions (artefact questions and the control question left out, as in S6). Outer design as in S6: 5 repeats of 5-fold, stratified on the 10 subtypes and grouped by lesion; 3-fold inner CV inside each outer training set. Penalty grid: 21 values of C from 1e-05 to 1 (lambda = 1/C from 1 to 100,000).

## How to read this

- *Inner* scores are what a selection rule sees: the model is fitted on two thirds of an outer training set and scored on the remaining third, averaged over the three inner folds.
- *Outer* scores are on the outer test fold, images the rule never saw. The outer curve over all penalties is drawn only to show how flat or peaked it is; picking the penalty from it would be cheating.
- *Fold mean* is the mean over the 25 outer test folds (about 160 images each for benign vs malignant). *Pooled* is the score of one repeat with its 5 test folds put together (all 800 or 1,000 images once); its spread over the 5 repeats is what a single 5x3 nested CV, without repeats, would vary by.

## Check against S6

Using S6's 7-value grid and log loss, this stage reproduces S6's chosen penalty and outer score in 100 of 100 outer folds (all).

## Selection rules

### s1_2c (auroc)

| source | rule | fold mean ± sd | pooled per repeat: mean (min to max) | lambda chosen: median (range); most common (share of folds) |
|---|---|---|---|---|
| Directional Score | lowest inner log loss (as in S6) | 0.840 ± 0.030 | 0.839 (0.832 to 0.842) | 33 (20 to 50); 33 (52%) |
| Directional Score | best inner AUROC / balanced accuracy | 0.839 ± 0.031 | 0.837 (0.829 to 0.842) | 20 (10 to 50); 20 (40%) |
| Directional Score | strongest penalty within 1 SE of lowest log loss | 0.829 ± 0.030 | 0.827 (0.822 to 0.834) | 100 (50 to 500); 50 (28%) |
| Directional Score | S6's 7-value grid, log loss | 0.839 ± 0.030 | 0.838 (0.832 to 0.843) | 33 (33 to 100); 33 (88%) |
| P(yes) | lowest inner log loss (as in S6) | 0.820 ± 0.034 | 0.819 (0.814 to 0.826) | 100 (50 to 333); 100 (64%) |
| P(yes) | best inner AUROC / balanced accuracy | 0.819 ± 0.034 | 0.817 (0.812 to 0.826) | 100 (33 to 200); 100 (68%) |
| P(yes) | strongest penalty within 1 SE of lowest log loss | 0.785 ± 0.043 | 0.771 (0.746 to 0.790) | 2,000 (100 to 100,000); 5,000 (16%) |
| P(yes) | S6's 7-value grid, log loss | 0.819 ± 0.034 | 0.818 (0.813 to 0.826) | 100 (33 to 333); 100 (76%) |

### flat_10c (balanced accuracy)

| source | rule | fold mean ± sd | pooled per repeat: mean (min to max) | lambda chosen: median (range); most common (share of folds) |
|---|---|---|---|---|
| Directional Score | lowest inner log loss (as in S6) | 0.490 ± 0.026 | 0.490 (0.486 to 0.497) | 100 (50 to 100); 100 (56%) |
| Directional Score | best inner AUROC / balanced accuracy | 0.486 ± 0.025 | 0.486 (0.471 to 0.493) | 50 (2 to 333); 33 (28%) |
| Directional Score | strongest penalty within 1 SE of lowest log loss | 0.486 ± 0.027 | 0.486 (0.478 to 0.497) | 100 (100 to 333); 100 (56%) |
| Directional Score | S6's 7-value grid, log loss | 0.487 ± 0.026 | 0.487 (0.483 to 0.494) | 100 (100 to 100); 100 (100%) |
| P(yes) | lowest inner log loss (as in S6) | 0.488 ± 0.025 | 0.488 (0.477 to 0.498) | 100 (100 to 100); 100 (100%) |
| P(yes) | best inner AUROC / balanced accuracy | 0.482 ± 0.029 | 0.482 (0.473 to 0.499) | 100 (2 to 333); 100 (40%) |
| P(yes) | strongest penalty within 1 SE of lowest log loss | 0.480 ± 0.025 | 0.480 (0.470 to 0.487) | 333 (200 to 500); 200 (48%) |
| P(yes) | S6's 7-value grid, log loss | 0.488 ± 0.025 | 0.488 (0.477 to 0.498) | 100 (100 to 100); 100 (100%) |

## Shape of the curves

Mean over the 25 outer folds at each penalty: inner log loss, inner primary score, and the outer primary score (for reference only). The best value on each curve and how much the score changes over a factor of 10 in lambda around it.

| source | task | lambda at lowest inner log loss | lambda at best inner score | lambda at best outer score | outer score: best / at 10x weaker / at 10x stronger penalty |
|---|---|---|---|---|---|
| Directional Score | s1_2c | 33 | 20 | 20 | 0.841 / 0.830 / 0.828 |
| Directional Score | flat_10c | 100 | 50 | 20 | 0.492 / 0.472 / 0.484 |
| P(yes) | s1_2c | 100 | 100 | 100 | 0.821 / 0.804 / 0.803 |
| P(yes) | flat_10c | 100 | 100 | 100 | 0.488 / 0.479 / 0.455 |

Figures: `figures/penalty_curves.png`, `figures/chosen_lambda.png`, `figures/repeat_spread.png`.
