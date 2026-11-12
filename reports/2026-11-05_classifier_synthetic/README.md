# Classifier stage checked on simulated answers (2026-11-05)

**Question:** before any real answers exist for the 1,000 beta images, does the new classifier stage (S6, `stage=classify`) do what we need? Specifically: does it score models without leaking test images into training, does it find the questions that carry signal, and do the stage-1, stage-2 and cascade outputs make sense?

The labels here are the real beta-set labels. Only the answers are simulated. The numbers below test the code. They say nothing about how well QA-Emb will classify real images.

## Runs covered

| Run | Settings | What it is |
|---|---|---|
| `20261105-180246_synthetic_planted` | `SYN_planted`: simulated answers with planted informative questions, effect 0.75 on the log-odds scale; all 6 models, 6 tasks and 3 feature sources; 5 repeats of 5-fold CV | the main check |
| `20261105-191256_synthetic_null` | `SYN_null`: the same simulation with effect 0, so no question carries any signal; otherwise identical | the leakage check |
| `20261105-191031` to `-191225` (`_null_seed_check`, 6 runs) | `SYN_null` with seeds 1–6; ridge and lasso; the 2-class and indeterminate-subtype tasks only; P(yes) only | follow-up to one high null score (see below) |
| `20261105-183910_synthetic_null` | `SYN_null`, seed 0 | crashed while writing its summary (with nothing planted, the recovery table was empty); the code was fixed and the run repeated as `20261105-191256_synthetic_null`, in the same stage directory |
| `20261105-172500_synthetic_planted` | as the main run, before two fixes | superseded: random forest and MLP decisions were biased (see Changes) |
| `20261105-164145_debug`, `-164346_quick_all_models`, `-171133_refactor_check` | `SYN_quick` (one repeat, P(yes) only) | development runs |

Stage outputs: planted run in `outputs/qaemb/classify/04652347e0cd/`, null run in `outputs/qaemb/classify/636270338498/` (its summary is copied as [null_summary.md](null_summary.md)). Its readable summary is copied here as [planted_summary.md](planted_summary.md), along with [planted_metrics_summary.csv](planted_metrics_summary.csv), [planted_recovery.csv](planted_recovery.csv), [planted_cascade.csv](planted_cascade.csv) and selected figures in [figures/](figures/).

## What was simulated

Each image keeps its manifest label. For each image and question, we drew Claude's P(yes) from a logistic model. A question-specific baseline sets most answers near "no" (12–15% yes overall, as in the proof of concept). A shift shared by all questions of one topic on one image makes questions within a topic correlate. Each answer also gets its own noise. The stated confidence rises with how far P(yes) is from 0.5, with noise, matching the proof-of-concept relation (r = 0.84). The Directional and Coherence Scores were then computed from P(yes) and confidence exactly as S4 now does for real answers.

Of the 446 non-artefact questions, 68 were made informative:
- 20 are more often "yes" in the three malignant subtypes, and half as much in the two indeterminate subtypes;
- 8 are more often "yes" in the two indeterminate subtypes;
- 40 are each more often "yes" in a single subtype (4 per subtype).

One in five of the planted effects points the other way, towards "no". The size of the effect, 0.75 on the log-odds scale, was chosen so that the 2-class task lands near AUROC 0.9 rather than at the ceiling. At an effect of 1.5 (the first debug run), every model reached 0.998 and the models could not be told apart.

## Validity checks

**No leakage.** With no informative questions (`20261105-191256_synthetic_null`), every model on every task should score at chance. Across the 6 models, 3 feature sources and 6 tasks (fold means over 25 test folds, seed 0):

| Task | Chance | Range of fold-mean scores |
|---|---|---|
| benign vs malignant, AUROC | 0.50 | 0.51–0.55 |
| benign / indeterminate / malignant, balanced accuracy | 0.33 | 0.32–0.36 |
| 10 subtypes in one model, balanced accuracy | 0.10 | 0.10–0.12 |
| benign subtypes (5), balanced accuracy | 0.20 | 0.18–0.23 |
| malignant subtypes (3), balanced accuracy | 0.33 | 0.29–0.35 |
| indeterminate subtypes (2), balanced accuracy | 0.50 | 0.50–0.60 |
| cascade, 10 subtypes, balanced accuracy | 0.10 | 0.08–0.12 |

One task stood out: the two indeterminate subtypes, where ridge, the random forest and the MLP all scored 0.58–0.59 on P(yes), and ridge did so for all three sources. Everything here shares one simulated dataset of 200 images, so these scores are not independent evidence. We repeated the task with ridge and the lasso on six newly simulated datasets (seeds 1–6). Ridge scored 0.54, 0.46, 0.48, 0.49, 0.51 and 0.46, which average 0.49 and fall on both sides of chance; the lasso scored 0.49–0.51. So the 0.59 belongs to that one draw, and we see no sign of leakage. With nothing to find, the lasso keeps no questions at all and predicts the same for every image. Its AUROC is then at chance and its macro-F1 falls below chance, which is the correct behaviour. The lesson for the real data: a single score on the 200 indeterminate images can be off by about ±0.08 by chance alone.

**The planted questions are found.** For each task we ranked the questions by each importance measure. We then asked how often a planted question that matters for that task ranks above an uninformative one. This is the AUROC of the ranking: 0.5 means the measure finds nothing, 1 means every relevant question ranks above every uninformative one. P(yes) features:

| Task (relevant questions) | ridge \|weight\| | lasso: times kept (of 25) | SHAP, lasso | SHAP, ridge | SHAP, random forest | SHAP, gradient boosting | SHAP, MLP |
|---|---|---|---|---|---|---|---|
| benign vs malignant (52) | 0.79 | 0.80 | 0.77 | 0.77 | 0.83 | 0.82 | 0.77 |
| benign / indeterminate / malignant (68) | 0.80 | 0.79 | 0.81 | 0.79 | 0.84 | 0.82 | 0.80 |
| 10 subtypes in one model (68) | 1.00 | 0.93 | 0.99 | 0.93 | 0.96 | 0.97 | 0.94 |
| benign subtypes (20) | 1.00 | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 |
| malignant subtypes (12) | 1.00 | 0.99 | 0.99 | 0.94 | 0.91 | 0.92 | 0.94 |
| indeterminate subtypes (8) | 0.95 | 0.95 | 0.91 | 0.94 | 0.99 | 0.97 | 0.93 |

For stage 1 the figure is lower than it first looks, because "relevant" includes the 40 single-subtype questions. A question that marks one benign subtype differs between benign and malignant by only a fifth of its effect. Counting only the questions planted for stage 1 (the 20 malignancy questions for the 2-class task, those plus the 8 indeterminate questions for the 3-class task), every measure separates them almost perfectly. Ridge weight and mean SHAP rank give 1.00, and the lasso count gives 0.997 and 0.983. The lasso kept the malignancy questions in all 25 of 25 fits (median), against a median of 2 of 25 for uninformative questions. The importance tables therefore do what we want: they rank strong signal first and weak signal next.

## Results on simulated answers

Fold means ± SD over the 25 test folds, P(yes) features. AUROC for the 2-class task; balanced accuracy (the average over classes of the share predicted correctly) for the others.

| Task | Lasso | Ridge | Elastic net | Random forest | Gradient boosting | MLP |
|---|---|---|---|---|---|---|
| benign vs malignant (AUROC) | 0.93 ± 0.02 | 0.89 ± 0.02 | 0.92 ± 0.02 | 0.93 ± 0.02 | 0.92 ± 0.02 | 0.89 ± 0.02 |
| benign / indeterminate / malignant | 0.71 ± 0.03 | 0.63 ± 0.04 | 0.70 ± 0.03 | 0.69 ± 0.03 | 0.67 ± 0.04 | 0.64 ± 0.04 |
| 10 subtypes in one model | 0.36 ± 0.03 | 0.27 ± 0.03 | 0.35 ± 0.03 | 0.28 ± 0.04 | 0.30 ± 0.02 | 0.28 ± 0.03 |
| benign subtypes (5) | 0.41 ± 0.04 | 0.30 ± 0.04 | 0.39 ± 0.05 | 0.38 ± 0.03 | 0.35 ± 0.05 | 0.29 ± 0.04 |
| malignant subtypes (3) | 0.55 ± 0.06 | 0.41 ± 0.07 | 0.54 ± 0.06 | 0.48 ± 0.05 | 0.49 ± 0.06 | 0.42 ± 0.06 |
| indeterminate subtypes (2) | 0.68 ± 0.07 | 0.67 ± 0.07 | 0.68 ± 0.07 | 0.66 ± 0.07 | 0.66 ± 0.06 | 0.66 ± 0.06 |

Full tables, with all metrics and SDs and the Directional and Coherence versions, are in [planted_summary.md](planted_summary.md).

- **The lasso and elastic net lead, and ridge trails, but only because of how we built the simulation.** We planted a few strong questions among many empty ones, which is the situation the lasso is built for. Whether real Claude answers look like this is exactly what the real run will tell us, so this ordering should not be carried over.
- **The two-stage cascade matched the flat 10-class model** (lasso: 0.37 against 0.36). About 27% of images were sent to the wrong stage-1 class. Among the images routed correctly, half got the right subtype. 43% of all wrong subtype calls were already wrong at stage 1. These are the numbers to watch on real data: they say whether stage 1 or stage 2 is the bottleneck.
- **The soft cascade is not reliable for models with flat probabilities.** The soft cascade multiplies the stage-1 and stage-2 probabilities. When the stage-2 probabilities are nearly uniform, it favours whichever group has the fewest subtypes (indeterminate has 2, benign 5). The random forest's soft cascade dropped to 0.20 while its hard routing gave 0.33. The hard route should be the main cascade score.
- **Indeterminate images scored by the 2-class model** sat between benign and malignant (lasso: median P(malignant) 0.45, against 0.15 for benign and 0.84 for malignant images), as the simulation built in ([figure](figures/planted_indeterminate_2c_orig.png)).
- **The three feature sources** gave nearly the same scores. This is expected, since the simulation ties confidence to P(yes) much as Claude does (see the next section).

Figures: [observed vs predicted, 2-class](figures/planted_calibration_s1_2c.png), [observed vs predicted, 3-class](figures/planted_calibration_s1_3c.png), [each model against ridge, per fold](figures/planted_model_vs_ridge_orig.png), [importance ranks, coefficient vs SHAP and model vs model](figures/planted_importance_ranks_orig.png), [cascade confusion matrix](figures/planted_confusion_cascade.png). Every scatterplot carries the y = x line.

## Changes made because of this check

1. **MLP: early stopping removed.** With early stopping, scikit-learn's MLP holds out 10% of the training images to decide when to stop. With a few hundred training images it stopped almost at once and scored at chance on the 5 benign subtypes (0.20–0.22), where the lasso reached 0.46. Without early stopping and with stronger L2 penalties (1, 10 or 30, chosen by inner CV), it reached about 0.30. An L-BFGS solver was no better and took far longer.
2. **Random forest and MLP probabilities put on equal class priors.** The logistic models and gradient boosting are fitted with balanced class weights, so their probabilities treat the classes equally. The random forest's probabilities still followed the training class mix (50% benign, 20% indeterminate, 30% malignant), even though the class weights were set. So picking its most probable class called 933 of 1,000 images benign, and its 3-class balanced accuracy was 0.40 despite ranking the classes as well as the lasso (macro AUROC 0.861 against 0.867). We now divide the random forest and MLP probabilities by the training class shares and renormalise. Its 3-class balanced accuracy went to 0.69. In the observed-vs-predicted plot, the forest's probabilities are still squeezed into a narrow middle range (0.35–0.55), which is typical of forests. Its probabilities should not be read at face value.
3. **All outer fits run in one pool.** One full run is 2,700 outer fits (3 sources × 6 tasks × 6 models × 25 folds). Run model by model, it left most of the 72 cores idle. The full run now takes about 37 minutes on 72 cores (about 36 CPU-hours). The slowest parts are the elastic net, the random forest and the MLP's SHAP values.

## Directional and Coherence Scores on the proof-of-concept answers

This section supports FINDINGS 14. It uses all topic-mode Claude answers we have: 2 images (ISIC_0000147, melanoma; ISIC_0000215, benign), 500 questions each, without the negations. The answer stores are `6e8c9772f7bb`, `671d5cdf75dd`, `0bdfd5a56734` and `64593f651dc6` (Opus, replicates 1–4), `ed429c314d29`, `a9184b405973`, `9c4bb8c5f5fc` and `86f5ca4a5151` (Opus, shuffled question order, replicates 1–4), and `b0aea9414af7` (Sonnet, ISIC_0000147 only). That makes 8,500 answers.

Without the negations, the Directional Score is (2 P(yes) − 1) × confidence. The Coherence Score is (2 P(yes) − 1) × (1 − | |2 P(yes) − 1| − confidence |), so it differs from the Directional Score only where the stated probability and the stated confidence disagree. Claude's confidence follows the strength of its answer closely (r = 0.84 between |2 P(yes) − 1| and confidence). In these answers the two scores correlate at r = 0.98. Running S4 on one Opus replicate stripped of its negations gives the same figure (0.982). S4 now reports both correlations for every run, so the beta set will show whether this holds on 1,000 images.

## What this does not tell us

- Nothing about real performance, or which model will do best on real answers. The ordering above follows from how the simulation was built.
- Whether Claude's errors are like our noise. Real errors are likely to be correlated across questions in ways the topic shift only roughly imitates, for example image quality affecting every answer.
- Anything about dermoscopic against clinical images. The manifest does not record image type.
