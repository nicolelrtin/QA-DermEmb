# Classifiers on Sonnet's answers for the 1,000 beta images (2026-11-11)

**Question.** Now that Sonnet has answered the 500 questions on all 1,000 beta images, how well do the classifiers
built on those answers tell the lesion classes apart? Specifically: do they beat Sonnet simply being asked "Is this a
malignant lesion?", does the two-stage design (group first, then subtype) beat one 10-class model, does the Directional
or Coherence Score add anything over P(yes), and which questions carry the signal?

This is plan item 4 in PROJECT.md. Everything here is cross-validated on the beta set; there is no external test set
yet, so these are estimates for the method, not for a frozen model.

## Runs covered

| Run | Experiments (meaning) | Stage hashes |
|---|---|---|
| [20261111-174157_s6-sonnet1000](../../outputs/qaemb/runs/20261111-174157_s6-sonnet1000/report.md) | `D1_train` (all 1,000 beta images), `I1_claude_max` (images shrunk only, to at most 3.6 MP), `Q0_orig` (500 questions as written plus the control question DQC01, no negations), `M_topic` (one prompt per topic), `ANS_claude_sonnet` (Claude Sonnet 5.5 through the Batch API, effort low), `G1` (whole image), `R1` (replicate 1). Stage `classify` with the settings in `conf/base.yaml` (all six model families, all six tasks, sources P(yes), Directional and Coherence Score). | manifest `531882139b22`, answer `2e3804059ebe`, embed `6cd6f95ebcbd`, classify `670746415c57` |

The answers come from run `20261111-095000_api-full1000` ([2026-11-11 API report](../2026-11-11_api_pilot/README.md)).
No API calls were made here.

**What changed before the run.** S6 now scores the control question as a benchmark (`control_benchmark` in
`qaemb/stages/s6_classify.py`, stage version 3). Its P(yes) is used directly as P(malignant), with nothing fitted, on
exactly the benign-vs-malignant test folds the classifiers use, with the same metrics. The control question is never a
feature.

## Set-up in brief

446 questions are used as features: the 500, less the 54 marked as artefact questions (about the photograph rather
than the lesion), and less the control question. Each of six model families (lasso, ridge and elastic-net logistic
regression, random forest, gradient boosting, a small neural network) is scored by 5 repeats of 5-fold
cross-validation, with folds stratified on the 10 subtypes and grouped by lesion. Its settings are tuned by a further
3-fold cross-validation inside each training set, so no image is ever scored by a model, or a tuning step, that saw it.
Each number below is the mean ± standard deviation over the 25 test folds. The run took 41 minutes on the lab
machine's 72 cores.

## Validity checks

- **Complete inputs.** All 1,000 images have an answer to every one of the 446 questions under all three scores; the
  stage stops if any is missing.
- **The benchmark is scored on the same images as the classifiers.** Each of the 5 repeats puts all 800 benign or
  malignant images in a test fold exactly once, for the benchmark as for the classifiers, so the two can be compared
  fold by fold.
- **The code was checked beforehand on simulated answers** ([2026-11-05 report](../2026-11-05_classifier_synthetic/README.md)).
  With no signal planted, every score sat at chance, so test images do not leak into training. The only change since
  is the benchmark, which fits nothing.
- **Tuning ranges.** For the three logistic models the chosen penalty is almost always inside the grid. The neural
  network chose the strongest penalty offered (alpha 30) in 47% of fits, and gradient boosting mostly chose its
  simplest trees. Both might do a little better with a wider grid towards simpler models; this does not bias the
  scores, which are all measured on unseen folds.

## Results

### Benign against malignant (800 images; the 200 indeterminate ones are scored but never fitted)

**The benchmark.** Asked directly, Sonnet's P(yes) to "Is this a malignant lesion?" gives an AUROC of
**0.79 ± 0.04**. That is, a malignant image gets a higher P(yes) than a benign one 79% of the time. At a threshold that
catches 95% of the malignant lesions, it correctly clears only **26% ± 9%** of the benign ones.

**The classifiers beat it, mainly in specificity.** All 18 combinations of model and score reach AUROC 0.81–0.84.
The best by the pre-set main score is ridge regression on the Directional Score: **AUROC 0.84 ± 0.03**, and
**44% ± 10%** of benign lesions cleared at 95% sensitivity.

Compared fold by fold with the benchmark:
- **AUROC:** ridge on the Directional Score is higher by 0.05 on average, and higher in 23 of the 25 folds.
- **Specificity at 95% sensitivity:** it is higher by 0.19, in 25 of 25 folds.

So, at the sensitivity a screening tool needs, the 446 questions clear about 18 more benign lesions in every 100 than
asking Sonnet for the diagnosis. Every model family beats the benchmark; gradient boosting by the least (+0.02 AUROC).

| | AUROC | specificity at 95% sensitivity |
|---|---|---|
| Benchmark: Sonnet asked directly (DQC01) | 0.79 ± 0.04 | 0.26 ± 0.09 |
| Ridge, P(yes) | 0.82 ± 0.03 | 0.44 ± 0.12 |
| Ridge, Directional Score | **0.84 ± 0.03** | 0.44 ± 0.10 |
| Ridge, Coherence Score | 0.83 ± 0.03 | 0.43 ± 0.10 |
| Random forest, P(yes) | 0.83 ± 0.03 | 0.41 ± 0.10 |
| Neural network, Directional Score | 0.83 ± 0.03 | 0.45 ± 0.11 |

**Directional Score against P(yes).** For the three logistic models, the Directional Score raised AUROC by about 0.02
over P(yes), in 22–24 of 25 folds. That small gain is consistent, and it means the stated confidence adds information.
For the trees and the neural network it made no consistent difference.

**The Coherence Score adds nothing over the Directional Score.** It was slightly lower for every logistic model (by
0.01; ridge lower in all 25 folds). This was expected from their near-identical values
(FINDINGS 14).

**Indeterminate lesions.** The 2-class model, never trained on them, places the indeterminate epidermal lesions on the
malignant side (median P(malignant) 0.58–0.70, depending on model and score) and the indeterminate melanocytic lesions on the benign side (0.20–0.28).
The benchmark puts both on the benign side (median P(yes) 0.33 and 0.23).
Figure: [indeterminate_2c_orig.png](figures/indeterminate_2c_orig.png).

### Three groups, ten subtypes and the two-stage design

Chance for balanced accuracy is 0.33 for 3 classes and 0.10 for 10.

- **Benign / indeterminate / malignant (s1_3c):** balanced accuracy 0.54–0.60. The best is the random forest on
  P(yes), 0.60 ± 0.03, which finds 49% of benign, 62% of indeterminate and 68% of malignant lesions. Stage 1 is the
  weak point.
- **Subtypes within a known group (stage 2, with the true group):** benign 0.69–0.73 (5 subtypes), malignant
  0.70–0.75 (3), indeterminate 0.84–0.88 (2). Once the group is known, telling subtypes apart is far easier.
- **All ten subtypes at once (flat_10c):** 0.45–0.50; the best is the neural network on the Directional Score, 0.50 ±
  0.03. Vascular lesions are the easiest (82 of 100 correct for ridge on P(yes)) and fibro-histiocytic lesions the
  hardest (34). Indeterminate melanocytic lesions are often called benign melanocytic or flat pigmentation.
  Figure: [confusion_flat_10c.png](figures/confusion_flat_10c.png).
- **The two-stage cascade does worse than the flat model.** Routing each image through stage 1 and then the subtype
  model gives balanced accuracy 0.41–0.45, against 0.45–0.50 for the flat model with the same family and score.
  Stage 1 sends about 43% of images to the wrong group, and about three quarters of the cascade's wrong subtype calls
  had already been sent to the wrong group. With this stage 1, two stages lose more than they gain.

### Which questions carry the signal

Full tables: [classify_summary.md](classify_summary.md) and [question_ranking_pyes.csv](question_ranking_pyes.csv).

For benign against malignant, the questions every model ranks highest include lesion features one would expect, such as:
- a homogeneous pigmented pattern (DQ020, towards benign);
- colour asymmetry and variegation (DQ399, DQ038, towards malignant);
- looking like a vascular lesion (DQ472, towards benign).

But the first-ranked question is about the skin around the lesion: "Are signs of chronic sun damage visible around the
lesions?" (DQ359, towards malignant). Several others near the top are also about the patient's skin rather than the
lesion:
- rough skin (DQ484);
- skin tone, as the individual typology angle (DQ220);
- dyspigmentation (DQ383).

In the 10-class task, a question that asks outright whether the image shows "demographic cues (e.g., age-related skin
changes) that a model could exploit" (DQ410) is in the top ten. Of the top 20 questions for benign against malignant,
16 are about the whole image and 4 about a local part of the lesion.

Sun damage and age are real risk factors for the malignant epidermal and adnexal classes, so this is not an error. But
it means part of the classifier's accuracy comes from the patient's skin, not from the lesion, and that part may not
carry over to a test set with a different patient mix or framing.

## What this means

- The QA-Emb route works on real answers and is better than asking Sonnet for the diagnosis. The gain is modest in
  AUROC (0.84 against 0.79) and large in specificity at 95% sensitivity (0.44 against 0.26). This rests on 1,000 images
  and one replicate of the answers.
- The Directional Score is the main score to carry forward. The Coherence Score can be dropped from the main results.
- The two-stage design does not pay off yet, because stage 1 misroutes too many images.
- Before cutting the questions down (plan item 5), we should know how much of the accuracy depends on questions about
  the skin around the lesion. Rerunning with only the lesion questions (`exclude_scopes`, or a list of
  surrounding-skin questions) would show this.

## Files

- [classify_summary.md](classify_summary.md): the stage's full readable summary (all models, tasks and scores; top ten
  questions per task). Its figure links point to `outputs/qaemb/classify/670746415c57/figures/`.
- [metrics_summary.csv](metrics_summary.csv): every metric, mean and SD over test folds and pooled.
- [benchmark_summary.csv](benchmark_summary.csv), [benchmark_folds.csv](benchmark_folds.csv): the control question
  on the same folds.
- [cascade.csv](cascade.csv): the two-stage cascade per model, score and repeat.
- [question_ranking_pyes.csv](question_ranking_pyes.csv): all importance measures per question and task, P(yes).
- [figures/](figures/): calibration (2-class), confusion matrices (3-class, 10-class, cascade), indeterminate lesions on
  the 2-class scale, importance ranks across models, each model against ridge fold by fold.
- Out-of-fold predictions per image: `outputs/qaemb/classify/670746415c57/oof_predictions.parquet`.
