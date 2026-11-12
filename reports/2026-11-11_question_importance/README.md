# Which questions the classifiers relied on (2026-11-11)

**Question.** In the first classifier run on real answers
([2026-11-11 classifier report](../2026-11-11_classifier_sonnet1000/README.md)), which of the 446 questions carried
the signal? Is the ranking stable enough to interpret? And do the important questions mean what they say, that is, does
Sonnet see the feature in the images of the class the question points to?

## Runs covered

| Run | Experiments (meaning) | Stage hashes |
|---|---|---|
| [20261111-183626_s7-importance](../../outputs/qaemb/runs/20261111-183626_s7-importance/report.md) | `D1_train` (all 1,000 beta images), `I1_claude_max` (images shrunk only, to at most 3.6 MP), `Q0_orig` (500 questions as written plus the control question, no negations), `M_topic` (one prompt per topic), `ANS_claude_sonnet` (Claude Sonnet 5.5 through the Batch API, effort low), `G1` (whole image), `R1` (replicate 1). Stage `importance`. | embed `6cd6f95ebcbd`, classify `670746415c57`, importance `43a20169f0c0` |

An earlier run of the same stage the same evening (`20261111-183515_s7-importance`, importance `5b8532b6fdb0`) showed
only the share of images answered yes. We replaced it once we saw that this hides most of the signal (see below);
its results are otherwise the same.

**What changed before the run.** We added a new stage, S7 importance (`qaemb/stages/s7_importance.py`, run with
`stage=importance`; settings under `importance:` in `conf/base.yaml`), and three figure helpers in
`qaemb/models/plots.py`. The stage refits nothing. It reads the importance values S6 measured on the test folds and
puts them next to Sonnet's answers. The main tables use the Directional Score, the score the classifier report
recommends carrying forward; the CSV files have all three scores.

## How importance is measured

- **Consensus rank.** Each of the six models (lasso, ridge, elastic net, random forest, gradient boosting, neural
  network) ranks the questions by how much they moved its predictions on the test folds (mean absolute SHAP value).
  The consensus is the average of the six ranks; 1 is the most important.
- **Lasso kept.** The share of the 25 cross-validation fits in which the lasso kept the question. A question kept in
  every fit is reliably useful, not chosen by chance in one split.
- **AUROC alone.** How well one question's score, used on its own with no model, separates malignant from benign
  across the 800 labelled images. 0.5 is chance; for comparison, Sonnet asked "Is this a malignant lesion?" scores
  0.79. This is computed once on all images, so it describes the answers rather than being a test score.
- **Mean P(yes) by class.** Sonnet's stated P(yes), averaged over the images of a class.

## Validity checks

- **The models broadly agree on the order of the questions.** The median rank correlation between one model's ranking
  and the others' is 0.57–0.82 depending on the task (1 would be identical, 0 unrelated). Ridge and the elastic net
  agree with the rest most (0.58–0.86). The two tree models agree least (0.36–0.69), as expected: trees spread
  importance unevenly over correlated questions.
- **The top questions are kept in almost every fit.** Of the 20 top questions for benign against malignant, the lasso
  kept 19 in at least 80% of the 25 fits, and it kept all 20 for the 3-class, 10-class and benign-subtype tasks. The
  exception is telling apart the two indeterminate subtypes: only 6 of its top 20 pass this bar, so its ranking beyond
  the first few questions should not be interpreted.
- **The ranking depends on the score used.** Rankings from P(yes) and from the Directional Score correlate at
  0.68–0.82, but share only 40–55% of their top 20. Questions near the top under one score are usually near the top
  under the other, but the exact order differs. We therefore report every ranking with the score it comes from.

## Results

### No single question comes close to asking directly; the classifier gains by combining many weak ones

The strongest single question for benign against malignant is "Does the skin surface appear rough?" (DQ484), with
AUROC alone 0.72. Sonnet asked for the diagnosis scores 0.79. 98 of the 446 questions reach 0.65 on their own, and 163
are below 0.55. Ridge regression combines them to 0.84.

The consensus rank agrees only loosely with single-question strength (rank correlation 0.3). Several strong single
questions rank low because other questions already carry the same information. Examples: chrysalis structures (DQ146,
0.71 alone, rank 232) and scaly plaques on sun-exposed skin (DQ353, 0.70, rank 239). Some pairs among the top questions
are near-duplicates, such as the two vascular questions DQ472 and DQ301 (correlation 0.89). Those pairs are listed in
[importance_summary.md](importance_summary.md).

### Most of the signal is in how doubtful Sonnet's "no" is

Sonnet answered yes on fewer than 5% of the images in every class for **309 of the 446 questions**. Eight of the top
20 for benign against malignant are among them. One is "Are dotted vessels visible within a pigment network?"
(DQ033): the mean P(yes) is 0.03 for both benign and malignant images, yet on its own it separates them with an AUROC
of 0.67. What carries the information is whether Sonnet says "no" with P(yes) 0.02 or 0.10.

For interpretation, this matters as much as anything else here. A question can be important without Sonnet ever
saying it sees the feature. Its doubt seems to rise with its overall impression of how worrying the lesion looks,
whatever the question asks. The marker questions for each subtype show both kinds:

- **Markers that mean what they say.**
  - Vascular lesions: "Does the lesion resemble a vascular lesion?" gets a mean P(yes) of 0.63, against 0.05–0.19 in
    every other subtype.
  - Benign melanocytic lesions: "Is the melanocytic lesion flat and subtle, without obvious melanoma features?" gets
    0.50, against 0.27 elsewhere.
  - Malignant epidermal lesions: "Does the skin surface appear rough?" gets 0.55, against 0.36.
  - Between the two indeterminate subtypes: erythema around the lesion (65% answered yes for epidermal, 8% for
    melanocytic) and dark brown or black colour (8% against 41%).
- **Markers that only make sense as graded doubt.**
  - "Are pseudostellate scars visible?" for fibro-histiocytic lesions: 0.04 against 0.03.
  - "Does the image show skin with visible features of obesity?" for indeterminate melanocytic lesions: 0.03
    everywhere.
  - "Does the image show more than 50 common naevi?" for flat melanotic pigmentation: 0.02 against 0.02.
  - These differences are tiny but consistent. The lasso kept the obesity question in 88% of fits.

Figures: [subtype_markers.png](figures/subtype_markers.png) (mean P(yes) of each marker question in every subtype)
and [top_s1_2c.png](figures/top_s1_2c.png) (mean P(yes) in benign and malignant images for the top 20).

Three of the top 20 for benign against malignant push the model the opposite way to how they point on their own: the
bluish-purple, vascular-lesion and milky-red globule questions (DQ479, DQ301, DQ054). They correct for correlated
questions and should not be read alone.

### What the important questions are about

For benign against malignant, the top 20 under the Directional Score fall into two kinds.

Questions about the lesion, as one would expect:
- towards benign: a homogeneous pigmented pattern (DQ020) and parallel brown lines (DQ191);
- towards malignant: colour asymmetry (DQ399), pink colour in a pigmented lesion (DQ025), an off-centre darkest region
  (DQ454), patchy brown globules (DQ173), grey-white or blue-grey structureless areas (DQ071, DQ074), and dotted
  vessels (DQ033).

Questions about the skin around the lesion:
- towards malignant: rough skin (DQ484, ranked first), chronic sun damage (DQ359), dyspigmentation (DQ383) and uneven
  pigmentation of the normal skin (DQ480);
- towards benign: lighter skin tone (DQ220).

That several of these surrounding-skin questions sit near the top confirms the concern raised in the classifier
report. By the annotated scope, though, questions about the whole lesion or image carry only slightly more than their
share of the importance (1.04–1.16 times), and the metadata questions carry less (0.64 times for benign against
malignant). So the surrounding-skin signal is concentrated in a handful of questions, not spread across a whole
category. Those questions are not tagged as a separate scope, which is why a lesion-only rerun needs a hand-made list
rather than the existing `scope` field.

**By topic** ([topics_s1_2c.png](figures/topics_s1_2c.png), [topics_flat_10c.png](figures/topics_flat_10c.png)):
- **More than their share:** "Dermoscopy terminology consensus" (37 questions; 1.45 times its share for benign
  against malignant), "Skin surface texture & microrelief" (1.83) and "Field cancerization & actinic damage scoring"
  (1.65 in the 3-class task).
- **Far less than its share:** "Special sites (acral, face/LM, nail, mucosa)", the largest topic with 63 questions
  (14% of the set), carries only 2–6% of the importance in every task (0.14–0.41 times its share). Few of the beta images
  come from those sites. This topic is the obvious first place to cut the question set, if the external test set has
  the same mix.

## What this means

- The ranking is stable enough to interpret for the 2-class, 3-class, 10-class and benign-subtype tasks. It is not
  stable enough for the two indeterminate subtypes beyond the first few questions.
- No small set of questions does the work. The classifier's edge over asking directly comes from many weak questions
  together, and a third of the 20 most important ones are features Sonnet almost never says it sees. A reader should
  not take a high importance to mean "this feature was present". Whether the graded "no" reflects real visual evidence
  or a general impression is an open question. It affects how readable the classifier really is.
- For cutting the questions down (plan item 5), the near-duplicate pairs and the special-sites topic are the natural
  first candidates. A lesion-only rerun (a hand-made list of the questions about the surrounding skin and the patient,
  since the `scope` field does not separate them) should come first.

## Files

- [importance_summary.md](importance_summary.md): the stage's full summary: top 20 questions for every task, marker
  questions for every subtype, topics, scopes, near-duplicate pairs.
- [consensus.csv](consensus.csv): every question, task and score, with all ranks, weights, yes shares, mean P(yes)
  per class and AUROC alone.
- [subtype_markers.csv](subtype_markers.csv), [topics.csv](topics.csv), [scopes.csv](scopes.csv),
  [redundancy.csv](redundancy.csv), [agreement.csv](agreement.csv), [source_agreement.csv](source_agreement.csv).
- [figures/](figures/).
