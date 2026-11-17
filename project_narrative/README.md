# Project narrative (2026-11-16)

[index.html](index.html) is the project told in the order we intend for the paper: what Sonnet was asked and how it
answered, what the questions measure, the classifier and how it compares with other models, the ten subtypes, and the
questions that carry the decision. It is a self-contained page of about 6.5 MB; open it in any browser. The reports in
[reports/](../reports/README.md), and above all the full [Answer Structure Explorer](../reports/2026-11-15_answer_structure/explorer.html),
are its supplement. The page links to them from its Methods tab.

The structure follows Nicole's notes of 2026-11-16 (tabs and the figures kept under each). The overview carries a
data-driven version of her graphical abstract (`graphical abstract.png` at the repository root), in which each panel opens
the chapter that develops it.

## Run covered

| Run | Experiments | Hashes |
|---|---|---|
| [20261116-220131_s9-explore-narrative](../outputs/qaemb/runs/20261116-220131_s9-explore-narrative/report.md) | `D1_train` (all 1,000 beta images), `I1_claude_max` (images shrunk only, to at most 3.6 MP), `Q0_orig` (500 questions as written plus the control question, no negations), `M_topic` (one prompt per topic), `ANS_claude_sonnet` (Claude Sonnet 5.5 through the Batch API, effort low), `G1` (whole image), `R1` (replicate 1). Stage `explore`. | explore `81acb51f0bca` |

The data on the page are identical to those of the explorer built in run `20261116-211352_s9-explore-cutoffs` (explore
`a887c4d68972`); we checked that the embedded payloads match apart from the run hash. Nothing was recomputed: the page
reads the same S7 importance, S10 ridge, S11 subtypes and S13 benchmark outputs as the explorer.

## How it is built

Stage S9 (`qaemb/stages/s9_explore.py`, version 4) fills two templates with the same data:
`qaemb/templates/explore.html` gives `explorer.html` and `qaemb/templates/narrative.html` gives `narrative.html`. To
rebuild after an upstream change:

```
python -m qaemb.run stage=explore +exp=[D1_train,I1_claude_max,Q0_orig,M_topic,ANS_claude_sonnet,G1,R1] tag=<text>
cp outputs/qaemb/explore/<hash>/narrative.html project_narrative/index.html
```

The narrative template was assembled once from the explorer's (the chart code is shared verbatim) and is now maintained
as its own file; a change to a chart in one template is not carried to the other.

## Chapters and figure concordance

| Narrative | Content | Explorer |
|---|---|---|
| Overview | Abstract, graphical abstract, one line per chapter | Overview (flow diagram replaced) |
| I Answers | Fig. 1.1 distribution of stated P(yes); Box 1.1 the Directional Score | Fig. 1.1; Fig. 1.3 |
| II Questions | Fig. 2.1 variance per component against parallel analysis; Fig. 2.2 agglomerative (Ward) clustering with the correlation matrix and dendrogram; Fig. 2.3 the clusters and their subtype profiles | Fig. 3.1; the correlation-matrix panel of Ch. II; Fig. 3.3 |
| III Modeling, part i | Fig. 3.1 ROC; Box 3.1 controls against leakage; Fig. 3.2 out-of-fold P(malignant) by subtype | Fig. 5.1; the leakage list of Ch. V; Fig. 5.3 |
| III Modeling, part ii | Table 3.3 ten-subtype models on the same folds (three tasks, with SkinGPT-R1); Table 3.4 two, three or ten classes; Table 3.5 threshold metrics by model | Tables 9.1; 9.2 and 9.3; 9.9 |
| IV Classification, part i | Figs. 4.1 to 4.3: performance by subtype, confusion matrix, nature of the errors | Figs. 7.1 to 7.3 |
| IV Classification, part ii | Figs. 4.4 to 4.8: naevus against melanoma | Figs. 8.1 to 8.5 |
| V Importance | Fig. 5.1 the twenty highest-ranked questions; Fig. 5.2 subtype markers (ten-subtype ridge); Fig. 5.3 ranking of all 446 clinical questions (new); Figs. 5.4 and 5.5 the two worked cases (new visual) | Figs. 6.1, 6.2; Ch. X |
| Methods | Methods, limitations and links to the supplement | Ch. XI |

Left in the explorer only: topic coherence (Ch. II), the component–covariate table (Fig. 3.2), the question and image
maps, image structure (Ch. IV), the paired-fold comparison (Fig. 5.2), questionnaire size (Fig. 5.4), the 27-question
lasso model (Fig. 6.3), the further comparison tables (9.4 onward) and the choice of P(malignant) cutoff (Table 9.10,
Fig. 9.11). Selecting a question or an image anywhere on the narrative page opens a card with its details, in place of
the explorer's maps.

## Changes beyond moving figures

- **Naevus against melanoma** now leads with the ten-subtype model (AUROC 0.910) rather than the pair model (0.901),
  following the decision that the paper reports this distinction as read from the ten-subtype model. Figs. 4.6 and 4.7
  still show the pair model, since only its per-image scores are in the S11 outputs.
- **Worked cases.** The waterfall (a running total in log-odds) is replaced by three readings of the same arithmetic:
  where the prediction starts and ends on the probability scale; the total evidence toward each side as a tug of war,
  with the net shift; and the six strongest questions on each side, each with the image's answer placed beside the average
  benign and the average malignant answer. The numbers are those of the explorer's Ch. X.
- **Question ranking.** Fig. 5.3 lists every clinical question by consensus rank, searchable and sortable, with a CSV
  download.

## Open points on the structure, for discussion

1. "Replace with agglomerative clustering": we read this as making the question clustering the figure for this chapter.
   Ward clustering, which the explorer already used, is agglomerative, so Fig. 2.2 shows it directly. If something else
   was meant (another linkage, clustering on the wording, or a dendrogram labelled by cluster theme), it can be added.
2. "6.1, 6.2 (Ridge 10 only)": we kept Fig. 6.1 as is (consensus over six families, two-class task) and Fig. 6.2 (ten-
   subtype ridge), and dropped the lasso views. If the intent is that every importance figure come from the ten-subtype
   ridge, Fig. 5.1 and the ranking need a ten-subtype importance measure computed in a stage.
3. "9.3 How many classes": the page shows Table 9.2 and 9.3 together as Table 3.4. Table 9.2 is the comparable one
   (patient-grouped folds, as everywhere else); 9.3 uses lesion-grouped folds and adds three classes.
4. The classifier in Ch. III and the cases in Ch. V are the two-class model, while Ch. IV and the paper's emphasis are
   the ten subtypes. Table 3.4 shows the ten-subtype model does at least as well on benign against malignant, so a single
   model throughout is possible. It would need the ROC, the strip plot and the cases recomputed from the ten-subtype
   model.
5. The choice of P(malignant) cutoff (explorer Table 9.10, Fig. 9.11) is not in the notes. Table 3.5's "95%
   sensitivity" setting reads the cutoff on the test fold itself, which is slightly optimistic (its P ≥ 0.5 setting is
   fixed in advance and is fine). For the paper the cutoff chosen in the training split is the honest figure; we suggest
   bringing Table 9.10 into Ch. III.
6. The title, "Questions as a Representation", is a working title.
