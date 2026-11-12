# QA-DermEmb: project notes

Nicole Tin, AbbasiLab, UCSF. Started 2026-10-08; QA-Emb pipeline since 2026-10-23.

This file records where the project stands: what we are trying to do, the decisions currently in force and why, and what comes next. What we have learned from experiments is in [FINDINGS.md](FINDINGS.md). The detailed evidence is in the reports under [reports/](reports/README.md).

## Aim

Represent each image by a model's answers to about 500 curated yes/no dermatology questions (QA-Emb; Benara et al. 2024, arXiv 2405.16714). Then classify malignant against benign from those answers, and show where in the image the influential questions apply. The point is a classifier whose features a dermatologist can read and check.

## How the records fit together

- [inputs/handoff/HANDOFF_QA-Emb-Derm.md](inputs/handoff/HANDOFF_QA-Emb-Derm.md) is the original plan from 2026-10-23. It is kept as written, as a record of where we started, and is not updated. Several of its "locked" decisions have since changed; the current ones are below. (Not in git.)
- This file holds the current state: decisions, pipeline status, plan, open questions.
- [FINDINGS.md](FINDINGS.md) is the running list of results that changed what we do, in the order we found them.
- [reports/](reports/README.md) has one folder per batch of experiments with the full numbers, figures and the runs they cover. [reports/experiments_log.md](reports/experiments_log.md) lists every individual run.

### Keeping them up to date

After each batch of experiments:
1. Write the report in `reports/<date>_<name>/` and add it to `reports/README.md`.
2. If the batch changed what we believe or what we do, add an entry to FINDINGS.md. Each entry gives the result in a sentence or two, what it rests on (how many images, which runs) and the link to the report. Numbers live in the report; the entry quotes only the one or two that matter.
3. Update this file where a decision, the pipeline status or the plan has changed. When a decision is replaced, keep the old one in the decision history with the date and the reason.
4. Commit the report and the updated notes together.

Ideas and decisions that come out of discussion rather than a run (for example, a model we decide not to use) are recorded the same way, with the date and the reasoning.

## Current decisions

| Topic | Current decision | Since | Basis |
|---|---|---|---|
| Answering model | Claude Sonnet 5.5, effort low, via the Messages API on the lab's Claude Console organisation (key active since 2026-11-11), through the Batch API for large runs. Opus 5.5 kept as a check on a subset of about 50 images. Qwen2.5-VL-7B and SkinGPT-R1 kept as comparisons. | 2026-11-05 | Sonnet agreed with Opus on 94% of answers at about 60% of the cost (FINDINGS 11); decided with the PI. The API also gives control over caching, which `claude -p` does not (FINDINGS 10). |
| Image size for Claude | Whole images shrunk to at most 3.6 MP and 2,576 px on the long edge, never enlarged, sent as JPEG (`I1_claude_max`). The local models keep 0.4 MP. | 2026-11-11 | This is the most Sonnet 5.5 uses: image tokens stop growing at about 4,700 above about 3.6 MP. Fine dermoscopic structures may need the detail, and it costs about $90 more on the 1,000 images with the batch discount ([2026-11-11 pilot](reports/2026-11-11_api_pilot/README.md)). |
| Control question | DQC01 "Is this a malignant lesion?" asked in its own topic ("Control") with the 500 questions; never a classifier feature. Its P(yes) is the zero-shot benchmark. | 2026-11-11 | Asking it in its own prompt keeps it from influencing the other answers. Keeping it out of the features stops the classifiers from just repeating Claude's diagnosis. |
| API spending | At most $500 on the Console key. Each submission is projected beforehand and stops if it would take the total past $400 until we approve it; every response is logged in `outputs/qaemb/claude_spend.csv`. We cannot set a Console spend limit ourselves (not organisation owners), so the pipeline's check is the only guard. Spent by 2026-11-11: $206.80. | 2026-11-11 | Agreed budget for the beta set. |
| How questions are asked | One prompt per topic (`M_topic`, 41 prompts per image), the 500 questions as written only (`Q0_orig`). Negated questions dropped. | 2026-11-05 | Topic mode: close to one question per prompt at about one-ninth of the cost (FINDINGS 6), with a slight lean towards "no" that we accept (FINDINGS 12). Negations: Claude's stated confidence reproduces what they add (FINDINGS 9), and dropping them halves the cost. |
| All questions in one prompt (`M_all`) | Dropped. | 2026-10-30 | Claude often stopped before answering all 500 (FINDINGS 7). |
| Answer format and scores | Claude states P(yes) and a confidence for each question. Without negations, the confidence takes the place of Net_P, with its sign taken from the answer: Net_P = sign(2 P(yes) − 1) × confidence. The Directional Score (= (2 P(yes) − 1) × confidence) and the Coherence Score (= (2 P(yes) − 1) × (1 − \|\|2 P(yes) − 1\| − confidence\|)) both follow from P(yes) and confidence. Both are kept as features alongside P(yes). Local models still give P(yes) from the Yes/No word probabilities. | 2026-11-05 | Without negations, the Directional Score shrinks an answer when Claude says it is unsure. The Coherence Score shrinks it when the stated probability and the stated confidence disagree. Whether that difference matters is one of the things the classifiers will show. On the 2 proof-of-concept images the two scores correlate at r = 0.98, so they may turn out nearly the same. |
| Labels and tasks | Stage 1: benign vs malignant (`label_2c`; the 200 indeterminate images are not fitted but are scored), and benign / indeterminate / malignant (`label_3c`). Stage 2: subtypes (`label_10c`) within each stage-1 class, one model each for benign (5 subtypes), malignant (3) and indeterminate (2). A flat 10-class model is the baseline for the two-stage design. | 2026-11-05 | Each stage-1 class has its own subtypes, hence two stages. Stage 2 is scored both with the true stage-1 class and with the predicted one, which shows how much error stage 1 passes on. |
| Evaluation | Nested cross-validation on all 1,000 beta images: 5 repeats of 5-fold, stratified on the 10 subtypes and grouped by `lesion_id`; hyperparameters by 3-fold CV inside each training set. Main scores fixed in advance: AUROC and specificity at 95% malignant sensitivity for the 2-class task; balanced accuracy and macro-F1 for the others. The external test set, when it arrives, is used once, on the final model. | 2026-11-05 | No test set yet. Fixing the main scores before seeing results keeps the choice of "best model" honest. Questions chosen by a cutoff on CV results give an optimistic CV score, so that cutoff is confirmed on the external test set. |
| Classifiers | L1 (lasso), L2 (ridge) and elastic-net logistic regression, random forest, gradient-boosted trees and a one-hidden-layer MLP, all in scikit-learn. Question importance from standardised weights, the number of fits in which the lasso keeps a question, and SHAP. PyTorch is kept for the later patch and question model (Track N). | 2026-11-05 | 800–1,000 images by about 450 questions does not need PyTorch, and one scikit-learn pipeline keeps scaling and tuning inside the folds. Ridge keeps every question and ranks them; only the lasso and elastic net drop questions. |
| Benchmarks | The same images and folds for: SkinGPT-R1 zero-shot diagnosis; Sonnet asked directly "Is this a malignant lesion?" (the control question, from 2026-11-11; a "benign?" question was planned but not added); random question subsets of the same size; PanDerm/DermLIP image features. Compared as performance against number of features rather than at one matched size. | 2026-11-05 | Direct questions to Sonnet show how much the 500-question route costs in accuracy compared with asking for the diagnosis. |
| Development order | Smallest test first (1–10 images) before any scale-up. The classifiers were built and checked on simulated answers before the real answers exist. | 2026-10-23, extended 2026-11-05 | Handoff. |
| PanDerm / DermLIP | Not an answerer. Image feature extractor for structure maps and a benchmark (linear probe). A scoring-only use of DermLIP is in the backlog. | 2026-10-23, revisited 2026-11-05 | It matches images to text but cannot produce an answer (FINDINGS 13). |

### Decision history

- 2026-10-23. Answerer set to Qwen2.5-VL-7B (handoff). Replaced on 2026-10-30 by Claude Opus 5.5.
- 2026-10-27. Stateless chosen as the only usable mode for Qwen. Still true for the local models; for Claude, topic mode replaced it on 2026-10-30.
- 2026-10-29. Planned to run Claude on the Batch API. Not possible without a separate Console account, so all Claude runs so far went through `claude -p` on the subscription.
- 2026-10-30. `M_all` dropped after the 10-image Claude pilot was stopped.
- 2026-10-30. Answerer set to Claude Opus 5.5 through `claude -p`. Replaced on 2026-11-05 by Sonnet 5.5 through the API, for cost; Opus is kept as a check on a subset.
- 2026-10-30. Questions asked with their negations. Dropped on 2026-11-05; the stated confidence takes their place (FINDINGS 9).
- 2026-10-29. Net_P, P_w, Directional and Coherence Scores defined from the question and its negation. From 2026-11-05, for Claude, they are defined from P(yes) and the stated confidence (Answer format, above).
- 2026-10-23. Images resized to 0.4 MP for every answerer (the Qwen budget). For Claude replaced on 2026-11-11 by Sonnet's own limit, 3.6 MP.
- 2026-10-23. Label `label_2c` only, and a grouped 70/15/15 split (seed 0). Replaced on 2026-11-05 by the 2-, 3- and 10-class tasks and nested cross-validation on all 1,000 images, since no test set exists yet. The 70/15/15 split is still what the older S5 models (W0, W1, P0–P3) use.

## Pipeline status

| Stage | Status |
|---|---|
| S0 manifest, S1 prepare (resize, patch grid) | Built and used. Shrink-only resizing for Claude added 2026-11-11. |
| S2 answer (Qwen, SkinGPT-R1, Claude via `claude -p`, Claude API) | Built. The Claude queue runs under cron and pauses at the usage limits. The API path ran for the first time on 2026-11-11 (2 images, direct requests), with the spending ledger and checkpoint; the batch path then ran on 10 and on all 1,000 beta images. |
| S4 embed (Net_P, P_w, Directional_Score, Coherence_Score) | Built. Directional and coherence scores added as features `F_dir` and `F_coh` on 2026-11-04. |
| S4 without negations | Built 2026-11-05: Net_P, Directional and Coherence Scores from P(yes) and the stated confidence, plus the confidence itself as a feature source (`conf`). |
| S5 classifiers (W0, W1, P0–P3) | Built and run only on the 10 smoke images, where the metrics carry no information. Whole-image work now goes through S6. |
| S6 classify (`stage=classify`) | Built 2026-11-05: nested CV of the six model families on all tasks, the two-stage cascade, importance tables, figures and a readable summary. Checked on simulated answers (`SYN_planted`, `SYN_null`; FINDINGS 15). A full run takes about 37 minutes on 72 cores. Not yet run on real answers. |
| Relevance maps, attribution and viewer, consistency | Not built. |

Data: the beta set has 1,000 ISIC images (10 classes × 100), all on disk since 2026-11-11 in `data/train_10class_100each/` (same manifest as the beta folder). All 1,000 have been answered by Sonnet through the API (501 questions, topic mode, full image size; embed `6cd6f95ebcbd`, 2026-11-11).

## Plan

### Next

1. Done 2026-11-05: S6 checked on simulated answers (FINDINGS 15). The planted questions are found, and every score sits at chance when nothing is planted.
2. Done 2026-11-11: the 1,000 beta images are on disk. Still to do: fetch the ISIC `image_type` metadata (dermoscopic or clinical) for each image, so results can be split by image type.
3. Done 2026-11-11 ([report](reports/2026-11-11_api_pilot/README.md)): the 2-image API pilot ($0.78), 10 images through the Batch API ($2.39), then Sonnet, topic mode, no negations, at full image size (`ANS_claude_sonnet`, `M_topic`, `Q0_orig`, `I1_claude_max`) on the 1,000 beta images, with the control question in the same run ($203.63, every answer complete). Still to do: Opus on about 50 of them to measure agreement (about $22).
4. Before S6 on real answers: report the control question's AUROC and specificity at 95% sensitivity on the same folds, as the benchmark row (pooled over the 800 labelled images: AUROC 0.79, specificity 22% at 95% sensitivity). Then run S6 on the real answers (all tasks, all models, P(yes), Directional and Coherence Scores) and write one report covering stage 1, stage 2 and question importance.
5. Refinement: from the best model by the fixed main scores, cut down the questions (top-k by importance, or a cutoff) and find how few keep performance. Compare with random subsets of the same size and the image-feature baselines.
6. External test set when it arrives: the frozen models, plus SkinGPT-R1 zero-shot diagnosis on the same images.

### Later

- Distinguishing whether the Coherence Score adds anything beyond the Directional Score (from the S6 results).
- Patch-level answers and the patch classifiers (P0–P3, grid sizes 2, 4, 8). Cost scales with the number of patches, so this depends on the cost work below.
- Relevance maps, deletion test, PanDerm structure clusters, attribution and the hover viewer (S6–S7).
- Small neural model over patch and question tokens (Track N).
- Ways to make answering cheaper:
  - Direct API with each image cached once and reused for all of its prompts, instead of `claude -p`, which gives no control over caching (FINDINGS 10).
  - Distillation: train a head on PanDerm image features to predict Claude's answers, as in the QA-Emb paper.
  - DermLIP as a local scoring answerer (proposed ID `ANS_dermlip`). Details in the handoff backlog, §8. In short: score each question as a pair of yes/no statements against the image. It is almost free to run, but CLIP models largely ignore negation, the questions would need rewriting as statements, and the scores are not calibrated. First test would be agreement with Claude on the 2 proof-of-concept images.
  - DermoGPT (Qwen3-VL-8B trained on DermoInstruct) as a local model that does generate answers. Derm1M itself has no such model yet; revisit if Derm1M_Instruct is released.
- Baselines: ABCD questions only (DQ001–DQ004).
- Other answerers: MedGemma, other Qwen sizes.

### Dropped or on hold

- `M_all` for Claude (2026-10-30).
- Negated questions for Claude (2026-11-05).
- Log-polar input: deferred until lesions can be centred.

## Open questions

- Are the negated phrasings ("Does the lesion lack …") partly to blame for the local models' contradictions? Rewriting a sample as positive opposites ("Is the border regular?") would tell us. Raised 2026-10-27; not yet tested.
- Do Claude's stated probabilities and confidences mean the same thing across images, or only rank answers within an image? This matters more now that confidence replaces the negated questions.
- Review of the drafted `scope` and `kind` annotations on the question set (137 borderline calls flagged).
- Where the external test set will come from, and how far its sources and its mix of dermoscopic and clinical images differ from the beta set.
- Whether to bring in ISIC 2018 Task 2 attribute masks, or in-house masks, to check the relevance maps.
