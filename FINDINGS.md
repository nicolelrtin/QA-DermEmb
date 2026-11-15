# QA-DermEmb: findings

A running record of results that changed what we believe or what we do, in the order we found them. Each entry says what we saw, how much it rests on, and what we did about it. The full numbers are in the linked report. Entries are not rewritten later: when a result is overturned or qualified, a new entry says so and the old one gets a note pointing to it.

Current decisions and the plan are in [PROJECT.md](PROJECT.md).

Almost everything so far comes from 1 to 10 images. Treat the results as pilot observations until they hold on the 1,000-image beta set.

Terms used below:
- A question pair is a question and its negated version ("Is the border irregular?" and "Does the lesion lack an irregular border?").
- A contradiction is the same answer to both halves of a pair.
- Net_P is P(yes) to the question minus P(yes) to its negation, from −1 (clearly no) to +1 (clearly yes).

---

## 2026-10-27: Qwen smoke test, 10 images

**1. For Qwen, only one question per prompt gives usable answers.** When several questions share a prompt, Qwen drifts towards "No" as it works through the list. Mean P(yes) fell from 0.18 for the first question of a topic prompt to 0.004 for the last. The grouped modes were also 4–7 times slower, because the image is already reused when questions are asked one at a time. We chose stateless for Qwen.
Report: [2026-10-27_smoke10](reports/2026-10-27_smoke10/README.md).

**2. Even asked one at a time, Qwen contradicts itself on most question pairs.** It gave the same answer to a question and its negation for 79% of pairs, mostly "no" to both. 211 of the 500 questions were contradicted on all 10 images. Part of this looks like phrasing: "lack …" negations seem to pull the answer towards "No" whatever the image shows. This is still an open question (see PROJECT.md).
Report: [2026-10-27_smoke10](reports/2026-10-27_smoke10/README.md).

## 2026-10-29: local answerers compared, 10 images

**3. SkinGPT-R1 is more coherent than Qwen but still contradicts itself on about half the pairs** (47% stateless, against 79% for Qwen). It also drifts much less within a grouped prompt. The two models agree on only 72% of yes/no answers. Neither looked good enough to build features on, which is what pushed us to try Claude.
Report: [2026-10-29_answerers](reports/2026-10-29_answerers/README.md).

## 2026-10-30: Claude proof of concept, 2 images, 4 replicates

Images ISIC_0000147 (melanoma) and ISIC_0000215 (benign). Written up with figures as a letter: [letter.md](reports/2026-10-30_claude_poc/letter.md).

**4. Claude almost never contradicts itself.** 3–5% of pairs, against 78–92% for Qwen and 50–61% for SkinGPT-R1 on the same images. This is the main reason we switched to Claude as the answerer.

**5. Claude's answers are repeatable.** Across 4 replicates, the same verdict came back every time for 92–95% of pairs, and a typical answer moved by 0.02–0.03. Its stated confidence was lower for the answers that did change between replicates (Spearman ρ = −0.72 to −0.79), so the confidence carries real information.

**6. Asking Claude one topic per prompt gives nearly the same answers as one question per prompt, at about one-ninth of the cost.** Net_P correlated at r = 0.92 between the two modes, with 91% identical verdicts. We adopted topic mode for Claude. Entry 12 qualifies this.

**7. Asked all 500 questions in one prompt, Claude often stops early.** In the 10-image pilot, only 3 of the 6 prompts we tried (3 images, questions and negations asked separately) ever returned all 500 answers, some only after several attempts. The others stopped after 322–447. We stopped the pilot and dropped this mode. The reply format now requires exactly one answer per question; that fix works for topic prompts but was never tested on the 500-question prompt.

**8. SkinGPT-R1's problem is negation, not the image.** Taking Claude as the reference, SkinGPT-R1's answers to the questions agreed beyond chance (Cohen's κ = 0.36), but its answers to the negations did not (κ = 0.05). It is also sensitive to question order: when the questions within a topic were shuffled differently in each of 4 replicates, the same verdict came back in all four for only 26% of pairs (Claude: 93%). For this model, P(yes) to the question alone, or negations rewritten as positive opposites, may be the better input.

**9. For Claude, the negated questions may be unnecessary.** Its P(yes) to the question weighted by its stated confidence reproduced the question-and-negation score almost exactly (ρ = 0.96). Dropping the negations would halve the cost of every run. Not yet decided; it is the first item in the plan.

## 2026-11-04: cost

**10. The same Claude run can cost 2.5 times more or less.** Identical stateless prompts cost $0.009–0.022 each through `claude -p`. Token counts were nearly the same; the difference is how much of each prompt was served from Anthropic's cache, which `claude -p` does not let us control. A direct API integration that caches each image once might fix this, but we have not measured it. Measured subscription capacity: a 5-hour window holds about $105 of list-price use, a week about $1,000.
Report: [2026-11-04_token_usage](reports/2026-11-04_token_usage/README.md).

**11. Sonnet is close to Opus but not interchangeable, at about 60% of the cost.** On one image (ISIC_0000147, topic mode), Sonnet gave the same yes/no answer as Opus for 94% of questions. Opus agrees with itself 98% of the time, so the gap is about 3–4 times Opus's own run-to-run noise. Sonnet said "no" more often (13.8% yes against 16.8–17.8%), and many of the biggest disagreements were about the photograph (lighting, framing, hair) rather than the lesion. Whether this matters for classification is the next test: both models on the 10 smoke images, compared on W0.
Report: [2026-11-04_sonnet_test](reports/2026-11-04_sonnet_test/README.md).

Projected cost of 500 questions without negations on the 1,000-image beta set, topic mode, at list price: Opus about $615–1,226, Sonnet about $370–740. Stateless would be about ten times more. Local models would take under 2 GPU-hours, but their answers are not usable as they stand (entries 2, 3, 8).

## 2026-11-05

**12. Topic prompts make Claude slightly more negative than one-question prompts.** Going back over the 2-image proof of concept answer by answer: 94 of 1,000 answers changed verdict between the two modes (66 on the melanoma, 28 on the benign image), and another 148 moved by 0.3 or more without changing verdict. Of these 242, 229 were lower in topic mode. So the agreement in entry 6 hides a consistent shift towards "no". We are keeping topic mode for its cost, but this should be checked again on more images, and kept in mind when comparing to any stateless results.
Files: [claude_stateless_vs_topic.xlsx](reports/2026-10-30_claude_poc/claude_stateless_vs_topic.xlsx), listed in the [2026-10-30 report](reports/2026-10-30_claude_poc/README.md).

**13. Derm1M does not provide a question-answering model.** The Derm1M GitHub mentions visual question answering, but that describes what the dataset could be used for. The released models (DermLIP) only match images to text and cannot produce an answer. The instruction-tuning data, Derm1M_Instruct, was still unreleased at the commit we use (2026-08-11), and we found no sign of a release since. DermLIP could still score questions as statements; this is in the backlog as a way to make answering cheaper (PROJECT.md, Later).
No report; from discussion and a reading of the Derm1M repo.

**14. Without the negated questions, the Directional and Coherence Scores may carry nearly the same information.** Now that the negations are dropped, both scores are computed from P(yes) and Claude's stated confidence (PROJECT.md, Answer format). On every topic-mode answer we have (2 images, 8 Opus and 1 Sonnet answer sets, 8,500 answers), Claude's confidence closely follows how far its P(yes) is from 0.5 (r = 0.84). As a result, the two scores correlate at r = 0.98. The Coherence Score can only differ where P(yes) and confidence disagree, and on these images they rarely do. We are keeping both as features, because the beta set may show more disagreement and the classifiers will show whether it matters. This is 2 images only.
Report: [2026-11-05_classifier_synthetic](reports/2026-11-05_classifier_synthetic/README.md), section on the scores.

**15. The classifier stage passes its checks on simulated answers, after two changes to the model settings.** We simulated answers for the 1,000 beta images, keeping their real labels, with 68 of 446 questions made informative. Every importance measure (ridge weight, the number of fits in which the lasso keeps a question, and SHAP for each model) ranked the planted questions first. When nothing was made informative, every model scored at chance on every task, so test images are not leaking into training. On one simulated dataset, the two indeterminate subtypes scored 0.58–0.59 with ridge, the random forest and the MLP; ridge scored 0.46–0.54 on six other datasets. That is also a measure of the chance spread to expect on the 200 indeterminate images. Two settings had to change. The MLP's early stopping halted training almost at once on a few hundred images. And the random forest's probabilities followed the training class mix, so it called almost every image benign in the 3-class task (balanced accuracy 0.40) while ranking images as well as the lasso. The forest and MLP probabilities are now put on equal class priors, like those of the class-weighted models (forest: 0.69). A full run takes about 37 minutes on the lab machine's 72 cores. On simulated data the lasso and elastic net led, but only because we planted a few strong questions among many empty ones. This says nothing about real answers.
Report: [2026-11-05_classifier_synthetic](reports/2026-11-05_classifier_synthetic/README.md).

## 2026-11-11: first runs through the Claude API, 2 to 1,000 images

**16. The direct API answerer works and costs less than we projected; the full beta run should cost about $240.** All 84 topic prompts for the 2 proof-of-concept images came back complete on the first attempt, for $0.78. On the melanoma image, the API answers matched the earlier `claude -p` Sonnet answers on 94.8% of the 500 questions, about the same as Sonnet against Opus. We also measured how many tokens Sonnet 5.5 charges for an image: about pixels / 750, levelling off at about 4,700 above about 3.6 MP. We therefore now send images at that size and no smaller (PROJECT.md, Image size for Claude), which is about five times the image tokens of the old 0.4 MP setting. Even so, the full run on 1,000 images is projected at about $205 for Sonnet with the batch discount, against $370–740 projected on 2026-11-04 for `claude -p` (entry 10). Most of the difference is that the direct API has no Claude Code overhead in each prompt, and that batches cost half. This is 2 images; the 10-image batch run will check the numbers again.
Report: [2026-11-11_api_pilot](reports/2026-11-11_api_pilot/README.md).

**17. Sonnet answered every question on the 1,000 beta images, within budget, and asked directly it separates malignant from benign with an AUROC of 0.79.** One Message Batch of 42,000 topic prompts (500 questions plus the control question, images at 3.6 MP) came back complete on the first attempt, for $203.63; total spent on the key is $206.80 of $500. The control question "Is this a malignant lesion?" gave an AUROC of 0.79 on the 800 labelled images, but reaching 95% sensitivity leaves only 22% specificity, partly because Sonnet uses only a few coarse P(yes) values. This is the bar the 500-question classifiers have to beat. Sonnet answered yes to 9.5% of the questions, at almost the same rate for benign and malignant images.
Run `20261111-095000_api-full1000`. Report: [2026-11-11_api_pilot](reports/2026-11-11_api_pilot/README.md), section on the 1,000-image run.

## 2026-11-11: classifiers on real answers, 1,000 images

**18. On real answers, the 446 questions beat asking Sonnet for the diagnosis, mostly at high sensitivity; the Directional Score is the score to keep, and the two-stage design does not yet pay off.** For benign against malignant, ridge regression on the Directional Score reached AUROC 0.84 and cleared 44% of benign lesions at 95% sensitivity, against 0.79 and 26% for Sonnet's P(yes) to "Is this a malignant lesion?" on the same folds. It did better in 23 and 25 of the 25 folds. For the logistic models the Directional Score was about 0.02 AUROC better than P(yes) in most folds, and the Coherence Score was never better than the Directional Score (as expected from entry 14). For the subtypes, the cascade (group first, then subtype) did worse than one 10-class model, because stage 1 sends about 43% of images to the wrong group. The highest-ranked questions include several about the skin around the lesion (chronic sun damage first, also skin tone and roughness). Part of the accuracy may therefore come from the patient rather than the lesion. This is 1,000 images, one replicate of answers, cross-validated; there is no external test yet.
Run `20261111-174157_s6-sonnet1000`. Report: [2026-11-11_classifier_sonnet1000](reports/2026-11-11_classifier_sonnet1000/README.md).

**19. The classifiers' edge comes from many weak questions, and much of it from how doubtful Sonnet's "no" is.** No single question separates benign from malignant as well as asking Sonnet for the diagnosis: the best, "Does the skin surface appear rough?", reaches AUROC 0.72 on its own, against 0.79. Combined, they reach 0.84 (entry 18). Sonnet answered yes on fewer than 5% of images in every class for 309 of the 446 questions, including 8 of the top 20. For these, what differs between classes is whether the "no" comes with P(yes) 0.02 or 0.10. Some marker questions mean what they say (vascular lesions, flat and subtle naevi, rough skin in malignant epidermal lesions). Others only work as graded doubt (pseudostellate scars, signs of obesity). So a question's importance cannot simply be read as "Sonnet saw this feature". The ranking itself is stable: the six models broadly agree, and the lasso keeps 19 of the top 20 in at least 80% of fits. The exception is telling apart the two indeterminate subtypes. The questions about the skin around the lesion (rough skin, sun damage, skin tone) are concentrated among the top ones rather than spread over a whole category. The special-sites topic, 63 questions, carries only 2–6% of the importance.
Run `20261111-183626_s7-importance`. Report: [2026-11-11_question_importance](reports/2026-11-11_question_importance/README.md).

## 2026-11-14: choosing the ridge penalty

**20. The ridge penalty is well chosen already, and its exact value hardly matters; we keep tuning on log loss and keep the 5 repeats.** Lambda (= 1/C in scikit-learn) is the strength of the L2 penalty on the standardised question weights, chosen by 3-fold cross-validation on log loss inside each outer training set. On a grid three times finer than S6's (lambda 1 to 100,000), the outer AUROC for benign against malignant on the Directional Score has one broad optimum between lambda 20 and 100, and the nested choice (mostly lambda 33) gives 0.840, against 0.841 for the best value in hindsight and 0.839 with S6's coarser grid. Choosing lambda by AUROC instead gave the same or slightly lower scores and much more scattered choices, because AUROC and balanced accuracy on an inner fold of about 210 images move in steps. The rule that takes the strongest penalty within one standard error of the best was worse (0.829, and 0.785 on P(yes)). Inner log loss on so few images can be swung by one confidently wrong image, which is what threw that rule off. Over the 5 repeats of the outer split, AUROC ranged from 0.832 to 0.842, so a single 5×3 nested CV would vary by about 0.01 with the shuffle alone. This is 1,000 images, one replicate of answers, ridge only. The stage reproduced S6's choice and score in all 100 folds it covers.
Run `20261114-185921_s8-ridge-tune`. Report: [2026-11-14_ridge_tuning](reports/2026-11-14_ridge_tuning/README.md).

## 2026-11-15: structure of the answers

**21. Questions of one topic are answered alike far beyond what their wording explains, which may be a cost of asking them in one prompt.** Across the 1,000 beta images, two questions of the same topic correlate at r = 0.55 on average, against 0.10 for questions of different topics and 0.12 for random groups of the same sizes. How alike two questions are worded agrees only weakly with how alike they are answered (rank correlation 0.13 to 0.17), and at equal wording similarity same-topic pairs are about 0.4 more correlated than different-topic pairs. The special-sites topic is the clearest case: 63 questions on nails, palms, face and mucosa correlate at 0.75. Either a topic describes one underlying feature, or Sonnet answers a whole prompt from one overall impression. Only asking the same questions one per prompt can tell these apart (Plan item D). The answers carry about 27 dimensions above noise (parallel analysis), and 60 principal components hold 90% of the variance. This is one replicate of Sonnet's topic-mode answers.
Run `20261115-105528_s9-explore`. Report: [2026-11-15_answer_structure](reports/2026-11-15_answer_structure/README.md).

**22. Image type and institution show in the answers, clinical photographs are concentrated in the non-benign classes, and our fold grouping does not group anything.** Clustered by their answers, the images follow the subtype most (adjusted mutual information 0.23), then the contributing institution (0.09) and image type (0.06). The institution association is as strong within benign images alone (0.09), so it is not only that institutions sent different diagnoses. According to the ISIC metadata, 160 of the 1,000 images are clinical photographs or total-body-photography tiles, not dermoscopy: 6% of benign images against 24% of malignant and 29% of indeterminate ones. Separately, every image has its own `lesion_id`, so grouping the cross-validation folds by it ties no images together. ISIC records a patient for 472 images, and 43 patients have more than one lesion (100 images), which can currently fall on both sides of a fold. Since the skin around the lesion carries much of the signal (entries 18, 19), the cross-validated scores may be somewhat optimistic until the folds are grouped by patient.
Run `20261115-105528_s9-explore`; metadata from the ISIC Archive API, 2026-11-15. Report: [2026-11-15_answer_structure](reports/2026-11-15_answer_structure/README.md).

