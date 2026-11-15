# Literature-derived answer vectors for 500 dermatology questions

Prompt for Claude Science, version 1, 2026-11-12. QA-DermEmb project, AbbasiLab, UCSF.

---

## Who we are and why we need this

We represent each skin-lesion image by a model's answers to 500 yes/no dermatology questions (for example "Is there a blue-white veil?", "Are arborizing vessels present?"), and train a classifier on those answers to tell malignant from benign lesions. The questions are attached as `DermQs_Source-1015FT_Prompt-C_N-500.csv` (columns: `Question ID`, `Topic`, `Question`, `Sources cited (Local ID)`, `Grounding`; 41 topics).

Our image model answers each question with two numbers: the probability that the answer is yes, and its confidence in that answer. We need answers in the same form that come from the published literature instead of from our images: for documented cases, and for the mechanisms described in the literature, what would the 500 answers be? We will use the result as an independent reference against our image-based classifiers. Your job is to build that reference faithfully. It does not have to make any classifier look good, and a sparse or uncertain answer that is correctly marked as such is more useful to us than a complete one.

You will not see our image answers, classifier results or which questions our models rank highly, and you should not look for them. The reference has to be independent of them.

## What a "mechanistic" link means here

A visible feature does not cause malignancy. But many dermoscopic and clinical features are the visible consequence of a specific malignant process, with a documented histopathological correlate. Examples: an atypical pigment network corresponds to irregular, confluent junctional nests of atypical melanocytes; arborizing vessels in BCC correspond to dilated dermal vessels over tumour nests; blue-white veil corresponds to dermal melanin under compact orthokeratosis. We call these **mechanistic** links.

Other features are associated with malignancy only statistically or through context: signs of chronic sun damage around a lesion, patient age, body site, lesion size on its own. They make malignancy more likely, but they are not produced by the tumour. We call these **associated** links.

Use these strengths:

| Strength | Meaning | Required evidence |
|---|---|---|
| 1 | Mechanistic: the feature is the visible correlate of a documented process in the lesion (or of its absence). | A source that states the histopathological or biological correlate, ideally a dermoscopic–histopathologic correlation study, a consensus paper or a review that cites one. |
| 0.5 | Associated: the feature changes the probability of the diagnosis, but no mechanism is described, or it reflects the patient or the surrounding skin rather than the lesion. | A source reporting the association (frequency, odds ratio, likelihood ratio, or inclusion in a validated algorithm without a stated correlate). |
| 0 | No link found for this target. | None needed. |

Every link also has a **direction**: `+` if the feature's presence points towards the malignant diagnosis, `−` if its presence points towards a benign one (for example a regular, symmetrical network, milia-like cysts, a central white patch in dermatofibroma). Benign-pointing features matter as much as malignant ones.

The boundary between 1 and 0.5 is the part we expect to disagree about. When you are unsure, choose 0.5, set `needs_review = yes` and say why in one sentence. We will review the borderline cases together before you finish.

## How answers are given

Every answer to a question has two numbers, as in our image pipeline:

- `p_yes`: the probability, from 0 to 1, that the answer to the question as worded is yes.
- `confidence`: how firmly that answer rests on the evidence, from 0 (no evidence at all) to 1 (stated plainly and unambiguously).

They are separate. "The lesion showed no vessels" is p_yes near 0 with high confidence. "Possibly a few dotted vessels" is p_yes around 0.6 with moderate confidence. A question the source says nothing about has p_yes 0.5 and confidence 0; do not write a row for it, we fill those in ourselves.

Use these anchors, so that numbers mean the same thing across cases and across readings:

| Evidence | p_yes | confidence |
|---|---|---|
| Stated present, without hedging | 0.95 | 0.9 |
| Stated absent, without hedging ("no", "absent", "lacked") | 0.05 | 0.9 |
| Hedged in the text ("possible", "subtle", "focal", "suggestive of") | 0.6–0.75 or 0.25–0.4 | 0.5–0.7 |
| Only in a figure caption | as stated | at most 0.7 |
| Follows directly from a stated feature ("flat macule" so "not raised") | as implied | at most 0.6 |
| Mechanistic reading, strength-1 link entailed by the stated histopathology | as entailed | 0.6–0.8, lower if the link is to a related but not identical finding |
| Nothing in the source | no row | (0) |

Move away from the anchors only for a reason, and give the reason in `note`.

## Target diagnoses

Our images come from the ISIC 10-class taxonomy (100 images each). Map everything you find to these classes. Our reading of what each class contains is in brackets; tell us where the literature suggests it is wrong.

Malignant: MALIGNANT MELANOCYTIC PROLIFERATIONS (melanoma, in situ and invasive, all subtypes); MALIGNANT ADNEXAL EPITHELIAL PROLIFERATIONS – FOLLICULAR (mostly basal cell carcinoma); MALIGNANT EPIDERMAL PROLIFERATIONS (cutaneous squamous cell carcinoma, including in situ / Bowen disease).

Indeterminate: INDETERMINATE MELANOCYTIC PROLIFERATIONS (atypical or dysplastic naevi, melanocytic tumours of uncertain potential); INDETERMINATE EPIDERMAL PROLIFERATIONS (actinic keratosis and similar).

Benign: BENIGN MELANOCYTIC PROLIFERATIONS (naevi); FLAT MELANOTIC PIGMENTATIONS – NOT MELANOCYTIC NEVUS (solar lentigo, lentigines, flat seborrhoeic keratosis and similar); BENIGN SOFT TISSUE PROLIFERATIONS – FIBRO-HISTIOCYTIC (dermatofibroma); BENIGN SOFT TISSUE PROLIFERATIONS – VASCULAR (haemangioma, angiokeratoma, pyogenic granuloma); INFLAMMATORY OR INFECTIOUS DISEASES.

Annotations in Phase 2 are made for the three malignant classes, each against its usual benign mimics, and once for malignant against benign overall.

## How to work

Work in five phases. **Stop at the end of Phase 1 and after the pilot of Phases 2 and 3**, report, and wait for us. Write every output as a file with exactly the columns given below; we will load them with code.

### Phase 1. Find the literature (stop and report)

Search PubMed and other indexed sources. We are looking for three kinds of paper, in this order of value:

1. **Dermoscopic–histopathologic correlation studies, consensus papers and terminology papers** that state what each feature corresponds to in tissue. These support strength 1.
2. **Feature-frequency studies**: diagnostic-accuracy or cohort studies that report how often each feature occurs in a histopathologically confirmed diagnosis and in its mimics (for example "blue-white veil in 51% of melanomas and 7% of naevi"). These let us build realistic answer vectors, not only textbook ones.
3. **Case reports and case series** with histopathological confirmation, a written description of the clinical and dermoscopic appearance detailed enough to answer many of the 500 questions, a description of the histopathology, and preferably a figure showing the lesion. Prefer series and reports of typical presentations. Case reports are biased towards unusual presentations (amelanotic, featureless, rare sites); include some, but label them `atypical_presentation = yes` so we can analyse them separately.

The attached `dermal_lesion_sources.csv` (131 sources, used to write the questions) is a starting point. Note in the output which of your sources are already in it. Overlap is expected; it does not disqualify a source, but we need to know about it, since a question written from a paper will trivially agree with that paper.

Aim for about 30–60 papers of the first two kinds and about 100–150 cases of the third, spread over all ten classes, with at least 15 cases per malignant class and at least 8 per other class. Cover both dermoscopic and clinical (naked-eye) descriptions, and fair and dark skin where the literature allows.

Output `corpus_candidates.csv`:
`source_id, kind (correlation | consensus | frequency | case_report | case_series | review), authors, year, title, journal, doi, pmid, target_classes (semicolon-separated), image_modality (dermoscopic | clinical | both), n_cases, histopath_confirmed (yes | no | partial), histopath_described (yes | no), has_lesion_figure (yes | no | unknown), atypical_presentation (yes | no), open_access (yes | no | unknown), in_existing_sources (Local ID or blank), why_included (one sentence)`

Check that every DOI or PMID resolves and that the paper is not retracted, and say so in the report. Do not list a paper you cannot find in an index. Then stop. We will obtain the full texts and give them back to you.

### Phase 2. Annotate the 500 questions (pilot first)

From the full texts, annotate each question for each target. Use only the supplied full texts plus well-established consensus documents you can cite. For each annotation with strength above 0, quote the sentence that supports it.

Output `question_annotations.csv`, one row per question and target:
`question_id, target (melanoma | bcc | scc | malignant_vs_benign), direction (+ | − | blank if strength 0), strength (1 | 0.5 | 0), mechanism (one sentence: the tissue or biological correlate, or "association only"), visible_in (dermoscopic | clinical | both), source_ids, quote, quote_location (section or page), needs_review (yes | no), note`

Questions that are not about the lesion itself (surrounding skin, patient, body site, image acquisition) can be at most 0.5. Questions whose meaning is ambiguous get `needs_review = yes` with the ambiguity stated.

**Pilot:** annotate the 30 questions we name when we reply to Phase 1 (they will span several topics), then stop and report, together with the Phase 3 pilot, so we can agree on the 1 / 0.5 boundary before you do the rest.

### Phase 3. Two text readings of each case (pilot first)

Each case is answered twice from its text, in two separate passes that must not borrow from each other. A third reading, from the case's figure alone, is made later by our own image pipeline (Phase 4), so in Phases 2 and 3 **do not look at the figure images**. Tables and figure captions are text and may be used.

**Presence reading.** Answer from what the paper says the lesion *looks like*: the clinical and dermoscopic description. Do not use the diagnosis or the histopathology to fill in features. The most important rule: **answer only what the paper states.** Most of the 500 features will not be mentioned for any one case. Those have no row (confidence 0), not a "no". Do not fill in a feature because the diagnosis usually has it; that would turn the case into a copy of the textbook and defeat the purpose. A feature that follows directly from a stated one may be answered with `basis = implied`, with the stated feature quoted.

**Mechanistic reading.** Answer from what the paper says the lesion *is*: the histopathology (and the diagnosis only as far as it is confirmed by histopathology), combined with the strength-1 links from Phase 2. Do not use the described appearance. A question is answered only if a stated histopathological finding produces or rules out the feature through a strength-1 link for that diagnosis; quote both the histopathology sentence and the Phase 2 annotation it relies on (by `question_id` and `target`). Everything else has no row. This reading asks what the case's pathology says the image should show; the presence reading asks what the authors saw. We will compare them.

Output `cases.csv`, one row per case:
`case_id, source_id, diagnosis_as_reported, histopath_confirmed, mapped_class (one of the 10 classes, exact spelling), age, sex, body_site, fitzpatrick_or_skin_tone, image_modality_described, atypical_presentation, n_answered_presence, n_answered_mechanistic, note`

and `case_answers.csv`, one row per case, reading and question, **only for questions with some evidence**:
`case_id, reading (presence | mechanistic), question_id, p_yes, confidence, basis (stated | stated_absent | hedged | caption | implied | histopath_entailed), quote, quote_location, annotation_ref (mechanistic reading only: question_id and target of the Phase 2 row used), note`

**Pilot:** do 3 cases first (one melanoma, one BCC, one benign), both readings, then stop and report alongside the Phase 2 pilot.

### Phase 4. Figures for the image reading

Only after both text readings are finished for every case, so the images cannot influence them: for each case with a figure of the lesion, save each relevant panel as a separate image file at the highest resolution the full text provides, and list it.

Output `case_figures.csv`, one row per panel:
`case_id, figure_file (file name you saved), figure_label (e.g. "Figure 2B"), modality (dermoscopic | clinical | histopathology), has_overlays (yes | no: arrows, letters, scale bars or circles drawn on the image), approx_resolution (width × height in pixels), same_lesion_as_text (yes | unsure), note`

Do not answer the questions from the figures yourself. We will run them through the same model, prompt and settings as our image set, so that the image reading is made by the same instrument as the classifier's training data. Histopathology panels are listed but will not be answered.

### Phase 5. Feature frequencies and canonical profiles

From the frequency studies, extract `feature_frequencies.csv`, one row per question, class and study where a frequency was reported:
`question_id, mapped_class, comparison_class (if the study gives one), frequency (0–1), n_lesions, image_modality, source_id, quote, quote_location, match_quality (exact | close | partial: how closely the study's feature matches our question wording)`

Then write `canonical_profiles.csv`: for each of the ten classes, the answers a textbook example would give, from the Phase 2 annotations and the frequency table only:
`mapped_class, question_id, p_yes, confidence, rule (the one-line reason, e.g. "strength-1 + feature for melanoma, present in 62% (n = 240)"), source_ids`

Where a frequency is known, p_yes is that frequency and confidence reflects how well the study's feature matches the question and how large the study was. Leave a question out where the literature does not support a typical answer. We will generate sampled vectors ourselves from these tables, so you do not need to.

## Rules that apply throughout

- Cite by DOI or PMID and quote verbatim. We will check a random sample of quotes against the full texts; an invented or paraphrased quote is worse than an empty cell.
- Do not use our question CSV's `Sources cited` column as evidence that a link exists. It records where a question's wording came from, not what the paper shows.
- No evidence means no row (p_yes 0.5, confidence 0), never p_yes 0. A "no" needs a source that says the feature is absent, or, in the mechanistic reading, a strength-1 link that rules it out.
- Keep `methods_log.md` as you go: search strings, dates, databases, numbers found and excluded with reasons, decisions you made on unclear cases, and anything in our class mapping or question wording that seems wrong.
- When you stop for review, report in plain prose: what you did, how many sources and cases, how the cases are spread over the ten classes and over dermoscopic and clinical descriptions, how many questions each reading answered per case on average, the main uncertainties, and what you need from us.
