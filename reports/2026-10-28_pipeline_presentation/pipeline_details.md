# QA-Emb pipeline: detailed notes for the lab presentation

Companion to the slide deck **QA-Emb Dermoscopy Pipeline** (https://claude.ai/artifact/VZybPKN9MagXsMHbRP83QQ).
The sections follow the slides. Each covers what the slide says, the details behind it, where the numbers
come from, and questions the audience is likely to ask. Every number comes from the 2026-10-27 smoke
test ([../2026-10-27_smoke10/README.md](../2026-10-27_smoke10/README.md)) unless noted.

---

## 1–2. Goal and method

**Idea.** QA-Emb (Benara et al. 2024, *Crafting Interpretable Embeddings by Asking LLMs Questions*,
arXiv 2405.16714) replaces an opaque embedding with the answers of a language model to a fixed list of
yes/no questions. Every dimension is a sentence, so a linear model's coefficients read as "irregular blotch
present → towards malignant". We apply this to dermoscopy with a vision-language model (VLM).

- **Target:** malignant = 1, benign = 0. It is configurable (`label.*` in `conf/base.yaml`).
- **Answerer:** Qwen2.5-VL-7B-Instruct, chosen in the handoff and swappable in config (`ANS_qwen3b`, etc.).
- **PanDerm** is *not* an answerer. It is a CLIP-style image encoder and cannot answer yes/no questions.
  It is planned as an image feature extractor for the structure maps (S6) and, later, as a
  linear-probe baseline.
- **Longer-term output:** for each influential question, show where in the image it applies (S6–S7, not built).

**Likely questions**
- *Why not fine-tune a classifier?* That would be accurate but not interpretable. QA-Emb trades some
  accuracy for features a dermatologist can audit. The planned baselines (PanDerm linear probe,
  ABCD-only questions, random question subsets) will measure that trade-off.
- *Why a general VLM?* It is the simplest first model. Other answerers (MedGemma, SkinGPT, other Qwen sizes)
  are in the backlog and are one config file each.

## 3. Pipeline stages

| Stage | Module | Does | Main outputs |
|---|---|---|---|
| S0 Manifest | `qaemb/stages/s0_manifest.py` | Reads the existing beta manifest; maps labels; grouped stratified split; per-image SHA-256 and original size | `manifest.parquet` |
| S1 Prepare | `qaemb/stages/s1_prepare.py` | Aspect-preserving resize to 401,408 px (512 vision tokens, sides multiples of 28); optional G×G grid (optional 50% overlap) with crops cut from the full-resolution image and enlarged to the same budget | `units.parquet`, `images/*.png` |
| S2 Answer (+S3 timing) | `qaemb/stages/s2_answer.py`, `qaemb/answerers/qwen_vl.py` | P(yes) for every (unit, question, original/complement); timing per item | `answers_raw.parquet`, `timing.parquet` |
| S4 Embed | `qaemb/stages/s4_embed.py` | Pairs originals with complements: `p_adj`, `incoherence`, verdicts; feature matrices per source | `answers.parquet`, `X_image_{orig,neg,adj}.parquet`, `X_patch_*.npy`, `answers_readable.csv` |
| S5 Classify | `qaemb/stages/s5_model.py`, `qaemb/models/` | W0, W1, P0–P3; metrics; coefficients | `metrics.json`, `predictions.parquet`, `coefficients.csv`, `curves.csv`, `selected_questions.csv` (W1) |
| S6–S7 | not built | Relevance maps, deletion test, PanDerm structures, attribution, hover viewer | `maps/*.npz`, viewer HTML |
| S8 (optional) | not built | Paraphrase / test-retest consistency figure | — |

Stages communicate only through files. Each output folder is `outputs/qaemb/<stage>/<hash>/`, where the
hash covers the stage's settings, its inputs' hashes and a code `VERSION` constant (bumped when a
stage's output changes for the same settings).

## 4. Data

- **Beta set:** `data/train_10class_100each - beta/` has 1,000 ISIC images in 10 classes × 100, and `manifest.csv` with
  `file, isic_id, lesion_id, label_10c, label_4c, label_3c, label_2c`.
- **Label used:** `label_2c` gives malignant 300 (melanoma 100, BCC 100, SCC 100), benign 500 (benign
  melanocytic, fibro-histiocytic, vascular, flat melanotic, inflammatory/infectious) and empty for the
  200 indeterminate images (indeterminate epidermal and melanocytic proliferations).
- **Indeterminate handling** (`label.indeterminate`): `holdout` is the default. Label −1; the images are
  answered and embedded, never fitted or used in metrics, and their predictions are reported
  separately. The alternatives are `drop`, `benign` and `malignant`.
- **Splits:** `StratifiedGroupKFold` grouped by `lesion_id` (no patient IDs), 70/15/15, seed 0.
  All 1,000 lesion IDs are unique, so grouping has no effect on this set. `_s_ISIC_…` lesion IDs are
  singletons. The smoke set (`D0_smoke10`) uses leave-one-out.
- **Smoke set:** the 10 images currently on disk, one per class: 5 benign, 3 malignant,
  2 indeterminate. ISIC IDs are in `conf/exp/data/D0_smoke10.yaml`.
- **Provenance recorded:** source "ISIC Archive, 10-class beta subset"; colleague preprocessing has no
  documented transform, and the files appear to be at native ISIC resolution (962×674 to 6721×4521 in
  the smoke set); the pipeline's only transform is the resize in S1.

## 5. Questions

- **Source:** `2026-10-15 Generated Questions/DermQs_Source-1015FT_Prompt-C_N-500_Negated.csv`, read-only.
  Columns are `Question ID, Topic, Question, Sources cited (Local ID), Grounding, original`.
  500 originals (`original = 1`) and their 500 negated complements (`original = 0`, same ID),
  for 1,000 items. There are 41 topics, with 1 to 63 questions each (about 12 on average).
- **Annotations:** `…_annotations.csv` (sidecar, keyed by Question ID) holds a drafted `scope` and `kind`,
  applied to both versions of a question. 137 borderline calls have an `annotation_note`.
  - scope: local 233 (visible in a patch), global 223 (needs the whole lesion), metadata 44 (not in
    pixels: absolute size without a scale, body site, palpation, history).
  - kind: clinical 446, artefact 54 (watermarks, rulers, hair, vignetting, image quality).
  - Artefact questions are audit features, excluded from classifiers by default (`model.exclude_kinds`).
    Patch models use local questions only.
- **Question-set hash:** a hash of the sorted (ID, original, text) triples. Editing annotations does not
  change it; editing any question text or adding questions does, and so does selecting a subset
  (`Q9_smoke50`, `Q1_local`, `Q2_selected`).
- **Questions flagged as hard to answer from one image** (from the annotation pass):
  - computed image statistics: DQ493–498;
  - whole-body or clinical-photo questions: DQ335–350;
  - a single anomalous pixel: DQ444;
  - certainty rather than presence: DQ447–448.

## 6. Answering

- **Prompt** (your edited templates, `qaemb/prompts/stateless_whole_v1.txt`):
  "This is an image of skin. / {question} / Answer with Yes or No." in Qwen's chat format, with the image first.
  The patch template says "This is an image of a patch of skin. Answer about this region only."
- **P(yes):** at the first assistant token, P(yes) = P(Yes-tokens) / (P(Yes-tokens) + P(No-tokens)), where the
  tokens are {Yes, yes, YES} and {No, no, NO}. `yesno_mass` = P(Yes or No) is stored as a check that
  the model is answering; it was ≥ 0.99 in stateless mode.
- **Image reuse (prefix KV cache):** the chat header, the image tokens and the fixed instruction are identical
  for every question, so they are processed once per image (about 340 ms) and the key/value cache is
  copied for each question. Questions run 32 per forward pass (`answer.compute_batch`). Qwen2.5-VL uses 3-D
  rotary positions for image tokens, so continuations get explicit position IDs (prefix length + offset +
  the model's rope delta).
- **Correctness checks:** tokenizing prefix + question separately gives exactly the same tokens as the
  full prompt. Cached P(yes) matches an uncached forward pass within 0.004 (fp16 noise), and the
  check is rerun at the start of every stateless answer stage.
- **Cost:** 9.8 ms per item including the image, on a Quadro RTX 8000 (48 GB, Turing; fp16 because
  Turing has no native bf16 and no FlashAttention-2).
- **Example on the slide:** ISIC_0000147 (malignant), DQ002 "Is the lesion border irregular?",
  P(yes) = 0.52.

## 7. The adjusted answer

- `p_adj = P(yes | Q) − P(yes | not Q)`, in [−1, 1]. This is the default feature (`F_adj`). Alternatives:
  `F_adj_bin` (p_adj ≥ 0), `F_orig` (originals only), `F_orig_bin`.
- `incoherence = P(yes | Q) + P(yes | not Q) − 1`: 0 for a perfectly complementary pair, > 0 yes to both,
  < 0 no to both. Smoke mean (stateless): −0.56.
- **Verdict** (threshold 0.5 on each version): yes, no, contradiction: yes to both, contradiction: no to both.
- **If asked for a probability:** (1 + p_adj) / 2 is the mean of P(yes | Q) and P(no | not Q). It is a
  linear rescaling, so linear models and AUROC are unchanged.

## 8. Answer modes

| Mode | ID | Prompts per image | How |
|---|---|---|---|
| stateless | `M_stateless` | 1,000 | one item per sequence |
| topic | `M_topic` | 82 (41 topics × original/complement) | one prompt per (original, topic) |
| all | `M_all` | 2 | one prompt with all originals, one with all complements |

In the batched modes the JSON answer skeleton `{"1": "…", "2": "…"}` is forced, and P(yes) is read at
each answer slot. The model's own Yes/No (argmax) is fed back before the next key. `answer.shuffle=true`
shuffles order within each prompt (order-effect test). The earlier random batches of 10 (`M_batch*`)
were removed at your request.

## 9–11. Smoke-test results

10 images × 1,000 items = 10,000 answers per mode. Source: `reports/2026-10-27_smoke10/mode_comparison/summary.md`.

| Mode | s per item | Answered yes | Same answer to Q and not Q | Mean \|p_adj\| | Binary agreement with stateless | Pearson r vs stateless |
|---|---|---|---|---|---|---|
| stateless | 0.0098 | 13.2% | 78.9% | 0.21 | — | — |
| topic | 0.043 | 3.9% | 93.2% | 0.10 | 88.2% | 0.50 |
| all | 0.071 | 0.3% | 99.4% | 0.01 | 86.9% | 0.16 |

- **Drift within a prompt:** mean P(yes) at the first vs last position is 0.18 → 0.004 (topic) and 0.28 → 0.002 (all).
- **Verdict shares:**

  | Mode | No to both | Consistent yes | Consistent no | Yes to both |
  |---|---|---|---|---|
  | stateless | 76.2% | 13.7% | 7.4% | 2.7% |
  | topic | 92.7% | 4.6% | 2.2% | 0.6% |
  | all | 99.4% | 0.6% | 0% | 0% |
- **Contradicted everywhere:** 211 of 500 questions are contradicted on all 10 images (stateless).
- **Why batched is slower:** stateless runs each question as a short continuation, 32 in parallel, while
  batched modes need one sequential decoding step per answer.

## 12. Wording effect (DQ002)

Stateless values for all 10 images (the slide shows the 8 labelled ones):

| Image | Label | P(yes) "irregular?" | P(yes) "lacks irregular?" | p_adj | Verdict |
|---|---|---|---|---|---|
| ISIC_0000147 | malignant | 0.52 | 0.02 | +0.50 | yes |
| ISIC_0012762 | malignant | 0.65 | 0.02 | +0.63 | yes |
| ISIC_0014042 | malignant | 0.18 | 0.20 | −0.02 | no to both |
| ISIC_0000215 | benign | 0.57 | 0.02 | +0.54 | yes |
| ISIC_0009983 | benign | 0.85 | 0.00 | +0.85 | yes |
| ISIC_0011775 | benign | 0.06 | 0.07 | 0.00 | no to both |
| ISIC_0012084 | benign | 0.03 | 0.04 | −0.01 | no to both |
| ISIC_0012087 | benign | 0.03 | 0.64 | −0.61 | no |
| ISIC_0011054 | indeterminate | 0.46 | 0.01 | +0.45 | no to both |
| ISIC_0012195 | indeterminate | 0.33 | 0.09 | +0.24 | no to both |

The complement is ≤ 0.09 on 8 of the 10 images. **Proposed test:** rewrite a sample of complements as
positive opposites ("Is the border regular?") and compare contradiction rates on the 10 images.

## 13. Consistently answered questions

Ranked by |mean p_adj| over the 10 images, stateless (`compare_by_question.csv`). The slide omits two
audit-style questions: DQ410 (visible demographic cues, −0.86, drafted as clinical) and DQ406
(colour cast, −0.61, artefact).

| Question | Mean p_adj | Contradicted (of 10) |
|---|---|---|
| DQ382 thick and leathery skin | −0.87 | 0 |
| DQ016 irregular blotch | +0.77 | 0 |
| DQ383 dyspigmentation | +0.75 | 1 |
| DQ171 pinkish-white area | +0.74 | 2 |
| DQ003 heterogeneous colour | +0.74 | 1 |
| DQ101 verrucous | −0.69 | 0 |
| DQ473 uneven patchy pigmentation | +0.69 | 1 |
| DQ038 colour variegation | +0.66 | 1 |
| DQ009 ugly duckling | +0.64 | 0 |
| DQ425 lesion colour distinct from skin | +0.60 | 2 |

With 10 images this shows reliability, not predictive value.

## 14. Classifiers

- **Common to all:** features are `p_adj` by default; questions with `kind` in `model.exclude_kinds`
  (artefact) are dropped; classes are balanced; label −1 images are excluded from fitting and metrics but
  get predictions (split `holdout`).
- **W0:** standardise, then L2 logistic regression. C ∈ {0.001, 0.01, 0.1, 1, 10} is chosen by grouped
  stratified CV AUROC on the training set (ties go to stronger regularisation). Regularisation path in
  `curves.csv`.
- **W1:** elastic net (saga) over C × l1_ratio ∈ {0.1, 0.5, 0.9}, chosen by inner grouped CV (nested inside
  the outer split). Stability selection: 50 half-subsamples of training groups; a question is selected
  if it is non-zero in ≥ 60% of them → `selected_questions.csv`, usable as `Q2_selected`.
- **P0–P3 (multiple-instance):**
  - patch score s_k = w·x_k + b; patch probability p_k = sigmoid(s_k);
  - P0 takes the mean of the p_k; P1 uses max, top-k (k = 2) or noisy-OR 1 − Π(1 − p_k);
  - P2 uses attention: a = softmax(v·tanh(U x_k)), image probability = sigmoid(Σ a_k s_k);
  - P3 learns weights per grid position (translation-dependent; only meaningful if lesions are centred).
  - Training: Adam, weight decay 0.01, BCE with class weights, up to 200 epochs, early stopping on
    validation loss (patience 30) when a validation split exists. Patch scores are saved for coarse maps.
- **Metrics (every model):** AUROC, AUPRC, balanced accuracy, sensitivity at 90% specificity, Brier,
  ECE (10 bins), confusion matrix; train/val curves.
- **Smoke results:** leave-one-out AUROCs between 0 and 0.67 on 8 images. These are not informative. W1's
  0.000 is a known leave-one-out artefact: heavily shrunk models give near-constant predictions, which
  rank exactly backwards.

## 15. Reproducibility and reports

- **Command shape:** `python -m qaemb.run stage=<manifest|prepare|answer|embed|model|all> +exp=[IDs] tag=<text> [key=value]`.
  `dry_run=true` prints the resolved config and which stages are cached.
- **Groups and IDs** (`conf/exp/<group>/<ID>.yaml`):

  | Group | IDs |
  |---|---|
  | data | `D0_smoke10`, `D1_train` |
  | qset | `Q0_all`, `Q1_local`, `Q2_selected`, `Q9_smoke50` |
  | answerer | `ANS_qwen7b`, `ANS_qwen3b` |
  | mode | `M_stateless`, `M_topic`, `M_all` |
  | feat | `F_adj`, `F_adj_bin`, `F_orig`, `F_orig_bin` |
  | grid | `G1`, `G2`/`G4`/`G8` (+`_ov`) |
  | model | `W0`, `W1`, `P0`, `P1_max`, `P1_topk`, `P1_noisyor`, `P2`, `P3` |

  Two IDs from one group are rejected.
- **Per run:** `outputs/qaemb/runs/<run_id>/run_meta.json` (full config, stage hashes, model revision
  `cc594898…`, library versions, GPU, git commit, seeds, wall time), `report.md`, a row in
  `experiments_log.md`, and an MLflow run (`outputs/mlflow.db`).
- **Per batch of experiments:** a write-up in `reports/<date>_<name>/`.
- **Comparisons:** `python -m qaemb.report compare <run_id> … --out reports/<name>`.

## 16. Cost and next steps

- **Cost formula:** images × items × 0.0098 s.
  - Whole images, full beta set: 1,000 × 1,000 × 0.0098 s ≈ **2.7 h**.
  - Plus a 4×4 grid on local questions: 1,000 × 16 × 466 more items, ≈ **23 h** total.
  - G2 ≈ 7.8 h total and G8 ≈ 84 h total, by the same formula.
- **Next steps:**
  1. Test positively phrased complements on the 10 images.
  2. Copy in the remaining 990 images (`D1_train` stops with a missing-files error until then); run W0 and W1.
  3. Patch models on G4 with `Q1_local`.
  4. S6 relevance maps: gradient-weighted attention on the Yes token (needs `answerer.attn=eager`), plus
     a deletion test (blur the top-relevance regions, re-ask, measure the drop in P(yes)).
  5. PanDerm structure clusters (k-means on patch tokens), attribution maps and the hover viewer (S7).

## 17. Open questions for the group

- **Negations:** positive opposites or literal negations?
- **The 44 metadata-scope questions:** drop them, or keep them as noise controls?
- **Ground truth for relevance maps:** ISIC 2018 Task 2 attribute masks, or in-house masks (e.g. ring masks)?
- **Patients:** with no patient IDs, is grouping by lesion ID acceptable?

## Where everything is

| What | Path |
|---|---|
| Code | `qaemb/` (stages, answerer, models, `report.py`, `docs.py`) |
| Config and experiments | `conf/base.yaml`, `conf/exp/` |
| Questions and annotations | `2026-10-15 Generated Questions/DermQs_Source-1015FT_Prompt-C_N-500_Negated.csv`, `…_annotations.csv` |
| Smoke-test write-up and comparison CSVs | `reports/2026-10-27_smoke10/` |
| Run reports and raw outputs (not in git) | `outputs/qaemb/` |
| Original plan | `handoff/HANDOFF_QA-Emb-Derm.md` |
