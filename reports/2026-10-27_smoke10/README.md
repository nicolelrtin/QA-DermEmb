# Smoke test: 10 images, 500 question pairs (2026-10-27)

**Purpose:** check the QA-Emb pipeline end to end on the 10 smoke images (one per class) before scaling.
This covers the answer modes, the question/complement pairs, timing, and the whole-image and patch classifiers.
With 8 labelled images (5 benign, 3 malignant; the 2 indeterminate images are scored but held out),
**classifier metrics are not meaningful**. The useful outputs are the answers themselves, the timing,
and the consistency diagnostics.

## Where to look

| What | File |
|---|---|
| All answers for the 10 images in the three modes, side by side (one row per image x question) | [mode_comparison/compare_answers.csv](mode_comparison/compare_answers.csv) |
| Per question: mean p_adj and contradiction rate in each mode | [mode_comparison/compare_by_question.csv](mode_comparison/compare_by_question.csv) |
| Mode comparison tables (timing, agreement, verdict shares) | [mode_comparison/summary.md](mode_comparison/summary.md) |
| Per-run reports (config, stage outputs, metrics, top questions) | runs listed below, in `outputs/qaemb/runs/<run_id>/report.md` |

Columns in `compare_answers.csv`: `question` and `question_neg` are the original question and its complement.
For each mode (`__stateless`, `__topic`, `__all`) there are four values:
- `p_yes_original` and `p_yes_complement`: P(yes) to each version.
- `p_adj = p_yes_original − p_yes_complement`, in [−1, 1].
- `verdict`: `yes` or `no` when the two answers are complementary, otherwise `contradiction: yes/no to both`.

`outputs/` is not in git (it is regenerated on the workstation). This folder is, so the CSVs here are the copy to keep.

## What was run

All runs use Qwen2.5-VL-7B-Instruct (fp16) and your edited prompts (`qaemb/prompts/*_v1.txt`).
Questions are `DermQs_Source-1015FT_Prompt-C_N-500_Negated.csv`: 500 originals plus 500 complements.

| Run report | Experiments | What it tests |
|---|---|---|
| [20261027-153315_smoke-oct27](../../outputs/qaemb/runs/20261027-153315_smoke-oct27/report.md) | `D0_smoke10, Q0_all, ANS_qwen7b, M_stateless, G1, F_adj, W0` | one question per sequence |
| [20261027-153324_smoke-oct27](../../outputs/qaemb/runs/20261027-153324_smoke-oct27/report.md) | `… M_topic …` | one prompt per (original, topic): 82 prompts per image |
| [20261027-153332_smoke-oct27](../../outputs/qaemb/runs/20261027-153332_smoke-oct27/report.md) | `… M_all …` | one prompt per original value: 2 prompts per image |
| [20261027-153341_smoke-oct27](../../outputs/qaemb/runs/20261027-153341_smoke-oct27/report.md) | `D0_smoke10, Q1_local, ANS_qwen7b, M_stateless, G2, F_adj, P0` | whole image + 2x2 patches, 233 local question pairs, patch MIL model |

Regenerate this comparison with:
```bash
python -m qaemb.report compare 20261027-153315_smoke-oct27 20261027-153324_smoke-oct27 20261027-153332_smoke-oct27 \
    --out reports/2026-10-27_smoke10/mode_comparison
```

## Results

### 1. Stateless is the only mode that gives usable answers

| Mode | Prompts per image | Seconds per item | Answered yes | Same answer to Q and not-Q | Mean \|p_adj\| |
|---|---|---|---|---|---|
| stateless | 1,000 | **0.0098** | 13.2% | **78.9%** | **0.21** |
| topic | 82 | 0.043 | 3.9% | 93.2% | 0.10 |
| all | 2 | 0.071 | 0.3% | 99.4% | 0.01 |

In the batched modes the model drifts to "No" as it works through a prompt. Mean P(yes) falls from
0.18 to 0.004 between the first and last question of a topic prompt, and from 0.28 to 0.002 in `all`.
Batched modes are also 4–7x slower, because each answer needs its own decoding step and the image is
already reused in stateless mode. Topic and `all` agree with stateless on 88% and 87% of binary
answers, but mostly because all three say "No".

### 2. Even stateless, most question pairs are answered "No" to both

Stateless verdicts over 10 images x 500 questions: 76% "no to both", 14% yes, 7% no, 3% "yes to both".
211 of 500 questions are contradicted on all 10 images. These questions therefore contribute
p_adj ≈ 0 everywhere, which is the "not confident" signal the complements were designed to catch.

The complement wording matters. For example, for DQ002 "Is the lesion border irregular?" the complement
"Does the lesion lack an irregular border?" gets P(yes) ≤ 0.09 on 8 of 10 images, including two where
the original is also ≤ 0.06. "Lack …" and "Is it not the case that …" phrasings seem to push the model towards "No"
regardless of the image. Some of the contradiction is likely the phrasing, not the image.

### 3. Questions the model answers consistently

These 12 questions have the largest |mean p_adj| (stateless; positive = confident yes, negative = confident no):

| Question | Mean p_adj | Contradiction rate |
|---|---|---|
| DQ382 Does the skin appear thick and leathery? | −0.87 | 0.0 |
| DQ410 Visible demographic cues a model could exploit? | −0.86 | 0.0 |
| DQ016 Is an irregular blotch present? | 0.77 | 0.0 |
| DQ383 Is dyspigmentation visible? | 0.75 | 0.1 |
| DQ171 Is a pinkish-white area present? | 0.74 | 0.2 |
| DQ003 Is the lesion colour heterogeneous? | 0.74 | 0.1 |
| DQ101 Is the lesion verrucous (wart-like)? | −0.69 | 0.0 |
| DQ473 Uneven pigmentation, patchy dark and light areas? | 0.69 | 0.1 |
| DQ038 Is colour variegation present across the lesion? | 0.66 | 0.1 |
| DQ009 Does one pigmented lesion look different from the others ("ugly duckling")? | 0.64 | 0.0 |
| DQ406 Visible colour cast across the image? | −0.61 | 0.1 |
| DQ425 Lesion colour distinct from surrounding skin? | 0.60 | 0.2 |

The full ranking is in `compare_by_question.csv`.

### 4. Timing and full-data cost (stateless)

- 9.8 ms per item including the image prefill, which takes about 340 ms per image and is then reused.
- Cached P(yes) matches an uncached forward pass to within 0.004.
- Estimated full beta set (1,000 images): whole images only (1,000 items each) about **2.7 h**.
  With a 4x4 patch grid on the 233 local pairs, about **23 h**.

### 5. Classifiers (plumbing only)

W0 (L2 logistic on p_adj, 446 non-artefact questions) and P0 (shared-weight patch model, mean
aggregator, 217 local non-artefact questions) both train and score. W1 and the P1/P2/P3 variants also
run. Leave-one-out AUROCs (0.4–0.6) on 8 images carry no information.

## Suggested next steps

1. Use **stateless** as the answer mode. Keep `topic` only as the order-effect comparison.
2. Review the complement phrasings. A quick test: rewrite a sample of complements as positive
   statements ("Is the lesion border regular?" instead of "Does the lesion lack an irregular border?")
   and re-run the contradiction rate on these 10 images.
3. Then copy in the remaining 990 images and run W0 on the full beta set (about 2.7 h).

## Earlier runs (superseded)

The 2026-10-23 smoke runs (`outputs/qaemb/runs/20261023-*`) used the original prompts, no complements,
and random batches of 10. They are kept in `experiments_log.md` but have no report.md.
