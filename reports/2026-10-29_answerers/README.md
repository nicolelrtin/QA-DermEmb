# Answerer comparison: Qwen2.5-VL vs SkinGPT-R1 vs Claude Opus 5.5 (2026-10-29)

**Status:** Qwen and SkinGPT-R1 are complete. Claude is pending: Anthropic credentials are not yet set up on the
workstation, so no API calls have been made.

**Design** (agreed 2026-10-29):
- **Images and questions:** the 10 smoke images, whole image only, and the 500 question pairs (original + complement).
- **Modes:** stateless, topic and all; originals and complements are always asked in separate prompts.
- **Metrics:** P(yes) to the original and complement, `p_adj`, the new `P_w = P(yes|original) × |p_adj|`,
  and (Claude only) stated confidence.
- **Qwen and SkinGPT:** P(yes) is read from the Yes/No logits. They are deterministic, so one run each.
  Qwen answers were reused from 2026-10-27; only P_w was added.
- **Claude:** stated P(yes) and confidence as forced JSON. Opus 5.5, effort low, Batch API, 4 replicates per mode.
- **SkinGPT-R1:** a reasoning model. Its reasoning is skipped by starting its reply for it with
  `<think></think><answer> Final answer:`, so the next word it writes is the answer.
- **Readout check (Qwen and SkinGPT):** P(yes) is only meaningful if the model's next word is actually
  "Yes" or "No" rather than something else (e.g. "Final" or "The"). We record how much of the model's
  next-word probability goes to "Yes" or "No" combined. For SkinGPT it is 100% on average (fewer than 0.1% of
  answers below 90%), so its P(yes) values are real answers. This says nothing about *which* answer it
  gave: SkinGPT says yes to about a third of questions and no to the rest.

## Where to look

| What | File |
|---|---|
| Qwen and SkinGPT side by side, every image × question, all modes | [local_models/compare_answers.csv](local_models/compare_answers.csv) |
| Per question: mean p_adj, mean P_w, contradiction rate per run | [local_models/compare_by_question.csv](local_models/compare_by_question.csv) |
| Comparison tables | [local_models/summary.md](local_models/summary.md) |
| Per-run reports, now with a Tokens section | `outputs/qaemb/runs/20261029-*_answerers-oct29/report.md` |

## Results so far (10 images × 1,000 items per run)

| Answerer / mode | Same answer to Q and not Q | Mean \|p_adj\| | Mean P_w | Answered yes | P(yes) first → last in a prompt | Seconds per item | Input / output tokens |
|---|---|---|---|---|---|---|---|
| Qwen / stateless | 78.9% | 0.21 | 0.08 | 13.2% | — | 0.0098 | 247,694 / 10,000 |
| Qwen / topic | 93.2% | 0.10 | 0.03 | 3.9% | 0.18 → 0.004 | 0.043 | 184,544 / 60,520 |
| Qwen / all | 99.4% | 0.01 | 0.004 | 0.3% | 0.28 → 0.002 | 0.071 | 176,164 / 77,740 |
| **SkinGPT-R1 / stateless** | **47.1%** | **0.50** | **0.20** | 38.5% | — | 0.012 | 367,694 / 10,000 |
| SkinGPT-R1 / topic | 61.0% | 0.39 | 0.08 | 27.5% | 0.31 → 0.30 | 0.043 | 191,924 / 60,520 |
| SkinGPT-R1 / all | 70.4% | 0.29 | 0.09 | 15.7% | 0.51 → 0.40 | 0.071 | 176,344 / 77,740 |

**Verdicts, stateless:**
- Qwen: 76% no to both, 14% yes, 7% no, 3% yes to both.
- SkinGPT-R1: 35% no to both, 22% yes, 31% no, 12% yes to both.

**Readings:**
- SkinGPT-R1 answers question/complement pairs far more coherently than Qwen: more than half of its stateless
  pairs are complementary, against about a fifth for Qwen. Its contradictions are also more balanced between
  "yes to both" and "no to both".
- SkinGPT-R1 shows much less drift towards "No" within a batched prompt (0.31 → 0.30 in topic mode).
  Stateless is still its most coherent mode.
- The two models agree only moderately: stateless Pearson r 0.60, and 72% of binary answers the same.
- Token counts for these models count the image once per image (about 500 tokens), because it is processed
  once and reused. Claude resends the image with every request, so its per-question tokens will be much higher.

Classifier numbers (leave-one-out, 8 images) remain uninformative.

## Pending: Claude Opus 5.5

1. **Pilot:** 1 image, direct calls, all three modes, to measure actual tokens and cost before the batch.
2. **Full run:** 4 replicates × 3 modes on the Batch API (`ANS_claude_opus`, `R1`–`R4`).
3. **Reports:** `python -m qaemb.report replicates` per mode (raw and aggregated values), then `compare` with the
   replicate means against Qwen and SkinGPT.
