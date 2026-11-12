# First runs through the Claude API: pilot, cost projection and the 1,000-image beta set (2026-11-11)

**Purpose.** Before the 1,000-image run, we checked that the direct API answerer works end to end, measured what it
costs per prompt, and projected the full run from those measurements. Three things changed on the same day and were
tested together: the answering images are now sent at the largest size Sonnet uses, a control question was added, and
every API response is now priced and logged against a spending checkpoint.

## Runs covered

| Run | Experiments (meaning) | Stage hashes | Prompt |
|---|---|---|---|
| [20261111-093530_api-pilot](../../outputs/qaemb/runs/20261111-093530_api-pilot/report.md) | `D0_poc2` (ISIC_0000147, melanoma; ISIC_0000215, benign), `I1_claude_max` (images shrunk only, to at most 3.6 MP and 2,576 px long edge), `Q0_orig` (500 questions as written plus the control question, no negations), `M_topic` (one prompt per topic, 42 per image), `ANS_claude_sonnet` (Claude Sonnet 5.5 through the Messages API, effort low), `G1` (whole image), `R1` (replicate 1). Direct requests, not a batch (`claude.use_batch=false`). | manifest `6599738ba8e0`, prepare `96e084c41283`, answer `5a8fbfee003d`, embed `b83eecf9364f` | `claude_batched_whole_v1` |

Two further runs used the same settings: the 10-image batch check (below) and the full beta set,
[20261111-095000_api-full1000](../../outputs/qaemb/runs/20261111-095000_api-full1000/report.md) (`D1_train`, the
1,000 beta images with the train/validation/test split grouped by lesion; stage hashes manifest `531882139b22`, prepare
`c8f171e55274`, answer `2e3804059ebe`, embed `6cd6f95ebcbd`; Message Batch `msgbatch_01HGdNdQPBZGuexQCgKjjDpA`).

Comparison run: the Sonnet answers from `claude -p` on 2026-11-04 (run `20261104-221846_sonnet-test`, answer stage
`b0aea9414af7`, ISIC_0000147 only, images at 0.4 MP).

## What changed before the run

**Image size.** Until now, S1 resized every image to 0.4 MP, the budget chosen for Qwen. Sonnet 5.5 can use far more.
We measured with the free token-counting endpoint how many tokens one image costs at several sizes (the largest beta
image, ISIC_0011775, 6721 × 4521, resized). Tokens grow as about pixels / 750 and stop growing at about 4,700 tokens
once the image passes about 3.6 MP; larger images are shrunk by the API itself.

| Size sent | Megapixels | Image tokens |
|---|---|---|
| 728 × 490 | 0.36 | 471 |
| 1872 × 1259 | 2.36 | 3,018 |
| 2310 × 1553 | 3.59 | 4,651 |
| 2400 × 1614 | 3.87 | 4,707 |
| 3000 × 2018 | 6.05 | 4,707 |

So "the highest resolution in the training data" in practice means Sonnet's own limit. The new setting
`I1_claude_max` shrinks each image to at most 3.6 MP and 2,576 px on the long edge, never enlarges it, and saves it as
JPEG at quality 95. Across the 1,000 beta images this gives on average 2,665 image tokens per prompt (it was about 520
before). 43% of the images reach the limit, and 26% are under 0.4 MP and are sent at their original size; at the old
setting these small images had been enlarged.

**Control question.** We added DQC01, "Is this a malignant lesion?", as a new question set dated 2026-11-11
(`inputs/questions/2026-11-11/`; the 2026-10-15 files are unchanged). It has its own topic, "Control", so it is asked
in a prompt of its own and the other questions cannot influence its answer, nor it theirs. Its kind is `control`, and
control questions are never used as classifier features (`feature_mask` in `qaemb/stages/s5_model.py`); otherwise the
classifiers would simply learn to repeat Claude's diagnosis. Its answer is the zero-shot benchmark.

**Spending checkpoint.** Every billed API response is now priced from its token counts at list price (batch at half)
and written to `outputs/qaemb/claude_spend.csv`, one row per response, shared by all runs. Before each submission the
cost is projected: the text of 40 sampled requests is counted with the free token-counting endpoint, image tokens are
estimated from the pixel count, and output tokens from the answers seen so far, plus a 10% margin. A submission that
would take the total past $400 stops until it is approved on the command line (`claude.budget.approved_usd=<amount>`).
Nothing is submitted past $500. The ledger only sees this pipeline, so the Console spend limit is still the hard
backstop.

## Validity checks

- **Every request returned a complete answer the first time.** All 84 prompts came back with a numeric P(yes) and
  confidence for every question; none had to be resent and none was refused. A failure here would mean missing
  answers, which the stage would have resent and then stopped on.
- **The projection errs on the high side.** It estimated $0.97 including the 10% margin ($0.88 without it); the
  measured cost was $0.78.
- **The API answers agree with the earlier `claude -p` answers.** On ISIC_0000147 the two gave the same yes/no
  answer to 474 of 500 questions (94.8%), with a rank correlation of 0.96 between their P(yes), and the same share of
  "yes" answers (13.8%). This is about the agreement we saw between Sonnet and Opus, even though the image was
  now sent at 2048 × 1536 instead of 728 × 532. So neither the new path nor the larger image changed the answers
  much. One image only.

## Results

| | ISIC_0000147 (melanoma) | ISIC_0000215 (benign) | Both |
|---|---|---|---|
| Image sent | 2048 × 1536 (its original size) | 962 × 674 (its original size) | |
| Input tokens per prompt (median) | 4,646 | 1,451 | |
| Share of the 500 questions answered yes | 14% | 12% | 13% |
| Control question: P(yes), confidence | 0.90, 0.8 | 0.25, 0.5 | |
| Cost | | | **$0.78** |

Per prompt, input was about 410 tokens of text, plus 21 per question, plus the image. Output was about 10 tokens
plus 25 per question. The median request took 2.8 s.

The control question went the right way on both images. Two images say nothing about how well it works.

## Second run: 10 images through the Batch API

Run [20261111-094616_api-batch10](../../outputs/qaemb/runs/20261111-094616_api-batch10/report.md): `D0_smoke10` (the
10 smoke images, one per diagnostic class) with the same settings as above, sent as one Message Batch (420 requests;
stage hashes: prepare `bc5f7d0faaa2`, answer `f7059ab01aff`, embed `cc665a9429a1`).

- **Validity:** all 420 prompts complete on the first attempt; no missing answers. The batch finished within minutes.
- **Cost:** $2.39 at batch price, against a projection of $2.70 including the 10% margin. These images are larger
  than the beta average (about 4,150 input tokens per prompt against an expected 3,300).
- **Repeatability at full image size.** The two pilot images were answered a second time in this batch. The two
  runs gave the same yes/no answer to 97.4% (ISIC_0000147) and 98.6% (ISIC_0000215) of the 501 questions, with rank
  correlations of 0.98 and 0.97 and a mean change in P(yes) of 0.02. This is as repeatable as Opus was with itself
  (98%, [2026-11-04 Sonnet test](../2026-11-04_sonnet_test/README.md)). Two images.
- **Control question.** P(yes) to "Is this a malignant lesion?" was highest for the melanoma (0.90) and the malignant
  epidermal proliferation (0.85). The malignant adnexal (follicular) lesion got 0.40, and the inflammatory lesion and
  the indeterminate epidermal lesion 0.55 each. One image per class; this only shows the answers vary sensibly.
- **Share of yes answers** ranged from 7% to 14% per image, the same lean towards "no" as before.

## Projection for the 1,000-image beta set

From the measured token counts and the mean image size of the beta set: about 140 M input and 13 M output tokens for
Sonnet, 42 prompts per image.

| Run | Prompts | List price | With the batch discount |
|---|---|---|---|
| Sonnet, 1,000 images | 42,000 | $409 | **$205** |
| Opus on 50 images (assuming 30% more output, as with `claude -p`) | 2,100 | $45 | $22 |
| 10-image batch check before the full run | 420 | $4 | $2 |
| Resent requests (assume 5%) | | $23 | $11 |
| **Total** | | **about $480** | **about $240** |

Without the batch discount the plan would come close to the $500 limit, so the full run must go through the Batch
API (`claude.use_batch: true`, the default).

**Time.** Resizing the 1,000 images takes minutes. Uploading them to the Files API, one at a time, should take
15–20 minutes. Most batches finish within an hour; Anthropic allows up to 24 hours.

## Third run: the 1,000-image beta set

Run [20261111-095000_api-full1000](../../outputs/qaemb/runs/20261111-095000_api-full1000/report.md), the plan's main
Sonnet run: `D1_train` (all 1,000 beta images), `I1_claude_max`, `Q0_orig`, `M_topic`, `ANS_claude_sonnet`, `G1`, `R1`,
sent as one Message Batch of 42,000 requests (42 topic prompts per image). It ran unattended from 09:50 to 12:44; almost
all of that was waiting for the batch.

**Validity checks.**

- **Every answer is there.** All 42,000 prompts came back complete on the first attempt; no request had to be resent.
  The embed stage holds 501,000 answers (1,000 images × 501 questions), each with a P(yes) and a confidence, with no
  gaps and no duplicates. A failure would have shown as missing answers and a second, smaller batch.
- **The cost stayed inside the budget.** The spending check projected $227 for this batch including the 10% margin; it
  cost **$203.63** (138.9 M input tokens, 12.9 M output tokens, batch price). Each response is one row in
  `outputs/qaemb/claude_spend.csv`. **Total spent on the API key so far: $206.80** of the $500 budget, so the $400
  checkpoint was never reached. The Opus check on 50 images (about $22) still fits.

**Results.**

- **Sonnet says "yes" less often on the full set.** Of the 500 questions, 9.5% were answered yes on average (13% on
  the 2 pilot images, 7–14% on the 10 smoke images). The share is almost the same for benign (9.1%) and malignant
  (9.9%) images, so the raw count of yes answers says little about the diagnosis; whatever signal there is must lie in
  which questions are answered yes.
- **Zero-shot benchmark: the control question.** P(yes) to "Is this a malignant lesion?", on the 800 images with a
  benign or malignant label (500 benign, 300 malignant; the 200 indeterminate images are left out), separates the two
  classes with an AUROC of **0.79**: a randomly chosen malignant image gets a higher P(yes) than a randomly chosen
  benign one 79% of the time. Taking P(yes) of 0.5 or more as "malignant", Sonnet finds 56% of the malignant lesions
  (sensitivity) and correctly clears 83% of the benign ones (specificity). To find 95% of the malignant lesions, the cut
  has to drop to 0.12, and then only 22% of the benign lesions are cleared. Sonnet's P(yes) is coarse (it mostly uses
  values such as 0.12, 0.2, 0.3 and 0.6), which limits how finely the cut can be set. These figures pool all 800
  images; the benchmark row for the classifiers will be computed on the same cross-validation folds as S6.

The answers are ready for S6 (classify): features in `outputs/qaemb/embed/6cd6f95ebcbd/` (`X_image_*.parquet`), with
DQC01 excluded from the features by `feature_mask`.
