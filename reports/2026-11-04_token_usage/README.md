# Token usage of every experiment so far, and the projection for 1,000 images (2026-11-04)

**What this covers:**
- **Measured:** tokens, answering time and Claude's list-price cost for every answer stage run to date.
- **Projected:** the next step, all 500 questions **without negations** on the 1,000 images of the beta set.

**How it was produced:**
- **Data:** the full per-stage table is [token_usage_by_stage.csv](token_usage_by_stage.csv); the projection is
  [projection.csv](projection.csv).
- **Script:** both come from [token_usage.py](token_usage.py) (`poetry run python reports/2026-11-04_token_usage/token_usage.py`).
- **Where the numbers come from:**
  - Tokens are read from each answer stage's `timing.parquet`, one row per prompt.
  - Claude's cost is the list-price figure that `claude -p` reports for each call, logged per run in
    `outputs/qaemb/queue/queue.log` and `pilot_topic_R1.log`.
  - On the Pro subscription nothing was charged; the dollar figures are what the same calls would cost at API list price.

**Terms:**
- A **prompt** is one call to the model.
- An **item** is one question or negated question answered for one image.
- **Stateless** (`M_stateless`) asks one question per prompt.
- **Topic** (`M_topic`) asks all of a topic's questions in one prompt (41 topics), with questions and negations in separate prompts.

## Totals to date (35 answer stages)

| Answerer | Answer stages | Prompts | Items answered | Input tokens | Output tokens | Answering time | List price |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 via `claude -p` | 12 | 9,312 | 24,000 | 19.9 M (8.5 M read from cache, 11.4 M written to cache) | 2.86 M | 13.5 h | **$151.52** (not charged: Pro subscription) |
| Qwen2.5-VL-7B (local GPU) | 16 | 54,546 | 91,450 | 2.0 M | 0.30 M | 0.7 h | – |
| SkinGPT-R1 (local GPU) | 7 | 11,496 | 38,000 | 0.9 M | 0.20 M | 0.4 h | – |

Not included in this table is the stopped 10-image all-questions-in-one-prompt Claude pilot (`M_all`, answer stage
`077a36214317`). It made 17 attempts, used 0.23 M input and 0.09 M output tokens, and its cost was not logged.

**Why a Claude prompt costs so much more than a local one.**
- **Claude** receives the image as about 1,600 tokens in every prompt, plus a fixed preamble of 576 tokens that
  `claude -p` adds. It also writes its answer as JSON with a confidence for each question.
- **The local models** encode the image once per image and reuse it for every question; the counts above exclude
  those image tokens. Each answer is a single word, Yes or No, read from the model's probabilities.

## Claude runs

All 12 runs used images `D0_poc2` (2 images: ISIC_0000147 and ISIC_0000215), questions `Q0_all` (500 questions plus
their 500 negations), the whole image (`G1`), and Claude Opus 5.5 at effort low (`ANS_claude_opus_cli`).

| Run | Mode | Replicate | Prompts | Input tokens | of which cache read / write | Output tokens | Time | List price | Per prompt |
|---|---|---|---|---|---|---|---|---|---|
| `20261030-005209_claude-cli-poc` | stateless | R1 | 2,000 | 4.17 M | 0.20 M / 3.97 M | 576 k | 2.8 h | $43.60 | $0.0218 |
| `20261030-013448_claude-cli-poc` | stateless | R2 | 2,000 | 4.05 M | 3.31 M / 0.74 M | 535 k | 2.7 h | $17.56 | $0.0088 |
| `20261030-051502_claude-cli-poc` | stateless | R3 | 2,000 | 4.07 M | 2.97 M / 1.10 M | 523 k | 2.7 h | $20.12 (paused once; two parts) | $0.0101 |
| `20261030-052021_claude-cli-poc` | stateless | R4 | 2,000 | 4.00 M | 0.42 M / 3.58 M | 572 k | 2.9 h | $40.18 | $0.0201 |
| `20261030-002157_claude-cli-poc` | topic | R1 | 164 | 490 k | 97 k / 393 k | 86 k | 18 min | $4.91 | $0.0299 |
| `20261030-003848_claude-cli-poc` | topic | R2 | 164 | 505 k | 162 k / 342 k | 80 k | 17 min | $4.41 | $0.0269 |
| `20261030-004312_claude-cli-poc` | topic | R3 | 164 | 515 k | 374 k / 141 k | 80 k | 17 min | $2.83 | $0.0173 |
| `20261030-004738_claude-cli-poc` | topic | R4 | 164 | 537 k | 447 k / 90 k | 81 k | 17 min | $2.46 | $0.0150 |
| `20261030-141502_claude-shuffle` | topic, shuffled (`M_topic_shuf`) | R1 | 164 | 404 k | 81 k / 323 k | 80 k | 16 min | $4.20 | $0.0256 |
| `20261030-141920_claude-shuffle` | topic, shuffled | R2 | 164 | 382 k | 124 k / 258 k | 81 k | 17 min | $3.71 | $0.0226 |
| `20261030-142338_claude-shuffle` | topic, shuffled | R3 | 164 | 410 k | 137 k / 273 k | 83 k | 17 min | $3.87 | $0.0236 |
| `20261030-142804_claude-shuffle` | topic, shuffled | R4 | 164 | 382 k | 133 k / 249 k | 82 k | 17 min | $3.67 | $0.0224 |

**The same run can cost 2.5× more or less.** Identical prompts sent with `claude -p` cost between $0.0088 and $0.0218
per stateless prompt. Token counts were almost the same; the difference is how much of each prompt Anthropic's side
served from its cache (cheap) rather than wrote to it (more expensive). `claude -p` gives no control over that. A direct
API integration could cache each image once and reuse it for all of that image's questions, but that is not measured here.

**Subscription capacity, as measured.** Usage percentages come from the subscription's rate-limit reports and are
rounded to whole percent.
- **5-hour window:** across the 8 runs that started and ended in the same window, a full window held about **$105** of
  list-price usage.
- **Weekly limit:** the weekly figure rose from 3% to 17% over $142 of runs, so a full week holds about **$1,000**.
- **These are conservative.** The same allowance also paid for this Claude Code session and any other Claude use at the time.

## Local-model runs

All ran on one Quadro RTX 8000. Data is `D0_smoke10` (10 smoke images, one per diagnostic class) or `D0_poc2`
(the 2 images above). The question sets are:
- `Q0_all`: 500 questions plus negations;
- `Q9_smoke50`: 50 questions plus negations, the early smoke tests;
- `Q1_local`: the 233 questions about a local region, used for the patch runs.

The grid is the whole image (`G1`) except for the two patch runs, which used `G2` (whole image plus 2 × 2 patches).

| Answerer | Mode | Data, questions, grid | Runs (answer stage) | Items | Input tokens | Output tokens | Time |
|---|---|---|---|---|---|---|---|
| Qwen | stateless | 10 images, `Q0_all`, `G1` | `20261029-152402_answerers-oct29` and 5 earlier (`f8daf7e76c0e`) | 10,000 | 248 k | 10 k | 98 s |
| Qwen | topic | 10 images, `Q0_all`, `G1` | `20261029-152411_answerers-oct29` and 3 earlier (`0b679433bc77`) | 10,000 | 185 k | 61 k | 7.1 min |
| Qwen | all in one prompt (`M_all`) | 10 images, `Q0_all`, `G1` | `20261029-152420_answerers-oct29` and 3 earlier (`4b2dcc28e194`) | 10,000 | 176 k | 78 k | 11.8 min |
| Qwen | topic, shuffled, R1–R4 | 2 images, `Q0_all`, `G1` | `20261030-135524`, `-135855`, `-140216`, `-140536` (`_local-shuffle`) | 2,000 each | 37 k each | 12 k each | about 80 s each |
| Qwen | stateless, patches | 10 images, `Q1_local`, `G2` | `20261027-121433_neg-smoke3` and 2 more (`b133bd2881dd`) | 23,300 | 569 k | 23 k | 3.9 min |
| Qwen | (superseded smoke runs, 2026-10-23 and -27) | various | 8 answer stages, see the CSV | 30,150 | 0.69 M | 0.08 M | 11 min |
| SkinGPT-R1 | stateless | 10 images, `Q0_all`, `G1` | `20261029-152430_answerers-oct29` | 10,000 | 368 k | 10 k | 2.0 min |
| SkinGPT-R1 | topic | 10 images, `Q0_all`, `G1` | `20261029-152656_answerers-oct29` | 10,000 | 192 k | 61 k | 7.1 min |
| SkinGPT-R1 | all in one prompt | 10 images, `Q0_all`, `G1` | `20261029-153429_answerers-oct29` | 10,000 | 176 k | 78 k | 11.9 min |
| SkinGPT-R1 | topic, shuffled, R1–R4 | 2 images, `Q0_all`, `G1` | `20261030-135709`, `-140040`, `-140400`, `-140712` (`_local-shuffle`) | 2,000 each | 38 k each | 12 k each | about 75 s each |

## Projection: 500 questions, no negations, 1,000 images (500,000 items)

**Method:**
- **Claude:** tokens per prompt are taken from the original-question prompts of the four fixed-order replicates for
  each mode, so negations are excluded. The price range is the lowest, median and highest cost per prompt among those
  four runs.
- **Local models:** time and tokens per item come from the 10-image runs.
- **Topic prompts:** 41 per image, one per topic.

| Answerer | Mode | Prompts | Input tokens | Output tokens | List price (low / median / high) | On the Pro subscription | Local GPU time |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | **topic** | 41,000 | 130 M | 18 M | **$615 / $906 / $1,226** | about 6–12 full 5-hour windows; **about 0.8–1.6 weeks** of the weekly allowance, with the queue stopping at 75% of it | – |
| Claude Opus 5.5 | stateless | 500,000 | 1,028 M | 130 M | **$4,400 / $7,550 / $10,900** | about **6–15 weeks** of the weekly allowance | – |
| Qwen2.5-VL-7B | stateless | 500,000 | 12 M | 0.5 M | – | – | **1.4 h** |
| Qwen2.5-VL-7B | topic | 41,000 | 9 M | 3 M | – | – | 6.0 h |
| SkinGPT-R1 | stateless | 500,000 | 18 M | 0.5 M | – | – | **1.7 h** |
| SkinGPT-R1 | topic | 41,000 | 10 M | 3 M | – | – | 5.9 h |

**Notes:**
- **Topic prompts are much cheaper for Claude.** They need about one-twelfth of the prompts and one-eighth of the input
  tokens of stateless, because the image is sent once per topic rather than once per question. Fig. 6 of the
  [2026-10-30 letter](../2026-10-30_claude_poc/letter.md) found that dropping the negations and keeping Claude's stated
  confidence loses almost nothing (ρ = 0.96).
- **Keeping the negations doubles every number in the table.**
- **The weekly limit is shared with all other Claude use on the same account,** including Claude Code. The Batch API
  would be cheaper than the list price but needs a separate Console account; check current API pricing before budgeting.
- **Correction:** on 2026-11-04 I estimated that Claude topic prompts for 1,000 images would take "several weeks" of the
  Pro allowance. The measured figure is about 0.8–1.6 weeks without negations; stateless is the one that takes many weeks.
- **These are estimates.** They extrapolate from 2 images (Claude) and 10 images (local models); image size and the
  number of questions per topic set the token counts.
