# Claude Sonnet 5.5 vs Opus 5.5 on one image: agreement and cost (2026-11-04)

**Question:** would the cheaper Sonnet give the same answers as Opus, and what would it cost on the 1,000-image set?

## Runs covered

| Run | Answerer | Settings |
|---|---|---|
| `20261104-221846_sonnet-test` | Claude Sonnet 5.5 through `claude -p`, effort low (`ANS_claude_sonnet_cli`) | `D0_poc1` (1 image, ISIC_0000147, melanoma); `Q0_all` (500 questions + 500 negations); `M_topic` (one prompt per topic; questions and negations in separate prompts); `G1` (whole image); `R1`; prompt `claude_batched_whole_v1` (b23c348132ac), the same as Opus |
| `20261030-002157`, `-003848`, `-004312`, `-004738` (`_claude-cli-poc`) | Claude Opus 5.5 through `claude -p`, effort low (`ANS_claude_opus_cli`) | the same settings, replicates `R1`–`R4`, run on 2 images; only ISIC_0000147 is used here |

**Validity check:** Sonnet answered all 82 prompts (1,000 items) completely on the first attempt.

## Results

**How to read the comparison.** Opus does not give exactly the same answers twice, so the ceiling is how well one Opus
replicate agrees with the mean of the other three. Sonnet is compared with each Opus replicate and with the Opus mean.

| Measure (1,000 items, ISIC_0000147) | Opus vs itself (ceiling) | Sonnet vs one Opus replicate | Sonnet vs Opus mean |
|---|---|---|---|
| Rank agreement, P(Y\|Qo) (Spearman ρ) | 0.98 | 0.93 | 0.93 |
| Rank agreement, Net_P | 0.99 | 0.92 | 0.92 |
| Rank agreement, Directional_Score | 0.99 | 0.92 | 0.93 |
| Rank agreement, Coherence_Score | 0.99 | 0.92 | 0.92 |
| Rank agreement, stated confidence (Qo) | 0.96 | 0.82 | 0.84 |
| Same yes/no answer to the question | 98% | 94% | 94% |
| Same verdict (yes / no / contradiction) | 96% | 89% | 88% |
| Same three-way answer, Directional_Score | 97% | 86% | 86% |
| Same three-way answer, Coherence_Score | 96% | 90% | 91% |
| Mean change in Net_P | 0.04 | 0.13 | 0.13 |

The three-way answer is yes (score ≥ 0.1), no (≤ −0.1) or no clear answer.

- **Sonnet is close to Opus but not interchangeable.** It ranks the questions almost the same way (ρ = 0.92–0.93) and
  gives the same yes/no answer to 94% of questions. That gap is about 3–4 times Opus's own replicate noise (mean change
  in Net_P 0.13 against 0.04).
- **Sonnet leans towards "no":**
  - It answered yes to 13.8% of questions, against 16.8–17.8% for Opus.
  - Where the two differ by more than 0.3 in Net_P, Sonnet is lower twice as often (44 items) as higher (22).
  - It contradicted itself on 6.6% of pairs (Opus: 3.6–6.4%).
  - Its stated confidence was slightly lower (0.65 against 0.70).
- **Where they disagree most** ([sonnet_vs_opus.csv](sonnet_vs_opus.csv)):
  - Many of the largest gaps are about the image rather than the lesion: "too large for the aperture", "cross-polarized
    lighting", "strong edges outside the lesion", "hair present".
  - Others are borderline clinical calls: "pigmented nodule", "pigment network absent", "atypical naevus ≥ 5 mm".

## Cost

| | Per prompt (list price) | This test |
|---|---|---|
| Sonnet, topic | **$0.0133** (1 run) | 82 prompts, $1.09; 5-hour usage 2% → 3%, weekly 1% |
| Opus, topic | $0.0150–0.0299, median $0.0221 (4 runs) | – |

On this run Sonnet cost about **60% of Opus's median per prompt**. Opus's cost varied 2× between identical runs
(see [token usage](../2026-11-04_token_usage/README.md)), so a single Sonnet run gives only a rough ratio.

**Projection: 500 questions, no negations, 1,000 images.** The Sonnet range applies Opus's run-to-run spread to
Sonnet's measured price. Stateless assumes the same Sonnet/Opus price ratio as topic.

| Answerer, mode | Prompts | List price (low / point / high) | On the Pro subscription, if usage scales with list price |
|---|---|---|---|
| **Sonnet, topic** | 41,000 | **$370 / $545 / $740** | about 0.5–1 week |
| Opus, topic | 41,000 | $615 / $906 / $1,226 | about 0.8–1.6 weeks |
| Sonnet, stateless (assumed ratio) | 500,000 | about $2,650 / $4,550 / $6,550 | about 3.5–9 weeks |
| Opus, stateless | 500,000 | $4,400 / $7,550 / $10,900 | about 6–15 weeks |

Subscription limits may not weigh Sonnet and Opus by their list prices. This test moved the 5-hour figure by only one
percentage point, too little to measure the rate; a 10-image Sonnet run would.

## Takeaway

Sonnet would cut the cost by roughly 40% and agrees with Opus on about 94% of yes/no answers. It is not a drop-in
replacement, because it says "no" more often and contradicts itself somewhat more. Whether that gap matters is a
classification question. Two options:
- **Run both on a small labelled set**, for example the 10 smoke images, and compare W0 performance.
- **Use Sonnet only for scale-up runs** where Opus's extra accuracy is less important.

## Files

- [agreement.csv](agreement.csv): every comparison: each Opus replicate against the others, and Sonnet against each
  replicate and the mean.
- [sonnet_vs_opus.csv](sonnet_vs_opus.csv): per question: the Opus mean and Sonnet's values, with both verdicts.
- [numbers.json](numbers.json) and [compare_sonnet.py](compare_sonnet.py)
  (`poetry run python reports/2026-11-04_sonnet_test/compare_sonnet.py 20261104-221846_sonnet-test`).
