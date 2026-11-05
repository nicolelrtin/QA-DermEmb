# Experiment reports

One folder per experiment batch, newest first. Each folder has a README with the purpose, the runs, the results and
links to the data. Every individual run also writes `outputs/qaemb/runs/<run_id>/report.md`
(indexed in [experiments_log.md](experiments_log.md)).

| Date | Report | Summary |
|---|---|---|
| 2026-11-04 | [Token usage and 1,000-image projection](2026-11-04_token_usage/README.md) | 35 answer stages to date: Claude 19.9 M input / 2.9 M output tokens, $152 list price (on the subscription); 500 questions without negations on 1,000 images: Claude topic about $615–1,226 or 0.8–1.6 weeks of Pro, stateless $4,400–10,900; local models under 2 GPU-hours stateless |
| 2026-10-30 | [Claude proof of concept: 2 images, 3 answerers](2026-10-30_claude_poc/README.md); [letter with Figs 1–5](2026-10-30_claude_poc/letter.md) | Claude contradicts itself on 3–4% of question pairs (Qwen 78–92%, SkinGPT-R1 50–54%); 92–95% identical verdicts over 4 replicates; topic mode matches stateless (r = 0.92) at ~1/9 of the cost |
| 2026-10-29 | [Answerer comparison (in progress)](2026-10-29_answerers/README.md) | SkinGPT-R1 answers question pairs far more coherently than Qwen (47% vs 79% same answer to Q and not-Q); Claude pending credentials |
| 2026-10-28 | [Pipeline presentation notes](2026-10-28_pipeline_presentation/pipeline_details.md) | Detailed companion to the lab-meeting deck (https://claude.ai/artifact/VZybPKN9MagXsMHbRP83QQ) |
| 2026-10-27 | [Smoke test: 10 images, 500 question pairs](2026-10-27_smoke10/README.md) | Stateless is the only usable mode; 79% of question pairs get the same answer to Q and not-Q; ~2.7 h for the full beta set |
