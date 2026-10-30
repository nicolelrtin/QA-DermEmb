# Replicates: claude-opus-cli/topic (4 runs)

## Runs covered

| run                              | label                    | experiments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | prompt                                   |   images |
|:---------------------------------|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|---------:|
| `20261030-002157_claude-cli-poc` | claude-opus-cli/topic/R1 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_topic`: one prompt per topic, listing that topic's questions; originals and complements in separate prompts<br>`G1`: whole image only<br>`R1`: replicate 1 (Claude only; local models are deterministic) | `claude_batched_whole_v1` (b23c348132ac) |        2 |
| `20261030-003848_claude-cli-poc` | claude-opus-cli/topic/R2 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_topic`: one prompt per topic, listing that topic's questions; originals and complements in separate prompts<br>`G1`: whole image only<br>`R2`: replicate 2                                               | `claude_batched_whole_v1` (b23c348132ac) |        2 |
| `20261030-004312_claude-cli-poc` | claude-opus-cli/topic/R3 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_topic`: one prompt per topic, listing that topic's questions; originals and complements in separate prompts<br>`G1`: whole image only<br>`R3`: replicate 3                                               | `claude_batched_whole_v1` (b23c348132ac) |        2 |
| `20261030-004738_claude-cli-poc` | claude-opus-cli/topic/R4 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_topic`: one prompt per topic, listing that topic's questions; originals and complements in separate prompts<br>`G1`: whole image only<br>`R4`: replicate 4                                               | `claude_batched_whole_v1` (b23c348132ac) |        2 |

## Summary

| value                 |    mean |   mean SD across replicates |
|:----------------------|--------:|----------------------------:|
| p_yes_original        |  0.1905 |                      0.0174 |
| p_yes_complement      |  0.7602 |                      0.0257 |
| p_adj                 | -0.5697 |                      0.0325 |
| p_w                   |  0.0983 |                      0.0132 |
| confidence_original   |  0.7359 |                      0.0335 |
| confidence_complement |  0.6978 |                      0.0371 |

- Image × question pairs with the same verdict in every replicate: **94.6%**
- Mean pairwise Pearson r of p_adj between replicates: **0.992**

- [replicates_raw.csv](replicates_raw.csv): every replicate's values, one row per replicate x image x question.
- [replicates_summary.csv](replicates_summary.csv): mean and SD per image x question, modal verdict and its share.
