# Replicates: claude-opus-cli/stateless (4 runs)

## Runs covered

| run                              | label                        | experiments                                                                                                                                                                                                                                                                                                                                                                                                                                                        | prompt                                     |   images |
|:---------------------------------|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|---------:|
| `20261030-005209_claude-cli-poc` | claude-opus-cli/stateless/R1 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_stateless`: one question per prompt<br>`G1`: whole image only<br>`R1`: replicate 1 (Claude only; local models are deterministic) | `claude_stateless_whole_v1` (5ab94b38280f) |        2 |
| `20261030-013448_claude-cli-poc` | claude-opus-cli/stateless/R2 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_stateless`: one question per prompt<br>`G1`: whole image only<br>`R2`: replicate 2                                               | `claude_stateless_whole_v1` (5ab94b38280f) |        2 |
| `20261030-051502_claude-cli-poc` | claude-opus-cli/stateless/R3 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_stateless`: one question per prompt<br>`G1`: whole image only<br>`R3`: replicate 3                                               | `claude_stateless_whole_v1` (5ab94b38280f) |        2 |
| `20261030-052021_claude-cli-poc` | claude-opus-cli/stateless/R4 | `D0_poc2`: 2 smoke images, ISIC_0000147 and ISIC_0000215 (Claude proof of concept)<br>`Q0_all`: all 500 questions, each asked as the original and as its negated complement (1,000 items)<br>`ANS_claude_opus_cli`: Claude Opus 5.5 through claude -p on the Claude subscription (effort low); stated P(yes) and confidence<br>`M_stateless`: one question per prompt<br>`G1`: whole image only<br>`R4`: replicate 4                                               | `claude_stateless_whole_v1` (5ab94b38280f) |        2 |

## Summary

| value                 |    mean |   mean SD across replicates |
|:----------------------|--------:|----------------------------:|
| p_yes_original        |  0.2532 |                      0.0215 |
| p_yes_complement      |  0.6827 |                      0.0267 |
| p_adj                 | -0.4295 |                      0.0367 |
| p_w                   |  0.132  |                      0.0154 |
| confidence_original   |  0.7196 |                      0.0308 |
| confidence_complement |  0.6914 |                      0.0344 |

- Image × question pairs with the same verdict in every replicate: **92.1%**
- Mean pairwise Pearson r of p_adj between replicates: **0.991**

- [replicates_raw.csv](replicates_raw.csv): every replicate's values, one row per replicate x image x question.
- [replicates_summary.csv](replicates_summary.csv): mean and SD per image x question, modal verdict and its share.
