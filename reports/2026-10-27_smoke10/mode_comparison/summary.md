# Answer-mode comparison: stateless, topic, all

Runs: `20261027-153315_smoke-oct27`, `20261027-153324_smoke-oct27`, `20261027-153332_smoke-oct27`. Reference for agreement columns: `stateless`.

| run                                     | mode      |   prompts | shuffle   |     n |   sec_per_query |   prefix_ms |   yesno_mass_min |   frac_yes |   pearson_vs_ref |   bin_agree_vs_ref |   mean_abs_dp_vs_ref |   contradiction_rate |   mean_abs_p_adj | p_yes_first_vs_last_pos   |
|:----------------------------------------|:----------|----------:|:----------|------:|----------------:|------------:|-----------------:|-----------:|-----------------:|-------------------:|---------------------:|---------------------:|-----------------:|:--------------------------|
| stateless (20261027-153315_smoke-oct27) | stateless |         0 | False     | 10000 |          0.0098 |     338.75  |           0.9923 |     0.1323 |           1      |             1      |               0      |               0.7886 |           0.2125 | nan                       |
| topic (20261027-153324_smoke-oct27)     | topic     |        82 | False     | 10000 |          0.0428 |     358.032 |           0.9366 |     0.0394 |           0.5025 |             0.8819 |               0.158  |               0.9324 |           0.0982 | 0.179 / 0.004             |
| all (20261027-153332_smoke-oct27)       | all       |         2 | False     | 10000 |          0.0711 |     357.505 |           0.9987 |     0.003  |           0.1588 |             0.8693 |               0.2137 |               0.994  |           0.0098 | 0.284 / 0.002             |

- [compare_answers.csv](compare_answers.csv): one row per image x question; P(yes) to the original and complement, p_adj and verdict for each run side by side.
- [compare_by_question.csv](compare_by_question.csv): per question, mean p_adj and contradiction rate per run.

Share of verdicts (whole image, all images x questions):

|                            |   stateless |   topic |   all |
|:---------------------------|------------:|--------:|------:|
| contradiction: no to both  |       0.762 |   0.927 | 0.994 |
| yes                        |       0.137 |   0.046 | 0.006 |
| no                         |       0.074 |   0.022 | 0     |
| contradiction: yes to both |       0.027 |   0.006 | 0     |
