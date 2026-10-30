# Answer-mode comparison: qwen2.5-vl-7b/stateless, qwen2.5-vl-7b/topic, qwen2.5-vl-7b/all, skingpt-r1/stateless, skingpt-r1/topic, skingpt-r1/all

Runs: `20261029-152402_answerers-oct29`, `20261029-152411_answerers-oct29`, `20261029-152420_answerers-oct29`, `20261029-152430_answerers-oct29`, `20261029-152656_answerers-oct29`, `20261029-153429_answerers-oct29`. Reference for agreement columns: `qwen2.5-vl-7b/stateless`.

| run                                                       | mode      |   prompts | shuffle   |     n |   sec_per_query |   prefix_ms |   yesno_mass_min |   frac_yes |   pearson_vs_ref |   bin_agree_vs_ref |   mean_abs_dp_vs_ref |   contradiction_rate |   mean_abs_p_adj |   mean_p_w |   tokens_in |   tokens_out | p_yes_first_vs_last_pos   |
|:----------------------------------------------------------|:----------|----------:|:----------|------:|----------------:|------------:|-----------------:|-----------:|-----------------:|-------------------:|---------------------:|---------------------:|-----------------:|-----------:|------------:|-------------:|:--------------------------|
| qwen2.5-vl-7b/stateless (20261029-152402_answerers-oct29) | stateless |         0 | False     | 10000 |          0.0098 |     338.75  |           0.9923 |     0.1323 |           1      |             1      |               0      |               0.7886 |           0.2125 |     0.0824 |      247694 |        10000 | nan                       |
| qwen2.5-vl-7b/topic (20261029-152411_answerers-oct29)     | topic     |        82 | False     | 10000 |          0.0428 |     358.032 |           0.9366 |     0.0394 |           0.5025 |             0.8819 |               0.158  |               0.9324 |           0.0982 |     0.0284 |      184544 |        60520 | 0.179 / 0.004             |
| qwen2.5-vl-7b/all (20261029-152420_answerers-oct29)       | all       |         2 | False     | 10000 |          0.0711 |     357.505 |           0.9987 |     0.003  |           0.1588 |             0.8693 |               0.2137 |               0.994  |           0.0098 |     0.0044 |      176164 |        77740 | 0.284 / 0.002             |
| skingpt-r1/stateless (20261029-152430_answerers-oct29)    | stateless |         0 | False     | 10000 |          0.0123 |     331.151 |           0.664  |     0.3848 |           0.6037 |             0.7157 |               0.2618 |               0.4712 |           0.504  |     0.202  |      367694 |        10000 | nan                       |
| skingpt-r1/topic (20261029-152656_answerers-oct29)        | topic     |        82 | False     | 10000 |          0.0426 |     354.413 |           0.2986 |     0.2746 |           0.3113 |             0.7275 |               0.2774 |               0.6096 |           0.3851 |     0.0809 |      191924 |        60520 | 0.307 / 0.300             |
| skingpt-r1/all (20261029-153429_answerers-oct29)          | all       |         2 | False     | 10000 |          0.0713 |     357.928 |           0.9979 |     0.1565 |           0.154  |             0.7782 |               0.2736 |               0.7042 |           0.2912 |     0.088  |      176344 |        77740 | 0.511 / 0.403             |

- [compare_answers.csv](compare_answers.csv): one row per image x question; P(yes) to the original and complement, p_adj, P_w, stated confidence (Claude) and verdict for each run side by side. Averaged replicate groups (`/meanN`) use the mean of each value; the table's timing and token columns describe their first replicate.
- [compare_by_question.csv](compare_by_question.csv): per question, mean p_adj and contradiction rate per run.

Share of verdicts (whole image, all images x questions):

|                            |   qwen2.5-vl-7b/stateless |   qwen2.5-vl-7b/topic |   qwen2.5-vl-7b/all |   skingpt-r1/stateless |   skingpt-r1/topic |   skingpt-r1/all |
|:---------------------------|--------------------------:|----------------------:|--------------------:|-----------------------:|-------------------:|-----------------:|
| contradiction: no to both  |                     0.762 |                 0.927 |               0.994 |                  0.351 |              0.53  |            0.696 |
| yes                        |                     0.137 |                 0.046 |               0.006 |                  0.224 |              0.09  |            0.104 |
| no                         |                     0.074 |                 0.022 |               0     |                  0.305 |              0.301 |            0.192 |
| contradiction: yes to both |                     0.027 |                 0.006 |               0     |                  0.12  |              0.079 |            0.009 |
