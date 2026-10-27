# Experiments log

| run_id | experiments | stage hashes | key metric | notes |
|---|---|---|---|---|
| 20261023-203004_smoke1 | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_stateless,G1 | man:820e1f pre:b9ddf6 ans:e5f540 | s/q=0.0153 | smoke1 |
| 20261023-203212_smoke1 | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_stateless,G1,F_prob,W0 | man:820e1f pre:b9ddf6 ans:e5f540 emb:bf4e7a mod:c3c593 | loo_AUROC=0.667 | smoke1 |
| 20261023-203222_smoke1 | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_stateless,G1,F_bin,W1 | man:820e1f pre:b9ddf6 ans:e5f540 emb:bf4e7a mod:b5a40f | loo_AUROC=0.000 | smoke1 |
| 20261023-203243_smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1 | man:820e1f pre:b9ddf6 ans:c9cb0b | s/q=0.0097 | smoke2 |
| 20261023-203351_smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_batch10,G1 | man:820e1f pre:b9ddf6 ans:7ce942 | s/q=0.0368 | smoke2 |
| 20261023-203715_smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_batch10_shuf,G1 | man:820e1f pre:b9ddf6 ans:174906 | s/q=0.0427 | smoke2 |
| 20261023-204105_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P0 | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:09d25a | loo_AUROC=0.600 | smoke3 |
| 20261023-204410_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P1_max | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:c2035d | loo_AUROC=0.667 | smoke3 |
| 20261023-204423_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P1_topk | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:5b22be | loo_AUROC=0.600 | smoke3 |
| 20261023-204436_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P1_noisyor | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:3d5170 | loo_AUROC=0.600 | smoke3 |
| 20261023-204448_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P2 | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:915032 | loo_AUROC=0.333 | smoke3 |
| 20261023-204503_smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_prob,P3 | man:820e1f pre:b38181 ans:2805dc emb:8590ea mod:5e5495 | loo_AUROC=0.333 | smoke3 |
| 20261027-114946_neg-smoke | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_stateless,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:8f8438 emb:c4cedb mod:ec864c | loo_AUROC=0.667 | neg-smoke |
| 20261027-115018_neg-smoke | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_topic,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:957972 emb:dce663 mod:e259e9 | loo_AUROC=0.467 | neg-smoke |
| 20261027-115135_neg-smoke | D0_smoke10,Q9_smoke50,ANS_qwen7b,M_all,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:4d73be emb:4b162a mod:f1c35f | loo_AUROC=0.200 | neg-smoke |
| 20261027-115245_neg-smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:f8daf7 emb:ad4dd1 mod:18fb2b | loo_AUROC=0.400 | neg-smoke2 |
| 20261027-115445_neg-smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_topic,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:0b6794 emb:cd3a9e mod:04f3a8 | loo_AUROC=0.467 | neg-smoke2 |
| 20261027-120223_neg-smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_all,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:4b2dcc emb:4d1ce6 mod:30d142 | loo_AUROC=0.600 | neg-smoke2 |
| 20261027-121433_neg-smoke3 | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_adj,P0 | man:820e1f pre:b38181 ans:b133bd emb:d08e46 mod:29f98d | loo_AUROC=0.533 | neg-smoke3 |
| 20261027-121854_neg-smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1,F_adj,W1 | man:820e1f pre:b9ddf6 ans:f8daf7 emb:ad4dd1 mod:8f9b2d | loo_AUROC=0.000 | neg-smoke2 |
| 20261027-121924_neg-smoke2 | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1,F_orig,W0 | man:820e1f pre:b9ddf6 ans:f8daf7 emb:ad4dd1 mod:318a45 | loo_AUROC=0.400 | neg-smoke2 |
| [20261027-153217_smoke-oct27](outputs/qaemb/runs/20261027-153217_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:f8daf7 emb:21675a mod:bd5292 | loo_AUROC=0.400 | smoke-oct27 |
| [20261027-153226_smoke-oct27](outputs/qaemb/runs/20261027-153226_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_topic,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:0b6794 emb:ef71ae mod:a51066 | loo_AUROC=0.467 | smoke-oct27 |
| [20261027-153235_smoke-oct27](outputs/qaemb/runs/20261027-153235_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_all,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:4b2dcc emb:8eb1c3 mod:f9750b | loo_AUROC=0.600 | smoke-oct27 |
| [20261027-153243_smoke-oct27](outputs/qaemb/runs/20261027-153243_smoke-oct27/report.md) | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_adj,P0 | man:820e1f pre:b38181 ans:b133bd emb:16480b mod:c80d83 | loo_AUROC=0.533 | smoke-oct27 |
| [20261027-153315_smoke-oct27](outputs/qaemb/runs/20261027-153315_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_stateless,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:f8daf7 emb:47163b mod:ba50d4 | loo_AUROC=0.400 | smoke-oct27 |
| [20261027-153324_smoke-oct27](outputs/qaemb/runs/20261027-153324_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_topic,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:0b6794 emb:4a0425 mod:b5c6b3 | loo_AUROC=0.467 | smoke-oct27 |
| [20261027-153332_smoke-oct27](outputs/qaemb/runs/20261027-153332_smoke-oct27/report.md) | D0_smoke10,Q0_all,ANS_qwen7b,M_all,G1,F_adj,W0 | man:820e1f pre:b9ddf6 ans:4b2dcc emb:bc2eeb mod:cf07b9 | loo_AUROC=0.600 | smoke-oct27 |
| [20261027-153341_smoke-oct27](outputs/qaemb/runs/20261027-153341_smoke-oct27/report.md) | D0_smoke10,Q1_local,ANS_qwen7b,M_stateless,G2,F_adj,P0 | man:820e1f pre:b38181 ans:b133bd emb:5e0ec2 mod:4f9c45 | loo_AUROC=0.533 | smoke-oct27 |
