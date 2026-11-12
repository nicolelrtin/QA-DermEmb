# Which questions the classifiers relied on

Main feature source: Directional Score (`dir`). Every table is also in the CSV files for P(yes), Directional Score, Coherence Score. Nothing is refitted here; the importance values are the S6 ones, measured on the test folds.

How to read the columns:
- **Consensus rank**: each of the six models ranks the questions by how much they moved its predictions on the test folds (mean |SHAP|); the consensus is the mean of the six ranks, re-ranked. 1 = most important.
- **Ridge weight**: the standardised ridge regression weight. For benign vs malignant, positive pushes towards malignant. Its size depends on the other questions in the model, so read it alongside the others.
- **Lasso kept**: the share of the 25 outer fits in which the lasso kept the question. Close to 100% means the question is reliably useful, not picked by chance in one split.
- **Mean P(yes) in <class>**: Sonnet's P(yes) averaged over the images of that class. Many questions are almost never answered yes; for those the difference lies in how doubtful the "no" is (say 0.05 against 0.2), which the share answered yes cannot show. That share is in consensus.csv (`yes_rate_*`).
- **Rarely yes**: fewer than 5% of the images in every class were answered yes.
- **AUROC alone** (2-class only): how well this one question's score separates malignant from benign by itself, with no model: 0.5 is no better than chance, above 0.5 means higher scores go with malignant, below 0.5 with benign. Computed once on all 800 images; it describes the answers, it is not a test score.
- **Towards**: the class the question pushes towards inside the model: for 2-class tasks the sign of the ridge weight; for multiclass tasks the class with the largest ridge weight. For 2-class tasks, a † marks questions whose direction inside the model is the opposite of their direction alone (AUROC alone). Such a question corrects for others it is correlated with, and should not be read on its own.

## Validity checks

These show whether the ranking is a property of the answers or an accident of one model or one split. If the models disagreed, or the lasso kept the top questions only in a few fits, the ranking would not be worth interpreting.

| task | median rank correlation between models | models' top 20 shared with the consensus top 20 (median) | consensus top 20 kept by the lasso in at least 80% of fits |
|---|---|---|---|
| s1_2c | 0.66 | 55% | 19 of 20 |
| s1_3c | 0.63 | 50% | 20 of 20 |
| flat_10c | 0.82 | 52% | 20 of 20 |
| s2_benign | 0.73 | 62% | 20 of 20 |
| s2_malignant | 0.57 | 57% | 16 of 20 |
| s2_indeterminate | 0.62 | 50% | 6 of 20 |

Rank correlation of 1 means the models order the questions identically; 0 means unrelated. Per model:

| model             |   flat_10c |   s1_2c |   s1_3c |   s2_benign |   s2_indeterminate |   s2_malignant |
|:------------------|-----------:|--------:|--------:|------------:|-------------------:|---------------:|
| MLP               |       0.82 |    0.66 |    0.63 |        0.73 |               0.62 |           0.58 |
| elastic net       |       0.86 |    0.74 |    0.67 |        0.86 |               0.74 |           0.58 |
| gradient boosting |       0.69 |    0.41 |    0.49 |        0.63 |               0.44 |           0.55 |
| lasso (L1)        |       0.82 |    0.66 |    0.63 |        0.73 |               0.62 |           0.5  |
| random forest     |       0.56 |    0.37 |    0.36 |        0.5  |               0.44 |           0.53 |
| ridge (L2)        |       0.84 |    0.79 |    0.75 |        0.75 |               0.66 |           0.75 |

Agreement between feature sources (rank correlation of the consensus ranks; share of the top 20 in common):

| task             | source_a   | source_b   |   rank_corr |   top20_overlap |
|:-----------------|:-----------|:-----------|------------:|----------------:|
| s1_2c            | orig       | dir        |        0.68 |            0.55 |
| s1_2c            | orig       | coh        |        0.69 |            0.65 |
| s1_2c            | dir        | coh        |        0.84 |            0.6  |
| s1_3c            | orig       | dir        |        0.73 |            0.55 |
| s1_3c            | orig       | coh        |        0.74 |            0.65 |
| s1_3c            | dir        | coh        |        0.89 |            0.65 |
| flat_10c         | orig       | dir        |        0.82 |            0.4  |
| flat_10c         | orig       | coh        |        0.85 |            0.7  |
| flat_10c         | dir        | coh        |        0.94 |            0.6  |
| s2_benign        | orig       | dir        |        0.78 |            0.6  |
| s2_benign        | orig       | coh        |        0.79 |            0.7  |
| s2_benign        | dir        | coh        |        0.92 |            0.7  |
| s2_malignant     | orig       | dir        |        0.77 |            0.5  |
| s2_malignant     | orig       | coh        |        0.75 |            0.75 |
| s2_malignant     | dir        | coh        |        0.9  |            0.7  |
| s2_indeterminate | orig       | dir        |        0.74 |            0.5  |
| s2_indeterminate | orig       | coh        |        0.69 |            0.55 |
| s2_indeterminate | dir        | coh        |        0.9  |            0.55 |

## s1_2c: top 20 questions (Directional Score)

|   rank | ID    | question                                                                      | scope   | towards   |   ridge weight | lasso kept   |   mean P(yes) benign |   mean P(yes) malignant |   AUROC alone | rarely yes   |
|-------:|:------|:------------------------------------------------------------------------------|:--------|:----------|---------------:|:-------------|---------------------:|------------------------:|--------------:|:-------------|
|      1 | DQ484 | Does the skin surface appear rough?                                           | local   | malignant |          0.227 | 100%         |                 0.32 |                    0.45 |          0.72 |              |
|      2 | DQ020 | Does the lesion show a pigmented homogeneous global pattern?                  | global  | benign    |         -0.166 | 100%         |                 0.25 |                    0.14 |          0.38 |              |
|      3 | DQ359 | Are signs of chronic sun damage visible around the lesions?                   | global  | malignant |          0.145 | 100%         |                 0.3  |                    0.46 |          0.7  |              |
|      4 | DQ399 | Is colour asymmetry within the lesion high?                                   | global  | malignant |          0.15  | 100%         |                 0.38 |                    0.45 |          0.65 |              |
|      5 | DQ479 | Does the lesion look bluish-purple, suggesting low blood oxygen saturation?   | global  | benign †  |         -0.217 | 100%         |                 0.1  |                    0.09 |          0.55 | yes          |
|      6 | DQ025 | Is pink colour present in a pigmented lesion?                                 | local   | malignant |          0.138 | 100%         |                 0.25 |                    0.33 |          0.64 |              |
|      7 | DQ173 | Are brown globules arranged in a patchy distribution?                         | global  | malignant |          0.14  | 96%          |                 0.07 |                    0.08 |          0.64 | yes          |
|      8 | DQ420 | Is the lesion clearly separable from the background for segmentation?         | global  | benign    |         -0.11  | 92%          |                 0.66 |                    0.56 |          0.38 |              |
|      9 | DQ499 | Does the lesion show a periodic, repeating texture?                           | local   | benign    |         -0.11  | 88%          |                 0.14 |                    0.1  |          0.41 | yes          |
|     10 | DQ074 | Is a combined brown and blue-grey structureless area present in the lesion?   | local   | malignant |          0.112 | 80%          |                 0.05 |                    0.08 |          0.6  | yes          |
|     11 | DQ454 | Is the lesion's darkest region located off-centre?                            | global  | malignant |          0.157 | 100%         |                 0.44 |                    0.5  |          0.59 |              |
|     12 | DQ383 | Is dyspigmentation visible?                                                   | global  | malignant |          0.146 | 100%         |                 0.54 |                    0.6  |          0.57 |              |
|     13 | DQ301 | Does the lesion resemble a vascular lesion?                                   | global  | benign †  |         -0.137 | 76%          |                 0.21 |                    0.14 |          0.53 |              |
|     14 | DQ054 | Are small pink clods (milky-red globules) present in the lesion?              | local   | benign †  |         -0.202 | 100%         |                 0.06 |                    0.06 |          0.62 | yes          |
|     15 | DQ191 | Are thin, curved, parallel brown lines present?                               | local   | benign    |         -0.18  | 100%         |                 0.08 |                    0.06 |          0.45 | yes          |
|     16 | DQ071 | Is a grey-white structureless zone present in the lesion?                     | local   | malignant |          0.106 | 84%          |                 0.05 |                    0.08 |          0.68 | yes          |
|     17 | DQ497 | Does lesion texture differ markedly between the red and blue colour channels? | global  | malignant |          0.14  | 88%          |                 0.47 |                    0.47 |          0.52 |              |
|     18 | DQ480 | Is epidermal pigmentation visibly heterogeneous across the normal skin?       | global  | malignant |          0.107 | 84%          |                 0.32 |                    0.4  |          0.62 |              |
|     19 | DQ220 | Is the individual typology angle (ITA) of the visible normal skin above 28°?  | global  | benign    |         -0.104 | 84%          |                 0.75 |                    0.71 |          0.39 |              |
|     20 | DQ033 | Are dotted vessels visible within a pigment network?                          | local   | malignant |          0.095 | 84%          |                 0.03 |                    0.03 |          0.67 | yes          |

8 of these 20 questions were rarely answered yes in any class. 3 point the other way inside the model than alone (†).

Figure: [topics_s1_2c.png](figures/topics_s1_2c.png); [top_s1_2c.png](figures/top_s1_2c.png)

## s1_3c: top 20 questions (Directional Score)

|   rank | ID    | question                                                                         | scope    | towards       |   ridge weight | lasso kept   |   mean P(yes) benign |   mean P(yes) indeterminate |   mean P(yes) malignant | rarely yes   |
|-------:|:------|:---------------------------------------------------------------------------------|:---------|:--------------|---------------:|:-------------|---------------------:|----------------------------:|------------------------:|:-------------|
|      1 | DQ020 | Does the lesion show a pigmented homogeneous global pattern?                     | global   | benign        |          0.087 | 100%         |                 0.25 |                        0.26 |                    0.14 |              |
|      2 | DQ359 | Are signs of chronic sun damage visible around the lesions?                      | global   | malignant     |          0.072 | 96%          |                 0.3  |                        0.46 |                    0.46 |              |
|      3 | DQ057 | Are follicular plugs present in the lesion?                                      | local    | indeterminate |          0.072 | 100%         |                 0.05 |                        0.07 |                    0.06 | yes          |
|      4 | DQ499 | Does the lesion show a periodic, repeating texture?                              | local    | benign        |          0.065 | 100%         |                 0.14 |                        0.13 |                    0.1  | yes          |
|      5 | DQ184 | Are clothing fibres or hairs adherent to the lesion surface (a clue to ulcerati… | local    | malignant     |          0.079 | 100%         |                 0.06 |                        0.04 |                    0.09 | yes          |
|      6 | DQ055 | Is a structureless pink zone (milky-red area) present in the lesion?             | local    | malignant     |          0.081 | 96%          |                 0.13 |                        0.11 |                    0.19 | yes          |
|      7 | DQ304 | Is a vascular blush (pink veil) present?                                         | local    | malignant     |          0.066 | 80%          |                 0.16 |                        0.15 |                    0.24 |              |
|      8 | DQ301 | Does the lesion resemble a vascular lesion?                                      | global   | benign        |          0.075 | 92%          |                 0.21 |                        0.08 |                    0.14 |              |
|      9 | DQ173 | Are brown globules arranged in a patchy distribution?                            | global   | malignant     |          0.081 | 96%          |                 0.07 |                        0.08 |                    0.08 | yes          |
|     10 | DQ151 | Is a pigment network absent in a pigmented lesion?                               | global   | benign        |          0.102 | 100%         |                 0.33 |                        0.34 |                    0.31 |              |
|     11 | DQ484 | Does the skin surface appear rough?                                              | local    | malignant     |          0.096 | 100%         |                 0.32 |                        0.39 |                    0.45 |              |
|     12 | DQ046 | Are grey or blue dots arranged in a peripheral clustered or linear pattern?      | global   | malignant     |          0.068 | 84%          |                 0.04 |                        0.05 |                    0.06 | yes          |
|     13 | DQ333 | Is a large congenital melanocytic naevus visible?                                | metadata | benign        |          0.063 | 100%         |                 0.08 |                        0.07 |                    0.07 | yes          |
|     14 | DQ054 | Are small pink clods (milky-red globules) present in the lesion?                 | local    | benign        |          0.095 | 100%         |                 0.06 |                        0.05 |                    0.06 | yes          |
|     15 | DQ479 | Does the lesion look bluish-purple, suggesting low blood oxygen saturation?      | global   | benign        |          0.1   | 100%         |                 0.1  |                        0.06 |                    0.09 | yes          |
|     16 | DQ188 | Are follicular plugs present in a flat, scaly lesion on sun-damaged skin?        | local    | indeterminate |          0.066 | 92%          |                 0.04 |                        0.07 |                    0.06 | yes          |
|     17 | DQ497 | Does lesion texture differ markedly between the red and blue colour channels?    | global   | malignant     |          0.07  | 80%          |                 0.47 |                        0.44 |                    0.47 |              |
|     18 | DQ031 | Is grey colour present in the lesion?                                            | local    | malignant     |          0.053 | 80%          |                 0.08 |                        0.07 |                    0.12 |              |
|     19 | DQ126 | Is the lesion annular?                                                           | global   | malignant     |          0.062 | 96%          |                 0.06 |                        0.05 |                    0.07 | yes          |
|     20 | DQ421 | Is the melanocytic lesion flat and subtle, without obvious melanoma features?    | global   | benign        |          0.057 | 92%          |                 0.32 |                        0.38 |                    0.19 |              |

11 of these 20 questions were rarely answered yes in any class.

## flat_10c: top 20 questions (Directional Score)

|   rank | ID    | question                                                                         | scope   | towards                      |   ridge weight | lasso kept   | rarely yes   |
|-------:|:------|:---------------------------------------------------------------------------------|:--------|:-----------------------------|---------------:|:-------------|:-------------|
|      1 | DQ043 | Are white dots or clods clustered or disseminated across the lesion?             | local   | indeterminate epidermal      |          0.1   | 100%         | yes          |
|      2 | DQ363 | Is the skin immediately surrounding the lesion erythematous?                     | global  | malignant epidermal          |          0.104 | 100%         |              |
|      3 | DQ472 | Does the lesion look like a vascular neoplasm (uniform red to purple colour)?    | global  | benign vascular              |          0.135 | 100%         |              |
|      4 | DQ193 | Does the lesion have a scalloped border?                                         | global  | indeterminate melanocytic    |          0.085 | 100%         |              |
|      5 | DQ454 | Is the lesion's darkest region located off-centre?                               | global  | malignant melanocytic        |          0.16  | 100%         |              |
|      6 | DQ301 | Does the lesion resemble a vascular lesion?                                      | global  | benign vascular              |          0.128 | 100%         |              |
|      7 | DQ471 | Is the lesion redder than the surrounding skin, suggesting higher haemoglobin c… | global  | benign melanocytic           |          0.067 | 100%         |              |
|      8 | DQ190 | Does the lesion have a sharply demarcated border?                                | global  | benign vascular              |          0.076 | 100%         |              |
|      9 | DQ484 | Does the skin surface appear rough?                                              | local   | malignant epidermal          |          0.085 | 100%         |              |
|     10 | DQ314 | Are red dots and globules present?                                               | local   | benign vascular              |          0.088 | 100%         |              |
|     11 | DQ386 | Is citrine (yellowish, thickened) skin visible?                                  | global  | flat melanotic pigmentation  |          0.061 | 100%         | yes          |
|     12 | DQ383 | Is dyspigmentation visible?                                                      | global  | malignant melanocytic        |          0.072 | 92%          |              |
|     13 | DQ497 | Does lesion texture differ markedly between the red and blue colour channels?    | global  | malignant epidermal          |          0.107 | 100%         |              |
|     14 | DQ489 | Is facial skin texture uneven?                                                   | global  | flat melanotic pigmentation  |          0.066 | 100%         |              |
|     15 | DQ499 | Does the lesion show a periodic, repeating texture?                              | local   | flat melanotic pigmentation  |          0.109 | 100%         |              |
|     16 | DQ480 | Is epidermal pigmentation visibly heterogeneous across the normal skin?          | global  | malignant adnexal follicular |          0.083 | 100%         |              |
|     17 | DQ321 | Are only slightly blurred pink vessels of the normal dermal plexus visible?      | global  | malignant adnexal follicular |          0.068 | 100%         |              |
|     18 | DQ397 | Is the blue-yellow (b*) contrast between lesion and surrounding skin low?        | global  | indeterminate epidermal      |          0.075 | 100%         |              |
|     19 | DQ376 | Do the naevi in the image show an eclipse pattern (tan centre with a brown rim)? | global  | benign fibro-histiocytic     |          0.09  | 100%         | yes          |
|     20 | DQ304 | Is a vascular blush (pink veil) present?                                         | local   | malignant adnexal follicular |          0.069 | 100%         |              |

3 of these 20 questions were rarely answered yes in any class.

Figure: [topics_flat_10c.png](figures/topics_flat_10c.png)

## s2_benign: top 20 questions (Directional Score)

|   rank | ID    | question                                                                      | scope    | towards                     |   ridge weight | lasso kept   | rarely yes   |
|-------:|:------|:------------------------------------------------------------------------------|:---------|:----------------------------|---------------:|:-------------|:-------------|
|      1 | DQ472 | Does the lesion look like a vascular neoplasm (uniform red to purple colour)? | global   | benign vascular             |          0.121 | 100%         |              |
|      2 | DQ480 | Is epidermal pigmentation visibly heterogeneous across the normal skin?       | global   | flat melanotic pigmentation |          0.102 | 100%         |              |
|      3 | DQ385 | Is melasma visible?                                                           | global   | flat melanotic pigmentation |          0.091 | 100%         | yes          |
|      4 | DQ454 | Is the lesion's darkest region located off-centre?                            | global   | benign fibro-histiocytic    |          0.192 | 100%         |              |
|      5 | DQ301 | Does the lesion resemble a vascular lesion?                                   | global   | benign vascular             |          0.115 | 100%         |              |
|      6 | DQ489 | Is facial skin texture uneven?                                                | global   | flat melanotic pigmentation |          0.095 | 100%         |              |
|      7 | DQ043 | Are white dots or clods clustered or disseminated across the lesion?          | local    | benign fibro-histiocytic    |          0.139 | 100%         | yes          |
|      8 | DQ386 | Is citrine (yellowish, thickened) skin visible?                               | global   | flat melanotic pigmentation |          0.08  | 100%         | yes          |
|      9 | DQ080 | Are white structureless areas present (a correlate of dermal fibrosis)?       | local    | benign fibro-histiocytic    |          0.098 | 100%         |              |
|     10 | DQ388 | Are pseudostellate scars visible?                                             | local    | benign fibro-histiocytic    |          0.088 | 100%         | yes          |
|     11 | DQ304 | Is a vascular blush (pink veil) present?                                      | local    | benign fibro-histiocytic    |          0.068 | 100%         | yes          |
|     12 | DQ190 | Does the lesion have a sharply demarcated border?                             | global   | benign vascular             |          0.086 | 92%          |              |
|     13 | DQ390 | Are solar lentigines visible?                                                 | local    | flat melanotic pigmentation |          0.079 | 100%         |              |
|     14 | DQ316 | Are punctate haemorrhages (haemorrhagic dots) present?                        | local    | benign vascular             |          0.09  | 100%         |              |
|     15 | DQ357 | Are scaly lesions present on the scalp?                                       | metadata | flat melanotic pigmentation |          0.079 | 100%         | yes          |
|     16 | DQ474 | Does the lesion contain both brown (melanin-like) and red (blood-like) areas? | global   | benign melanocytic          |          0.079 | 100%         |              |
|     17 | DQ363 | Is the skin immediately surrounding the lesion erythematous?                  | global   | inflammatory or infectious  |          0.072 | 100%         |              |
|     18 | DQ424 | Is the background skin texture prominent?                                     | global   | inflammatory or infectious  |          0.146 | 100%         |              |
|     19 | DQ097 | Is the depigmented lesion free of scale?                                      | global   | benign melanocytic          |          0.097 | 100%         | yes          |
|     20 | DQ101 | Is the lesion verrucous (wart-like)?                                          | local    | inflammatory or infectious  |          0.08  | 100%         | yes          |

8 of these 20 questions were rarely answered yes in any class.

## s2_malignant: top 20 questions (Directional Score)

|   rank | ID    | question                                                                         | scope    | towards                      |   ridge weight | lasso kept   |   mean P(yes) malignant adnexal follicular |   mean P(yes) malignant epidermal |   mean P(yes) malignant melanocytic | rarely yes   |
|-------:|:------|:---------------------------------------------------------------------------------|:---------|:-----------------------------|---------------:|:-------------|-------------------------------------------:|----------------------------------:|------------------------------------:|:-------------|
|      1 | DQ193 | Does the lesion have a scalloped border?                                         | global   | malignant melanocytic        |          0.055 | 100%         |                                       0.07 |                              0.1  |                                0.21 | yes          |
|      2 | DQ400 | Is the lesion's minor-axis diameter greater than 6 mm?                           | metadata | malignant epidermal          |          0.067 | 100%         |                                       0.52 |                              0.68 |                                0.69 |              |
|      3 | DQ160 | Is pink coloration sparse or absent in a lesion with BCC-like structures?        | global   | malignant adnexal follicular |          0.063 | 96%          |                                       0.15 |                              0.12 |                                0.11 | yes          |
|      4 | DQ067 | Are curved vessels present in the lesion?                                        | local    | malignant adnexal follicular |          0.055 | 100%         |                                       0.11 |                              0.08 |                                0.05 | yes          |
|      5 | DQ482 | Does a thick keratin layer make the lesion surface look white or opaque?         | local    | malignant epidermal          |          0.036 | 84%          |                                       0.15 |                              0.29 |                                0.14 |              |
|      6 | DQ397 | Is the blue-yellow (b*) contrast between lesion and surrounding skin low?        | global   | malignant epidermal          |          0.052 | 100%         |                                       0.49 |                              0.51 |                                0.27 |              |
|      7 | DQ306 | Are dotted vessels present?                                                      | local    | malignant epidermal          |          0.049 | 100%         |                                       0.13 |                              0.25 |                                0.12 |              |
|      8 | DQ222 | Does the visible skin have a yellow (rather than red) undertone?                 | global   | malignant adnexal follicular |          0.059 | 100%         |                                       0.31 |                              0.23 |                                0.3  |              |
|      9 | DQ363 | Is the skin immediately surrounding the lesion erythematous?                     | global   | malignant epidermal          |          0.048 | 84%          |                                       0.56 |                              0.72 |                                0.35 |              |
|     10 | DQ472 | Does the lesion look like a vascular neoplasm (uniform red to purple colour)?    | global   | malignant adnexal follicular |          0.058 | 100%         |                                       0.12 |                              0.07 |                                0.05 | yes          |
|     11 | DQ204 | Is a collarette of scale present around the lesion?                              | global   | malignant epidermal          |          0.038 | 60%          |                                       0.09 |                              0.17 |                                0.06 | yes          |
|     12 | DQ101 | Is the lesion verrucous (wart-like)?                                             | local    | malignant epidermal          |          0.035 | 88%          |                                       0.04 |                              0.06 |                                0.04 | yes          |
|     13 | DQ082 | Are blue-grey or black ovoid nests of varying size present (correlates of derma… | local    | malignant adnexal follicular |          0.049 | 92%          |                                       0.09 |                              0.05 |                                0.08 | yes          |
|     14 | DQ432 | Does the lesion border contain notches or indentations?                          | global   | malignant epidermal          |          0.05  | 88%          |                                       0.35 |                              0.41 |                                0.57 |              |
|     15 | DQ370 | Does the image show any naevus of at least 5 mm?                                 | metadata | malignant melanocytic        |          0.031 | 32%          |                                       0.12 |                              0.09 |                                0.45 |              |
|     16 | DQ499 | Does the lesion show a periodic, repeating texture?                              | local    | malignant epidermal          |          0.055 | 84%          |                                       0.08 |                              0.11 |                                0.12 | yes          |
|     17 | DQ293 | Is a small elevated lesion uniformly black?                                      | global   | malignant adnexal follicular |          0.051 | 84%          |                                       0.02 |                              0.02 |                                0.02 | yes          |
|     18 | DQ079 | Are blue-grey dots present (a correlate of pigment incontinence)?                | local    | malignant adnexal follicular |          0.038 | 76%          |                                       0.16 |                              0.1  |                                0.24 |              |
|     19 | DQ198 | Is a white network (brown holes intersected by white lines) present?             | local    | malignant epidermal          |          0.043 | 76%          |                                       0.05 |                              0.06 |                                0.07 | yes          |
|     20 | DQ361 | Are multiple lesions from the same patient visible in the image?                 | global   | malignant melanocytic        |          0.076 | 100%         |                                       0.09 |                              0.11 |                                0.14 |              |

10 of these 20 questions were rarely answered yes in any class.

## s2_indeterminate: top 20 questions (Directional Score)

|   rank | ID    | question                                                                         | scope    | towards                   |   ridge weight | lasso kept   |   mean P(yes) indeterminate epidermal |   mean P(yes) indeterminate melanocytic |   AUROC alone | rarely yes   |
|-------:|:------|:---------------------------------------------------------------------------------|:---------|:--------------------------|---------------:|:-------------|--------------------------------------:|----------------------------------------:|--------------:|:-------------|
|      1 | DQ363 | Is the skin immediately surrounding the lesion erythematous?                     | global   | indeterminate epidermal   |         -0.078 | 100%         |                                  0.58 |                                    0.19 |          0.08 |              |
|      2 | DQ024 | Is dark brown or black colour present in the lesion?                             | local    | indeterminate melanocytic |          0.076 | 100%         |                                  0.16 |                                    0.43 |          0.84 |              |
|      3 | DQ176 | Are whitish structureless areas present?                                         | local    | indeterminate epidermal   |         -0.057 | 48%          |                                  0.26 |                                    0.1  |          0.11 |              |
|      4 | DQ373 | Is there a single darker or larger 'ugly duckling' lesion among the naevi visib… | global   | indeterminate melanocytic |          0.083 | 100%         |                                  0.09 |                                    0.25 |          0.86 |              |
|      5 | DQ154 | Does the lesion appear scar-like?                                                | global   | indeterminate epidermal   |         -0.062 | 72%          |                                  0.1  |                                    0.05 |          0.1  | yes          |
|      6 | DQ043 | Are white dots or clods clustered or disseminated across the lesion?             | local    | indeterminate epidermal   |         -0.077 | 100%         |                                  0.16 |                                    0.09 |          0.25 | yes          |
|      7 | DQ397 | Is the blue-yellow (b*) contrast between lesion and surrounding skin low?        | global   | indeterminate epidermal   |         -0.057 | 84%          |                                  0.57 |                                    0.29 |          0.12 |              |
|      8 | DQ155 | Does the lesion appear as an erythematous plaque or patch?                       | global   | indeterminate epidermal   |         -0.042 | 8%           |                                  0.35 |                                    0.08 |          0.09 |              |
|      9 | DQ417 | Is the lesion hypopigmented relative to the surrounding skin?                    | global   | indeterminate epidermal   |         -0.06  | 64%          |                                  0.16 |                                    0.04 |          0.1  | yes          |
|     10 | DQ371 | Are most naevi in the image located on sun-exposed skin?                         | metadata | indeterminate melanocytic |          0.053 | 40%          |                                  0.21 |                                    0.28 |          0.73 |              |
|     11 | DQ333 | Is a large congenital melanocytic naevus visible?                                | metadata | indeterminate epidermal † |         -0.066 | 64%          |                                  0.08 |                                    0.06 |          0.58 | yes          |
|     12 | DQ353 | Are scaly plaques present on sun-exposed skin?                                   | local    | indeterminate epidermal   |         -0.042 | 40%          |                                  0.34 |                                    0.12 |          0.1  |              |
|     13 | DQ083 | Do milky-red areas and blurred vessels suggest deep tumoral vessels?             | local    | indeterminate epidermal   |         -0.041 | 4%           |                                  0.17 |                                    0.07 |          0.12 | yes          |
|     14 | DQ365 | Is at least one clinically atypical naevus visible in the image?                 | global   | indeterminate melanocytic |          0.038 | 8%           |                                  0.15 |                                    0.36 |          0.86 |              |
|     15 | DQ103 | Is lichenification (thickened skin with accentuated markings) present?           | local    | indeterminate epidermal   |         -0.046 | 48%          |                                  0.06 |                                    0.03 |          0.13 | yes          |
|     16 | DQ332 | Is there a soft, diffuse overgrowth within a congenital naevus?                  | metadata | indeterminate epidermal   |         -0.065 | 76%          |                                  0.04 |                                    0.03 |          0.46 | yes          |
|     17 | DQ193 | Does the lesion have a scalloped border?                                         | global   | indeterminate melanocytic |          0.058 | 84%          |                                  0.1  |                                    0.22 |          0.83 |              |
|     18 | DQ453 | Is the lesion's appearance dominated by one compact region?                      | global   | indeterminate melanocytic |          0.061 | 24%          |                                  0.4  |                                    0.58 |          0.73 |              |
|     19 | DQ108 | Are white striae arranged in an annular or circular pattern?                     | global   | indeterminate epidermal   |         -0.05  | 16%          |                                  0.06 |                                    0.03 |          0.14 | yes          |
|     20 | DQ126 | Is the lesion annular?                                                           | global   | indeterminate epidermal   |         -0.044 | 8%           |                                  0.06 |                                    0.04 |          0.17 | yes          |

9 of these 20 questions were rarely answered yes in any class. 1 point the other way inside the model than alone (†).

## Marker questions for each subtype (10-class model)

For each subtype, the questions with the largest positive ridge weight for that subtype in the 10-class model, with Sonnet's answers in that subtype and in all the others (share answered yes; mean P(yes)). A good marker has a clearly higher P(yes) in its subtype than elsewhere.

| subtype                      | ID    | question                                                               |   ridge weight | lasso kept   | yes in subtype   | yes elsewhere   |   mean P(yes) in subtype |   mean P(yes) elsewhere |
|:-----------------------------|:------|:-----------------------------------------------------------------------|---------------:|:-------------|:-----------------|:----------------|-------------------------:|------------------------:|
| benign fibro-histiocytic     | DQ388 | Are pseudostellate scars visible?                                      |          0.097 | 100%         | 0%               | 0%              |                    0.041 |                   0.028 |
| benign fibro-histiocytic     | DQ376 | Do the naevi in the image show an eclipse pattern (tan centre with a…  |          0.089 | 96%          | 3%               | 0%              |                    0.073 |                   0.037 |
| benign fibro-histiocytic     | DQ346 | Are the pigmented macules non-hairy?                                   |          0.085 | 84%          | 21%              | 20%             |                    0.273 |                   0.252 |
| benign melanocytic           | DQ421 | Is the melanocytic lesion flat and subtle, without obvious melanoma f… |          0.073 | 100%         | 54%              | 24%             |                    0.496 |                   0.27  |
| benign melanocytic           | DQ097 | Is the depigmented lesion free of scale?                               |          0.073 | 68%          | 0%               | 1%              |                    0.077 |                   0.098 |
| benign melanocytic           | DQ476 | Does lesion contrast increase when the RGB image is converted to simu… |          0.073 | 100%         | 100%             | 98%             |                    0.644 |                   0.62  |
| benign vascular              | DQ472 | Does the lesion look like a vascular neoplasm (uniform red to purple…  |          0.135 | 100%         | 55%              | 1%              |                    0.505 |                   0.069 |
| benign vascular              | DQ301 | Does the lesion resemble a vascular lesion?                            |          0.128 | 100%         | 69%              | 1%              |                    0.626 |                   0.109 |
| benign vascular              | DQ293 | Is a small elevated lesion uniformly black?                            |          0.088 | 100%         | 4%               | 0%              |                    0.069 |                   0.017 |
| flat melanotic pigmentation  | DQ416 | Is the lesion located on a nail or mucosa?                             |          0.135 | 100%         | 5%               | 0%              |                    0.075 |                   0.036 |
| flat melanotic pigmentation  | DQ094 | Does the lesion have a smooth, shiny surface?                          |          0.077 | 68%          | 2%               | 7%              |                    0.175 |                   0.21  |
| flat melanotic pigmentation  | DQ364 | Does the image show more than 50 common naevi?                         |          0.073 | 96%          | 0%               | 0%              |                    0.024 |                   0.017 |
| inflammatory or infectious   | DQ412 | Is the skin in the image dark (a group where diagnostic accuracy is l… |          0.085 | 96%          | 2%               | 0%              |                    0.061 |                   0.035 |
| inflammatory or infectious   | DQ112 | Does the lesion surface appear rippled?                                |          0.083 | 96%          | 1%               | 0%              |                    0.1   |                   0.066 |
| inflammatory or infectious   | DQ486 | Are wrinkles closely spaced (short wrinkle pitch)?                     |          0.079 | 88%          | 15%              | 7%              |                    0.277 |                   0.223 |
| indeterminate epidermal      | DQ043 | Are white dots or clods clustered or disseminated across the lesion?   |          0.098 | 100%         | 3%               | 0%              |                    0.164 |                   0.117 |
| indeterminate epidermal      | DQ482 | Does a thick keratin layer make the lesion surface look white or opaq… |          0.078 | 96%          | 10%              | 4%              |                    0.214 |                   0.145 |
| indeterminate epidermal      | DQ057 | Are follicular plugs present in the lesion?                            |          0.075 | 72%          | 1%               | 0%              |                    0.092 |                   0.054 |
| indeterminate melanocytic    | DQ193 | Does the lesion have a scalloped border?                               |          0.079 | 100%         | 5%               | 1%              |                    0.224 |                   0.132 |
| indeterminate melanocytic    | DQ481 | Does the image show skin with visible features of obesity (e.g., stri… |          0.067 | 88%          | 0%               | 0%              |                    0.03  |                   0.03  |
| indeterminate melanocytic    | DQ298 | Are pigment remnants arranged in a cobblestone pattern?                |          0.067 | 96%          | 1%               | 0%              |                    0.087 |                   0.062 |
| malignant adnexal follicular | DQ222 | Does the visible skin have a yellow (rather than red) undertone?       |          0.093 | 100%         | 15%              | 15%             |                    0.306 |                   0.292 |
| malignant adnexal follicular | DQ082 | Are blue-grey or black ovoid nests of varying size present (correlate… |          0.081 | 96%          | 3%               | 1%              |                    0.093 |                   0.056 |
| malignant adnexal follicular | DQ160 | Is pink coloration sparse or absent in a lesion with BCC-like structu… |          0.079 | 96%          | 0%               | 0%              |                    0.147 |                   0.111 |
| malignant epidermal          | DQ355 | Are scaly lesions present on the ear helix?                            |          0.109 | 100%         | 2%               | 0%              |                    0.057 |                   0.029 |
| malignant epidermal          | DQ484 | Does the skin surface appear rough?                                    |          0.084 | 100%         | 59%              | 21%             |                    0.549 |                   0.356 |
| malignant epidermal          | DQ290 | Are dotted vessels present together with an inverse (negative) networ… |          0.073 | 96%          | 0%               | 0%              |                    0.053 |                   0.032 |
| malignant melanocytic        | DQ454 | Is the lesion's darkest region located off-centre?                     |          0.083 | 76%          | 59%              | 40%             |                    0.546 |                   0.454 |
| malignant melanocytic        | DQ361 | Are multiple lesions from the same patient visible in the image?       |          0.083 | 84%          | 8%               | 4%              |                    0.135 |                   0.11  |
| malignant melanocytic        | DQ212 | Is the lesion on a less pigmented acral site of a person with dark sk… |          0.076 | 100%         | 0%               | 0%              |                    0.037 |                   0.028 |

Figure: [subtype_markers.png](figures/subtype_markers.png)

## Topics and scopes

*Importance share*: each model's |SHAP| scaled to sum to 1, averaged over models, summed over the topic. *Enrichment* is that share divided by the topic's share of the questions: above 1, the topic carries more than its size would suggest.

**s1_2c**, 10 largest topics

| topic                                          |   n_questions |   importance_share |   question_share |   enrichment |
|:-----------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Dermoscopy terminology consensus               |            37 |              0.12  |            0.083 |        1.45  |
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) |            50 |              0.109 |            0.112 |        0.975 |
| Dermoscopy algorithms for melanoma             |            30 |              0.085 |            0.067 |        1.264 |
| Inflammatory dermoscopy & general dermatology  |            36 |              0.072 |            0.081 |        0.887 |
| Vascular structures in dermoscopy              |            21 |              0.048 |            0.047 |        1.027 |
| Special sites (acral, face/LM, nail, mucosa)   |            63 |              0.048 |            0.141 |        0.339 |
| Photoaging scales & image grading              |            16 |              0.041 |            0.036 |        1.132 |
| Seborrhoeic keratosis & benign mimics          |            15 |              0.037 |            0.034 |        1.112 |
| ILDS morphology / lesion terminology           |            20 |              0.036 |            0.045 |        0.796 |
| Skin surface texture & microrelief             |             8 |              0.033 |            0.018 |        1.834 |

**s1_3c**, 10 largest topics

| topic                                          |   n_questions |   importance_share |   question_share |   enrichment |
|:-----------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Dermoscopy terminology consensus               |            37 |              0.111 |            0.083 |        1.338 |
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) |            50 |              0.11  |            0.112 |        0.979 |
| Dermoscopy algorithms for melanoma             |            30 |              0.078 |            0.067 |        1.155 |
| Inflammatory dermoscopy & general dermatology  |            36 |              0.068 |            0.081 |        0.843 |
| Vascular structures in dermoscopy              |            21 |              0.055 |            0.047 |        1.168 |
| Special sites (acral, face/LM, nail, mucosa)   |            63 |              0.05  |            0.141 |        0.356 |
| ILDS morphology / lesion terminology           |            20 |              0.043 |            0.045 |        0.951 |
| Photoaging scales & image grading              |            16 |              0.04  |            0.036 |        1.124 |
| Field cancerization & actinic damage scoring   |            10 |              0.037 |            0.022 |        1.651 |
| Seborrhoeic keratosis & benign mimics          |            15 |              0.034 |            0.034 |        1.018 |

**flat_10c**, 10 largest topics

| topic                                          |   n_questions |   importance_share |   question_share |   enrichment |
|:-----------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) |            50 |              0.107 |            0.112 |        0.957 |
| Dermoscopy terminology consensus               |            37 |              0.093 |            0.083 |        1.123 |
| Dermoscopy algorithms for melanoma             |            30 |              0.065 |            0.067 |        0.974 |
| Inflammatory dermoscopy & general dermatology  |            36 |              0.062 |            0.081 |        0.769 |
| Photoaging scales & image grading              |            16 |              0.051 |            0.036 |        1.432 |
| Vascular structures in dermoscopy              |            21 |              0.051 |            0.047 |        1.078 |
| Seborrhoeic keratosis & benign mimics          |            15 |              0.046 |            0.034 |        1.369 |
| ILDS morphology / lesion terminology           |            20 |              0.046 |            0.045 |        1.021 |
| Special sites (acral, face/LM, nail, mucosa)   |            63 |              0.045 |            0.141 |        0.317 |
| Spitz / pink / amelanotic & paediatric lesions |            15 |              0.041 |            0.034 |        1.217 |

**s2_benign**, 10 largest topics

| topic                                          |   n_questions |   importance_share |   question_share |   enrichment |
|:-----------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Dermoscopy terminology consensus               |            37 |              0.085 |            0.083 |        1.024 |
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) |            50 |              0.082 |            0.112 |        0.733 |
| Photoaging scales & image grading              |            16 |              0.068 |            0.036 |        1.888 |
| Dermoscopy algorithms for melanoma             |            30 |              0.067 |            0.067 |        1.003 |
| Inflammatory dermoscopy & general dermatology  |            36 |              0.065 |            0.081 |        0.811 |
| Special sites (acral, face/LM, nail, mucosa)   |            63 |              0.057 |            0.141 |        0.406 |
| Vascular structures in dermoscopy              |            21 |              0.052 |            0.047 |        1.094 |
| ILDS morphology / lesion terminology           |            20 |              0.051 |            0.045 |        1.141 |
| Seborrhoeic keratosis & benign mimics          |            15 |              0.05  |            0.034 |        1.498 |
| Spitz / pink / amelanotic & paediatric lesions |            15 |              0.049 |            0.034 |        1.459 |

**s2_malignant**, 10 largest topics

| topic                                          |   n_questions |   importance_share |   question_share |   enrichment |
|:-----------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) |            50 |              0.122 |            0.112 |        1.087 |
| Dermoscopy terminology consensus               |            37 |              0.097 |            0.083 |        1.164 |
| Dermoscopy algorithms for melanoma             |            30 |              0.069 |            0.067 |        1.023 |
| Seborrhoeic keratosis & benign mimics          |            15 |              0.063 |            0.034 |        1.868 |
| Inflammatory dermoscopy & general dermatology  |            36 |              0.054 |            0.081 |        0.669 |
| Special sites (acral, face/LM, nail, mucosa)   |            63 |              0.051 |            0.141 |        0.359 |
| ILDS morphology / lesion terminology           |            20 |              0.041 |            0.045 |        0.916 |
| Vascular structures in dermoscopy              |            21 |              0.04  |            0.047 |        0.847 |
| Spitz / pink / amelanotic & paediatric lesions |            15 |              0.038 |            0.034 |        1.131 |
| Nevus count & atypical mole phenotype          |             9 |              0.036 |            0.02  |        1.797 |

**s2_indeterminate**, 10 largest topics

| topic                                              |   n_questions |   importance_share |   question_share |   enrichment |
|:---------------------------------------------------|--------------:|-------------------:|-----------------:|-------------:|
| Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA)     |            50 |              0.142 |            0.112 |        1.268 |
| Dermoscopy algorithms for melanoma                 |            30 |              0.082 |            0.067 |        1.225 |
| Inflammatory dermoscopy & general dermatology      |            36 |              0.069 |            0.081 |        0.855 |
| Dermoscopy terminology consensus                   |            37 |              0.069 |            0.083 |        0.83  |
| Nevus count & atypical mole phenotype              |             9 |              0.063 |            0.02  |        3.12  |
| Seborrhoeic keratosis & benign mimics              |            15 |              0.051 |            0.034 |        1.505 |
| ILDS morphology / lesion terminology               |            20 |              0.048 |            0.045 |        1.067 |
| Lesion-body context & patient-level AI             |             2 |              0.044 |            0.004 |        9.816 |
| Nevus phenotypes / signature nevus / ugly duckling |             5 |              0.039 |            0.011 |        3.493 |
| Congenital nevi & vascular anomalies/birthmarks    |            11 |              0.038 |            0.025 |        1.56  |

Enrichment by scope (local = a part of the lesion; global = the whole lesion or image; metadata = facts about the patient or site that the image may show):

| scope    |   flat_10c |   s1_2c |   s1_3c |   s2_benign |   s2_indeterminate |   s2_malignant |
|:---------|-----------:|--------:|--------:|------------:|-------------------:|---------------:|
| global   |       1.11 |    1.11 |    1.04 |        1.15 |               1.16 |           1.1  |
| local    |       0.93 |    0.98 |    0.97 |        0.93 |               0.86 |           0.91 |
| metadata |       0.88 |    0.64 |    0.99 |        0.69 |               1.01 |           1.05 |

## Near-duplicate questions among each task's top 20

Pairs whose scores correlate at |r| ≥ 0.7 over the 1,000 images. The models can split weight between such questions, so either one may stand for both.

| task             | source   | question_a   | question_b   |    r |
|:-----------------|:---------|:-------------|:-------------|-----:|
| s1_2c            | dir      | DQ074        | DQ071        | 0.74 |
| s1_3c            | dir      | DQ055        | DQ304        | 0.83 |
| s1_3c            | dir      | DQ484        | DQ188        | 0.7  |
| s1_3c            | dir      | DQ046        | DQ031        | 0.73 |
| flat_10c         | dir      | DQ363        | DQ304        | 0.78 |
| flat_10c         | dir      | DQ472        | DQ301        | 0.89 |
| flat_10c         | dir      | DQ301        | DQ471        | 0.73 |
| flat_10c         | dir      | DQ471        | DQ304        | 0.75 |
| s2_benign        | dir      | DQ472        | DQ301        | 0.89 |
| s2_benign        | dir      | DQ480        | DQ390        | 0.72 |
| s2_benign        | dir      | DQ385        | DQ386        | 0.71 |
| s2_benign        | dir      | DQ385        | DQ388        | 0.7  |
| s2_benign        | dir      | DQ385        | DQ390        | 0.71 |
| s2_benign        | dir      | DQ386        | DQ388        | 0.73 |
| s2_benign        | dir      | DQ304        | DQ363        | 0.78 |
| s2_malignant     | dir      | DQ082        | DQ079        | 0.77 |
| s2_indeterminate | dir      | DQ363        | DQ155        | 0.76 |
| s2_indeterminate | dir      | DQ024        | DQ365        | 0.72 |
| s2_indeterminate | dir      | DQ176        | DQ154        | 0.8  |
| s2_indeterminate | dir      | DQ176        | DQ417        | 0.75 |
| s2_indeterminate | dir      | DQ176        | DQ353        | 0.72 |
| s2_indeterminate | dir      | DQ176        | DQ083        | 0.73 |
| s2_indeterminate | dir      | DQ373        | DQ365        | 0.72 |
| s2_indeterminate | dir      | DQ155        | DQ353        | 0.71 |
| s2_indeterminate | dir      | DQ333        | DQ332        | 0.92 |
| s2_indeterminate | dir      | DQ108        | DQ126        | 0.92 |

## Files

- consensus.csv: every question, task and source; all ranks, weights, yes shares, AUROC alone
- agreement.csv, source_agreement.csv: the validity checks above
- subtype_markers.csv, topics.csv, scopes.csv, redundancy.csv
