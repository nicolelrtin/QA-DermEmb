# S9 explore: structure of the answers

1000 images, 500 questions (clinical, artefact; the control question left out). Score: Directional Score. Open `explorer.html` in a browser for the interactive version.

## Questions answered alike: topics

- Mean correlation between two questions of the same topic: 0.55; between questions of different topics: 0.10. With the topic labels shuffled (same topic sizes, 1000 times) the same-topic mean is 0.12 on average and at most 0.14 (p = 0.0010).
- Questions citing a common paper (2856 pairs): mean r 0.46, against 0.12 for random pairs (95th percentile 0.13).
- Topics more cohesive than 95% of random groups of the same size: 23 of 37 (topics with at least 2 questions).

- At equal wording similarity (qwen3, 10 bins), same-topic pairs are more correlated than different-topic pairs in 10 of 10 bins (median difference 0.43).

## Answers against wording

- Rank correlation between answer correlation and text similarity (qwen3), over all question pairs: 0.126 (shuffled 95th percentile 0.024).
- Rank correlation between answer correlation and text similarity (panderm), over all question pairs: 0.166 (shuffled 95th percentile 0.019).

Agreement of the 20 answer clusters with other groupings (adjusted mutual information; 0 = chance, 1 = identical; in brackets the 95th percentile with one grouping shuffled):

- answer clusters vs topic: 0.51 (0.01)
- answer clusters vs text clusters (qwen3): 0.36 (0.01)
- answer clusters vs text clusters (panderm): 0.35 (0.01)
- answer clusters vs grounding: 0.02 (0.01)
- answer clusters vs first cited source: 0.36 (0.02)
- text clusters (qwen3) vs topic: 0.38 (0.01)
- text clusters (panderm) vs topic: 0.30 (0.02)

## How many independent dimensions

- Components needed for 50 / 80 / 90 / 95% of the variance: 4 / 25 / 60 / 112.
- Components whose variance beats the same answers shuffled across images (parallel analysis): 27.
- Participation ratio (effective number of equally strong dimensions): 11.4.
- The first component carries 21% of the variance; separates malignant from benign with AUROC 0.64.

| component | variance | AUROC malignant | subtype | image type | institution | body site | sex | age |
|---|---|---|---|---|---|---|---|---|
| 1 | 20.7% | 0.64 | 0.52 | 0.09 | 0.19 | 0.05 | 0.02 | 0.07 |
| 2 | 17.2% | 0.34 | 0.20 | 0.02 | 0.03 | 0.04 | 0.00 | 0.07 |
| 3 | 7.6% | 0.57 | 0.13 | 0.02 | 0.05 | 0.02 | 0.02 | 0.00 |
| 4 | 5.3% | 0.37 | 0.13 | 0.15 | 0.07 | 0.04 | 0.01 | 0.03 |
| 5 | 4.0% | 0.43 | 0.13 | 0.14 | 0.22 | 0.03 | 0.01 | 0.01 |
| 6 | 3.6% | 0.53 | 0.05 | 0.28 | 0.17 | 0.01 | 0.00 | 0.01 |
| 7 | 2.5% | 0.51 | 0.05 | 0.01 | 0.08 | 0.01 | 0.00 | 0.00 |
| 8 | 2.2% | 0.64 | 0.08 | 0.11 | 0.20 | 0.01 | 0.00 | 0.00 |
| 9 | 2.1% | 0.47 | 0.18 | 0.04 | 0.04 | 0.07 | 0.00 | 0.00 |
| 10 | 1.5% | 0.49 | 0.04 | 0.00 | 0.04 | 0.03 | 0.02 | 0.01 |

(Share of each component's variance explained by each variable: eta squared, or r squared for age.)

## Images: what the answer profiles are organised by

Ward clustering of the images on the first 27 principal components. Adjusted mutual information between the 10 image clusters and each variable, overall and within benign or malignant images only:

| variable | AMI | shuffled 95th | within benign | within malignant |
|---|---|---|---|---|
| subtype | 0.23 | 0.00 | 0.31 | 0.18 |
| group | 0.05 | 0.00 | n/a | n/a |
| image type | 0.06 | 0.00 | 0.03 | 0.05 |
| institution | 0.09 | 0.01 | 0.09 | 0.08 |
| body site | 0.03 | 0.00 | 0.02 | 0.00 |
| sex | 0.02 | 0.00 | 0.01 | 0.01 |
| age (decade) | 0.04 | 0.01 | 0.04 | 0.04 |

## The 20 question clusters

| cluster | questions | mean r within | AUROC of cluster score | main topics | most central question |
|---|---|---|---|---|---|
| 1 | 48 | 0.88 | 0.51 | Special sites (acral, face/LM, nail, muc (48) | Are splinter haemorrhages visible under the nail? |
| 2 | 14 | 0.91 | 0.51 | Special sites (acral, face/LM, nail, muc (14) | Are rhomboidal structures present around follicles? |
| 3 | 38 | 0.61 | 0.47 | Dermoscopy algorithms for melanoma (11); Seborrhoeic keratosis & benign mimics (5); Nevus count & atypical mole phenotype (5) | Is an atypical pigment network present? |
| 4 | 23 | 0.55 | 0.64 | Keratinocyte tumour dermoscopy (BCC/SCC/ (9); Dermoscopy algorithms for melanoma (3); Dermoscopy terminology consensus (3) | Are blue, black and grey dots present together? |
| 5 | 20 | 0.56 | 0.60 | Dermoscopy algorithms for melanoma (7); Dermoscopy terminology consensus (7); Keratinocyte tumour dermoscopy (BCC/SCC/ (2) | Are grey or blue dots arranged in a peripheral clustered or linear pattern? |
| 6 | 20 | 0.61 | 0.58 | Dermoscopy algorithms for melanoma (6); Texture descriptors & radiomics for lesi (6); Clinical melanoma criteria (ABCDE, EFG,  (3) | Is there asymmetry in the distribution of dermoscopic structures? |
| 7 | 23 | 0.63 | 0.56 | ILDS morphology / lesion terminology (12); Spitz / pink / amelanotic & paediatric l (4); Clinical melanoma criteria (ABCDE, EFG,  (2) | Is the lesion dome-shaped? |
| 8 | 32 | 0.73 | 0.68 | Vascular structures in dermoscopy (20); Keratinocyte tumour dermoscopy (BCC/SCC/ (5); Dermoscopy algorithms for melanoma (1) | Are hairpin vessels present? |
| 9 | 44 | 0.66 | 0.68 | Keratinocyte tumour dermoscopy (BCC/SCC/ (28); Inflammatory dermoscopy & general dermat (5); Seborrhoeic keratosis & benign mimics (5) | Is an ivory-white peripheral zone present? |
| 10 | 18 | 0.22 | 0.44 | Shortcut / artefact learning (markings,  (9); Colour constancy & acquisition standards (2); Nevus count & atypical mole phenotype (1) | Are two or more different artefact types present in the image? |
| 11 | 21 | 0.16 | 0.38 | Anatomic site & body-region lesion densi (4); Skin tone measurement (Fitzpatrick, Monk (3); Public datasets & challenges (3) | Is the lesion clearly separable from the background for segmentation? |
| 12 | 14 | 0.77 | 0.55 | Skin of colour presentation (9); Skin tone measurement (Fitzpatrick, Monk (3); Fairness & skin-tone bias (2) | Is the contrast between epidermal structures and background low because of dark skin? |
| 13 | 15 | 0.35 | 0.60 | Smartphone / consumer apps & teledermato (5); Skin tone measurement (Fitzpatrick, Monk (2); Total body photography & 3D TBP (2) | Is the image resolution too low to resolve lesion detail? |
| 14 | 18 | 0.58 | 0.58 | Congenital nevi & vascular anomalies/bir (11); Segmentation & handcrafted ABCD features (4); Clinical melanoma criteria (ABCDE, EFG,  (1) | Does a facial haemangioma involve both sides of the face? |
| 15 | 10 | 0.80 | 0.54 | Distribution & configuration (Blaschko,  (10) | Do the lesions follow a dermatomal pattern? |
| 16 | 34 | 0.12 | 0.53 | Shortcut / artefact learning (markings,  (6); Colour constancy & acquisition standards (4); Segmentation & handcrafted ABCD features (3) | Does the lighting visibly alter the apparent skin tone? |
| 17 | 26 | 0.75 | 0.66 | Inflammatory dermoscopy & general dermat (26) | Is perifollicular tubular scaling present? |
| 18 | 40 | 0.41 | 0.67 | Dermoscopy terminology consensus (22); Spitz / pink / amelanotic & paediatric l (7); Keratinocyte tumour dermoscopy (BCC/SCC/ (3) | Are vessels arranged in a serpiginous (string-of-pearls) line? |
| 19 | 16 | 0.57 | 0.63 | Photoaging scales & image grading (11); Skin surface texture & microrelief (2); Field cancerization & actinic damage sco (1) | Does the skin show loss of elasticity (sagging)? |
| 20 | 26 | 0.50 | 0.68 | Field cancerization & actinic damage sco (8); ILDS morphology / lesion terminology (4); Skin surface texture & microrelief (3) | Are scaly plaques present on sun-exposed skin? |
