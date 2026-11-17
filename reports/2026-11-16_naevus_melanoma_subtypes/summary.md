# S11 subtypes: the 10-subtype model in detail, and naevus against melanoma

## Ten subtypes (S10 ridge, all questions, folds grouped by patient; mean over 5 repeats, range in brackets)

Balanced accuracy 0.479 (0.469 to 0.487); chance 0.10. Top-2 accuracy 0.685, top-3 0.806. Of the errors, 69% stay within the same lineage (melanocytic or not) and 25% within the same malignancy group. Collapsed to benign against malignant, the 10-subtype probabilities reach AUROC 0.843 (two-class model, pooled per repeat: 0.837).

| subtype | group | lineage | n | sensitivity | precision | F1 | AUROC one vs rest | sensitivity, dermoscopic | most often taken for |
|---|---|---|---|---|---|---|---|---|---|
| benign fibro-histiocytic | benign | non-melanocytic | 100 | 0.30 (0.27 to 0.32) | 0.39 | 0.34 | 0.802 | 0.30 | benign melanocytic (13%) |
| benign melanocytic | benign | melanocytic | 100 | 0.75 (0.72 to 0.76) | 0.56 | 0.64 | 0.943 | 0.76 | indeterminate melanocytic (8%) |
| benign vascular | benign | non-melanocytic | 100 | 0.81 (0.80 to 0.82) | 0.77 | 0.79 | 0.961 | 0.81 | malignant adnexal follicular (5%) |
| flat melanotic pigmentation | benign | melanocytic | 100 | 0.48 (0.47 to 0.50) | 0.43 | 0.45 | 0.876 | 0.51 | indeterminate melanocytic (18%) |
| inflammatory or infectious | benign | non-melanocytic | 100 | 0.43 (0.39 to 0.47) | 0.47 | 0.45 | 0.839 | 0.45 | malignant epidermal (15%) |
| indeterminate epidermal | indeterminate | non-melanocytic | 100 | 0.36 (0.34 to 0.40) | 0.42 | 0.39 | 0.840 | 0.27 | malignant epidermal (15%) |
| indeterminate melanocytic | indeterminate | melanocytic | 100 | 0.30 (0.28 to 0.34) | 0.37 | 0.33 | 0.866 | 0.28 | flat melanotic pigmentation (27%) |
| malignant adnexal follicular | malignant | non-melanocytic | 100 | 0.41 (0.38 to 0.43) | 0.42 | 0.42 | 0.862 | 0.43 | malignant epidermal (12%) |
| malignant epidermal | malignant | non-melanocytic | 100 | 0.52 (0.49 to 0.55) | 0.44 | 0.48 | 0.883 | 0.49 | indeterminate epidermal (13%) |
| malignant melanocytic | malignant | melanocytic | 100 | 0.42 (0.37 to 0.46) | 0.43 | 0.42 | 0.830 | 0.40 | benign melanocytic (13%) |

Confusion matrix (rows: true subtype; counts averaged over repeats): `tenclass_confusion.csv`, `figures/tenclass_confusion.png`.

## Naevus against melanoma

| score | AUROC, pooled per repeat: mean (range) | AUROC on the averaged score [95% CI, patient bootstrap] | specificity at 95% sensitivity: mean (range) | [95% CI] | dermoscopic only: AUROC, specificity |
|---|---|---|---|---|---|
| pair model (naevus vs melanoma only) | 0.901 (0.884 to 0.910) | 0.906 [0.863, 0.945] | 0.56 (0.53 to 0.62) | [0.22, 0.74] | 0.915, 0.56 |
| 10-subtype model | 0.910 (0.900 to 0.917) | 0.911 [0.869, 0.949] | 0.58 (0.53 to 0.67) | [0.32, 0.76] | 0.921, 0.59 |
| benign-vs-malignant model | 0.898 (0.892 to 0.902) | 0.903 [0.858, 0.944] | 0.47 (0.45 to 0.48) | [0.28, 0.73] | 0.900, 0.47 |
| Sonnet asked "malignant?" | 0.883 (0.883 to 0.883) | 0.883 [0.834, 0.928] | 0.31 (0.31 to 0.31) | [0.22, 0.66] | 0.896, 0.32 |

Pair model: median lambda 50 over the outer fits; final model (all 200 images) lambda 33.

Paired AUROC differences (averaged scores; patient bootstrap on the same resamples):

| a | b | a − b | 95% CI | share of resamples a > b | repeats a > b |
|---|---|---|---|---|---|
| pair | direct | +0.023 | [-0.013, +0.058] | 89% | 5 of 5 |
| tenclass | direct | +0.028 | [-0.004, +0.060] | 96% | 5 of 5 |
| twoclass | direct | +0.020 | [-0.020, +0.062] | 84% | 5 of 5 |
| pair | tenclass | -0.005 | [-0.017, +0.008] | 24% | 1 of 5 |
| pair | twoclass | +0.003 | [-0.029, +0.034] | 60% | 3 of 5 |
| tenclass | twoclass | +0.008 | [-0.024, +0.039] | 70% | 4 of 5 |

### Breakdown (AUROC, mean over repeats; [95% CI])

| subset | naevi | melanomas | pair model (naevus vs melanoma only) | 10-subtype model | benign-vs-malignant model | Sonnet asked "malignant?" |
|---|---|---|---|---|---|---|
| melanoma: Melanoma, NOS | 100 | 40 | 0.898 [0.83, 0.95] | 0.917 [0.86, 0.96] | 0.889 [0.81, 0.95] | 0.911 [0.85, 0.96] |
| melanoma: Melanoma Invasive | 100 | 30 | 0.916 [0.87, 0.97] | 0.913 [0.85, 0.97] | 0.922 [0.86, 0.98] | 0.849 [0.76, 0.93] |
| melanoma: Melanoma in situ | 100 | 25 | 0.915 [0.86, 0.97] | 0.923 [0.86, 0.98] | 0.876 [0.81, 0.95] | 0.892 [0.82, 0.95] |
| melanoma: Melanoma metastasis | 100 | 5 | 0.767 [0.44, 0.97] | 0.767 [0.46, 0.97] | 0.933 [0.90, 0.99] | 0.815 [0.52, 0.99] |
| naevi excised (histopathology) | 15 | 100 | 0.807 [0.69, 0.92] | 0.785 [0.66, 0.90] | 0.848 [0.74, 0.94] | 0.785 [0.64, 0.90] |
| naevi not excised (consensus, follow-up, other) | 85 | 100 | 0.918 [0.88, 0.96] | 0.932 [0.89, 0.96] | 0.907 [0.87, 0.95] | 0.900 [0.86, 0.94] |
| dermoscopic images only | 97 | 90 | 0.915 [0.88, 0.96] | 0.921 [0.88, 0.96] | 0.900 [0.86, 0.95] | 0.896 [0.85, 0.94] |

### Atypical melanocytic lesions on the pair model's scale

Threshold catching 95% of melanomas: 0.206. Above it: naevi 41%, atypical melanocytic 76%, melanomas 95%. Median score: naevi 0.146, atypical 0.479, melanomas 0.866.

- Atypical intraepithelial melanocytic proliferation (n = 37): median 0.644, 84% above the threshold
- Atypical melanocytic neoplasm (n = 62): median 0.375, 71% above the threshold
- Atypical proliferative nodules in congenital melanocytic nevus (n = 1): median 0.217, 100% above the threshold

### Questions that separate naevus from melanoma most, each alone (AUROC on the 200 images)

| question | topic | AUROC alone | towards | pair-model weight |
|---|---|---|---|---|
| Are shiny white structures present? | Dermoscopy algorithms for melanoma | 0.860 | melanoma | +0.037 |
| Are white structureless areas present (a correlate of dermal fibrosis)? | Histopathologic correlates of dermoscopic structures | 0.852 | melanoma | +0.066 |
| Is colour asymmetry within the lesion high? | Total body photography & 3D TBP | 0.849 | melanoma | +0.084 |
| Are regression structures present? | Dermoscopy algorithms for melanoma | 0.849 | melanoma | +0.085 |
| Are blue-grey or black ovoid nests of varying size present (correlates of dermal tumour islands)? | Histopathologic correlates of dermoscopic structures | 0.842 | melanoma | -0.007 |
| Does the lesion contain more than two colours? | Dermoscopy algorithms for melanoma | 0.839 | melanoma | +0.047 |
| Is an atypical vascular pattern present? | Dermoscopy algorithms for melanoma | 0.838 | melanoma | +0.059 |
| Do shiny white structures coincide with hypopigmented lines in the lesion? | Spitz / pink / amelanotic & paediatric lesions | 0.836 | melanoma | +0.035 |
| Are short white streaks (chrysalis structures) present? | Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) | 0.836 | melanoma | +0.026 |
| Is a central scar-like area present? | Inflammatory dermoscopy & general dermatology | 0.834 | melanoma | +0.090 |
| Are signs of chronic sun damage visible around the lesions? | Field cancerization & actinic damage scoring | 0.828 | melanoma | +0.103 |
| Does the lesion show a multicomponent global pattern? | Dermoscopy algorithms for melanoma | 0.825 | melanoma | +0.044 |
| Does the skin surface appear rough? | Skin surface texture & microrelief | 0.824 | melanoma | +0.043 |
| Are white dots or clods clustered or disseminated across the lesion? | Dermoscopy terminology consensus | 0.822 | melanoma | +0.098 |
| Are whitish structureless areas present? | Keratinocyte tumour dermoscopy (BCC/SCC/AK/KA) | 0.821 | melanoma | +0.026 |
| Are dotted vessels visible within a pigment network? | Dermoscopy algorithms for melanoma | 0.821 | melanoma | +0.046 |
| Are bright white streaks present? | Inflammatory dermoscopy & general dermatology | 0.819 | melanoma | +0.018 |
| Is a grey-white structureless zone present in the lesion? | Dermoscopy terminology consensus | 0.816 | melanoma | +0.070 |
| Is grey colour present in the lesion? | Dermoscopy algorithms for melanoma | 0.815 | melanoma | +0.077 |
| Are comedo-like structures filled with yellow keratinous plugs present? | Inflammatory dermoscopy & general dermatology | 0.812 | melanoma | -0.019 |
| Is there asymmetry in the distribution of dermoscopic structures? | Dermoscopy algorithms for melanoma | 0.811 | melanoma | +0.048 |
| Are shiny white clods present in the lesion? | Dermoscopy terminology consensus | 0.810 | melanoma | +0.068 |
| Is a structureless pink zone (milky-red area) present in the lesion? | Dermoscopy terminology consensus | 0.810 | melanoma | +0.053 |
| Are linear irregular vessels present? | Vascular structures in dermoscopy | 0.810 | melanoma | +0.055 |
| Are blue-grey dots present (a correlate of pigment incontinence)? | Histopathologic correlates of dermoscopic structures | 0.810 | melanoma | +0.016 |
| Does a thick keratin layer make the lesion surface look white or opaque? | Skin optical properties & chromophores | 0.809 | melanoma | +0.015 |
| Are vessels distributed randomly without a specific arrangement? | Dermoscopy terminology consensus | 0.808 | melanoma | +0.042 |
| Is at least one major 7-point checklist criterion present? | Dermoscopy algorithms for melanoma | 0.807 | melanoma | +0.049 |
| Does the lesion score 3 or more on the revised 7-point checklist? | Dermoscopy algorithms for melanoma | 0.807 | melanoma | +0.035 |
| Are erosions or ulcerations present in the lesion? | Dermoscopy terminology consensus | 0.806 | melanoma | +0.048 |
