"""Synthetic Claude answers with known informative questions, to test the classifiers before the real answers exist.

Each image gets the label it has in the manifest; only the answers are simulated. For image i of subtype k and
question q, a latent evidence score is
    e = mu_q + delta[k, q] + topic shift (shared by the questions of one topic on one image) + noise,
P(yes) = sigmoid(e), rounded to 0.01, and the stated confidence rises with how far P(yes) is from 0.5, plus noise.
mu_q is set so that about 12-15% of answers are "yes" and the confidence sits around 0.75, as in the 2-image Claude
proof of concept (15% yes; mean confidence 0.73; |2 P(yes) - 1| vs confidence r = 0.84).

delta is zero except for planted questions (chosen at random among clinical, non-artefact questions):
  malignancy   20 questions: +effect in the 3 malignant subtypes, +effect/2 in the 2 indeterminate subtypes
  indeterminate 8 questions: +effect in the 2 indeterminate subtypes
  subtype       4 questions per subtype (40): +effect in that subtype only
About 1 in 5 planted effects is negative (the answer turns towards "no"). With effect = 0 nothing is informative and
every classifier should score at chance: the leakage check. Directional and Coherence Scores are derived from the
simulated P(yes) and confidence exactly as S4 derives them from Claude's (stages/s4_embed.confidence_scores).
"""
import numpy as np
import pandas as pd

from .stages.s4_embed import confidence_scores

N_MALIG, N_INDET, N_PER_SUBTYPE = 20, 8, 4


def simulate(qt: pd.DataFrame, image_ids, subtype, group, effect: float, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """qt: question table (question_id, topic, kind). subtype: label_10c per image; group: label_3c per image.
    Returns ({source: DataFrame images x questions}, planted table with delta per subtype)."""
    rng = np.random.default_rng(seed)
    qids = list(qt.question_id)
    M = len(qids)
    subtypes = sorted(set(subtype))
    sub_group = dict(zip(subtype, group))
    delta = pd.DataFrame(0.0, index=subtypes, columns=qids)
    role = pd.Series("uninformative", index=qids)

    eligible = list(qt.question_id[qt.get("kind", pd.Series("clinical", index=qt.index)).ne("artefact")])
    picks = iter(rng.permutation(eligible))
    sign = lambda: 1.0 if rng.random() > 0.2 else -1.0  # noqa: E731

    for _ in range(N_MALIG):
        q, s = next(picks), sign()
        role[q] = "malignancy"
        for k in subtypes:
            g = sub_group[k]
            delta.loc[k, q] = s * effect if g == "malignant" else s * effect / 2 if g == "indeterminate" else 0.0
    for _ in range(N_INDET):
        q, s = next(picks), sign()
        role[q] = "indeterminate"
        for k in subtypes:
            if sub_group[k] == "indeterminate":
                delta.loc[k, q] = s * effect
    for k in subtypes:
        for _ in range(N_PER_SUBTYPE):
            q = next(picks)
            role[q] = f"subtype: {k}"
            delta.loc[k, q] = sign() * effect

    mu = rng.normal(-2.6, 1.2, M)
    topics = qt.get("topic", pd.Series("all", index=qt.index)).values
    utop, tix = np.unique(topics, return_inverse=True)
    n = len(image_ids)
    shift = rng.normal(0, 0.6, (n, len(utop)))[:, tix]
    e = mu[None] + delta.loc[list(subtype)].values + shift + rng.normal(0, 1.2, (n, M))
    p = np.clip(np.round(1 / (1 + np.exp(-e)), 2), 0.01, 0.99)
    conf = np.clip(np.round((0.45 + 0.5 * np.abs(2 * p - 1) + rng.normal(0, 0.08, p.shape)) * 20) / 20, 0.2, 0.98)

    P = pd.DataFrame(p, index=pd.Index(image_ids, name="image_id"), columns=qids)
    C = pd.DataFrame(conf, index=P.index, columns=qids)
    stacked = confidence_scores(P.stack(), C.stack())
    sources = {"orig": P, "conf": C, "adj": stacked["Net_P"].unstack()[qids], "w": stacked["P_w"].unstack()[qids],
               "dir": stacked["Directional_Score"].unstack()[qids], "coh": stacked["Coherence_Score"].unstack()[qids]}
    planted = delta.T.rename_axis("question_id").reset_index()
    planted.insert(1, "role", role.values)
    return sources, planted


def relevance(planted: pd.DataFrame, classes_to_subtypes: dict) -> pd.Series:
    """How strongly each question separates a task's classes in the simulation: the largest gap, over pairs of the
    task's classes, between the class means of delta (each class = the mean over its subtypes). 0 = uninformative."""
    d = planted.set_index("question_id").drop(columns="role")
    means = pd.DataFrame({c: d[list(subs)].mean(1) for c, subs in classes_to_subtypes.items()})
    return (means.max(1) - means.min(1)).rename("relevance")
