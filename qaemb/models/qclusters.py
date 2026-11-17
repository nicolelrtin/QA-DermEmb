"""Helpers for S15 qclusters: agglomerative clustering of the questions at a minimum correlation, and the checks
that ask whether the clusters reflect what is in the images or how the answers were produced."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score


def zscore(M: np.ndarray) -> np.ndarray:
    sd = M.std(0)
    sd[sd == 0] = 1.0
    return (M - M.mean(0)) / sd


def residualise(M: np.ndarray, design: list[np.ndarray]) -> np.ndarray:
    """M minus its least-squares fit on an intercept and the given columns (one regression per column of M)."""
    D = np.column_stack([np.ones(len(M)), *design])
    return M - D @ np.linalg.lstsq(D, M, rcond=None)[0]


def dummies(values) -> np.ndarray:
    return pd.get_dummies(pd.Series(values).astype(str)).values[:, 1:].astype(float)


def ranks(M: np.ndarray) -> np.ndarray:
    """Column-wise ranks (ties averaged); Pearson r of ranks is Spearman's rho."""
    return rankdata(M, axis=0)


def corr(M: np.ndarray) -> np.ndarray:
    """Pearson correlation between the columns of M (constant columns give 0). Applied to ranks it gives Spearman's
    rho, which is what the stage uses: most questions are answered "no" on nearly every image, and Pearson r between
    two such questions is then decided by whether their one or two "yes" images coincide (shuffled answers reached
    r = 0.70 on logit P(yes)), whereas rank correlation stays below 0.3."""
    Z = zscore(M)
    R = (Z.T @ Z) / len(Z)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1, 1)


def complete_tree(R: np.ndarray):
    """Complete-linkage agglomerative clustering on the distance 1 - r. Two clusters merge at the largest distance
    between their members, so cutting the tree at 1 - r_min leaves clusters in which every pair of questions
    correlates at r >= r_min."""
    D = np.clip(1 - R, 0, 2)
    np.fill_diagonal(D, 0)
    return linkage(squareform(D, checks=False), "complete", optimal_ordering=True)


def cut(L, r_min: float) -> np.ndarray:
    """Cluster labels 0..k-1, numbered in the left-to-right order of the tree's leaves."""
    lab = fcluster(L, 1 - r_min + 1e-12, "distance")
    order = leaves_list(L)
    first = {}
    for pos, q in enumerate(order):
        first.setdefault(lab[q], pos)
    remap = {c: i for i, c in enumerate(sorted(first, key=first.get))}
    return np.array([remap[c] for c in lab])


def pair_mean(M: np.ndarray, idx) -> float:
    idx = np.asarray(idx)
    if len(idx) < 2:
        return np.nan
    S = M[np.ix_(idx, idx)]
    return float(S[np.triu_indices(len(idx), 1)].mean())


def pair_min(M: np.ndarray, idx) -> float:
    idx = np.asarray(idx)
    if len(idx) < 2:
        return np.nan
    S = M[np.ix_(idx, idx)]
    return float(S[np.triu_indices(len(idx), 1)].min())


def coclustering(F: np.ndarray, r_min: float, rng, n_boot: int, strata) -> np.ndarray:
    """Share of bootstrap resamples of the images (stratified on subtype) in which each pair of questions falls in
    the same cluster. F: images x questions (raw values; ranked inside each resample)."""
    n, Q = F.shape
    C = np.zeros((Q, Q))
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        lab = cut(complete_tree(corr(ranks(F[idx]))), r_min)
        C += lab[:, None] == lab[None, :]
    return C / n_boot


def split_half(F: np.ndarray, r_min: float, rng, n_split: int, strata) -> list[float]:
    """Adjusted Rand index between the clusterings of two disjoint halves of the images (stratified on subtype)."""
    out = []
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for _ in range(n_split):
        a, b = [], []
        for g in groups:
            p = rng.permutation(g)
            a.append(p[: len(p) // 2]); b.append(p[len(p) // 2:])
        la = cut(complete_tree(corr(ranks(F[np.concatenate(a)]))), r_min)
        lb = cut(complete_tree(corr(ranks(F[np.concatenate(b)]))), r_min)
        out.append(adjusted_rand_score(la, lb))
    return out


def null_max_r(F: np.ndarray, rng, n_perm: int) -> list[float]:
    """Largest |r| between any two questions after shuffling each question's answers independently across images."""
    out = []
    for _ in range(n_perm):
        P = np.column_stack([rng.permutation(F[:, j]) for j in range(F.shape[1])])
        R = corr(ranks(P))
        np.fill_diagonal(R, 0)
        out.append(float(np.abs(R).max()))
    return out


def variance_by_prompt(C: pd.DataFrame, topic_of: dict) -> dict:
    """How much of the stated confidence is shared by the questions of one prompt. C: images x questions. After
    removing each image's mean and each question's mean (additive), the share of the remaining variance explained by
    the mean of the image's prompt (image x topic)."""
    M = C.values.astype(float)
    E = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    topics = np.array([topic_of[q] for q in C.columns])
    P = np.zeros_like(E)
    for t in np.unique(topics):
        cols = topics == t
        P[:, cols] = E[:, cols].mean(1, keepdims=True)
    total = M.var()
    return {"total_var": float(total), "image_share": float(M.mean(1).var() / total),
            "question_share": float(M.mean(0).var() / total), "prompt_share_of_residual": float(P.var() / E.var()),
            "prompt_share_of_total": float(P.var() / total)}
