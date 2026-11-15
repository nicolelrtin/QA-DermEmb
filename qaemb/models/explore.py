"""Unsupervised analysis of the answers for S9 explore (PROJECT.md, Plan items B and D).

Questions are compared by how they were answered across the images (Pearson correlation of the standardised scores),
and by their wording (cosine similarity of text embeddings). Images are compared by their answer profiles. Every
association is set against a permutation null, so "the clusters agree with the topics" has a chance level to beat.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, roc_auc_score


# ------------------------------------------------------------------ basics
def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns to mean 0, sd 1. Returns (Z, keep) where keep drops columns with no variation."""
    sd = X.std(0)
    keep = sd > 1e-9
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    return Z, keep


def tree(D: np.ndarray, method: str = "average"):
    """Hierarchical clustering on a square distance matrix (or, for ward, on the raw rows passed as D)."""
    if method == "ward":
        return linkage(D, method="ward")
    return linkage(squareform(D, checks=False), method=method)


def cuts(L, ks) -> dict[int, np.ndarray]:
    """Cluster labels (0-based) for each k, numbered in dendrogram leaf order so neighbouring clusters get
    neighbouring numbers."""
    order = dendrogram(L, no_plot=True)["leaves"]
    out = {}
    for k in ks:
        lab = fcluster(L, k, criterion="maxclust")
        first = {}
        for leaf in order:
            first.setdefault(lab[leaf], len(first))
        out[int(k)] = np.array([first[v] for v in lab])
    return out


def dendro_segments(L, labels_order=None) -> dict:
    """Dendrogram line segments for drawing: x in leaf units (0..n-1), y = merge height."""
    d = dendrogram(L, no_plot=True)
    ic = (np.array(d["icoord"]) - 5) / 10          # scipy puts leaf i at 10 i + 5
    return {"leaves": [int(v) for v in d["leaves"]], "x": np.round(ic, 2).tolist(),
            "y": np.round(np.array(d["dcoord"]), 4).tolist()}


# ------------------------------------------------------------------ agreement between partitions
def ami(a, b) -> float:
    return float(adjusted_mutual_info_score(a, b))


def agree(A, B, rng, n_perm: int) -> dict:
    """Adjusted mutual information and adjusted Rand index between two labelings of the same items (0 = no more
    agreement than chance, 1 = identical), with the 95th percentile of each when one labeling is shuffled. Items
    missing in either labeling are dropped."""
    A, B = pd.Series(A).reset_index(drop=True), pd.Series(B).reset_index(drop=True)
    ok = A.notna() & B.notna()
    A, B = A[ok].astype(str).values, B[ok].astype(str).values
    null_ami = [adjusted_mutual_info_score(A, rng.permutation(B)) for _ in range(n_perm)]
    return {"n": int(ok.sum()), "ami": ami(A, B), "ami_null95": float(np.percentile(null_ami, 95)),
            "ari": float(adjusted_rand_score(A, B))}


# ------------------------------------------------------------------ cohesion of groups of questions
def group_cohesion(R: np.ndarray, groups: np.ndarray, rng, n_perm: int) -> tuple[dict, pd.DataFrame]:
    """Mean correlation between questions of the same group, against random groupings of the same sizes.

    Overall: mean r over all same-group pairs, and over all different-group pairs; null = the same-group mean after
    shuffling the group labels (sizes kept). Per group: mean r over its pairs; null = mean r over random sets of
    questions of the same size."""
    Q = len(groups)
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    cats, g = np.unique(groups, return_inverse=True)
    M = np.zeros((Q, len(cats)))
    M[np.arange(Q), g] = 1
    n = M.sum(0)
    npairs = n * (n - 1)

    def within(Mx):
        return (Mx * (R0 @ Mx)).sum(0)          # per group: sum of r over its ordered pairs

    obs_t = within(M)
    tot_pairs = npairs.sum()
    obs_mean = obs_t.sum() / tot_pairs
    all_mean = R0.sum() / (Q * (Q - 1))
    between_mean = (R0.sum() - obs_t.sum()) / (Q * (Q - 1) - tot_pairs)
    null_mean, null_t = [], []
    for _ in range(n_perm):
        Mp = M[rng.permutation(Q)]
        wt = within(Mp)
        null_mean.append(wt.sum() / tot_pairs)
        null_t.append(np.where(npairs > 0, wt / np.maximum(npairs, 1), np.nan))
    null_mean = np.array(null_mean)
    null_t = np.array(null_t)
    overall = {"within_mean_r": float(obs_mean), "between_mean_r": float(between_mean), "all_pairs_mean_r": float(all_mean),
               "null_mean": float(null_mean.mean()), "null_95": float(np.percentile(null_mean, 95)),
               "null_max": float(null_mean.max()), "p_perm": float((1 + (null_mean >= obs_mean).sum()) / (1 + n_perm))}
    per = pd.DataFrame({"group": cats, "n_questions": n.astype(int),
                        "within_mean_r": np.where(npairs > 0, obs_t / np.maximum(npairs, 1), np.nan),
                        "null_mean": np.nanmean(null_t, 0), "null_05": np.nanpercentile(null_t, 5, 0),
                        "null_95": np.nanpercentile(null_t, 95, 0)})
    per["share_null_below"] = [(null_t[:, i] < per.within_mean_r.iloc[i]).mean() if npairs[i] > 0 else np.nan
                               for i in range(len(cats))]
    return overall, per


def pair_mask_cohesion(R: np.ndarray, mask: np.ndarray, rng, n_perm: int) -> dict:
    """Mean r over the question pairs flagged in a symmetric mask (e.g. 'cite a common paper'), against the same mask
    applied to the questions in random order."""
    Q = len(R)
    iu = np.triu_indices(Q, 1)
    m = mask[iu]
    obs = R[iu][m].mean()
    null = []
    for _ in range(n_perm):
        p = rng.permutation(Q)
        null.append(R[np.ix_(p, p)][iu][m].mean())
    null = np.array(null)
    return {"n_pairs": int(m.sum()), "mean_r": float(obs), "other_pairs_mean_r": float(R[iu][~m].mean()),
            "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
            "p_perm": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def text_vs_answer_pairs(R, T, same_topic, bins=10) -> pd.DataFrame:
    """Mean answer correlation of question pairs in bins of wording similarity, separately for pairs in the same topic
    and in different topics. If same-topic pairs are more correlated at equal wording similarity, being asked in the
    same prompt (or a topic-level feature the wording does not show) adds to the correlation."""
    Q = len(R)
    iu = np.triu_indices(Q, 1)
    r, t, s = R[iu], T[iu], same_topic[iu]
    edges = np.quantile(t, np.linspace(0, 1, bins + 1))
    b = np.clip(np.digitize(t, edges[1:-1]), 0, bins - 1)
    rows = []
    for i in range(bins):
        for flag, name in ((True, "same topic"), (False, "different topics")):
            sel = (b == i) & (s == flag)
            if sel.sum() >= 5:
                rows.append({"bin": i, "text_sim_mid": float(np.median(t[b == i])), "pairs": name,
                             "n_pairs": int(sel.sum()), "mean_r": float(r[sel].mean()),
                             "se": float(r[sel].std(ddof=1) / np.sqrt(sel.sum()))})
    return pd.DataFrame(rows)


def mantel(A: np.ndarray, B: np.ndarray, rng, n_perm: int) -> dict:
    """Spearman correlation between the upper triangles of two similarity matrices, with a permutation p-value."""
    from scipy.stats import rankdata

    Q = len(A)
    iu = np.triu_indices(Q, 1)
    a = rankdata(A[iu])
    obs = np.corrcoef(a, rankdata(B[iu]))[0, 1]
    null = []
    for _ in range(n_perm):
        p = rng.permutation(Q)
        null.append(np.corrcoef(a, rankdata(B[np.ix_(p, p)][iu]))[0, 1])
    null = np.array(null)
    return {"spearman": float(obs), "null_95": float(np.percentile(null, 95)),
            "p_perm": float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm))}


# ------------------------------------------------------------------ dimensions
def pca(Z: np.ndarray, rng, n_perm: int, n_keep: int = 50) -> dict:
    """PCA of the standardised answers, with parallel analysis: each question's answers are shuffled across images
    independently (keeping each question's distribution, destroying the correlations), and a component counts as
    real while its eigenvalue beats the 95th percentile of the shuffled ones."""
    n = len(Z)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = s ** 2 / (n - 1)
    ratio = ev / ev.sum()
    null = []
    for _ in range(n_perm):
        Zp = np.column_stack([rng.permutation(c) for c in Z.T])
        null.append(np.linalg.svd(Zp, compute_uv=False)[: n_keep] ** 2 / (n - 1))
    null95 = np.percentile(np.array(null), 95, axis=0)
    above = ev[: n_keep] > null95
    n_real = int(np.argmin(above)) if not above.all() else n_keep
    cum = np.cumsum(ratio)
    return {"eigen": ev, "ratio": ratio, "cum": cum, "null95": null95, "n_parallel": n_real,
            "n_for": {p: int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9, 0.95)},
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "scores": U * s, "loadings": Vt.T}


def eta2(y: np.ndarray, groups) -> float:
    """Share of the variance of y explained by a categorical variable (missing categories dropped)."""
    g = pd.Series(groups)
    ok = g.notna().values & ~pd.isna(y)
    if ok.sum() < 3 or g[ok].nunique() < 2:
        return float("nan")
    y, g = np.asarray(y)[ok], g[ok].values
    m = y.mean()
    ss_b = sum(((y[g == c].mean() - m) ** 2) * (g == c).sum() for c in np.unique(g))
    return float(ss_b / ((y - m) ** 2).sum())


def auroc(y01, s) -> float:
    ok = ~pd.isna(y01)
    y01 = np.asarray(y01)[ok].astype(int)
    if len(np.unique(y01)) < 2:
        return float("nan")
    return float(roc_auc_score(y01, np.asarray(s)[ok]))
