"""Question-set reducers for the ridge stage (S10). Each is a scikit-learn transformer, so it sits inside the model
pipeline and is refitted on every training split, inner and outer: which questions are kept never depends on a test
image. 

  Columns(idx)      a fixed list of questions (all questions, or a random subset drawn once without the data)
  TopK(k)           the k questions with the largest ANOVA F statistic between the classes, on the training split
                    only (uses the training labels, never test ones)
  ClusterReps(k)    Ward clustering of the questions on correlation distance (1 - r between their scores across the
                    training images), cut into k clusters; keeps the most typical question of each cluster (highest
                    mean correlation with the rest of its cluster). k questions, each still a readable question.
Only TopK uses labels.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, TransformerMixin


class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, idx=None):
        self.idx = idx

    def fit(self, X, y=None):
        self.idx_ = np.arange(X.shape[1]) if self.idx is None else np.asarray(self.idx)
        return self

    def transform(self, X):
        return X[:, self.idx_]


def cluster_representatives(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(chosen column indices in cluster order, cluster label of every column)."""
    sd = X.std(0)
    ok = sd > 1e-9
    Z = np.zeros_like(X, dtype=float)
    Z[:, ok] = (X[:, ok] - X[:, ok].mean(0)) / sd[ok]
    # Ward on columns scaled to unit length: squared distance = 2 (1 - r), i.e. Ward on correlation distance.
    lab = fcluster(linkage(Z.T / np.sqrt(len(Z)), method="ward"), k, criterion="maxclust")
    R = np.corrcoef(Z[:, ok].T)
    Rf = np.zeros((X.shape[1], X.shape[1]))
    Rf[np.ix_(ok, ok)] = R
    reps = []
    for c in np.unique(lab):
        mem = np.where(lab == c)[0]
        if len(mem) == 1:
            reps.append(mem[0])
            continue
        typ = (Rf[np.ix_(mem, mem)].sum(1) - 1) / (len(mem) - 1)
        reps.append(mem[int(np.argmax(typ))])
    return np.array(sorted(reps)), lab


class ClusterReps(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 27):
        self.k = k

    def fit(self, X, y=None):
        self.idx_, self.labels_ = cluster_representatives(np.asarray(X, float), self.k)
        return self

    def transform(self, X):
        return X[:, self.idx_]


class TopK(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 27):
        self.k = k

    def fit(self, X, y):
        from sklearn.feature_selection import f_classif

        F, _ = f_classif(X, y)
        F = np.nan_to_num(F, nan=-1.0)
        self.idx_ = np.sort(np.argsort(-F)[: self.k])
        return self

    def transform(self, X):
        return X[:, self.idx_]


class LassoK(BaseEstimator, TransformerMixin):
    """The k questions the lasso keeps first: L1 logistic regression (class-balanced, standardised answers) fitted
    along an increasing C until at least k questions have a non-zero weight; the k with the largest |weight| (largest
    over classes for multiclass tasks) are kept. Uses the training labels only. Unlike TopK, the lasso picks questions
    jointly, so near-duplicates do not crowd each other in."""

    def __init__(self, k: int = 60, seed: int = 0):
        self.k = k
        self.seed = seed

    def fit(self, X, y):
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X = np.asarray(X, float)
        sd = X.std(0)
        sd[sd < 1e-9] = 1
        Z = (X - X.mean(0)) / sd
        clf = LogisticRegression(l1_ratio=1.0, solver="saga", class_weight="balanced", max_iter=3000, tol=1e-3,
                                 warm_start=True, random_state=self.seed)
        W = None
        for C in np.logspace(-3, 0, 19):
            clf.set_params(C=C)
            clf.fit(Z, y)
            W = np.abs(clf.coef_).max(0)
            if (W > 1e-8).sum() >= self.k:
                break
        self.idx_ = np.sort(np.argsort(-W)[: self.k])
        return self

    def transform(self, X):
        return X[:, self.idx_]
