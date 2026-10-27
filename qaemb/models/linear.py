"""Whole-image linear models (Track W).

W0: L2 logistic regression, C chosen by grouped CV on the training images.
W1: elastic-net logistic regression, (C, l1_ratio) by grouped CV (nested inside the outer split),
    then stability selection over half-subsamples of training groups -> selected question subset.
Features are standardised; class_weight='balanced'.
"""
import itertools
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make(C: float, l1_ratio: float = 0.0, seed: int = 0):
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, class_weight="balanced",
                                                              max_iter=5000, random_state=seed))


def cv_score(X, y, groups, C, l1, folds, seed) -> float:
    """Mean grouped-CV AUROC on the training set; NaN when there are too few samples per class."""
    k = min(folds, int(np.bincount(y).min()))
    if k < 2:
        return float("nan")
    scores = []
    for tr, va in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        scores.append(roc_auc_score(y[va], make(C, l1, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def select(X, y, groups, mcfg, seed, l1_grid) -> tuple[dict, list[dict]]:
    """Grid over (C, l1_ratio); returns the best params and the regularisation path rows."""
    path = []
    for C, l1 in itertools.product(mcfg.Cs, l1_grid):
        model = make(C, l1, seed).fit(X, y)
        coef = model[-1].coef_[0]
        path.append({"C": C, "l1_ratio": l1, "cv_auroc": cv_score(X, y, groups, C, l1, mcfg.cv_folds, seed),
                     "train_auroc": float(roc_auc_score(y, model.predict_proba(X)[:, 1])) if len(np.unique(y)) == 2 else float("nan"),
                     "n_nonzero": int((np.abs(coef) > 1e-8).sum())})
    scored = [r for r in path if not np.isnan(r["cv_auroc"])]
    if scored:
        best = max(scored, key=lambda r: (round(r["cv_auroc"], 6), -r["C"]))  # ties -> stronger regularisation
    else:  # too few samples for inner CV (smoke runs): middle of the grid
        best = path[len(path) // 2]
    return {"C": best["C"], "l1_ratio": best["l1_ratio"]}, path


def stability(X, y, groups, params, rounds, seed) -> np.ndarray:
    """Selection frequency of each feature over half-subsamples of training groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    hits, done = np.zeros(X.shape[1]), 0
    for _ in range(rounds):
        keep = np.isin(groups, rng.choice(ug, size=max(2, len(ug) // 2), replace=False))
        if len(np.unique(y[keep])) < 2:
            continue
        coef = make(params["C"], params["l1_ratio"], seed).fit(X[keep], y[keep])[-1].coef_[0]
        hits += np.abs(coef) > 1e-8
        done += 1
    return hits / max(done, 1)


class LinearModel:
    def __init__(self, mcfg, seed: int):
        self.mcfg, self.seed = mcfg, seed
        self.l1_grid = [0.0] if mcfg.name == "W0" else list(mcfg.l1_ratios)

    def fit(self, X, y, groups, X_val=None, y_val=None):
        self.params, self.path = select(X, y, groups, self.mcfg, self.seed, self.l1_grid)
        if X_val is not None and len(np.unique(y_val)) == 2:
            for row in self.path:
                m = make(row["C"], row["l1_ratio"], self.seed).fit(X, y)
                row["val_auroc"] = float(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        self.model = make(self.params["C"], self.params["l1_ratio"], self.seed).fit(X, y)
        self.coef = self.model[-1].coef_[0]
        if self.mcfg.name == "W1":
            self.stability = stability(X, y, groups, self.params, self.mcfg.stability_rounds, self.seed)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
