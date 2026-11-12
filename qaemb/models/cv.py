"""Nested cross-validation for the whole-image classifiers (S6).

Model families (all scikit-learn; features standardised for the linear models and the MLP):
  l1   L1 (lasso) logistic regression        -- drops questions; selection counts come from here
  l2   L2 (ridge) logistic regression        -- keeps every question; ranks them by weight
  enet elastic-net logistic regression       -- in between, for correlated questions
  rf   random forest
  hgb  histogram gradient-boosted trees
  mlp  one-hidden-layer perceptron (Adam, L2 penalty, no early stopping)
Outer loop: repeated stratified group k-fold (stratified on the finest label, grouped by lesion). Inner loop: grid
search by log loss on the outer training set only. Every prediction we score is out-of-fold.

Per outer fit we keep: test-fold probabilities, probabilities for "extra" images outside the task (e.g. indeterminate
images for the 2-class model, other groups for the stage-2 models), the chosen hyperparameters, standardised
coefficients (linear models) and, on the first `shap_repeats` repeats, mean |SHAP| per question on the test fold.
SHAP units: log-odds for the linear models and hgb, probability for rf and mlp, so compare SHAP across models by rank.
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LINEAR = ("l1", "l2", "enet")
TREES = ("rf", "hgb")
# The logistic models and hgb are fitted with balanced class weights, so their probabilities are for equal class
# priors. The random forest's leaf probabilities still follow the training class mix (on simulated answers it called
# 933 of 1,000 images benign in the 3-class task, 2026-11-05) and the MLP has no class weights, so their probabilities
# are divided by the training class shares and renormalised. Every model's decisions then treat the classes equally.
PRIOR_CORRECT = ("rf", "mlp")


def make_model(name: str, seed: int):
    """(estimator, parameter grid) for a model family."""
    def logit(l1_ratio):
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(l1_ratio=l1_ratio, solver="saga", class_weight="balanced",
                                                    max_iter=5000, tol=1e-3, random_state=seed))])

    if name == "l1":
        return logit(1.0), {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
    if name == "l2":
        return logit(0.0), {"clf__C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]}
    if name == "enet":
        return logit(0.5), {"clf__C": [0.01, 0.03, 0.1, 0.3], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    if name == "rf":
        return (RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
                {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 5]})
    if name == "hgb":
        return (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=True,
                                               class_weight="balanced", random_state=seed),
                {"max_leaf_nodes": [7, 15], "min_samples_leaf": [10, 30]})
    if name == "mlp":
        # No early stopping: with a 10% validation split of a few hundred images it stopped almost at once and
        # scored at chance on simulated answers (2026-11-05). Strong L2 penalties instead.
        return (Pipeline([("scale", StandardScaler()),
                          ("clf", MLPClassifier(early_stopping=False, max_iter=500, random_state=seed))]),
                {"clf__hidden_layer_sizes": [(32,), (64,)], "clf__alpha": [1.0, 10.0, 30.0]})
    raise SystemExit(f"Unknown model {name!r}; choose from l1, l2, enet, rf, hgb, mlp")


def outer_splits(strat, groups, folds: int, repeats: int, seed: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    k = min(folds, int(np.bincount(np.unique(strat, return_inverse=True)[1]).min()))
    out = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(strat)), strat, groups)):
            out.append((r, f, tr, te))
    return out


def _as3d(v, n_classes: int) -> np.ndarray:
    """SHAP values -> [n, M, K'] with K' = 1 (positive class) for binary tasks."""
    v = np.asarray(v)
    if v.ndim == 2:
        v = v[:, :, None]
    if n_classes == 2 and v.shape[2] == 2:
        v = v[:, :, 1:]
    return v


def shap_abs(name, est, X_tr, X_te, n_classes, background, seed) -> np.ndarray:
    """Mean |SHAP| per (output, question) on the test fold: [K' x M]."""
    if name in LINEAR:
        sc, clf = est["scale"], est["clf"]
        Z = sc.transform(X_te) - sc.transform(X_tr).mean(0)
        v = Z[:, :, None] * clf.coef_.T[None]          # exact for a linear model with independent features
    else:
        import shap

        if name in TREES:
            v = shap.TreeExplainer(est).shap_values(X_te)
        else:
            rng = np.random.default_rng(seed)
            bg = X_tr[rng.choice(len(X_tr), min(background, len(X_tr)), replace=False)]
            ex = shap.PermutationExplainer(est.predict_proba, shap.maskers.Independent(bg), seed=seed)
            v = ex(X_te, max_evals=2 * X_te.shape[1] + 1, silent=True).values
    return np.abs(_as3d(v, n_classes)).mean(0).T


def fit_fold(name, X, y, groups, tr, te, X_extra, inner_folds, seed, want_shap, background):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    est, grid = make_model(name, seed)
    k = min(inner_folds, int(np.bincount(y[tr]).min()))
    gs = GridSearchCV(est, grid, scoring="neg_log_loss", cv=StratifiedGroupKFold(k, shuffle=True, random_state=seed),
                      n_jobs=1, refit=True, error_score="raise")
    gs.fit(X[tr], y[tr], groups=groups[tr])
    best = gs.best_estimator_
    prior = np.bincount(y[tr], minlength=int(y.max()) + 1) / len(tr)

    def proba(Z):
        P = best.predict_proba(Z)
        if name in PRIOR_CORRECT:                     # to equal class priors, like the class-weighted models
            P = P / prior
            P = P / P.sum(1, keepdims=True)
        return P

    out = {"te": te, "P": proba(X[te]), "params": gs.best_params_,
           "P_extra": proba(X_extra) if X_extra is not None and len(X_extra) else None}
    if name in LINEAR:
        out["coef"] = best["clf"].coef_.copy()       # [K' x M], standardised scale
    if want_shap:
        out["shap"] = shap_abs(name, best, X[tr], X[te], len(np.unique(y)), background, seed)
    return out


def task_jobs(name, X, y, strat, groups, X_extra, ccfg, seed) -> tuple[list, list]:
    """The outer splits of one (model, task) and one delayed fit per split, to be run in one shared pool."""
    splits = outer_splits(strat, groups, ccfg.outer_folds, ccfg.outer_repeats, seed)
    jobs = [delayed(fit_fold)(name, X, y, groups, tr, te, X_extra, ccfg.inner_folds, seed + 1000 * r + f,
                              r < ccfg.shap_repeats, ccfg.shap_background) for r, f, tr, te in splits]
    return splits, jobs


def assemble(y, n_extra, splits, fits, ccfg) -> dict:
    """Collect the outer fits of one (model, task): out-of-fold probabilities, extra-image probabilities averaged over
    the folds of each repeat, coefficients, selection counts and mean |SHAP|."""
    K = int(y.max()) + 1
    R = ccfg.outer_repeats
    oof = np.full((R, len(y), K), np.nan)
    extra = np.zeros((R, n_extra, K)) if n_extra else None
    folds_per_rep = np.zeros(R)
    fold_rows, coefs, shap_sum, shap_n = [], [], None, 0
    for (r, f, tr, te), fit in zip(splits, fits):
        oof[r, te] = fit["P"]
        if extra is not None:
            extra[r] += fit["P_extra"]
            folds_per_rep[r] += 1
        fold_rows.append({"repeat": r, "fold": f, "te": te, "params": fit["params"]})
        if "coef" in fit:
            coefs.append(fit["coef"])
        if "shap" in fit:
            w = len(te)
            shap_sum = fit["shap"] * w if shap_sum is None else shap_sum + fit["shap"] * w
            shap_n += w
    if extra is not None:
        extra /= folds_per_rep[:, None, None]
    coefs = np.stack(coefs) if coefs else None                              # [fits x K' x M]
    return {
        "oof": oof, "extra": extra, "folds": fold_rows,
        "coef": coefs,
        "selected": (np.abs(coefs) > 1e-8).any(1).sum(0) if coefs is not None else None,   # fits keeping each question
        "n_fits": len(splits),
        "shap": shap_sum / shap_n if shap_sum is not None else None,
    }


COST = {"enet": 0, "rf": 1, "mlp": 2, "l1": 3, "hgb": 4, "l2": 5}   # slowest first, for load balancing


def run_all(specs: dict, ccfg) -> dict:
    """specs[key] = (model, X, y, strat, groups, X_extra, seed). Runs every outer fit of every key in one pool."""
    plan, jobs = [], []
    for k in sorted(specs, key=lambda k: (COST.get(specs[k][0], 9), -len(specs[k][2]))):
        name, X, y, strat, groups, X_extra, seed = specs[k]
        splits, js = task_jobs(name, X, y, strat, groups, X_extra, ccfg, seed)
        plan.append((k, splits, len(jobs), len(js)))
        jobs += js
    print(f"  {len(jobs)} outer fits in one pool (n_jobs={ccfg.n_jobs})", flush=True)
    fits = Parallel(n_jobs=ccfg.n_jobs, verbose=2)(jobs)
    out = {}
    for k, splits, a, n in plan:
        y, X_extra = specs[k][2], specs[k][5]
        res = assemble(y, len(X_extra) if X_extra is not None else 0, splits, fits[a:a + n], ccfg)
        if res["shap"] is None:
            K = int(y.max()) + 1
            res["shap"] = np.full((1 if K == 2 else K, specs[k][1].shape[1]), np.nan)
        out[k] = res
    return out
