"""Penalty curves for one model family (S8 tune): every value of the grid, scored inside and outside each outer fold.

For each outer fold of the S6 design (same splits, same inner splits, same estimator and seeds as qaemb/models/cv.py),
and for every penalty value C on the grid:
  inner   fit on two thirds of the outer training set, score the other third, for each of the inner folds. These are
          the numbers a selection rule is allowed to see.
  outer   fit on the whole outer training set and score the outer test fold. Never used to choose C; kept so the curve
          on held-out images can be drawn next to the inner one, and so every rule's outer score comes from the same fits.
A selection rule then picks one C per outer fold from the inner numbers alone, and its outer score is read off.
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .cv import make_model, outer_splits

C_PARAM = {"l1": "clf__C", "l2": "clf__C"}


def inner_score(y, P, K) -> tuple[float, float]:
    """(log loss, primary score) on one validation fold: AUROC for two classes, balanced accuracy otherwise."""
    # Unclipped, exactly as GridSearchCV's neg_log_loss: clipping at 1e-7 changed the choice in 6 of 100 S6 folds,
    # where one confidently wrong image dominates an inner fold's log loss (2026-11-14).
    ll = float(log_loss(y, P, labels=list(range(K))))
    prim = float(roc_auc_score(y, P[:, 1])) if K == 2 else float(balanced_accuracy_score(y, P.argmax(1)))
    return ll, prim


def fit_one_C(name, C, X, y, groups, tr, te, inner_folds, seed):
    """Inner scores (one row per inner fold) and outer test-fold probabilities for one C and one outer fold."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    K = int(y.max()) + 1
    k = min(inner_folds, int(np.bincount(y[tr]).min()))
    # Same splitter and random_state as GridSearchCV in cv.fit_fold, so the inner folds are the ones S6 used.
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    inner = []
    for i, (a, b) in enumerate(cv.split(Xtr, ytr, gtr)):
        est, _ = make_model(name, seed)
        est.set_params(**{C_PARAM[name]: C})
        est.fit(Xtr[a], ytr[a])
        inner.append((i, *inner_score(ytr[b], est.predict_proba(Xtr[b]), K)))
    est, _ = make_model(name, seed)
    est.set_params(**{C_PARAM[name]: C})
    est.fit(Xtr, ytr)
    n_iter = int(np.max(est["clf"].n_iter_))
    return {"inner": inner, "P": est.predict_proba(X[te]), "n_iter": n_iter,
            "n_nonzero": int((np.abs(est["clf"].coef_) > 1e-8).any(0).sum())}


def penalty_curves(specs: dict, grid, ccfg, inner_folds, n_jobs) -> dict:
    """specs[key] = (model, X, y, strat, groups, seed). Returns per key: the outer splits and, per split and C, the
    fit_one_C result. All fits share one pool."""
    plan, jobs = [], []
    for key, (name, X, y, strat, groups, seed) in specs.items():
        splits = outer_splits(strat, groups, ccfg.outer_folds, ccfg.outer_repeats, seed)
        for si, (r, f, tr, te) in enumerate(splits):
            for C in grid:
                plan.append((key, si, C))
                jobs.append(delayed(fit_one_C)(name, C, X, y, groups, tr, te, inner_folds, seed + 1000 * r + f))
        specs[key] = (*specs[key], splits)
    print(f"  {len(jobs)} (outer fold x C) jobs, {inner_folds + 1} fits each (n_jobs={n_jobs})", flush=True)
    fits = Parallel(n_jobs=n_jobs, verbose=2)(jobs)
    out = {k: {"splits": specs[k][-1], "fits": {}} for k in specs}
    for (key, si, C), fit in zip(plan, fits):
        out[key]["fits"][(si, C)] = fit
    return out


def select(curve, rule: str) -> float:
    """Pick C from one outer fold's inner curve. curve: DataFrame with C, ll_mean, ll_se, prim_mean (one row per C).
      logloss  lowest mean inner log loss (what S6 does)
      primary  highest mean inner AUROC (2 classes) or balanced accuracy (more); ties go to the stronger penalty
      one_se   the strongest penalty whose mean log loss is within one standard error of the lowest"""
    c = curve.sort_values("C")
    if rule == "logloss":
        return float(c.C.values[np.argmin(c.ll_mean.values)])
    if rule == "primary":
        return float(c.C.values[np.argmax(c.prim_mean.values)])          # argmax takes the first = smallest C
    if rule == "one_se":
        i = int(np.argmin(c.ll_mean.values))
        ok = c.ll_mean.values <= c.ll_mean.values[i] + c.ll_se.values[i]
        return float(c.C.values[np.argmax(ok)])
    raise SystemExit(f"Unknown selection rule {rule!r}; choose from logloss, primary, one_se")
