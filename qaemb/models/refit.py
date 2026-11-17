"""Refitting S10's outer folds with the penalty S10 chose for each fold (used by S14 and S9).

S10 stores out-of-fold predictions and the chosen penalty per outer fold, but not the fold assignment or the fitted
fold models. The folds are rebuilt here exactly as S10 built them (outer_splits on the 10-subtype labels of the task's
images, grouped by patient, same seeds), and each fold model refitted on its training split with S10's penalty. The
grid search in S10 refits the best penalty on the same training split with the same seed, so the refitted model is the
one S10 scored; reproduction() checks this against S10's stored test-fold predictions.
"""
import warnings

import numpy as np
import pandas as pd

from .cv import outer_splits


def fold_penalties(ridge_dir, task: str) -> pd.Series:
    """S10's chosen C per (repeat, fold) for a task, all 446 questions, folds grouped by patient."""
    f = pd.read_csv(ridge_dir / "metrics_folds.csv")
    f = f[(f.task == task) & (f.set == "all") & (f.grouping == "patient") & (f.subset == "all images")]
    return f.set_index(["repeat", "fold"]).C.astype(float)


def s10_splits(strat, groups, cc, seed):
    """S10's outer folds for a task: strat = 10-subtype labels of the task's images, groups = S10's patient groups."""
    return outer_splits(strat, groups, cc.outer_folds, cc.outer_repeats, seed)


def inner_predictions(Xtr, ytr, gtr, C, seed, folds):
    """Cross-validated class probabilities [n x K] on the training images: inner folds grouped by patient and stratified
    on ytr, the outer fold's penalty. Used to choose cutoffs without touching the test fold."""
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold

    from ..stages.s10_ridge import make_pipe

    P = np.full((len(ytr), len(np.unique(ytr))), np.nan)
    pipe = make_pipe(("cols", None), seed).set_params(clf__C=C)
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(Xtr, ytr, gtr):
        P[te] = clone(pipe).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[te])
    return P


def refit_outer(X, y, groups, splits, Cs, base_seed, n_jobs, inner_folds=None, keep_model=False):
    """Refit every outer fold of S10 (ridge on all questions) with S10's penalty for that fold. Returns one dict per fold:
    r, f, tr, te, C, P_te (test-fold class probabilities), and P_tr_cv (inner cross-validated probabilities on the
    training split) when inner_folds is given, model when keep_model."""
    from joblib import Parallel, delayed

    from ..stages.s10_ridge import make_pipe

    def one(r, f, tr, te):
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        seed = base_seed + 1000 * r + f
        C = float(Cs.loc[(r, f)])
        pipe = make_pipe(("cols", None), seed).set_params(clf__C=C).fit(X[tr], y[tr])
        d = {"r": r, "f": f, "te": te, "tr": tr, "C": C, "P_te": pipe.predict_proba(X[te])}
        if inner_folds:
            d["P_tr_cv"] = inner_predictions(X[tr], y[tr], groups[tr], C, seed, inner_folds)
        if keep_model:
            d["model"] = pipe
        return d

    return Parallel(n_jobs=n_jobs)(delayed(one)(*s) for s in splits)


def reproduction(fits, oof, row_ids, cols) -> pd.DataFrame:
    """Largest absolute difference, per outer fold, between the refitted test-fold probabilities and S10's stored ones
    (oof: S10's out-of-fold rows of one task, set 'all', patient grouping; cols: its probability columns, in the order
    of the refitted model's classes)."""
    rep = []
    for F in fits:
        s10 = oof[oof.repeat == F["r"]].set_index("image_id")[cols].reindex(row_ids[F["te"]]).values
        rep.append({"repeat": F["r"], "fold": F["f"], "n_test": len(F["te"]),
                    "max_abs_diff": float(np.nanmax(np.abs(s10 - F["P_te"]))), "n_missing": int(np.isnan(s10).any(1).sum())})
    return pd.DataFrame(rep)
