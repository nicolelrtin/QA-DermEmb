"""S14 Thresholds: which P(malignant) cutoff to use, and how much the result depends on it.

AUROC needs no cutoff, but any statement about sensitivity or specificity does. Until now the specificity at 95%
sensitivity was read at the cutoff that reaches 95% on the test fold itself, i.e. chosen after seeing the test labels.
This stage asks what happens when the cutoff is fixed beforehand, as it must be on the external test set.

Model: S10's main model, ridge on the Directional Score of the 446 clinical questions, benign against malignant, folds
grouped by patient (5 repeats of 5-fold). The direct question ("Is this a malignant lesion?", Sonnet's P(yes)) is
treated the same way for reference.

1. S10's 25 outer folds are rebuilt (same stratification, groups and seeds) and each outer model refitted with the
   penalty S10 chose for that fold; the test-fold predictions are checked against S10's stored ones.
2. Rules fixed in advance, each turned into a P cutoff inside the training split only: cross-validated predictions
   on the training images (inner folds grouped by patient, same penalty), and the cutoff read from those.
     p50      P >= 0.5 (no tuning)
     youden   the cutoff maximising sensitivity + specificity on the training predictions
     sens90 / sens95 / sens975 / sens99   the highest cutoff that catches at least that share of the training
              malignancies (rule-out cutoffs)
     spec90 / spec95   the lowest cutoff that clears at least that share of the training benign lesions (rule-in)
   The cutoff is then applied unchanged to the test fold. For the sensitivity rules the hindsight version (cutoff
   read on the test fold, as reported so far) is kept alongside.
3. A sweep of fixed cutoffs (0.05 to 0.95): sensitivity, specificity, PPV, NPV, accuracy and the share of images
   called malignant on every test fold.
All scores per test fold, then mean, SD and range over the 25 folds; on all images and on dermoscopic images only.
PPV and NPV depend on prevalence: here 300 malignant of 800 (37.5%).

Outputs: rules_folds.csv, rules_summary.csv, sweep_folds.csv, sweep_summary.csv, reproduction.csv, figures/, summary.md.
"""
import warnings

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models.cv import outer_splits
from .s5_model import feature_mask
from .s6_classify import load_labels
from .s10_ridge import make_pipe

UPSTREAM = ["manifest", "embed", "ridge"]
VERSION = 1
RULES = [("p50", "P ≥ 0.5"), ("youden", "Youden (max sens + spec)"), ("sens90", "90% sensitivity"),
         ("sens95", "95% sensitivity"), ("sens975", "97.5% sensitivity"), ("sens99", "99% sensitivity"),
         ("spec90", "90% specificity"), ("spec95", "95% specificity")]
METRICS = ["cutoff", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1", "called_malignant"]


def key(cfg, ctx) -> dict:
    return {"thresholds": cfg.thresholds, "outer": [cfg.classify.outer_folds, cfg.classify.outer_repeats, cfg.classify.inner_folds],
            "label_columns": cfg.classify.label_columns, "exclude_kinds": cfg.model.exclude_kinds, "seed": cfg.seed,
            "source": cfg.ridge.source, "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "metadata_sha": file_hash(ROOT / cfg.explore.metadata)}


# ------------------------------------------------------------------ cutoffs
def cutoff_for(rule, y, p):
    """P cutoff for a rule, read from scores p with labels y (1 = malignant). An image is called malignant if p >= cutoff."""
    if rule == "p50":
        return 0.5
    s = np.unique(p)[::-1]                                     # candidate cutoffs, high to low
    pos, neg = p[y == 1], p[y == 0]
    sens = np.array([(pos >= c).mean() for c in s])
    spec = np.array([(neg < c).mean() for c in s])
    if rule == "youden":
        return float(s[np.argmax(sens + spec)])
    if rule.startswith("sens"):
        t = {"sens90": .90, "sens95": .95, "sens975": .975, "sens99": .99}[rule]
        return float(s[np.argmax(sens >= t)])                  # highest cutoff reaching the target
    if rule.startswith("spec"):
        t = {"spec90": .90, "spec95": .95}[rule]
        ok = np.where(spec >= t)[0]
        return float(s[ok[-1]]) if len(ok) else float(s[0])    # lowest cutoff still clearing the target
    raise ValueError(rule)


def scores_at(y, p, c) -> dict:
    call = p >= c
    tp, fn = int((call & (y == 1)).sum()), int((~call & (y == 1)).sum())
    tn, fp = int((~call & (y == 0)).sum()), int((call & (y == 0)).sum())
    d = lambda a, b: a / b if b else np.nan
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    ppv = d(tp, tp + fp)
    return {"cutoff": float(c), "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": d(tn, tn + fn),
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2,
            "f1": d(2 * tp, 2 * tp + fp + fn), "called_malignant": float(call.mean()), "n": int(len(y)), "n_malignant": int(y.sum())}


def inner_predictions(Xtr, ytr, gtr, C, seed, folds):
    """Cross-validated P(malignant) on the training images: inner folds grouped by patient, the outer fold's penalty."""
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold

    p = np.full(len(ytr), np.nan)
    pipe = make_pipe(("cols", None), seed).set_params(clf__C=C)
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(Xtr, ytr, gtr):
        p[te] = clone(pipe).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[te])[:, 1]
    return p


def summarise(df, by, metrics=METRICS):
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sd"] = (m, "std")
        agg[f"{m}_min"] = (m, "min")
        agg[f"{m}_max"] = (m, "max")
    return df.groupby(by, sort=False).agg(**agg).reset_index()


# ------------------------------------------------------------------ main
def run(cfg, out, inputs, ctx) -> dict:
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    from joblib import Parallel, delayed
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    tc, cc = cfg.thresholds, cfg.classify
    (out / "figures").mkdir(exist_ok=True)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    meta = pd.read_csv(ROOT / cfg.explore.metadata, dtype={"patient_id": str, "lesion_id": str}).set_index("isic_id")
    derm_all = meta.image_type.reindex(ids).eq("dermoscopic").values
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qids = list(qt[feature_mask(qt, cfg.model, patch=False)].question_id)
    X_all = pd.read_parquet(inputs["embed"] / f"X_image_{cfg.ridge.source}.parquet").loc[ids, qids].values.astype(float)
    ctrl_all = pd.read_parquet(inputs["embed"] / "X_image_orig.parquet").loc[ids, "DQC01"].values.astype(float)
    groups_all = pd.read_csv(inputs["ridge"] / "groups.csv").set_index("image_id").group.reindex(ids).values
    rows = lab.c3.isin(["benign", "malignant"]).values
    X, y, g = X_all[rows], (lab.c3.values[rows] == "malignant").astype(int), groups_all[rows]
    ctrl, derm, rid = ctrl_all[rows], derm_all[rows], np.array(ids)[rows]
    folds_tab = pd.read_csv(inputs["ridge"] / "metrics_folds.csv")
    Cs = folds_tab[(folds_tab.task == "s1_2c") & (folds_tab.set == "all") & (folds_tab.grouping == "patient")
                   & (folds_tab.subset == "all images")].set_index(["repeat", "fold"]).C
    oof = pd.read_parquet(inputs["ridge"] / "oof_predictions.parquet")
    oof = oof[(oof.task == "s1_2c") & (oof.set == "all") & (oof.grouping == "patient")]
    splits = outer_splits(lab.c10.values[rows], g, cc.outer_folds, cc.outer_repeats, cfg.seed)

    def one(r, f, tr, te):
        seed = cfg.seed + 1000 * r + f
        C = float(Cs.loc[(r, f)])
        pipe = make_pipe(("cols", None), seed).set_params(clf__C=C).fit(X[tr], y[tr])
        return {"r": r, "f": f, "te": te, "tr": tr, "C": C, "p_te": pipe.predict_proba(X[te])[:, 1],
                "p_tr_cv": inner_predictions(X[tr], y[tr], g[tr], C, seed, cc.inner_folds)}

    print(f"  refitting {len(splits)} outer folds with S10's penalties, {cc.inner_folds} inner fits each", flush=True)
    fits = Parallel(n_jobs=tc.n_jobs)(delayed(one)(*s) for s in splits)

    # 1. reproduction of S10's out-of-fold predictions
    rep = []
    for F in fits:
        s10 = oof[oof.repeat == F["r"]].set_index("image_id")["p::malignant"].reindex(rid[F["te"]]).values
        rep.append({"repeat": F["r"], "fold": F["f"], "n_test": len(F["te"]), "max_abs_diff": float(np.nanmax(np.abs(s10 - F["p_te"])))})
    rep = pd.DataFrame(rep)
    rep.to_csv(out / "reproduction.csv", index=False)
    worst = rep.max_abs_diff.max()
    print(f"  reproduction of S10 test predictions: largest difference {worst:.2e}", flush=True)
    if worst > tc.reproduce_tol:
        raise SystemExit(f"S14 could not reproduce S10's out-of-fold predictions (largest difference {worst:.3g}); "
                         "the folds or the fitting differ, so the cutoffs would not describe the S10 model.")

    # 2. rules (cutoff from the training split) and 3. sweep, on the ridge model and on the direct question
    sweep_c = np.round(np.arange(tc.sweep[0], tc.sweep[1] + 1e-9, tc.sweep[2]), 4)
    rule_rows, sweep_rows = [], []
    for F in fits:
        tr, te = F["tr"], F["te"]
        for score, p_tr, p_te in (("ridge", F["p_tr_cv"], F["p_te"]), ("direct question", ctrl[tr], ctrl[te])):
            for sub, m in (("all images", np.ones(len(te), bool)), ("dermoscopic only", derm[te])):
                base = {"score": score, "subset": sub, "repeat": F["r"], "fold": F["f"]}
                for rule, _ in RULES:
                    c = cutoff_for(rule, y[tr], p_tr)
                    rule_rows.append({**base, "rule": rule, "chosen_on": "training split", **scores_at(y[te][m], p_te[m], c)})
                    if rule.startswith("sens") or rule == "youden":
                        ch = cutoff_for(rule, y[te][m], p_te[m])
                        rule_rows.append({**base, "rule": rule, "chosen_on": "test fold (hindsight)", **scores_at(y[te][m], p_te[m], ch)})
                for c in sweep_c:
                    sweep_rows.append({**base, **scores_at(y[te][m], p_te[m], c)})
    rf = pd.DataFrame(rule_rows)
    sf = pd.DataFrame(sweep_rows)
    rf.to_csv(out / "rules_folds.csv", index=False)
    sf.to_csv(out / "sweep_folds.csv", index=False)
    rs = summarise(rf, ["score", "subset", "rule", "chosen_on"])
    rs.insert(3, "rule_label", rs.rule.map(dict(RULES)))
    ss = summarise(sf, ["score", "subset", "cutoff"], METRICS[1:])
    rs.to_csv(out / "rules_summary.csv", index=False)
    ss.to_csv(out / "sweep_summary.csv", index=False)

    figure(ss, rs, out / "figures" / "cutoff_sweep.png")
    write_summary(out, rs, ss, rep, len(fits), int(y.sum()), len(y))
    main = rs[(rs.score == "ridge") & (rs.subset == "all images") & (rs.rule == "sens95")].set_index("chosen_on")
    return {"n_folds": len(fits), "reproduction_max_diff": float(worst),
            "key_metric": (f"sens95 cutoff from training: sens {main.loc['training split', 'sensitivity_mean']:.3f}, "
                           f"spec {main.loc['training split', 'specificity_mean']:.3f} (hindsight spec "
                           f"{main.loc['test fold (hindsight)', 'specificity_mean']:.3f})")}


def figure(ss, rs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = {"sensitivity": "#e34948", "specificity": "#2a78d6", "ppv": "#94762c", "npv": "#1baf7a"}
    for ax, score in zip(axes, ["ridge", "direct question"]):
        d = ss[(ss.score == score) & (ss.subset == "all images")]
        for m, c in cols.items():
            ax.plot(d.cutoff, d[f"{m}_mean"], color=c, lw=2, label=m.upper() if len(m) == 3 else m)
            ax.fill_between(d.cutoff, d[f"{m}_min"], d[f"{m}_max"], color=c, alpha=.15, lw=0)
        r = rs[(rs.score == score) & (rs.subset == "all images") & (rs.chosen_on == "training split")].set_index("rule")
        for rule, ls in (("sens95", "--"), ("youden", ":")):
            ax.axvline(r.loc[rule, "cutoff_mean"], color="#4d4840", ls=ls, lw=1)
        ax.set_title(score)
        ax.set_xlabel("P cutoff (called malignant if P ≥ cutoff)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("mean over 25 test folds (band: range)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(out, rs, ss, rep, n_folds, n_mal, n):
    def f(v, d=2):
        return "n/a" if pd.isna(v) else f"{v:.{d}f}"

    L = ["# S14 thresholds: P(malignant) cutoffs chosen in the training split", "",
         f"Ridge on all 446 questions (S10 main model) and the direct question, benign against malignant, {n} images "
         f"({n_mal} malignant), {n_folds} outer folds. S10's test predictions reproduced to within {rep.max_abs_diff.max():.1e}.",
         "Each cutoff is read from cross-validated predictions on the training split and applied unchanged to the test fold; "
         "the hindsight rows read it on the test fold itself (as the specificity at 95% sensitivity was reported so far).",
         "Mean over the 25 test folds, with the range in brackets.", ""]
    for score in ["ridge", "direct question"]:
        for sub in ["all images", "dermoscopic only"]:
            d = rs[(rs.score == score) & (rs.subset == sub)]
            L += [f"## {score}, {sub}", "",
                  "| rule | cutoff chosen on | P cutoff | sensitivity | specificity | PPV | NPV | accuracy | called malignant |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for x in d.itertuples():
                L.append(f"| {x.rule_label} | {x.chosen_on} | {f(x.cutoff_mean)} ({f(x.cutoff_min)}–{f(x.cutoff_max)}) | "
                         f"{f(x.sensitivity_mean)} ({f(x.sensitivity_min)}–{f(x.sensitivity_max)}) | "
                         f"{f(x.specificity_mean)} ({f(x.specificity_min)}–{f(x.specificity_max)}) | {f(x.ppv_mean)} | "
                         f"{f(x.npv_mean)} | {f(x.accuracy_mean)} | {f(x.called_malignant_mean)} |")
            L.append("")
    L += ["## Fixed cutoffs, ridge, all images", "", "| P cutoff | sensitivity | specificity | PPV | NPV | accuracy | called malignant |",
          "|---|---|---|---|---|---|---|"]
    for x in ss[(ss.score == "ridge") & (ss.subset == "all images")].itertuples():
        L.append(f"| {x.cutoff:.2f} | {f(x.sensitivity_mean)} ({f(x.sensitivity_min)}–{f(x.sensitivity_max)}) | "
                 f"{f(x.specificity_mean)} ({f(x.specificity_min)}–{f(x.specificity_max)}) | {f(x.ppv_mean)} | {f(x.npv_mean)} | "
                 f"{f(x.accuracy_mean)} | {f(x.called_malignant_mean)} |")
    L += ["", "PPV and NPV depend on the share of malignant images (37.5% here) and will differ in other settings.",
          "", "![cutoff sweep](figures/cutoff_sweep.png)", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
