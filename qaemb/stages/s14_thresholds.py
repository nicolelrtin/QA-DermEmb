"""S14 Thresholds: which P(malignant) cutoff to use, and how much the result depends on it.

AUROC needs no cutoff, but any statement about sensitivity or specificity does. Until now the specificity at 95%
sensitivity was read at the cutoff that reaches 95% on the test fold itself, i.e. chosen after seeing the test labels.
This stage asks what happens when the cutoff is fixed beforehand, as it must be on the external test set.

Scores (score column):
  ridge            S10's two-class model, ridge on the Directional Score of the 446 clinical questions, benign against
                   malignant, folds grouped by patient (5 repeats of 5-fold), P(malignant).
  tenclass         S10's ten-subtype model (task flat_10c, the model the paper leads with), ridge on the same 446
                   questions trained on all 1,000 images including the indeterminate ones, read as benign against
                   malignant with S11's collapse_malignant (malignant subtypes' summed probability over their number,
                   against the benign subtypes' over theirs, renormalised so 0.5 = equal prior odds).
  direct question  Sonnet's P(yes) to "Is this a malignant lesion?", on the two-class folds, for reference.

1. S10's 25 outer folds are rebuilt for each task (same stratification, groups and seeds; qaemb/models/refit.py) and
   each outer model refitted with the penalty S10 chose for that fold; the test-fold predictions are checked against
   S10's stored ones (all ten probabilities for the ten-subtype model). The ten-subtype folds are S10's flat_10c folds,
   built over all 1,000 images, so they are not the two-class folds: tenclass rows are not paired fold by fold with the
   ridge and direct-question rows.
2. Rules fixed in advance, each turned into a P cutoff inside the training split only: cross-validated predictions
   on the training images (inner folds grouped by patient, same penalty), and the cutoff read from those. For the
   ten-subtype model the inner models are trained on all training images (as S10 trains it, inner folds stratified on
   the ten subtypes) and the cutoff is read from the collapsed scores of the benign and malignant training images only.
     p50      P >= 0.5 (no tuning)
     youden   the cutoff maximising sensitivity + specificity on the training predictions
     sens90 / sens95 / sens975 / sens99   the highest cutoff that catches at least that share of the training
              malignancies (rule-out cutoffs)
     spec90 / spec95   the lowest cutoff that clears at least that share of the training benign lesions (rule-in)
   The cutoff is then applied unchanged to the test fold. For the sensitivity rules the hindsight version (cutoff
   read on the test fold, as reported so far) is kept alongside.
3. A sweep of fixed cutoffs (0.05 to 0.95): sensitivity, specificity, PPV, NPV, accuracy and the share of images
   called malignant on every test fold.
All scores per test fold (benign and malignant test images only; the indeterminate ones have no benign/malignant
label), then mean, SD and range over the 25 folds; on all images and on dermoscopic images only.
PPV and NPV depend on prevalence: here 300 malignant of 800 (37.5%).

Outputs: rules_folds.csv, rules_summary.csv, sweep_folds.csv, sweep_summary.csv, reproduction.csv (task column: s1_2c,
flat_10c), figures/, summary.md.
"""

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models.refit import fold_penalties, refit_outer, reproduction, s10_splits
from .s5_model import feature_mask
from .s6_classify import define_tasks, load_labels
from .s11_subtypes import collapse_malignant

UPSTREAM = ["manifest", "embed", "ridge"]
VERSION = 2    # 2: + the ten-subtype model read as benign vs malignant (score 'tenclass')
SCORES = ["ridge", "tenclass", "direct question"]
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
    oof_all = pd.read_parquet(inputs["ridge"] / "oof_predictions.parquet")
    oof_all = oof_all[(oof_all.set == "all") & (oof_all.grouping == "patient")]
    all_ids = np.array(ids)

    # two-class model: the 800 benign and malignant images
    rows = lab.c3.isin(["benign", "malignant"]).values
    X, y, g = X_all[rows], (lab.c3.values[rows] == "malignant").astype(int), groups_all[rows]
    ctrl, derm, rid = ctrl_all[rows], derm_all[rows], all_ids[rows]
    splits = s10_splits(lab.c10.values[rows], g, cc, cfg.seed)
    print(f"  two classes: refitting {len(splits)} outer folds with S10's penalties, {cc.inner_folds} inner fits each", flush=True)
    fits = refit_outer(X, y, g, splits, fold_penalties(inputs["ridge"], "s1_2c"), cfg.seed, tc.n_jobs, cc.inner_folds)

    # ten-subtype model: all 1,000 images (S10 trains flat_10c on every image, indeterminate included)
    T10 = define_tasks(lab, ["flat_10c"])["flat_10c"]
    classes = T10["classes"]
    grp3 = {c: lab.c3[lab.c10 == c].iloc[0] for c in classes}
    y10 = np.array([classes.index(v) for v in lab.c10.values])
    bm_all = lab.c3.isin(["benign", "malignant"]).values
    yb_all = (lab.c3.values == "malignant").astype(int)
    splits10 = s10_splits(lab.c10.values, groups_all, cc, cfg.seed)
    print(f"  ten subtypes: refitting {len(splits10)} outer folds with S10's penalties, {cc.inner_folds} inner fits each", flush=True)
    fits10 = refit_outer(X_all, y10, groups_all, splits10, fold_penalties(inputs["ridge"], "flat_10c"), cfg.seed, tc.n_jobs,
                         cc.inner_folds)

    # 1. reproduction of S10's out-of-fold predictions
    rep = pd.concat([reproduction(fits, oof_all[oof_all.task == "s1_2c"], rid, ["p::benign", "p::malignant"]).assign(task="s1_2c"),
                     reproduction(fits10, oof_all[oof_all.task == "flat_10c"], all_ids, [f"p::{c}" for c in classes])
                     .assign(task="flat_10c")], ignore_index=True)
    rep.to_csv(out / "reproduction.csv", index=False)
    worst = rep.groupby("task").max_abs_diff.max()
    print(f"  reproduction of S10 test predictions: largest difference {worst.to_dict()}", flush=True)
    if worst.max() > tc.reproduce_tol or rep.n_missing.any():
        raise SystemExit(f"S14 could not reproduce S10's out-of-fold predictions (largest difference {worst.to_dict()}); "
                         "the folds or the fitting differ, so the cutoffs would not describe the S10 model.")

    # 2. rules (cutoff from the training split) and 3. sweep, on the three scores
    sweep_c = np.round(np.arange(tc.sweep[0], tc.sweep[1] + 1e-9, tc.sweep[2]), 4)
    # (score, repeat, fold, training labels, training scores, test labels, test scores, test dermoscopic)
    units = []
    for F in fits:
        tr, te = F["tr"], F["te"]
        units.append(("ridge", F, y[tr], F["P_tr_cv"][:, 1], y[te], F["P_te"][:, 1], derm[te]))
        units.append(("direct question", F, y[tr], ctrl[tr], y[te], ctrl[te], derm[te]))
    for F in fits10:
        tr, te = F["tr"][bm_all[F["tr"]]], F["te"][bm_all[F["te"]]]          # benign and malignant images only
        p_tr = collapse_malignant(F["P_tr_cv"], classes, grp3)[bm_all[F["tr"]]]
        p_te = collapse_malignant(F["P_te"], classes, grp3)[bm_all[F["te"]]]
        units.append(("tenclass", F, yb_all[tr], p_tr, yb_all[te], p_te, derm_all[te]))
    order = {s_: k for k, s_ in enumerate(SCORES)}
    units.sort(key=lambda u: order[u[0]])
    rule_rows, sweep_rows = [], []
    for score, F, y_tr, p_tr, y_te, p_te, d_te in units:
        for sub, m in (("all images", np.ones(len(y_te), bool)), ("dermoscopic only", d_te)):
            base = {"score": score, "subset": sub, "repeat": F["r"], "fold": F["f"]}
            for rule, _ in RULES:
                c = cutoff_for(rule, y_tr, p_tr)
                rule_rows.append({**base, "rule": rule, "chosen_on": "training split", **scores_at(y_te[m], p_te[m], c)})
                if rule.startswith("sens") or rule == "youden":
                    ch = cutoff_for(rule, y_te[m], p_te[m])
                    rule_rows.append({**base, "rule": rule, "chosen_on": "test fold (hindsight)", **scores_at(y_te[m], p_te[m], ch)})
            for c in sweep_c:
                sweep_rows.append({**base, **scores_at(y_te[m], p_te[m], c)})
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
    ten = rs[(rs.score == "tenclass") & (rs.subset == "all images") & (rs.rule == "sens95")].set_index("chosen_on")
    return {"n_folds": len(fits), "reproduction_max_diff": float(worst.max()),
            "key_metric": (f"sens95 cutoff from training: sens {main.loc['training split', 'sensitivity_mean']:.3f}, "
                           f"spec {main.loc['training split', 'specificity_mean']:.3f} (hindsight spec "
                           f"{main.loc['test fold (hindsight)', 'specificity_mean']:.3f}); ten-subtype model: sens "
                           f"{ten.loc['training split', 'sensitivity_mean']:.3f}, spec {ten.loc['training split', 'specificity_mean']:.3f}")}


def figure(ss, rs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    cols = {"sensitivity": "#e34948", "specificity": "#2a78d6", "ppv": "#94762c", "npv": "#1baf7a"}
    titles = {"ridge": "two-class ridge", "tenclass": "ten-subtype ridge, read as benign vs malignant",
              "direct question": "direct question"}
    for ax, score in zip(axes, SCORES):
        d = ss[(ss.score == score) & (ss.subset == "all images")]
        for m, c in cols.items():
            ax.plot(d.cutoff, d[f"{m}_mean"], color=c, lw=2, label=m.upper() if len(m) == 3 else m)
            ax.fill_between(d.cutoff, d[f"{m}_min"], d[f"{m}_max"], color=c, alpha=.15, lw=0)
        r = rs[(rs.score == score) & (rs.subset == "all images") & (rs.chosen_on == "training split")].set_index("rule")
        for rule, ls in (("sens95", "--"), ("youden", ":")):
            ax.axvline(r.loc[rule, "cutoff_mean"], color="#4d4840", ls=ls, lw=1)
        ax.set_title(titles[score], fontsize=10)
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

    w = rep.groupby("task").max_abs_diff.max()
    L = ["# S14 thresholds: P(malignant) cutoffs chosen in the training split", "",
         f"Ridge on all 446 questions (S10 two-class model), the S10 ten-subtype ridge read as benign against malignant "
         f"('tenclass': malignant subtypes' summed probability over their number against the benign subtypes', "
         f"renormalised) and the direct question, scored on the {n} benign and malignant images ({n_mal} malignant), "
         f"{n_folds} outer folds per model. S10's test predictions reproduced to within {w.get('s1_2c', np.nan):.1e} "
         f"(two-class) and {w.get('flat_10c', np.nan):.1e} (ten-subtype, all ten probabilities). The ten-subtype model "
         "has its own folds (S10's flat_10c folds over all 1,000 images), so its rows are not paired fold by fold with "
         "the two-class rows.",
         "Each cutoff is read from cross-validated predictions on the training split and applied unchanged to the test fold; "
         "the hindsight rows read it on the test fold itself (as the specificity at 95% sensitivity was reported so far).",
         "Mean over the 25 test folds, with the range in brackets.", ""]
    for score in SCORES:
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
    for score in ["ridge", "tenclass"]:
        L += [f"## Fixed cutoffs, {score}, all images", "",
              "| P cutoff | sensitivity | specificity | PPV | NPV | accuracy | called malignant |", "|---|---|---|---|---|---|---|"]
        for x in ss[(ss.score == score) & (ss.subset == "all images")].itertuples():
            L.append(f"| {x.cutoff:.2f} | {f(x.sensitivity_mean)} ({f(x.sensitivity_min)}–{f(x.sensitivity_max)}) | "
                     f"{f(x.specificity_mean)} ({f(x.specificity_min)}–{f(x.specificity_max)}) | {f(x.ppv_mean)} | "
                     f"{f(x.npv_mean)} | {f(x.accuracy_mean)} | {f(x.called_malignant_mean)} |")
        L.append("")
    L += ["", "PPV and NPV depend on the share of malignant images (37.5% here) and will differ in other settings.",
          "", "![cutoff sweep](figures/cutoff_sweep.png)", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
