"""S11 Subtypes: the 10-subtype classifier in detail, and naevus against melanoma (requested by the collaborating
dermatologist, 2026-11-16).

Ten subtypes
  Read from the S10 out-of-fold predictions (ridge on all 446 questions, folds grouped by patient, 5 repeats of 5-fold):
  confusion matrix (counts averaged over repeats), and per subtype the sensitivity (recall), precision, F1 and
  one-against-rest AUROC, each per repeat and then averaged; top-2 and top-3 accuracy; how many errors stay within the
  same lineage (melanocytic or not, from the ISIC 'melanocytic' field) and within the same malignancy group; and the
  10-subtype probabilities collapsed to benign against malignant, compared with the two-class model.

Naevus against melanoma
  The images of the two subtypes in subtypes.pair (benign melanocytic: 99 naevi and one lentigo simplex; malignant
  melanocytic: 100 melanomas). Four scores, all out-of-fold:
    pair model    ridge trained on these images only, same nested CV as S10 (patient groups, lambda by inner CV,
                  everything fitted inside the training folds). The atypical melanocytic images (subtypes.between)
                  are never fitted; each outer fit scores them.
    10 subtypes   S10's 10-subtype model, P(melanoma) / (P(melanoma) + P(naevus))
    2 classes     S10's benign-against-malignant model, P(malignant)
    direct        Sonnet's P(yes) to "Is this a malignant lesion?"
  AUROC and specificity (naevi correctly cleared) at 95% melanoma sensitivity, pooled per repeat (mean and range over
  repeats); then, on the scores averaged over the repeats, a bootstrap over patients for intervals and paired
  differences. Broken down by melanoma type (in situ, invasive, NOS, metastasis), by how the naevus diagnosis was
  confirmed (excised and read by histopathology, or not), and on dermoscopic images only. The questions that separate
  the two: each question's AUROC alone, and the weights of the pair model fitted on all 200 images (not scored).

Threshold metrics (for the explorer's metrics chapter)
  Sensitivity, specificity, accuracy, precision and F1, with AUROC alongside, for benign against malignant (ridge on
  all questions and on the lasso-chosen 120, 60 and 27, and the direct question) and naevus against melanoma (the four
  scores), at two thresholds: P >= 0.5, and the threshold that catches subtypes.sens_target of the positives on the
  same pooled predictions (chosen in hindsight, so optimistic). Ten subtypes: each image to its most probable subtype;
  per subtype sensitivity, specificity, precision and F1, and overall accuracy, balanced accuracy and macro-F1. All
  computed per repeat on the pooled out-of-fold predictions, then averaged (range over repeats kept), on all images and
  on dermoscopic images only.

Comparison over all runs
  model_comparison.md / .csv: the ten-subtype models on S10's folds (Sonnet's answers and the S13 SkinGPT-R1 benchmarks)
  first, then the training target (two classes, three, ten subtypes), every S6 model family and answer encoding, the S10
  ridge variants (fold grouping, question subsets) and the S11 naevus-against-melanoma scores; every distinct model
  numbered once, each scored the same way (AUROC; sensitivity,
  specificity, accuracy and F1 at P >= 0.5; specificity at the 95%-sensitivity threshold) on benign against malignant,
  naevus against melanoma and the ten subtypes.

Outputs: model_comparison.md, model_comparison.csv, metrics_binary.csv, metrics_10c.csv, tenclass_per_class.csv, tenclass_confusion.csv, tenclass_summary.csv, pair_scores.csv, pair_metrics.csv, pair_differences.csv,
pair_breakdown.csv, pair_between.csv, pair_questions.csv, pair_final_weights.csv, pair_extra.json (ROC points, threshold), figures/, summary.md.
"""
import json
import warnings

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import plots
from ..models.cv import outer_splits
from ..models.metrics import spec_at_sens
from .s5_model import feature_mask
from .s6_classify import load_labels
from .s10_ridge import fit_split, grid_values, make_pipe

UPSTREAM = ["manifest", "embed", "ridge", "classify", "bench"]
VERSION = 4    # 2: threshold metrics; 3: model_comparison over S6, S10, S11; 4: + S13 SkinGPT-R1, numbered models
SCORES = [("pair", "pair model (naevus vs melanoma only)"), ("tenclass", "10-subtype model"),
          ("twoclass", "benign-vs-malignant model"), ("direct", 'Sonnet asked "malignant?"')]


def key(cfg, ctx) -> dict:
    return {"subtypes": cfg.subtypes, "outer": [cfg.classify.outer_folds, cfg.classify.outer_repeats, cfg.classify.inner_folds],
            "ridge": {k: cfg.ridge[k] for k in ("source", "log10_C", "final_cv_folds")},
            "label_columns": cfg.classify.label_columns, "exclude_kinds": cfg.model.exclude_kinds, "seed": cfg.seed,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "metadata_sha": file_hash(ROOT / cfg.explore.metadata), "sens_target": cfg.classify.sens_target}


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def boot_idx(groups, n_boot, seed):
    """Bootstrap resamples of whole groups (patients): image indices per resample."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        yield np.concatenate([members[g] for g in pick])


def roc_points(y, s, max_pts=200):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    k = np.unique(np.linspace(0, len(fpr) - 1, min(max_pts, len(fpr))).round().astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in k]


# ------------------------------------------------------------------ ten subtypes
def collapse_malignant(P, classes, grp3) -> np.ndarray:
    """Ten-subtype probabilities read as benign against malignant: the malignant subtypes' summed probability divided
    by their number, against the benign subtypes' summed probability divided by theirs, renormalised, so that 0.5 again
    means equal prior odds (the model was trained with equal weight per subtype). Indeterminate subtypes enter neither
    sum. P: [n x K] in the order of `classes`; grp3: subtype -> benign / indeterminate / malignant."""
    P = np.asarray(P, float)
    g = np.array([grp3[c] for c in classes])
    m = P[:, g == "malignant"].sum(1) / (g == "malignant").sum()
    return m / (m + P[:, g == "benign"].sum(1) / (g == "benign").sum())


def tenclass(oof, lab, derm, lineage, classes):
    """Per-repeat confusion matrices and per-class scores of the S10 10-subtype model (all questions, patient groups)."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

    o = oof[(oof.task == "flat_10c") & (oof.set == "all") & (oof.grouping == "patient")]
    grp = {c: lab.c3[lab.c10 == c].iloc[0] for c in classes}
    rows, cms, summ = [], [], []
    for r, g in o.groupby("repeat"):
        P = g[[f"p::{c}" for c in classes]].values
        y = np.array([classes.index(v) for v in g.true])
        pred = P.argmax(1)
        cms.append(confusion_matrix(y, pred, labels=range(len(classes))))
        pr, rc, f1, n = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
        dm = derm.reindex(g.image_id.values).values
        for k, c in enumerate(classes):
            yk = (y == k).astype(int)
            rows.append({"repeat": r, "subtype": c, "group": grp[c], "n": int(n[k]), "sensitivity": rc[k], "precision": pr[k],
                         "f1": f1[k], "auroc_ovr": roc_auc_score(yk, P[:, k]),
                         "sensitivity_dermoscopic": float((pred[dm & (y == k)] == k).mean()) if (dm & (y == k)).any() else np.nan})
        rank = (-P).argsort(1)
        err = pred != y
        lin = np.array([lineage[c] for c in classes])
        g3 = np.array([grp[c] for c in classes])
        # collapse to benign vs malignant on the images that are benign or malignant
        bm = np.isin(g3[y], ["benign", "malignant"])
        pm = P[:, g3 == "malignant"].sum(1) / (P[:, g3 == "malignant"].sum(1) + P[:, g3 == "benign"].sum(1))
        summ.append({"repeat": r, "balanced_accuracy": rc.mean(), "accuracy": float((~err).mean()),
                     "top2": float((rank[:, :2] == y[:, None]).any(1).mean()), "top3": float((rank[:, :3] == y[:, None]).any(1).mean()),
                     "errors": int(err.sum()), "errors_same_lineage": float((lin[pred[err]] == lin[y[err]]).mean()),
                     "errors_same_group": float((g3[pred[err]] == g3[y[err]]).mean()),
                     "collapsed_2c_auroc": auc((g3[y[bm]] == "malignant").astype(int), pm[bm])})
    per = pd.DataFrame(rows)
    agg = per.groupby(["subtype", "group"], sort=False).agg(
        n=("n", "first"), **{f"{m}{s}": (m, f) for m in ("sensitivity", "precision", "f1", "auroc_ovr", "sensitivity_dermoscopic")
                             for s, f in (("", "mean"), ("_min", "min"), ("_max", "max"))}).reset_index()
    cm = pd.DataFrame(np.mean(cms, 0), index=classes, columns=classes)
    # main confusion of each subtype (the subtype it is most often mistaken for)
    off = cm.values.copy()
    np.fill_diagonal(off, 0)
    agg["most_confused_with"] = [classes[j] for j in off.argmax(1)]
    agg["share_to_most_confused"] = off.max(1) / cm.values.sum(1)
    return agg, cm, pd.DataFrame(summ)



# ------------------------------------------------------------------ threshold metrics
BIN_MODELS = [("all", "ridge, all 446 questions"), ("lasso120", "ridge, 120 lasso-chosen questions"),
              ("lasso60", "ridge, 60 lasso-chosen questions"), ("lasso27", "ridge, 27 lasso-chosen questions")]


def at_threshold(y, s, thr) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(s) >= thr
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    tn, fp = int((~pred & ~y).sum()), int((pred & ~y).sum())
    return {"threshold": float(thr), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp) if tp + fp else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan, "auroc": auc(y.astype(int), s),
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def thr_for_sens(y, s, sens) -> float:
    """Highest threshold at which at least `sens` of the positives score at or above it."""
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])[::-1]
    return float(pos[int(np.ceil(sens * len(pos))) - 1])


def binary_rows(task, model, name, y, S, derm, sens) -> list:
    """S: [repeats x images] scores. One row per (subset, rule, repeat)."""
    out = []
    for sub, m in (("all images", np.ones(len(y), bool)), ("dermoscopic only", derm)):
        for r, s in enumerate(S):
            for rule, thr in (("P >= 0.5", 0.5), (f"{sens:.0%} sensitivity", thr_for_sens(y[m], s[m], sens))):
                out.append({"task": task, "model": model, "name": name, "subset": sub, "rule": rule, "repeat": r,
                            **at_threshold(y[m], s[m], thr)})
    return out


def aggregate(df, keys) -> pd.DataFrame:
    vals = [c for c in df.columns if c not in keys + ["repeat"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, sort=False)[vals]
    a = g.mean()
    return a.join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max")).reset_index()


def tenclass_rows(oof, classes, derm) -> pd.DataFrame:
    out = []
    for fs, name in BIN_MODELS:
        o = oof[(oof.task == "flat_10c") & (oof.set == fs) & (oof.grouping == "patient")]
        for r, g in o.groupby("repeat"):
            y = np.array([classes.index(v) for v in g.true])
            pred = g[[f"p::{c}" for c in classes]].values.argmax(1)
            dm = derm.reindex(g.image_id.values).values
            for sub, m in (("all images", np.ones(len(y), bool)), ("dermoscopic only", dm)):
                yy, pp = y[m], pred[m]
                per = []
                for k, c in enumerate(classes):
                    tp, fn = ((pp == k) & (yy == k)).sum(), ((pp != k) & (yy == k)).sum()
                    fp, tn = ((pp == k) & (yy != k)).sum(), ((pp != k) & (yy != k)).sum()
                    row = {"model": fs, "name": name, "subset": sub, "repeat": r, "subtype": c, "n": int(tp + fn),
                           "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                           "precision": tp / (tp + fp) if tp + fp else np.nan, "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan}
                    per.append(row)
                out += per
                P = pd.DataFrame(per)
                out.append({"model": fs, "name": name, "subset": sub, "repeat": r, "subtype": "overall", "n": int(len(yy)),
                            "accuracy": float((pp == yy).mean()), "balanced_accuracy": P.sensitivity.mean(),
                            "sensitivity": P.sensitivity.mean(), "specificity": P.specificity.mean(),
                            "precision": P.precision.mean(), "f1": P.f1.mean()})
    return aggregate(pd.DataFrame(out), ["model", "name", "subset", "subtype"])



# ------------------------------------------------------------------ comparison over all runs (model_comparison.md)
FAMILIES = [("l2", "ridge (L2 logistic)"), ("l1", "lasso (L1 logistic)"), ("enet", "elastic net"), ("rf", "random forest"),
            ("hgb", "gradient boosting"), ("mlp", "MLP")]
SOURCES = [("dir", "Directional Score"), ("orig", "P(yes)"), ("coh", "Coherence Score")]
QSETS = [("all", "all 446 questions", 446), ("lasso120", "120, chosen jointly by the lasso", 120),
         ("lasso60", "60, chosen jointly by the lasso", 60), ("lasso27", "27, chosen jointly by the lasso", 27),
         ("top60", "60, best one at a time (F-test)", 60), ("top27", "27, best one at a time (F-test)", 27),
         ("reps60", "60, most typical per answer cluster", 60), ("reps27", "27, most typical per answer cluster", 27),
         ("pca60", "60 principal components (all 446 asked)", 60), ("pca27", "27 principal components (all 446 asked)", 27),
         ("rand120", "120 at random (mean of 5 draws)", 120), ("rand60", "60 at random (mean of 5 draws)", 60),
         ("rand27", "27 at random (mean of 5 draws)", 27)]


def _mat(d, ids, f):
    """Per-repeat matrix [repeats x len(ids)] of f(frame indexed by image_id)."""
    return np.vstack([f(h.set_index("image_id").reindex(ids)) for _, h in d.groupby("repeat")])


def bin_summary(y, S, sens) -> dict:
    """Pooled per repeat at P >= 0.5, then mean over repeats; plus specificity at the 95%-sensitivity threshold."""
    rows = [at_threshold(y, s, 0.5) | {"spec95": at_threshold(y, s, thr_for_sens(y, s, sens))["specificity"]} for s in S]
    return {k: float(np.mean([r[k] for r in rows])) for k in ("auroc", "sensitivity", "specificity", "accuracy", "f1", "spec95")}


def multi_summary(y, Ps) -> dict:
    from sklearn.metrics import roc_auc_score

    out = []
    for P in Ps:
        pred, K = P.argmax(1), P.shape[1]
        se = [((pred == k) & (y == k)).sum() / max((y == k).sum(), 1) for k in range(K)]
        sp = [((pred != k) & (y != k)).sum() / max((y != k).sum(), 1) for k in range(K)]
        f1 = [2 * ((pred == k) & (y == k)).sum() / max((pred == k).sum() + (y == k).sum(), 1) for k in range(K)]
        out.append({"auroc": roc_auc_score(y, P, multi_class="ovr", average="macro"), "accuracy": float((pred == y).mean()),
                    "sensitivity": float(np.mean(se)), "specificity": float(np.mean(sp)), "f1": float(np.mean(f1))})
    return {k: float(np.mean([r[k] for r in out])) for k in out[0]}


ANS = "Sonnet's answers to all 446 questions (Directional Score)"
DIRECT_Q = 'Sonnet asked "Is this a malignant lesion?"'
BENCH = [("questions", ANS),
         ("skingpt_emb:img_L28", "SkinGPT-R1 image embedding, layer 28 of 28 (prespecified)"),
         ("skingpt_emb:img_L21", "SkinGPT-R1 image embedding, layer 21"),
         ("skingpt_emb:img_L14", "SkinGPT-R1 image embedding, layer 14"),
         ("skingpt_emb:img_L0", "SkinGPT-R1 vision encoder output (layer 0)"),
         ("skingpt_emb:last_L28", "SkinGPT-R1 last prompt token, layer 28"),
         ("skingpt_diagnosis_fitted", "SkinGPT-R1's scores for 16 candidate diagnoses")]
SCORE = {("two classes", "benign vs malignant"): "P(malignant)",
         ("two classes", "naevus vs melanoma"): "P(malignant), on the 200 naevi and melanomas",
         ("three classes", "benign vs malignant"): "P(malignant) / (P(malignant) + P(benign))",
         ("three classes", "naevus vs melanoma"): "P(malignant) / (P(malignant) + P(benign)), on the 200 images",
         ("ten subtypes", "benign vs malignant"): "malignant subtypes summed against benign subtypes",
         ("ten subtypes", "naevus vs melanoma"): "P(melanoma) / (P(melanoma) + P(naevus))",
         ("ten subtypes", "ten subtypes"): "most probable subtype",
         ("naevus vs melanoma", "naevus vs melanoma"): "P(melanoma)",
         ("no training", "benign vs malignant"): "P(yes)", ("no training", "naevus vs melanoma"): "P(yes), on the 200 images",
         ("candidates", "benign vs malignant"): "malignant candidate labels summed",
         ("candidates", "naevus vs melanoma"): "malignant candidate labels summed, on the 200 images"}


def model_name(r) -> str:
    """One readable name per distinct model (features, classifier, training target, folds, pipeline)."""
    if r["classifier"] == "none":
        return r["features"]
    return f"{r['classifier']} on {r['features']}"


def model_comparison(out, oof10, oof6, oofb, classes, grp3, direct, pair_S, pid, ypair, neg, pos, sens, run_names) -> pd.DataFrame:
    """Every model of S6, S10, S11 and S13 scored the same way on the three tasks. Each distinct model (pipeline, input,
    classifier, training target, fold grouping) gets one number, so a model that appears in several tables is
    recognisable as the same one. Writes model_comparison.csv / model_comparison.md and returns the long table."""
    b = oof10[(oof10.task == "s1_2c") & (oof10.set == "all") & (oof10.grouping == "patient") & (oof10.repeat == 0)]
    ids2, y2 = b.image_id.values, (b.true.values == "malignant").astype(int)
    a = oof10[(oof10.task == "flat_10c") & (oof10.set == "all") & (oof10.grouping == "patient") & (oof10.repeat == 0)]
    ids10, y10 = a.image_id.values, np.array([classes.index(v) for v in a.true])
    pc = [f"p::{c}" for c in classes]
    ratio = lambda h: (h[f"p::{pos}"] / (h[f"p::{pos}"] + h[f"p::{neg}"])).values
    # ten subtypes collapsed to benign vs malignant: the summed probabilities, each group divided by its number of subtypes,
    # so that 0.5 again means equal prior odds (the model was trained with equal weight per subtype)
    collapse = lambda h: collapse_malignant(h[pc].values, classes, grp3)
    three = lambda h: (h["p::malignant"] / (h["p::malignant"] + h["p::benign"])).values
    pmal = lambda h: h["p::malignant"].values
    rows = []

    def ident(section, stage, features, classifier, target, folds):
        return {"section": section, "stage": stage, "features": features, "classifier": classifier, "target": target,
                "folds": folds}

    def add(idn, task, metrics, readout=None):
        rows.append(idn | {"task": task, "score": SCORE[(readout or idn["target"], task)]} | metrics)

    def add_model(idn, d=None, tasks=("benign vs malignant", "naevus vs melanoma", "ten subtypes"), ds=None, readout=None):
        """d: the model's out-of-fold predictions (one frame, or ds: several draws whose metrics are averaged)."""
        for task in tasks:
            mets = []
            for dd in (ds if ds is not None else [d]):
                if task == "ten subtypes":
                    Ps = [h.set_index("image_id").reindex(ids10)[pc].values for _, h in dd.groupby("repeat")]
                    mets.append(multi_summary(y10, Ps))
                    continue
                f = {"two classes": pmal, "three classes": three,
                     "ten subtypes": collapse if task == "benign vs malignant" else ratio}[readout or idn["target"]]
                ids_, y_ = (ids2, y2) if task == "benign vs malignant" else (pid, ypair)
                mets.append(bin_summary(y_, _mat(dd, ids_, f), sens))
            add(idn, task, pd.DataFrame(mets).mean().to_dict(), readout)

    s6 = lambda task, m, s: oof6[(oof6.task == task) & (oof6.model == m) & (oof6.source == s)]
    s10 = lambda task, fs, g="patient": oof10[(oof10.task == task) & (oof10.set == fs) & (oof10.grouping == g)]
    bn = lambda task, m: oofb[(oofb.task == task) & (oofb.set == m)]
    S6F, S10P, S10L = "lesion (S6 pipeline)", "patient", "lesion"
    BM, NM = ("benign vs malignant",), ("naevus vs melanoma",)
    zs = ident("M", "S12/S13", "SkinGPT-R1's own diagnosis (16 candidate labels), zero-shot", "none", "no training", "no training")
    dq = ident("M", "S2", DIRECT_Q, "none", "no training", "no training")

    # M. ten-subtype models, all on S10's patient-grouped folds (S13 reproduced the question model exactly)
    for m, feat in BENCH:
        add_model(ident("M", "S10" if m == "questions" else "S13", feat, "ridge", "ten subtypes", S10P), bn("flat_10c", m))
    add_model(zs, bn("flat_10c", "skingpt_diagnosis"), readout="ten subtypes")
    for task, ids_, y_ in (("benign vs malignant", ids2, y2), ("naevus vs melanoma", pid, ypair)):
        add(dq, task, bin_summary(y_, direct.reindex(ids_).values[None, :], sens))
    # T. training target: the same inputs trained on two classes, three classes or ten subtypes
    T = lambda st, feat, cl, tg, fo: ident("T", st, feat, cl, tg, fo)
    for m, feat in BENCH[:2] + BENCH[4:5] + BENCH[6:]:
        st = "S10" if m == "questions" else "S13"
        add_model(T(st, feat, "ridge", "two classes", S10P), bn("s1_2c", m), BM + NM)
        add_model(T(st, feat, "ridge", "ten subtypes", S10P), bn("flat_10c", m), BM + NM)
    # the zero-shot diagnosis read two ways: its candidate labels summed by group, or its ten-subtype probabilities
    for task, ids_, y_ in (("benign vs malignant", ids2, y2), ("naevus vs melanoma", pid, ypair)):
        add(zs | {"section": "T"}, task, bin_summary(y_, _mat(bn("s1_2c", "skingpt_diagnosis"), ids_, pmal), sens), "candidates")
    add_model(zs | {"section": "T"}, bn("flat_10c", "skingpt_diagnosis"), BM + NM, readout="ten subtypes")
    for tg, task in (("two classes", "s1_2c"), ("three classes", "s1_3c"), ("ten subtypes", "flat_10c")):
        add_model(ident("U", "S6", ANS, dict(FAMILIES)["l2"], tg, S6F), s6(task, "l2", "dir"), BM + NM)
    rows.append(T("S11", ANS, "ridge", "naevus vs melanoma", S10P) | {"task": "naevus vs melanoma", "score": "P(melanoma)"}
                | bin_summary(ypair, pair_S, sens))
    # B. model families (S6, Directional Score); C. answer encoding (S6)
    for m, n in FAMILIES:
        add_model(ident("B", "S6", ANS, n, "two classes", S6F), s6("s1_2c", m, "dir"), BM)
        add_model(ident("B", "S6", ANS, n, "ten subtypes", S6F), s6("flat_10c", m, "dir"), NM + ("ten subtypes",))
    for m, n in FAMILIES:
        for s, sn in SOURCES:
            feat = ANS if s == "dir" else f"Sonnet's answers to all 446 questions ({sn})"
            add_model(ident("C", "S6", feat, n, "two classes", S6F), s6("s1_2c", m, s), BM)
            add_model(ident("C", "S6", feat, n, "ten subtypes", S6F), s6("flat_10c", m, s), NM + ("ten subtypes",))
    # D. leakage controls (S10 ridge, all questions)
    for g, fo in (("lesion", S10L), ("patient", S10P)):
        add_model(ident("D", "S10", ANS, "ridge", "two classes", fo), s10("s1_2c", "all", g), BM)
        add_model(ident("D", "S10", ANS, "ridge", "ten subtypes", fo), s10("flat_10c", "all", g), NM + ("ten subtypes",))
    # E. number of questions (S10 ridge, patient groups)
    for fs, n, k in QSETS:
        feat = ANS if fs == "all" else f"Sonnet's answers, {n}"
        if fs.startswith("rand"):
            draws = sorted(oof10[oof10.set.str.startswith(fs + "_")].set.unique())
            add_model(ident("E", "S10", feat, "ridge", "two classes", S10P), tasks=BM, ds=[s10("s1_2c", d) for d in draws])
            add_model(ident("E", "S10", feat, "ridge", "ten subtypes", S10P), tasks=NM + ("ten subtypes",), ds=[s10("flat_10c", d) for d in draws])
        else:
            add_model(ident("E", "S10", feat, "ridge", "two classes", S10P), s10("s1_2c", fs), BM)
            add_model(ident("E", "S10", feat, "ridge", "ten subtypes", S10P), s10("flat_10c", fs), NM + ("ten subtypes",))
    # F. naevus against melanoma: the pair model and the scores derived from the general models
    rows.append(ident("F", "S11", ANS, "ridge", "naevus vs melanoma", S10P) | {"task": "naevus vs melanoma", "score": "P(melanoma)"}
                | bin_summary(ypair, pair_S, sens))
    add_model(ident("F", "S10", ANS, "ridge", "ten subtypes", S10P), s10("flat_10c", "all"), NM)
    add_model(ident("F", "S10", ANS, "ridge", "two classes", S10P), s10("s1_2c", "all"), NM)
    add(dq | {"section": "F"}, "naevus vs melanoma", bin_summary(ypair, direct.reindex(pid).values[None, :], sens))

    tab = pd.DataFrame(rows)
    tab["name"] = tab.apply(model_name, axis=1)
    idk = ["stage", "features", "classifier", "target", "folds"]
    first = tab.drop_duplicates(idk)[idk].reset_index(drop=True)
    first["no"] = np.arange(1, len(first) + 1)
    tab = tab.merge(first, on=idk, how="left")
    tab = tab[["section", "no", "name", "stage", "features", "classifier", "target", "folds", "task", "score",
               "auroc", "sensitivity", "specificity", "accuracy", "f1", "spec95"]]
    tab.to_csv(out / "model_comparison.csv", index=False)
    write_comparison(out, tab, sens, run_names)
    return tab


SECTIONS = {
    "M": ("Ten-subtype models on the same folds",
          "Each model was trained to name one of the ten subtypes, on the same patient-grouped folds (S10's design: 5 repeats "
          "of 5-fold cross-validation, every learned step inside the training folds). The two binary tasks are read from the "
          "same ten probabilities, so each row is one model scored three ways. Stage S13 refitted the question model on these "
          "folds and reproduced S10's fold results exactly. SkinGPT-R1's diagnosis was scored without its reasoning step (S13 "
          "report). The direct question is a reference with no training; it cannot name a subtype."),
    "T": ("Training target: two classes or ten subtypes",
          "The same inputs trained on two classes or on the ten subtypes, on S10's patient-grouped folds, all scored on the "
          "two binary tasks. A ten-subtype model is read as malignant subtypes against benign subtypes (each group's summed "
          "probability divided by its number of subtypes, so that 0.5 again means equal prior odds; this rescaling does not "
          "change AUROC). SkinGPT-R1's zero-shot diagnosis has no training target; it is shown read both ways. For naevus "
          "against melanoma the ridge trained on the 200 naevi and melanomas only is added."),
    "U": ("Training target in the S6 pipeline: two, three or ten classes",
          "The S6 ridge on all 446 questions trained on two classes, three (with indeterminate) or the ten subtypes. S6 folds "
          "were grouped by lesion, so these rows are comparable with each other but not with the patient-grouped tables."),
    "B": ("Model families",
          "The six model families of S6, on all 446 questions encoded as the Directional Score. S6 folds were grouped by "
          "`lesion_id`, which is unique per image, so a patient's other lesions can sit in the training folds (FINDINGS 22); "
          "compare these rows with each other, and with the leakage-control table for the effect of the grouping. Benign against malignant "
          "uses each family's two-class model; the other two tasks use its ten-subtype model, so each family is two models."),
    "C": ("Answer encoding",
          "The same six families on each of the three ways of turning Sonnet's answers into numbers: P(yes) as stated; the "
          "Directional Score (2 P(yes) − 1 times the stated confidence, standardised); the Coherence Score. S6 folds, grouped "
          "by lesion; two-class models for benign against malignant, ten-subtype models for the other tasks."),
    "D": ("Ridge with leakage controls",
          "S10 ridge on all 446 questions, with every learned step inside the training folds, and the folds grouped either by "
          "lesion (as in S6) or by ISIC patient. Two-class models for benign against malignant, ten-subtype models for the "
          "other tasks."),
    "E": ("Number of questions",
          "S10 ridge, folds grouped by patient, on subsets of the questions chosen inside each training fold, or at random. "
          "Two-class models for benign against malignant, ten-subtype models (trained on the same subset) for the other tasks."),
    "F": ("Scores built for naevus against melanoma",
          "S11: a ridge model trained on the 100 naevi and 100 melanomas only, against scores read from the general models "
          "on the same 200 images."),
}
SECTION_ORDER = "MTUBCDEF"
TASKS = [("benign vs malignant", "Benign against malignant (300 malignant, 500 benign images)",
          [("auroc", "AUROC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("accuracy", "Accuracy"),
           ("f1", "F1"), ("spec95", "Specificity at 95% sensitivity")]),
         ("naevus vs melanoma", "Naevus against melanoma (100 melanomas, 100 naevi)",
          [("auroc", "AUROC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("accuracy", "Accuracy"),
           ("f1", "F1"), ("spec95", "Specificity at 95% sensitivity")]),
         ("ten subtypes", "Ten subtypes (100 images each; chance accuracy 0.10)",
          [("auroc", "AUROC (mean one vs rest)"), ("sensitivity", "Sensitivity (mean)"), ("specificity", "Specificity (mean)"),
           ("accuracy", "Accuracy"), ("f1", "F1 (macro)")])]


def write_comparison(out, tab, sens, run_names):
    L = ["# Model metrics across all experiments", "",
         "Every model trained so far on the 1,000 beta images, from Sonnet's answers or from SkinGPT-R1, scored the same way "
         f"on three tasks. Generated by stage S11 from {run_names}; the numbers are in `model_comparison.csv`.", "",
         "**Each model has one number.** A model is one combination of pipeline stage, input, classifier, training target "
         "and fold grouping. The same number in two tables is the same fitted model scored on a different task; a different "
         "number is a different model, even when the name looks alike (for example, the question ridge trained on two classes "
         "and the one trained on ten subtypes). The column *Trained on* gives the training target and *Score* how the task "
         "is read from the model's output. The model index at the end lists every number.", "",
         "**How every number is computed.** Each image is scored by a model that did not see it in training (out-of-fold "
         "predictions; 5 repeats of 5-fold cross-validation). Within each repeat the predictions of all folds are pooled and "
         "the measure is computed once; the tables give the mean over the 5 repeats. Pooled values can differ by about 0.01 "
         "to 0.02 from the fold means quoted in earlier reports.", "",
         "**What the measures mean.**",
         "- **AUROC:** the chance that a randomly chosen positive case (malignant, melanoma) scores above a randomly chosen "
         "negative one, over all thresholds. 0.5 is chance, 1 is perfect.",
         "- **Sensitivity:** of the positive cases, the share the model flags (P ≥ 0.5).",
         "- **Specificity:** of the negative cases, the share it correctly leaves alone (P ≥ 0.5).",
         "- **Accuracy:** the share of all images classified correctly (P ≥ 0.5). For benign against malignant there are "
         "more benign images (500 against 300), so accuracy leans toward specificity.",
         "- **F1:** the harmonic mean of precision (the share of flagged cases that are truly positive) and sensitivity; high "
         "only when the model both finds the positives and does not over-call them.",
         f"- **Specificity at {sens:.0%} sensitivity:** the share of negatives left alone when the threshold is lowered until "
         f"{sens:.0%} of positives are caught. The threshold is chosen on the same predictions, so this is slightly optimistic. "
         "It is the project's pre-set main score for benign against malignant.",
         "- **Ten subtypes:** each image goes to its most probable subtype (no threshold). Sensitivity, specificity and F1 "
         "are computed per subtype and averaged. With 100 images per subtype, accuracy equals mean sensitivity (balanced "
         "accuracy).", "",
         "The trained models output probabilities for equal class priors (balanced class weights, or the random forest and "
         "MLP corrected to equal priors), so P ≥ 0.5 means the same for each. The zero-shot SkinGPT-R1 diagnosis and the "
         "direct question are not calibrated that way, so their threshold metrics depend on where their scores happen to sit; "
         "compare them by AUROC. Bold marks the best value in each column of each table (ties included; not marked where more "
         "than half the rows share it).", "",
         "Differences of 0.01 to 0.02 between models are within the spread over repeats and folds (about ±0.03 in AUROC for "
         "benign against malignant, ±0.04 for naevus against melanoma, where each class has only 100 images); the highest "
         "value in a column is not by itself evidence that the model is better.", ""]
    fmtv = lambda k, v: "n/a" if pd.isna(v) else f"{v:.3f}" if k == "auroc" else f"{v:.2f}"
    for i, sec in enumerate(SECTION_ORDER, 1):
        title, note = SECTIONS[sec]
        L += [f"## {i}. {title}", "", note, ""]
        for task, ttitle, cols in TASKS:
            g = tab[(tab.section == sec) & (tab.task == task)]
            if g.empty:
                continue
            shown = {k: [fmtv(k, v) for v in g[k]] for k, _ in cols}
            best = {k: max((float(v) for v in shown[k] if v != "n/a"), default=None) for k, _ in cols}
            # mark the best only where it singles out a minority of rows
            vary = {k: len(set(shown[k])) > 1 and sum(v != "n/a" and float(v) == best[k] for v in shown[k]) <= len(g) / 2
                    for k, _ in cols}
            L += [f"### {ttitle}", "", "| No. | Model | Trained on | Folds grouped by | Score | " + " | ".join(n for _, n in cols) + " |",
                  "|---|---|---|---|---|" + "---|" * len(cols)]
            for x in g.itertuples():
                cells = []
                for k, _ in cols:
                    s = fmtv(k, getattr(x, k))
                    cells.append(f"**{s}**" if s != "n/a" and len(g) > 1 and vary[k] and float(s) == best[k] else s)
                L.append(f"| {x.no} | {x.name} | {x.target} | {x.folds} | {x.score} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## Model index", "", "Every distinct model in the tables above, with the sections it appears in.", "",
          "| No. | Model | Trained on | Folds grouped by | Stage | Sections |", "|---|---|---|---|---|---|"]
    for no, g in tab.groupby("no", sort=True):
        x = g.iloc[0]
        secs = ", ".join(str(SECTION_ORDER.index(s) + 1) for s in sorted(g.section.unique(), key=SECTION_ORDER.index))
        L.append(f"| {no} | {x['name']} | {x.target} | {x.folds} | {x.stage} | {secs} |")
    (out / "model_comparison.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def run(cfg, out, inputs, ctx) -> dict:
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    from joblib import Parallel, delayed
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    sc, cc, rc = cfg.subtypes, cfg.classify, cfg.ridge
    (out / "figures").mkdir(exist_ok=True)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    meta = pd.read_csv(ROOT / cfg.explore.metadata, dtype={"patient_id": str, "lesion_id": str}).set_index("isic_id").reindex(ids)
    derm = pd.Series(meta.image_type.eq("dermoscopic").values, index=ids)
    groups = pd.read_csv(inputs["ridge"] / "groups.csv").set_index("image_id").group.reindex(ids).astype(str)
    oof = pd.read_parquet(inputs["ridge"] / "oof_predictions.parquet")
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qk = qt[feature_mask(qt, cfg.model, patch=False)].reset_index(drop=True)
    qids = list(qk.question_id)
    X = pd.read_parquet(inputs["embed"] / f"X_image_{rc.source}.parquet").loc[ids, qids].values.astype(float)
    P0 = pd.read_parquet(inputs["embed"] / "X_image_orig.parquet").loc[ids]
    direct = pd.Series(P0["DQC01"].values, index=ids)

    # ---------------- ten subtypes
    classes = sorted(lab.c10.unique(), key=lambda s: (["benign", "indeterminate", "malignant"].index(lab.c3[lab.c10 == s].iloc[0]), s))
    melc = meta.melanocytic                              # ISIC field; blank for some images, consistent within a subtype
    lineage = {c: ("melanocytic" if melc[(lab.c10 == c).values].dropna().astype(str).str.lower().eq("true").mean() > 0.5
                   else "non-melanocytic") for c in classes}
    per, cm, tsum = tenclass(oof, lab, derm, lineage, classes)
    per["lineage"] = per.subtype.map(lineage)
    per.to_csv(out / "tenclass_per_class.csv", index=False)
    cm.to_csv(out / "tenclass_confusion.csv")
    tsum.to_csv(out / "tenclass_summary.csv", index=False)
    twoc_ref = oof[(oof.task == "s1_2c") & (oof.set == "all") & (oof.grouping == "patient")]
    twoc_auc = [auc((g.true == "malignant").astype(int), g["p::malignant"]) for _, g in twoc_ref.groupby("repeat")]
    plots.confusion(cm.round().values, classes, "10 subtypes: confusion matrix (counts, mean over 5 repeats)",
                    out / "figures" / "tenclass_confusion.png")

    # ---------------- naevus against melanoma: the pair model
    neg, pos = sc.pair
    rows = lab.c10.isin([neg, pos]).values
    between = lab.c10.isin(list(sc.between)).values
    pid = np.array(ids)[rows]
    y = (lab.c10.values[rows] == pos).astype(int)
    g = groups.values[rows]
    grid = grid_values(*rc.log10_C)
    splits = outer_splits(y, g, cc.outer_folds, cc.outer_repeats, cfg.seed)
    warnings.filterwarnings("ignore")
    fits = Parallel(n_jobs=rc.n_jobs)(delayed(fit_split)(("cols", None), X[rows], y, g, tr, te, cc.inner_folds,
                                                         cfg.seed + 1000 * r + f, grid, None, X[between])
                                      for r, f, tr, te in splits)
    pair = np.full((cc.outer_repeats, rows.sum()), np.nan)
    lam = []
    for (r, f, tr, te), fit in zip(splits, fits):
        pair[r, te] = fit["P"][:, 1]
        lam.append(1 / fit["C"])
    btw = np.mean([fit["P_extra"][:, 1] for fit in fits], 0)

    # the other three scores on the same 200 images, per repeat
    def from_oof(task, f):
        d = oof[(oof.task == task) & (oof.set == "all") & (oof.grouping == "patient")]
        M = np.full((cc.outer_repeats, len(pid)), np.nan)
        for r, h in d.groupby("repeat"):
            h = h.set_index("image_id").reindex(pid)
            M[r] = f(h)
        return M
    S = {"pair": pair,
         "tenclass": from_oof("flat_10c", lambda h: h[f"p::{pos}"] / (h[f"p::{pos}"] + h[f"p::{neg}"])),
         "twoclass": from_oof("s1_2c", lambda h: h["p::malignant"]),
         "direct": np.tile(direct.reindex(pid).values, (cc.outer_repeats, 1))}
    mean = {k: np.nanmean(v, 0) for k, v in S.items()}
    sens = cc.sens_target
    dm = derm.reindex(pid).values

    met = []
    for k, name in SCORES:
        per_r = [(auc(y, s), spec_at_sens(y, s, sens)) for s in S[k]]
        per_rd = [(auc(y[dm], s[dm]), spec_at_sens(y[dm], s[dm], sens)) for s in S[k]]
        a = np.array(per_r)
        ad = np.array(per_rd)
        boots = np.array([(auc(y[i], mean[k][i]), spec_at_sens(y[i], mean[k][i], sens)) for i in boot_idx(g, sc.n_boot, cfg.seed)])
        met.append({"score": k, "name": name, "n_naevi": int((y == 0).sum()), "n_melanoma": int(y.sum()),
                    "auroc_repeat_mean": a[:, 0].mean(), "auroc_repeat_min": a[:, 0].min(), "auroc_repeat_max": a[:, 0].max(),
                    "spec95_repeat_mean": a[:, 1].mean(), "spec95_repeat_min": a[:, 1].min(), "spec95_repeat_max": a[:, 1].max(),
                    "auroc_avg_score": auc(y, mean[k]), "auroc_ci_lo": np.nanpercentile(boots[:, 0], 2.5),
                    "auroc_ci_hi": np.nanpercentile(boots[:, 0], 97.5), "spec95_avg_score": spec_at_sens(y, mean[k], sens),
                    "spec95_ci_lo": np.nanpercentile(boots[:, 1], 2.5), "spec95_ci_hi": np.nanpercentile(boots[:, 1], 97.5),
                    "auroc_dermoscopic": ad[:, 0].mean(), "spec95_dermoscopic": ad[:, 1].mean(),
                    "lambda_median": float(np.median(lam)) if k == "pair" else np.nan})
    met = pd.DataFrame(met)
    # paired differences on the same bootstrap resamples
    diffs = []
    pairs_cmp = [("pair", "direct"), ("tenclass", "direct"), ("twoclass", "direct"), ("pair", "tenclass"), ("pair", "twoclass"), ("tenclass", "twoclass")]
    B = list(boot_idx(g, sc.n_boot, cfg.seed + 1))
    for a_, b_ in pairs_cmp:
        d = np.array([auc(y[i], mean[a_][i]) - auc(y[i], mean[b_][i]) for i in B])
        per_r = [auc(y, S[a_][r]) - auc(y, S[b_][r]) for r in range(cc.outer_repeats)]
        diffs.append({"a": a_, "b": b_, "auroc_diff": auc(y, mean[a_]) - auc(y, mean[b_]), "ci_lo": np.nanpercentile(d, 2.5),
                      "ci_hi": np.nanpercentile(d, 97.5), "share_boot_above_0": float((d > 0).mean()),
                      "repeats_higher": int(sum(x > 0 for x in per_r))})
    diffs = pd.DataFrame(diffs)
    met.to_csv(out / "pair_metrics.csv", index=False)
    diffs.to_csv(out / "pair_differences.csv", index=False)

    # breakdown by melanoma type, naevus confirmation, image type
    d3 = meta.diagnosis_3.reindex(pid).fillna("unknown").values
    conf = meta.diagnosis_confirm_type.reindex(pid).fillna("not recorded").values
    excised = conf == "histopathology"
    br = []
    subsets = [(f"melanoma: {t}", (y == 0) | ((y == 1) & (d3 == t))) for t in pd.Series(d3[y == 1]).value_counts().index]
    subsets += [("naevi excised (histopathology)", (y == 1) | ((y == 0) & excised)),
                ("naevi not excised (consensus, follow-up, other)", (y == 1) | ((y == 0) & ~excised)),
                ("dermoscopic images only", dm)]
    for label, m in subsets:
        for k, name in SCORES:
            bs = [auc(y[m][i], mean[k][m][i]) for i in boot_idx(g[m], sc.n_boot, cfg.seed)]
            br.append({"subset": label, "n_naevi": int(((y == 0) & m).sum()), "n_melanoma": int(((y == 1) & m).sum()),
                       "score": k, "auroc": np.mean([auc(y[m], s[m]) for s in S[k]]),
                       "auroc_ci_lo": np.nanpercentile(bs, 2.5), "auroc_ci_hi": np.nanpercentile(bs, 97.5)})
    br = pd.DataFrame(br)
    br.to_csv(out / "pair_breakdown.csv", index=False)

    # per-image scores, and the threshold that catches 95% of melanomas (scores averaged over repeats)
    sc_df = pd.DataFrame({"image_id": pid, "subtype": lab.c10.values[rows], "melanoma": y, "diagnosis": d3,
                          "confirmation": conf, "image_type": meta.image_type.reindex(pid).values,
                          **{f"score_{k}": mean[k] for k, _ in SCORES}})
    sc_df.to_csv(out / "pair_scores.csv", index=False)
    thr = np.sort(mean["pair"][y == 1])[int(np.floor((1 - sens) * y.sum()))]
    b_ids = np.array(ids)[between]
    bt = pd.DataFrame({"image_id": b_ids, "subtype": lab.c10.values[between],
                       "diagnosis": meta.diagnosis_3.reindex(b_ids).fillna("unknown").values,
                       "image_type": meta.image_type.reindex(b_ids).values, "score_pair": btw})
    bt.to_csv(out / "pair_between.csv", index=False)
    btw_summary = {"threshold_95": float(thr),
                   "naevi_above": float((mean["pair"][y == 0] >= thr).mean()),
                   "melanoma_above": float((mean["pair"][y == 1] >= thr).mean()),
                   "between_above": float((btw >= thr).mean()),
                   "median": {"naevus": float(np.median(mean["pair"][y == 0])), "between": float(np.median(btw)),
                              "melanoma": float(np.median(mean["pair"][y == 1]))},
                   "by_diagnosis": {k: {"n": int(len(v)), "median": float(v.median()), "above": float((v >= thr).mean())}
                                    for k, v in bt.groupby("diagnosis").score_pair}}

    # questions that separate the two: alone, and in the pair model fitted on all 200 images (not scored)
    qa = np.array([auc(y, X[rows][:, j]) for j in range(len(qids))])
    gs = GridSearchCV(make_pipe(("cols", None), cfg.seed), {"clf__C": grid}, scoring="neg_log_loss", n_jobs=-1,
                      cv=StratifiedGroupKFold(rc.final_cv_folds, shuffle=True, random_state=cfg.seed), refit=True)
    gs.fit(X[rows], y, groups=g)
    w = gs.best_estimator_["clf"].coef_[0]
    qtab = qk[["question_id", "question", "topic"]].assign(auroc_alone=qa, towards=np.where(qa >= 0.5, "melanoma", "naevus"),
                                                           separation=np.abs(qa - 0.5), final_weight=w,
                                                           mean_naevus=X[rows][y == 0].mean(0), mean_melanoma=X[rows][y == 1].mean(0))
    qtab = qtab.sort_values("separation", ascending=False)
    qtab.to_csv(out / "pair_questions.csv", index=False)
    qtab.sort_values("final_weight")[["question_id", "question", "topic", "final_weight"]].to_csv(out / "pair_final_weights.csv", index=False)

    # threshold metrics: benign vs malignant (S10 sets and the direct question), naevus vs melanoma, ten subtypes
    rows_b = []
    for fs, name in BIN_MODELS:
        o = oof[(oof.task == "s1_2c") & (oof.set == fs) & (oof.grouping == "patient")]
        ids2 = o[o.repeat == 0].image_id.values
        M = np.vstack([h.set_index("image_id").reindex(ids2)["p::malignant"].values for _, h in o.groupby("repeat")])
        yb = (o[o.repeat == 0].true.values == "malignant").astype(int)
        rows_b += binary_rows("benign vs malignant", fs, name, yb, M, derm.reindex(ids2).values, sens)
        if fs == "all":
            rows_b += binary_rows("benign vs malignant", "direct", 'Sonnet asked "malignant?"', yb, direct.reindex(ids2).values[None, :],
                                  derm.reindex(ids2).values, sens)
    for k, name in SCORES:
        rows_b += binary_rows("naevus vs melanoma", k, name, y, S[k], dm, sens)
    mb = aggregate(pd.DataFrame(rows_b), ["task", "model", "name", "subset", "rule"])
    mb.to_csv(out / "metrics_binary.csv", index=False)
    m10 = tenclass_rows(oof, classes, derm)
    m10.to_csv(out / "metrics_10c.csv", index=False)
    oof6 = pd.read_parquet(inputs["classify"] / "oof_predictions.parquet")
    oofb = pd.read_parquet(inputs["bench"] / "oof_predictions.parquet")
    grp3 = {c: lab.c3[lab.c10 == c].iloc[0] for c in classes}
    model_comparison(out, oof, oof6, oofb, classes, grp3, direct, S["pair"], pid, y, neg, pos, sens,
                     f"S6 classify `{inputs['classify'].name}`, S10 ridge `{inputs['ridge'].name}`, "
                     f"S13 bench `{inputs['bench'].name}` and this stage")

    curves = {k: roc_points(y, mean[k]) for k, _ in SCORES}
    plots.pair_roc(curves, dict(SCORES), {k: auc(y, mean[k]) for k, _ in SCORES}, sens,
                   "Naevus against melanoma (100 + 100 images)", out / "figures" / "pair_roc.png")
    (out / "pair_extra.json").write_text(json.dumps({"between": btw_summary, "curves": curves, "final_lambda": 1 / gs.best_params_["clf__C"],
                                                     "twoclass_auroc_repeats": twoc_auc, "lineage": lineage}, indent=1))
    write_summary(out, per, cm, tsum, twoc_auc, met, diffs, br, btw_summary, qtab, 1 / gs.best_params_["clf__C"], sens)
    m = met.set_index("score")
    return {"tenclass_balanced_accuracy": float(tsum.balanced_accuracy.mean()), "tenclass_top2": float(tsum.top2.mean()),
            "pair_auroc": float(m.auroc_repeat_mean["pair"]), "pair_auroc_direct": float(m.auroc_repeat_mean["direct"]),
            "key_metric": f"naevus vs melanoma auroc={m.auroc_repeat_mean['pair']:.3f} (pair ridge); 10c bacc={tsum.balanced_accuracy.mean():.3f}"}


def _f(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_summary(out, per, cm, tsum, twoc_auc, met, diffs, br, btw, qtab, lam, sens):
    L = ["# S11 subtypes: the 10-subtype model in detail, and naevus against melanoma", "",
         "## Ten subtypes (S10 ridge, all questions, folds grouped by patient; mean over 5 repeats, range in brackets)", "",
         f"Balanced accuracy {tsum.balanced_accuracy.mean():.3f} ({tsum.balanced_accuracy.min():.3f} to {tsum.balanced_accuracy.max():.3f}); "
         f"chance 0.10. Top-2 accuracy {tsum.top2.mean():.3f}, top-3 {tsum.top3.mean():.3f}. Of the errors, "
         f"{tsum.errors_same_lineage.mean():.0%} stay within the same lineage (melanocytic or not) and "
         f"{tsum.errors_same_group.mean():.0%} within the same malignancy group. Collapsed to benign against malignant, the "
         f"10-subtype probabilities reach AUROC {tsum.collapsed_2c_auroc.mean():.3f} (two-class model, pooled per repeat: "
         f"{np.mean(twoc_auc):.3f}).", "",
         "| subtype | group | lineage | n | sensitivity | precision | F1 | AUROC one vs rest | sensitivity, dermoscopic | most often taken for |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for x in per.itertuples():
        L.append(f"| {x.subtype} | {x.group} | {x.lineage} | {x.n} | {_f(x.sensitivity)} ({_f(x.sensitivity_min)} to {_f(x.sensitivity_max)}) | "
                 f"{_f(x.precision)} | {_f(x.f1)} | {_f(x.auroc_ovr, 3)} | {_f(x.sensitivity_dermoscopic)} | "
                 f"{x.most_confused_with} ({x.share_to_most_confused:.0%}) |")
    L += ["", "Confusion matrix (rows: true subtype; counts averaged over repeats): `tenclass_confusion.csv`, "
          "`figures/tenclass_confusion.png`.", "",
          "## Naevus against melanoma", "",
          f"| score | AUROC, pooled per repeat: mean (range) | AUROC on the averaged score [95% CI, patient bootstrap] | "
          f"specificity at {sens:.0%} sensitivity: mean (range) | [95% CI] | dermoscopic only: AUROC, specificity |", "|---|---|---|---|---|---|"]
    for x in met.itertuples():
        L.append(f"| {x.name} | {_f(x.auroc_repeat_mean, 3)} ({_f(x.auroc_repeat_min, 3)} to {_f(x.auroc_repeat_max, 3)}) | "
                 f"{_f(x.auroc_avg_score, 3)} [{_f(x.auroc_ci_lo, 3)}, {_f(x.auroc_ci_hi, 3)}] | {_f(x.spec95_repeat_mean)} "
                 f"({_f(x.spec95_repeat_min)} to {_f(x.spec95_repeat_max)}) | [{_f(x.spec95_ci_lo)}, {_f(x.spec95_ci_hi)}] | "
                 f"{_f(x.auroc_dermoscopic, 3)}, {_f(x.spec95_dermoscopic)} |")
    L += ["", f"Pair model: median lambda {met.lambda_median.dropna().iloc[0]:.0f} over the outer fits; final model (all 200 images) lambda {lam:.0f}.", "",
          "Paired AUROC differences (averaged scores; patient bootstrap on the same resamples):", "",
          "| a | b | a − b | 95% CI | share of resamples a > b | repeats a > b |", "|---|---|---|---|---|---|"]
    for x in diffs.itertuples():
        L.append(f"| {x.a} | {x.b} | {x.auroc_diff:+.3f} | [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}] | {x.share_boot_above_0:.0%} | {x.repeats_higher} of 5 |")
    L += ["", "### Breakdown (AUROC, mean over repeats; [95% CI])", "", "| subset | naevi | melanomas | " +
          " | ".join(n for _, n in SCORES) + " |", "|---|---|---|" + "---|" * len(SCORES)]
    for s_, h in br.groupby("subset", sort=False):
        h = h.set_index("score")
        L.append(f"| {s_} | {h.n_naevi.iloc[0]} | {h.n_melanoma.iloc[0]} | " +
                 " | ".join(f"{_f(h.auroc[k], 3)} [{_f(h.auroc_ci_lo[k], 2)}, {_f(h.auroc_ci_hi[k], 2)}]" for k, _ in SCORES) + " |")
    L += ["", "### Atypical melanocytic lesions on the pair model's scale", "",
          f"Threshold catching {sens:.0%} of melanomas: {btw['threshold_95']:.3f}. Above it: naevi {btw['naevi_above']:.0%}, "
          f"atypical melanocytic {btw['between_above']:.0%}, melanomas {btw['melanoma_above']:.0%}. Median score: naevi "
          f"{btw['median']['naevus']:.3f}, atypical {btw['median']['between']:.3f}, melanomas {btw['median']['melanoma']:.3f}.", ""]
    for k, v in btw["by_diagnosis"].items():
        L.append(f"- {k} (n = {v['n']}): median {v['median']:.3f}, {v['above']:.0%} above the threshold")
    L += ["", "### Questions that separate naevus from melanoma most, each alone (AUROC on the 200 images)", "",
          "| question | topic | AUROC alone | towards | pair-model weight |", "|---|---|---|---|---|"]
    for x in qtab.head(30).itertuples():
        L.append(f"| {x.question} | {x.topic} | {x.auroc_alone:.3f} | {x.towards} | {x.final_weight:+.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
