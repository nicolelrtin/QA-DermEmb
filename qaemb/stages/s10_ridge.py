"""S10 Ridge: the main model (ridge on the Directional Score), evaluated with leakage controls, compared with trimmed
question sets, and refitted once on all images as the frozen final model (PROJECT.md, Plan items A and 5).

Leakage controls
  groups      folds are grouped by ISIC patient where ISIC records one, else by lesion; images whose pictures are
              near-identical (difference hash within ridge.dup_hamming bits) are merged into one group. Inner folds
              (choice of lambda) use the same groups.
  in-fold     scaling, lambda, and the choice of questions (cluster representatives, principal components) are all
              refitted inside every training split, inner and outer (qaemb/models/selectors.py). Nothing is chosen on
              images that are scored.
  final       the final models are fitted on all labelled images, lambda chosen by grouped CV on those images, and
              saved without any score: their performance is measured once, on the external test set.

Question sets
  all         the 446 clinical questions
  repsK       the most typical question of each of K clusters of questions answered alike (K = 27: the dimensions above
              shuffled noise in S9; K = 60: the components for 90% of the variance)
  pcaK        the first K principal components of all 446 answers (not readable, and still needs all 446 asked; the
              upper bound for K dimensions)
  randK       K questions drawn at random, ridge.n_random draws (the floor: what any K questions give)
  lassoK      the K questions the lasso keeps first on the training split (chosen jointly, with the training labels;
              then refitted with ridge)
  topK        the K questions that best separate the classes on the training images alone (ANOVA F-test, refitted in
              every training split; the only set chosen with labels, and still never with test labels)

Scores: AUROC and specificity at 95% malignant sensitivity (benign vs malignant), balanced accuracy and macro-F1
(10 subtypes); on all test images and on dermoscopic test images only. Sonnet's P(yes) to "Is this a malignant lesion?"
is scored on the same folds as a secondary reference. The 'all' set is also run with the old grouping by lesion, to show
what the grouping changes.

Outputs: groups.csv, duplicates.csv, metrics_folds.csv, summary.csv, paired.csv, reps_stability.csv,
final_models/ (joblib + coefficient tables + chosen lambda), figures/, summary.md.
"""
import json
import warnings

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import plots
from ..models.cv import outer_splits
from ..models.metrics import task_metrics
from ..models.selectors import ClusterReps, Columns, LassoK, TopK
from .s5_model import feature_mask
from .s6_classify import SOURCE_NAMES, define_tasks, load_labels

UPSTREAM = ["manifest", "embed"]
VERSION = 3


def key(cfg, ctx) -> dict:
    return {"ridge": cfg.ridge, "outer": [cfg.classify.outer_folds, cfg.classify.outer_repeats, cfg.classify.inner_folds],
            "label_columns": cfg.classify.label_columns, "exclude_kinds": cfg.model.exclude_kinds, "seed": cfg.seed,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "metadata_sha": file_hash(ROOT / cfg.explore.metadata), "sens_target": cfg.classify.sens_target,
            "sens_at_spec": cfg.model.sens_at_spec, "ece_bins": cfg.model.ece_bins}


def primary(task):
    return "auroc" if task == "s1_2c" else "balanced_accuracy"


def grid_values(lo, hi):
    g = [m * 10.0 ** d for d in range(int(lo), int(hi)) for m in (1, 2, 3, 5)] + [10.0 ** int(hi)]
    return sorted(set(np.round(g, 12)))


# ------------------------------------------------------------------ groups
def dhash(path, size=16):
    from PIL import Image

    im = Image.open(path)
    im.draft("L", (64, 64))
    a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), float)
    return (a[:, 1:] > a[:, :-1]).flatten()


def build_groups(man, meta, by: str, max_ham: int, n_jobs: int):
    """Group label per image, and the near-duplicate pairs found."""
    from joblib import Parallel, delayed

    ids = list(man.image_id)
    H = np.array(Parallel(n_jobs=n_jobs)(delayed(dhash)(p) for p in man.path))
    D = (H[:, None, :] != H[None, :, :]).sum(2)
    iu = np.triu_indices(len(ids), 1)
    close = np.where(D[iu] <= max_ham)[0]
    pairs = pd.DataFrame({"image_a": [ids[iu[0][k]] for k in close], "image_b": [ids[iu[1][k]] for k in close],
                          "hamming": D[iu][close]})
    base = man.patient_id.astype(str).values.copy()                      # S0 stores lesion_id as patient_id
    if by == "patient":
        pid = meta.patient_id.reindex(ids).values
        base = np.where(pd.notna(pid), pid.astype(str), base)
    parent = {g: g for g in base}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for a, b in zip(iu[0][close], iu[1][close]):
        parent[find(base[a])] = find(base[b])
    return np.array([find(g) for g in base]), pairs, int((D[iu] <= max_ham).sum())


# ------------------------------------------------------------------ models
def make_pipe(spec, seed, memory=None):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = LogisticRegression(l1_ratio=0.0, solver="saga", class_weight="balanced", max_iter=5000, tol=1e-3,
                             random_state=seed)
    kind, arg = spec
    # memory: the selection step does not depend on lambda, so it is fitted once per training split and reused
    if kind == "lasso":
        return Pipeline([("sel", LassoK(arg, seed)), ("scale", StandardScaler()), ("clf", clf)], memory=memory)
    if kind == "top":
        return Pipeline([("sel", TopK(arg)), ("scale", StandardScaler()), ("clf", clf)], memory=memory)
    if kind == "reps":
        return Pipeline([("sel", ClusterReps(arg)), ("scale", StandardScaler()), ("clf", clf)], memory=memory)
    if kind == "pca":
        return Pipeline([("scale", StandardScaler()), ("pca", PCA(arg, whiten=True, random_state=seed)), ("clf", clf)])
    return Pipeline([("sel", Columns(arg)), ("scale", StandardScaler()), ("clf", clf)])


def fit_split(spec, X, y, groups, tr, te, inner_folds, seed, grid, cache_dir=None, X_extra=None):
    import shutil
    import tempfile

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    k = min(inner_folds, int(np.bincount(y[tr]).min()))
    mem = tempfile.mkdtemp(dir=cache_dir) if spec[0] in ("lasso", "top", "reps") else None
    gs = GridSearchCV(make_pipe(spec, seed, mem), {"clf__C": grid}, scoring="neg_log_loss", n_jobs=1, refit=True,
                      cv=StratifiedGroupKFold(k, shuffle=True, random_state=seed), error_score="raise")
    gs.fit(X[tr], y[tr], groups=groups[tr])
    b = gs.best_estimator_
    sel = b["sel"].idx_ if "sel" in b.named_steps else None
    out = {"P": b.predict_proba(X[te]), "C": gs.best_params_["clf__C"], "sel": sel,
           "P_extra": b.predict_proba(X_extra) if X_extra is not None else None}
    if mem:
        shutil.rmtree(mem, ignore_errors=True)
    return out


def feature_specs(rc, n_q, rng) -> dict:
    specs = {}
    for fs in rc.feature_sets:
        if fs == "all":
            specs["all"] = ("cols", None)
        elif fs.startswith("lasso"):
            specs[fs] = ("lasso", int(fs[5:]))
        elif fs.startswith("top"):
            specs[fs] = ("top", int(fs[3:]))
        elif fs.startswith("reps"):
            specs[fs] = ("reps", int(fs[4:]))
        elif fs.startswith("pca"):
            specs[fs] = ("pca", int(fs[3:]))
        elif fs.startswith("rand"):
            K = int("".join(ch for ch in fs if ch.isdigit()))
            for d in range(rc.n_random):
                specs[f"{fs}_{d}"] = ("cols", np.sort(rng.choice(n_q, K, replace=False)))
    return specs


# ------------------------------------------------------------------ main
def run(cfg, out, inputs, ctx) -> dict:
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    from joblib import Parallel, delayed

    rc, cc = cfg.ridge, cfg.classify
    rng = np.random.default_rng(cfg.seed)
    (out / "figures").mkdir(exist_ok=True)
    (out / "final_models").mkdir(exist_ok=True)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    meta = pd.read_csv(ROOT / cfg.explore.metadata, dtype={"patient_id": str, "lesion_id": str}).set_index("isic_id")
    derm = meta.image_type.reindex(ids).eq("dermoscopic").values
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qk = qt[feature_mask(qt, cfg.model, patch=False)].reset_index(drop=True)
    qids = list(qk.question_id)
    X = pd.read_parquet(inputs["embed"] / f"X_image_{rc.source}.parquet").loc[ids, qids].values.astype(float)
    P = pd.read_parquet(inputs["embed"] / "X_image_orig.parquet").loc[ids]
    ctrl = P["DQC01"].values if "DQC01" in P.columns else None
    tasks = define_tasks(lab, list(rc.tasks))
    grid = grid_values(*rc.log10_C)

    # groups
    groups_p, dups, n_dup = build_groups(man, meta, rc.group_by, rc.dup_hamming, 16)
    groups_l = man.patient_id.astype(str).values
    gsz = pd.Series(groups_p).value_counts()
    pd.DataFrame({"image_id": ids, "group": groups_p, "lesion_id": groups_l, "subtype": lab.c10.values,
                  "image_type": meta.image_type.reindex(ids).values}).to_csv(out / "groups.csv", index=False)
    dups.to_csv(out / "duplicates.csv", index=False)
    print(f"  {len(gsz)} groups ({(gsz > 1).sum()} with several images, largest {gsz.max()}); "
          f"{n_dup} near-duplicate pairs", flush=True)

    specs = feature_specs(rc, len(qids), rng)
    cache = out / "_cache"
    cache.mkdir(exist_ok=True)
    runs = [(fs, "patient") for fs in specs] + ([("all", "lesion")] if rc.compare_lesion_grouping else [])
    plan, jobs = [], []
    for t, T in tasks.items():
        rows = T["rows"]
        y = np.array([T["classes"].index(v) for v in lab[T["col"]].values[rows]])
        for fs, gname in runs:
            g = (groups_p if gname == "patient" else groups_l)[rows]
            splits = outer_splits(lab.c10.values[rows], g, cc.outer_folds, cc.outer_repeats, cfg.seed)
            # indeterminate images are never fitted; the main model scores them in every outer fit
            Xe = X[T["extra"]] if (T["extra"] is not None and t == "s1_2c" and fs == "all" and gname == "patient") else None
            for r, f, tr, te in splits:
                plan.append((t, fs, gname, r, f, np.where(rows)[0][te]))
                jobs.append(delayed(fit_split)(specs[fs], X[rows], y, g, tr, te, cc.inner_folds, cfg.seed + 1000 * r + f, grid,
                                               str(cache), Xe))
    print(f"  {len(jobs)} outer fits, {len(grid)} penalties x {cc.inner_folds} inner folds each", flush=True)
    fits = Parallel(n_jobs=rc.n_jobs, verbose=2)(jobs)
    import shutil

    shutil.rmtree(cache, ignore_errors=True)

    # ---------------- metrics
    spec_, sens = cfg.model.sens_at_spec, cc.sens_target
    fold_rows, oof, sel_rows = [], {}, []
    for (t, fs, gname, r, f, te_abs), fit in zip(plan, fits):
        T = tasks[t]
        cls = T["classes"]
        yt = np.array([cls.index(v) for v in lab[T["col"]].values[te_abs]])
        base = {"task": t, "set": fs, "grouping": gname, "repeat": r, "fold": f, "C": fit["C"], "lambda": 1 / fit["C"]}
        for sub, m in (("all images", np.ones(len(te_abs), bool)), ("dermoscopic only", derm[te_abs])):
            if len(np.unique(yt[m])) < len(cls) and t == "s1_2c":
                continue
            met = task_metrics(yt[m], fit["P"][m], cls, spec_, sens, cfg.model.ece_bins)
            fold_rows.append({**base, "subset": sub, "n_test": int(m.sum()), **{k: v for k, v in met.items() if k != "confusion_matrix"}})
        key_ = (t, fs, gname, r)
        oof.setdefault(key_, {})
        for i, im in enumerate(te_abs):
            oof[key_][im] = fit["P"][i]
        if fit["sel"] is not None and specs[fs][0] in ("reps", "top", "lasso"):
            for q in fit["sel"]:
                sel_rows.append({"task": t, "set": fs, "repeat": r, "fold": f, "question_id": qids[q]})
    if ctrl is not None:                                  # control question on the same folds (s1_2c, patient groups)
        for (t, fs, gname, r, f, te_abs) in plan:
            if t != "s1_2c" or fs != "all" or gname != "patient":
                continue
            T = tasks[t]
            yt = np.array([T["classes"].index(v) for v in lab[T["col"]].values[te_abs]])
            Pc = np.column_stack([1 - ctrl[te_abs], ctrl[te_abs]])
            for sub, m in (("all images", np.ones(len(te_abs), bool)), ("dermoscopic only", derm[te_abs])):
                met = task_metrics(yt[m], Pc[m], T["classes"], spec_, sens, cfg.model.ece_bins)
                fold_rows.append({"task": t, "set": "control question", "grouping": gname, "repeat": r, "fold": f,
                                  "subset": sub, "n_test": int(m.sum()), **{k: v for k, v in met.items() if k != "confusion_matrix"}})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "metrics_folds.csv", index=False)

    # out-of-fold predictions (every set, every repeat) and the indeterminate images scored by the main model
    pr = []
    for (t, fs, gname, r), d in oof.items():
        cls = tasks[t]["classes"]
        idx = list(d)
        df = pd.DataFrame(np.array([d[i] for i in idx]), columns=[f"p::{c}" for c in cls])
        df.insert(0, "image_id", np.array(ids)[idx])
        pr.append(df.assign(task=t, set=fs, grouping=gname, repeat=r, true=lab[tasks[t]["col"]].values[idx]))
    pd.concat(pr, ignore_index=True).to_parquet(out / "oof_predictions.parquet", index=False)
    ext = [fit["P_extra"][:, 1] for (t, fs, gname, *_), fit in zip(plan, fits) if fit["P_extra"] is not None]
    if ext and "s1_2c" in tasks:
        e_ids = np.array(ids)[tasks["s1_2c"]["extra"]]
        pd.DataFrame({"image_id": e_ids, "subtype": lab.c10.values[tasks["s1_2c"]["extra"]],
                      "p_malignant": np.mean(ext, axis=0), "n_fits": len(ext)}).to_csv(out / "indeterminate_2c.csv", index=False)

    # pooled per repeat
    pooled = []
    for (t, fs, gname, r), d in oof.items():
        T = tasks[t]
        cls = T["classes"]
        idx = list(d)                                     # absolute image positions
        Pm = np.array([d[i] for i in idx])
        yt = np.array([cls.index(v) for v in lab[T["col"]].values[idx]])
        for sub, m in (("all images", np.ones(len(idx), bool)), ("dermoscopic only", derm[idx]), ("clinical only", ~derm[idx])):
            met = task_metrics(yt[m], Pm[m], cls, spec_, sens, cfg.model.ece_bins)
            pooled.append({"task": t, "set": fs, "grouping": gname, "repeat": r, "subset": sub, "n": int(m.sum()),
                           **{k: v for k, v in met.items() if k != "confusion_matrix"}})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "metrics_pooled_by_repeat.csv", index=False)

    summ = summarise(folds, pooled, sens)
    summ.to_csv(out / "summary.csv", index=False)
    paired = paired_vs_all(folds)
    paired.to_csv(out / "paired.csv", index=False)
    stab = pd.DataFrame(sel_rows)
    if len(stab):
        stab = (stab.groupby(["task", "set", "question_id"]).size().rename("times_chosen").reset_index()
                .merge(qk[["question_id", "question", "topic"]], on="question_id")
                .sort_values(["task", "set", "times_chosen"], ascending=[True, True, False]))
    stab.to_csv(out / "reps_stability.csv", index=False)

    # ---------------- final models (no score)
    finals = final_models(out / "final_models", rc, cc, cfg.seed, tasks, lab, X, qk, groups_p, specs, grid)
    plots.ridge_sets_figure(summ, out / "figures" / "performance_vs_questions.png", len(qids))
    write_summary(out, cfg, lab, qids, gsz, dups, summ, paired, stab, finals, derm, tasks, grid)
    main = summ[(summ.task == "s1_2c") & (summ.set == "all") & (summ.grouping == "patient") & (summ.subset == "all images")]
    return {"n_groups": int(len(gsz)), "n_near_duplicates": int(len(dups)),
            "key_metric": f"s1_2c auroc={main.fold_mean.iloc[0]:.3f} (ridge, dir, patient-grouped)" if len(main) else ""}


def summarise(folds, pooled, sens) -> pd.DataFrame:
    rows = []
    folds = folds.assign(family=folds.set.str.replace(r"_\d+$", "", regex=True))
    pooled = pooled.assign(family=pooled.set.str.replace(r"_\d+$", "", regex=True))
    for (t, fam, gname, sub), g in folds.groupby(["task", "family", "grouping", "subset"]):
        pm = primary(t)
        per_draw = g.groupby("set")[pm].mean()
        pr = pooled[(pooled.task == t) & (pooled.family == fam) & (pooled.grouping == gname) & (pooled.subset == sub)]
        row = {"task": t, "set": fam, "grouping": gname, "subset": sub, "metric": pm, "n_draws": g.set.nunique(),
               "fold_mean": g[pm].mean(), "fold_sd": g.groupby(["repeat", "fold"])[pm].mean().std(),
               "draw_min": per_draw.min(), "draw_max": per_draw.max(),
               "pooled_mean": pr[pm].mean(), "pooled_min": pr[pm].min(), "pooled_max": pr[pm].max(),
               "lambda_median": g["lambda"].median() if "lambda" in g and g["lambda"].notna().any() else np.nan}
        if t == "s1_2c":
            row["spec_at_sens"] = g[f"spec_at_sens{int(sens * 100)}"].mean()
        else:
            row["macro_f1"] = g["macro_f1"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_all(folds) -> pd.DataFrame:
    """Each question set against all 446 questions on the same outer folds (patient grouping): mean difference of the
    primary score and the number of folds in which the set scored higher."""
    rows = []
    f = folds[(folds.grouping == "patient")]
    for (t, sub), g in f.groupby(["task", "subset"]):
        pm = primary(t)
        ref = g[g.set == "all"].set_index(["repeat", "fold"])[pm]
        g = g.assign(family=g.set.str.replace(r"_\d+$", "", regex=True))
        for s, h in g[g.set != "all"].groupby("family"):
            d = (h.groupby(["repeat", "fold"])[pm].mean() - ref).dropna()      # random draws averaged per fold
            rows.append({"task": t, "subset": sub, "set": s, "mean_diff": d.mean(), "sd_diff": d.std(),
                         "folds_higher": int((d > 0).sum()), "n_folds": int(len(d))})
    return pd.DataFrame(rows)


def final_models(fdir, rc, cc, seed, tasks, lab, X, qk, groups, specs, grid) -> list:
    """Fit each final set on all labelled images of each task; lambda by grouped CV on those same images. Saved, never
    scored here."""
    import joblib
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    out = []
    for t, T in tasks.items():
        rows = T["rows"]
        y = np.array([T["classes"].index(v) for v in lab[T["col"]].values[rows]])
        for fs in rc.final_sets:
            gs = GridSearchCV(make_pipe(specs[fs], seed), {"clf__C": grid}, scoring="neg_log_loss", n_jobs=-1,
                              cv=StratifiedGroupKFold(rc.final_cv_folds, shuffle=True, random_state=seed), refit=True)
            gs.fit(X[rows], y, groups=groups[rows])
            b = gs.best_estimator_
            idx = b["sel"].idx_ if "sel" in b.named_steps else np.arange(X.shape[1])
            coef = b["clf"].coef_
            tab = qk.iloc[idx][["question_id", "question", "topic"]].reset_index(drop=True)
            if coef.shape[0] == 1:
                tab["weight_towards_" + T["classes"][1]] = coef[0]
            else:
                for c, name in enumerate(T["classes"]):
                    tab[f"weight_{name}"] = coef[c]
            name = f"{t}_{fs}"
            joblib.dump({"model": b, "question_ids": [qk.question_id[i] for i in idx], "all_question_ids": list(qk.question_id),
                         "classes": T["classes"], "source": rc.source, "C": gs.best_params_["clf__C"],
                         "note": "Fitted on all labelled beta images; lambda chosen by grouped CV on the same images. "
                                 "Do not score on the beta images. Input: Directional Score for all_question_ids, in order."},
                        fdir / f"{name}.joblib")
            tab.to_csv(fdir / f"{name}_weights.csv", index=False)
            out.append({"task": t, "set": fs, "n_questions": len(idx), "C": gs.best_params_["clf__C"],
                        "lambda": 1 / gs.best_params_["clf__C"], "n_images": int(rows.sum())})
    pd.DataFrame(out).to_csv(fdir / "final_models.csv", index=False)
    (fdir / "README.md").write_text(
        "# Final models (frozen)\n\nFitted on all labelled beta images, with lambda chosen by grouped cross-validation on "
        "those same images. Their performance must not be reported on the beta images (that would leak validation into "
        "test); it is measured once, on the external test set. The cross-validated estimate of what to expect is in "
        "../summary.md.\n\nLoad with `joblib.load(path)`; `model.predict_proba(X)` takes the Directional Scores of "
        "`all_question_ids` in that order (the selection step inside the model keeps `question_ids`).\n")
    return out


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_summary(out, cfg, lab, qids, gsz, dups, summ, paired, stab, finals, derm, tasks, grid):
    cc, rc = cfg.classify, cfg.ridge
    L = ["# S10 ridge: main model with leakage controls, trimmed question sets, final models", "",
         f"Ridge (L2 logistic regression) on the {SOURCE_NAMES.get(rc.source, rc.source)}, {len(qids)} clinical questions, "
         f"{len(lab)} images ({int(derm.sum())} dermoscopic). Nested CV: {cc.outer_repeats} repeats of {cc.outer_folds}-fold, "
         f"stratified on the 10 subtypes, grouped by {rc.group_by} (ISIC patient where recorded, else lesion; near-duplicate "
         f"pictures merged); lambda from {len(grid)} values by {cc.inner_folds}-fold grouped inner CV on log loss. Question "
         "selection and principal components are refitted inside every training split.", "",
         f"Groups: {len(gsz)} groups for {len(lab)} images; {(gsz > 1).sum()} groups hold more than one image (largest {gsz.max()}). "
         f"Near-duplicate image pairs (difference hash within {rc.dup_hamming} of 256 bits): {len(dups)}.", ""]
    for t in tasks:
        L += [f"## {t} ({primary(t).replace('_', ' ')})", "",
              "| question set | grouping | images | fold mean ± sd | pooled per repeat: mean (min to max) | " +
              ("specificity at 95% sensitivity |" if t == "s1_2c" else "macro-F1 |") + " lambda (median) |",
              "|---|---|---|---|---|---|---|"]
        s = summ[summ.task == t]
        order = ["all", "lasso27", "lasso60", "lasso120", "top27", "top60", "reps27", "reps60", "pca27", "pca60", "rand27", "rand60", "rand120", "control question"]
        for fam in order:
            for x in s[s.set == fam].itertuples():
                extra = x.spec_at_sens if t == "s1_2c" else x.macro_f1
                draws = f" (draws {_f(x.draw_min)} to {_f(x.draw_max)})" if x.n_draws > 1 else ""
                L.append(f"| {fam} | {x.grouping} | {x.subset} | {_f(x.fold_mean)} ± {_f(x.fold_sd)}{draws} | "
                         f"{_f(x.pooled_mean)} ({_f(x.pooled_min)} to {_f(x.pooled_max)}) | {_f(extra, 2)} | {_f(x.lambda_median, 0)} |")
        L.append("")
        p = paired[(paired.task == t)]
        if len(p):
            L += ["Against all questions on the same folds (patient grouping):", "",
                  "| set | images | mean difference | folds higher |", "|---|---|---|---|"]
            for x in p.itertuples():
                L.append(f"| {x.set} | {x.subset} | {x.mean_diff:+.3f} ± {x.sd_diff:.3f} | {x.folds_higher} of {x.n_folds} |")
            L.append("")
    if len(stab):
        L += ["## Stability of the chosen questions (refitted in each of the 25 outer training sets)", ""]
        for (t, fs), g in stab.groupby(["task", "set"]):
            K = int("".join(ch for ch in fs if ch.isdigit()))
            L.append(f"- {t}, {fs}: {int((g.times_chosen >= 20).sum())} of {K} questions chosen in at least 20 of 25 outer folds; "
                     f"{g.question_id.nunique()} different questions chosen overall.")
        L.append("")
    L += ["## Final models (fitted on all labelled images; not scored)", "", "| task | set | questions | images | lambda |", "|---|---|---|---|---|"]
    for f in finals:
        L.append(f"| {f['task']} | {f['set']} | {f['n_questions']} | {f['n_images']} | {f['lambda']:.0f} |")
    L.append("")
    (out / "summary.md").write_text("\n".join(L))
