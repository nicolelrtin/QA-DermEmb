"""S13 Bench: the main model (ridge on Sonnet's Directional Scores) against SkinGPT-R1 on the same outer folds
(PROJECT.md, Plan item E; Benchmarks (a)).

Methods, all scored on exactly the outer test folds of S10 (patient-grouped, 5 repeats of 5-fold, same seed):
  questions            ridge on the Directional Score of the 446 clinical questions: S10's 'all' set, refitted here with
                       S10's own pipeline and grid, and checked fold by fold against S10's stored scores (which also proves
                       the folds are the same).
  skingpt_emb:<name>   ridge (L2 logistic regression) on a SkinGPT-R1 image embedding from S12; scaling and lambda fitted
                       inside every outer training split exactly as for the questions (3-fold grouped inner CV, log loss).
                       The first name in bench.embeddings is the main one, fixed before any result was seen; the others
                       are secondary.
  skingpt_diagnosis    SkinGPT-R1's own diagnosis, zero-shot: its probability over the candidate labels (S12), summed into
                       P(malignant) for benign vs malignant and into the subtypes for the 10-class task (candidates with
                       no subtype, such as seborrhoeic keratosis, count as benign and are dropped for the 10 subtypes).
                       Nothing is fitted, so the folds only decide which images are scored.
  skingpt_diagnosis_fitted
                       ridge on SkinGPT-R1's log-probabilities for the candidate labels, fitted in the same folds: its
                       diagnosis, read out by a model trained on our labels (it can learn, for example, that "actinic
                       keratosis" on our images often means squamous cell carcinoma).
  control question     Sonnet's P(yes) to "Is this a malignant lesion?" (benign vs malignant only), as in S10.

Scores and subsets as in S10: AUROC and specificity at 95% sensitivity (benign vs malignant), balanced accuracy and
macro-F1 (10 subtypes), on all test images and on dermoscopic test images only; fold means with the spread over the 25
outer folds, pooled per repeat, and paired differences against the questions on the same folds.

Outputs: metrics_folds.csv, metrics_pooled_by_repeat.csv, summary.csv, paired.csv, lambda_choices.csv,
oof_predictions.parquet, check_s10.csv, figures/, summary.md.
"""
import warnings

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import plots
from ..models.cv import outer_splits
from ..models.metrics import task_metrics
from .s5_model import feature_mask
from .s6_classify import define_tasks, load_labels
from .s10_ridge import build_groups, fit_split, grid_values, primary, summarise

UPSTREAM = ["manifest", "embed", "ridge", "vlmfeat"]
VERSION = 1

NAMES = {"questions": "ridge on 446 question answers (Sonnet)", "skingpt_diagnosis": "SkinGPT-R1 diagnosis, zero-shot",
         "skingpt_diagnosis_fitted": "SkinGPT-R1 diagnosis, ridge readout", "control question": 'Sonnet asked "malignant?"'}


def emb_name(e: str) -> str:
    kind, layer = e.split("_L")
    if kind == "img" and layer == "0":
        return "SkinGPT-R1 vision encoder, ridge"
    where = "image tokens" if kind == "img" else "last token"
    return f"SkinGPT-R1 layer {layer} ({where}), ridge"


def method_name(m: str) -> str:
    return emb_name(m.split(":", 1)[1]) if m.startswith("skingpt_emb:") else NAMES.get(m, m)


def key(cfg, ctx) -> dict:
    r = cfg.ridge
    return {"bench": cfg.bench, "ridge": {k: r[k] for k in ("source", "group_by", "dup_hamming", "log10_C")},
            "outer": [cfg.classify.outer_folds, cfg.classify.outer_repeats, cfg.classify.inner_folds],
            "candidates": cfg.vlmfeat.candidates, "label_columns": cfg.classify.label_columns,
            "exclude_kinds": cfg.model.exclude_kinds, "seed": cfg.seed,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "metadata_sha": file_hash(ROOT / cfg.explore.metadata), "sens_target": cfg.classify.sens_target,
            "sens_at_spec": cfg.model.sens_at_spec, "ece_bins": cfg.model.ece_bins}


def fit_dense(X, y, groups, tr, te, inner_folds, seed, grid):
    """Ridge for dense, high-dimensional embeddings: the same model as S10 (L2 logistic regression, balanced classes,
    standardised inputs, lambda by grouped inner CV on log loss), solved with L-BFGS instead of SAGA, which converges
    far faster on thousands of correlated columns and reaches the same optimum."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=3000,
                                                random_state=seed))])
    k = min(inner_folds, int(np.bincount(y[tr]).min()))
    gs = GridSearchCV(pipe, {"clf__C": grid}, scoring="neg_log_loss", n_jobs=1, refit=True, error_score="raise",
                      cv=StratifiedGroupKFold(k, shuffle=True, random_state=seed))
    gs.fit(X[tr], y[tr], groups=groups[tr])
    return {"P": gs.best_estimator_.predict_proba(X[te]), "C": gs.best_params_["clf__C"]}


def diagnosis_probs(D: np.ndarray, cands, task: dict, lab_classes: list) -> np.ndarray:
    """Candidate log-probabilities -> class probabilities for one task (zero-shot)."""
    P = np.exp(D - D.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    if task["col"] == "c2":
        mal = np.array([c.group == "malignant" for c in cands])
        return np.column_stack([1 - P[:, mal].sum(1), P[:, mal].sum(1)])
    out = np.zeros((len(D), len(lab_classes)))
    for j, c in enumerate(cands):
        if c.subtype in lab_classes:
            out[:, lab_classes.index(c.subtype)] += P[:, j]
    missing = [s for s in lab_classes if s not in {c.subtype for c in cands}]
    if missing:
        raise SystemExit(f"vlmfeat.candidates name no label for subtypes {missing}")
    return out / out.sum(1, keepdims=True)


def run(cfg, out, inputs, ctx) -> dict:
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    from joblib import Parallel, delayed

    bc, rc, cc = cfg.bench, cfg.ridge, cfg.classify
    (out / "figures").mkdir(exist_ok=True)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    meta = pd.read_csv(ROOT / cfg.explore.metadata, dtype={"patient_id": str, "lesion_id": str}).set_index("isic_id")
    derm = meta.image_type.reindex(ids).eq("dermoscopic").values

    # features: questions (as S10), SkinGPT-R1 embeddings and candidate log-probabilities (S12)
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qk = qt[feature_mask(qt, cfg.model, patch=False)].reset_index(drop=True)
    Xq = pd.read_parquet(inputs["embed"] / f"X_image_{rc.source}.parquet").loc[ids, list(qk.question_id)].values.astype(float)
    ctrl = pd.read_parquet(inputs["embed"] / "X_image_orig.parquet").loc[ids]["DQC01"].values
    npz = np.load(inputs["vlmfeat"] / "embeddings.npz", allow_pickle=True)
    order = pd.Series(np.arange(len(npz["image_ids"])), index=npz["image_ids"].astype(str)).loc[ids].values
    E = {e: npz[e][order].astype(float) for e in bc.embeddings}
    dg = pd.read_parquet(inputs["vlmfeat"] / "diagnosis.parquet").set_index("image_id").loc[ids]
    cands = list(cfg.vlmfeat.candidates)
    D = dg[[c.label for c in cands]].values.astype(float)
    Dn = D - np.logaddexp.reduce(D, axis=1, keepdims=True)          # log P(label) within the candidates

    tasks = define_tasks(lab, list(bc.tasks))
    groups, _, _ = build_groups(man, meta, rc.group_by, rc.dup_hamming, 16)
    grid_q = grid_values(*rc.log10_C)
    grid_e = grid_values(*bc.log10_C)

    feats = {"questions": Xq, **{f"skingpt_emb:{e}": E[e] for e in bc.embeddings}, "skingpt_diagnosis_fitted": Dn}
    plan, jobs = [], []
    for t, T in tasks.items():
        rows = T["rows"]
        y = np.array([T["classes"].index(v) for v in lab[T["col"]].values[rows]])
        g = groups[rows]
        splits = outer_splits(lab.c10.values[rows], g, cc.outer_folds, cc.outer_repeats, cfg.seed)
        for m, X in feats.items():
            for r, f, tr, te in splits:
                plan.append((t, m, r, f, np.where(rows)[0][te]))
                seed = cfg.seed + 1000 * r + f
                if m == "questions":
                    jobs.append(delayed(fit_split)(("cols", None), X[rows], y, g, tr, te, cc.inner_folds, seed, grid_q))
                else:
                    jobs.append(delayed(fit_dense)(X[rows], y, g, tr, te, cc.inner_folds, seed, grid_e))
        for r, f, tr, te in splits:                                    # zero-shot scores on the same test folds
            te_abs = np.where(rows)[0][te]
            plan.append((t, "skingpt_diagnosis", r, f, te_abs))
            jobs.append({"P": diagnosis_probs(D[te_abs], cands, T, T["classes"]), "C": np.nan})
            if t == "s1_2c":
                plan.append((t, "control question", r, f, te_abs))
                jobs.append({"P": np.column_stack([1 - ctrl[te_abs], ctrl[te_abs]]), "C": np.nan})
    todo = [i for i, j in enumerate(jobs) if not isinstance(j, dict)]
    print(f"  {len(todo)} outer fits ({len(grid_e)} penalties for the embeddings, {len(grid_q)} for the questions)", flush=True)
    done = Parallel(n_jobs=bc.n_jobs, verbose=2)([jobs[i] for i in todo])
    fits = list(jobs)
    for i, d in zip(todo, done):
        fits[i] = d

    # ---------------- metrics
    spec_, sens = cfg.model.sens_at_spec, cc.sens_target
    fold_rows, oof, lam = [], {}, []
    for (t, m, r, f, te_abs), fit in zip(plan, fits):
        T = tasks[t]
        cls = T["classes"]
        yt = np.array([cls.index(v) for v in lab[T["col"]].values[te_abs]])
        base = {"task": t, "set": m, "grouping": "patient", "repeat": r, "fold": f, "C": fit["C"], "lambda": 1 / fit["C"]}
        for sub, msk in (("all images", np.ones(len(te_abs), bool)), ("dermoscopic only", derm[te_abs])):
            met = task_metrics(yt[msk], fit["P"][msk], cls, spec_, sens, cfg.model.ece_bins)
            fold_rows.append({**base, "subset": sub, "n_test": int(msk.sum()), **{k: v for k, v in met.items() if k != "confusion_matrix"}})
        oof.setdefault((t, m, r), {}).update({im: fit["P"][i] for i, im in enumerate(te_abs)})
        if not np.isnan(fit["C"]):
            grid = grid_q if m == "questions" else grid_e
            lam.append({"task": t, "set": m, "repeat": r, "fold": f, "lambda": 1 / fit["C"],
                        "at_grid_edge": bool(np.isclose(fit["C"], min(grid)) or np.isclose(fit["C"], max(grid)))})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "metrics_folds.csv", index=False)
    lam = pd.DataFrame(lam)
    lam.to_csv(out / "lambda_choices.csv", index=False)

    # the questions refitted here must reproduce S10 fold by fold (same folds, same model)
    s10 = pd.read_csv(inputs["ridge"] / "metrics_folds.csv")
    s10 = s10[(s10.set == "all") & (s10.grouping == "patient")]
    chk = folds[folds.set == "questions"].merge(s10, on=["task", "repeat", "fold", "subset"], suffixes=("", "_s10"))
    chk = chk.assign(metric=chk.task.map(primary))
    chk["diff"] = [abs(getattr(x, x.metric) - getattr(x, x.metric + "_s10")) for x in chk.itertuples()]
    chk[["task", "subset", "repeat", "fold", "metric", "diff", "n_test", "n_test_s10"]].to_csv(out / "check_s10.csv", index=False)
    if len(chk) != (folds.set == "questions").sum() or chk["diff"].max() > 1e-9 or (chk.n_test != chk.n_test_s10).any():
        raise RuntimeError(f"questions do not reproduce S10 (max difference {chk['diff'].max():.2e}); folds differ")

    pr, pooled = [], []
    for (t, m, r), d in oof.items():
        T = tasks[t]
        cls = T["classes"]
        idx = list(d)
        Pm = np.array([d[i] for i in idx])
        df = pd.DataFrame(Pm, columns=[f"p::{c}" for c in cls])
        df.insert(0, "image_id", np.array(ids)[idx])
        pr.append(df.assign(task=t, set=m, repeat=r, true=lab[T["col"]].values[idx]))
        yt = np.array([cls.index(v) for v in lab[T["col"]].values[idx]])
        for sub, msk in (("all images", np.ones(len(idx), bool)), ("dermoscopic only", derm[idx]), ("clinical only", ~derm[idx])):
            met = task_metrics(yt[msk], Pm[msk], cls, spec_, sens, cfg.model.ece_bins)
            pooled.append({"task": t, "set": m, "grouping": "patient", "repeat": r, "subset": sub, "n": int(msk.sum()),
                           **{k: v for k, v in met.items() if k != "confusion_matrix"}})
    oofdf = pd.concat(pr, ignore_index=True)
    oofdf.to_parquet(out / "oof_predictions.parquet", index=False)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "metrics_pooled_by_repeat.csv", index=False)
    summ = summarise(folds, pooled, sens)
    summ.to_csv(out / "summary.csv", index=False)
    paired = paired_vs(folds, "questions")
    paired.to_csv(out / "paired.csv", index=False)

    # zero-shot diagnosis: which candidate SkinGPT-R1 puts first, per subtype (all 1,000 images)
    top = pd.crosstab(lab.c10.values, np.array([c.label for c in cands])[D.argmax(1)])
    top.to_csv(out / "diagnosis_top_label_by_subtype.csv")

    methods = ["questions"] + [f"skingpt_emb:{e}" for e in bc.embeddings] + ["skingpt_diagnosis_fitted", "skingpt_diagnosis", "control question"]
    plots.bench_figure(summ, methods, method_name, out / "figures" / "benchmarks.png")
    roc = {}
    if "s1_2c" in tasks:
        sub = oofdf[(oofdf.task == "s1_2c") & (oofdf.repeat == 0)]
        main = ["questions", f"skingpt_emb:{bc.embeddings[0]}", "skingpt_diagnosis_fitted", "skingpt_diagnosis", "control question"]
        for m in main:
            s = sub[sub.set == m]
            roc[m] = ((s.true == "malignant").astype(int).values, s["p::malignant"].values)
        plots.bench_roc(roc, method_name, sens, out / "figures" / "roc_benign_vs_malignant.png")
    write_summary(out, cfg, lab, derm, summ, paired, lam, top, methods, tasks, len(qk), npz, cands)
    m0 = summ[(summ.task == "s1_2c") & (summ.subset == "all images")].set_index("set").fold_mean
    return {"key_metric": (f"s1_2c auroc questions={m0.get('questions', np.nan):.3f} "
                           f"skingpt_emb={m0.get('skingpt_emb:' + bc.embeddings[0], np.nan):.3f} "
                           f"skingpt_dx={m0.get('skingpt_diagnosis', np.nan):.3f}") if len(m0) else "",
            "s10_check_max_diff": float(chk["diff"].max())}


def paired_vs(folds, ref: str) -> pd.DataFrame:
    rows = []
    for (t, sub), g in folds.groupby(["task", "subset"]):
        pm = primary(t)
        base = g[g.set == ref].set_index(["repeat", "fold"])[pm]
        for s, h in g[g.set != ref].groupby("set"):
            d = (h.set_index(["repeat", "fold"])[pm] - base).dropna()
            rows.append({"task": t, "subset": sub, "set": s, "vs": ref, "mean_diff": d.mean(), "sd_diff": d.std(),
                         "folds_higher": int((d > 0).sum()), "n_folds": int(len(d))})
    return pd.DataFrame(rows)


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_summary(out, cfg, lab, derm, summ, paired, lam, top, methods, tasks, n_q, npz, cands):
    bc, cc = cfg.bench, cfg.classify
    hid = npz[bc.embeddings[0]].shape[1]
    L = ["# S13 bench: ridge on question answers against SkinGPT-R1, same folds", "",
         f"{len(lab)} images ({int(derm.sum())} dermoscopic); outer folds exactly as in S10 ({cc.outer_repeats} repeats of "
         f"{cc.outer_folds}-fold, stratified on the 10 subtypes, grouped by patient with near-duplicate pictures merged). "
         f"Questions: ridge on the Directional Score of {n_q} questions (S10 'all', reproduced fold by fold, see check_s10.csv). "
         f"SkinGPT-R1 embeddings ({hid} numbers per image): ridge with scaling and lambda refitted in every outer training "
         f"split; main embedding fixed in advance: {bc.embeddings[0]} ({emb_name(bc.embeddings[0])}). SkinGPT-R1 diagnosis: "
         f"its probability over {len(cands)} candidate labels, without reasoning.", ""]
    for t in tasks:
        L += [f"## {t} ({primary(t).replace('_', ' ')})", "",
              "| method | images | fold mean ± sd | pooled per repeat: mean (min to max) | " +
              ("specificity at 95% sensitivity |" if t == "s1_2c" else "macro-F1 |") + " lambda (median) |",
              "|---|---|---|---|---|---|"]
        s = summ[summ.task == t]
        for m in methods:
            for x in s[s.set == m].itertuples():
                extra = x.spec_at_sens if t == "s1_2c" else x.macro_f1
                L.append(f"| {method_name(m)} | {x.subset} | {_f(x.fold_mean)} ± {_f(x.fold_sd)} | "
                         f"{_f(x.pooled_mean)} ({_f(x.pooled_min)} to {_f(x.pooled_max)}) | {_f(extra, 2)} | {_f(x.lambda_median, 0)} |")
        L += ["", "Against the questions on the same folds (positive = the method scored higher):", "",
              "| method | images | mean difference | folds higher |", "|---|---|---|---|"]
        p = paired[paired.task == t]
        for m in methods[1:]:
            for x in p[p.set == m].itertuples():
                L.append(f"| {method_name(m)} | {x.subset} | {x.mean_diff:+.3f} ± {x.sd_diff:.3f} | {x.folds_higher} of {x.n_folds} |")
        L.append("")
    if len(lam):
        L += ["## Lambda", "", "| task | method | lambda median (range) | folds at the grid edge |", "|---|---|---|---|"]
        for (t, m), g in lam.groupby(["task", "set"]):
            L.append(f"| {t} | {method_name(m)} | {g['lambda'].median():.0f} ({g['lambda'].min():.0f} to {g['lambda'].max():.0f}) | "
                     f"{int(g.at_grid_edge.sum())} of {len(g)} |")
        L.append("")
    L += ["## SkinGPT-R1's first-choice label by subtype (all images)", "", top.to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(L))
