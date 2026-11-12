"""S6 Classify: nested cross-validation of six model families on the whole-image answers, for every task at once.

Tasks (labels from the source manifest CSV; folds stratified on label_10c and grouped by lesion):
  s1_2c            benign vs malignant. Indeterminate images are never fitted; each fold's model scores them.
  s1_3c            benign / indeterminate / malignant.
  flat_10c         all ten subtypes in one model (the baseline for the two-stage design).
  s2_<group>       subtypes within one stage-1 class, fitted and scored on that class only (true routing).
  cascade          s1_3c routes each image, then the s2 model of the predicted class names the subtype. Computed from the
                   out-of-fold predictions above, so no image is ever scored by a model that saw it.
Feature sources: orig = P(yes); dir / coh = Directional / Coherence Score (stages/s4_embed.py). Artefact questions are
left out (model.exclude_kinds). With classify.synthetic set, the answers are simulated (qaemb/synthetic.py) and the
stage also reports how well each importance measure recovers the planted questions.

Outputs: metrics_folds.csv, metrics_summary.csv, cascade.csv, oof_predictions.parquet, indeterminate_2c.csv,
importance.csv, question_ranking.csv, best_params.csv, [planted.csv, recovery.csv], figures/, summary.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import plots
from ..models.cv import run_all
from ..models.metrics import task_metrics
from ..questions import question_table
from .s5_model import feature_mask

UPSTREAM = ["manifest", "embed"]
VERSION = 2

SHORT = {
    "BENIGN MELANOCYTIC PROLIFERATIONS": "benign melanocytic",
    "BENIGN SOFT TISSUE PROLIFERATIONS - FIBRO-HISTIOCYTIC": "benign fibro-histiocytic",
    "BENIGN SOFT TISSUE PROLIFERATIONS - VASCULAR": "benign vascular",
    "FLAT MELANOTIC PIGMENTATIONS - NOT MELANOCYTIC NEVUS": "flat melanotic pigmentation",
    "INFLAMMATORY OR INFECTIOUS DISEASES": "inflammatory or infectious",
    "INDETERMINATE EPIDERMAL PROLIFERATIONS": "indeterminate epidermal",
    "INDETERMINATE MELANOCYTIC PROLIFERATIONS": "indeterminate melanocytic",
    "MALIGNANT ADNEXAL EPITHELIAL PROLIFERATIONS - FOLLICULAR": "malignant adnexal follicular",
    "MALIGNANT EPIDERMAL PROLIFERATIONS": "malignant epidermal",
    "MALIGNANT MELANOCYTIC PROLIFERATIONS": "malignant melanocytic",
}
GROUPS = ["benign", "indeterminate", "malignant"]
MODEL_NAMES = {"l1": "lasso (L1)", "l2": "ridge (L2)", "enet": "elastic net", "rf": "random forest",
               "hgb": "gradient boosting", "mlp": "MLP"}
SOURCE_NAMES = {"orig": "P(yes)", "dir": "Directional Score", "coh": "Coherence Score", "adj": "signed confidence",
                "conf": "confidence", "w": "P_w"}


def upstream(cfg) -> list[str]:
    """Synthetic runs need no images, so they read the manifest CSV directly and skip S0."""
    return [] if cfg.classify.synthetic else ["manifest", "embed"]


def manifest_frame(cfg, inputs) -> pd.DataFrame:
    if "manifest" in inputs:
        return pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    d = cfg.data
    src = pd.read_csv(ROOT / d.root / d.manifest, dtype=str, keep_default_na=False)
    if d.image_ids:
        src = src[src[d.id_column].isin(set(d.image_ids))]
    return pd.DataFrame({"image_id": src[d.id_column].values, "patient_id": src[cfg.split.group_column].values})


def key(cfg, ctx) -> dict:
    return {"classify": cfg.classify, "exclude_kinds": cfg.model.exclude_kinds, "exclude_scopes": cfg.model.exclude_scopes,
            "seed": cfg.seed, "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "qset_hash": ctx.qset_hash if cfg.classify.synthetic else None,
            "sens_at_spec": cfg.model.sens_at_spec, "ece_bins": cfg.model.ece_bins}


def primary(task: str) -> str:
    return "auroc" if task == "s1_2c" else "balanced_accuracy"


def load_labels(cfg, man: pd.DataFrame) -> pd.DataFrame:
    d, lc = cfg.data, cfg.classify.label_columns
    src = pd.read_csv(ROOT / d.root / d.manifest, dtype=str, keep_default_na=False).set_index(d.id_column)
    lab = pd.DataFrame({"image_id": man.image_id.values, "lesion": man.patient_id.values})
    lab["c2"] = src.loc[lab.image_id, lc.c2].str.strip().str.lower().replace("", "indeterminate").values
    lab["c3"] = src.loc[lab.image_id, lc.c3].str.strip().str.lower().values
    lab["c10"] = [SHORT.get(v.strip(), v.strip().lower()) for v in src.loc[lab.image_id, lc.c10]]
    bad = set(lab.c3) - set(GROUPS)
    if bad:
        raise SystemExit(f"{lc.c3} values outside {GROUPS}: {sorted(bad)}")
    return lab


def define_tasks(lab: pd.DataFrame, wanted) -> dict:
    sub_order = sorted(lab.c10.unique(), key=lambda s: (GROUPS.index(lab.c3[lab.c10 == s].iloc[0]), s))
    T = {
        "s1_2c": {"rows": lab.c3.isin(["benign", "malignant"]).values, "col": "c2", "classes": ["benign", "malignant"],
                  "extra": (lab.c3 == "indeterminate").values},
        "s1_3c": {"rows": np.ones(len(lab), bool), "col": "c3", "classes": GROUPS, "extra": None},
        "flat_10c": {"rows": np.ones(len(lab), bool), "col": "c10", "classes": sub_order, "extra": None},
    }
    for g in GROUPS:
        rows = (lab.c3 == g).values
        T[f"s2_{g}"] = {"rows": rows, "col": "c10", "classes": [s for s in sub_order if (lab.c10[rows] == s).any()],
                        "extra": ~rows}
    unknown = set(wanted) - set(T)
    if unknown:
        raise SystemExit(f"Unknown classify.tasks {sorted(unknown)}; choose from {list(T)}")
    return {t: T[t] for t in wanted}


def load_sources(cfg, ctx, inputs, man, lab, qt) -> tuple[dict, pd.DataFrame | None]:
    ids = list(man.image_id)
    if cfg.classify.synthetic:
        from ..synthetic import simulate

        X, planted = simulate(qt, ids, lab.c10.values, lab.c3.values, float(cfg.classify.synthetic.effect), cfg.seed)
        return {s: X[s] for s in cfg.classify.sources}, planted
    out = {}
    for s in cfg.classify.sources:
        f = inputs["embed"] / f"X_image_{s}.parquet"
        if not f.exists():
            raise SystemExit(f"{f.name} not in the embedding store; available: "
                             f"{sorted(p.stem.replace('X_image_', '') for p in inputs['embed'].glob('X_image_*.parquet'))}")
        out[s] = pd.read_parquet(f).loc[ids]
    return out, None


def run(cfg, out, inputs, ctx) -> dict:
    cc = cfg.classify
    man = manifest_frame(cfg, inputs)
    lab = load_labels(cfg, man)
    qt = (question_table(ctx.questions) if cc.synthetic else pd.read_parquet(inputs["embed"] / "questions.parquet"))
    keep = feature_mask(qt, cfg.model, patch=False)
    qk = qt[keep].reset_index(drop=True)
    qids = list(qk.question_id)
    sources, planted = load_sources(cfg, ctx, inputs, man, lab, qt)
    tasks = define_tasks(lab, list(cc.tasks))
    groups = lab.lesion.values
    spec, sens = cfg.model.sens_at_spec, cc.sens_target
    (out / "figures").mkdir(exist_ok=True)

    specs, meta = {}, {}
    for s, Xdf in sources.items():
        X = Xdf[qids].values.astype(np.float64)
        if np.isnan(X).any():
            raise SystemExit(f"Source {s} has missing answers")
        for t, T in tasks.items():
            rows, cls = T["rows"], T["classes"]
            y = lab[T["col"]].values[rows]
            yi = np.array([cls.index(v) for v in y])
            Xe = X[T["extra"]] if T["extra"] is not None else None
            meta[t] = (rows, cls, y, yi)
            for m in cc.models:
                specs[(s, t, m)] = (m, X[rows], yi, lab.c10.values[rows], groups[rows], Xe, cfg.seed)
    results = run_all(specs, cc)

    fold_rows, pooled_rows, pred_rows, imp_rows, param_rows, indet_rows = [], [], [], [], [], []
    for (s, t, m), res in results.items():
        rows, cls, y, yi = meta[t]
        T = tasks[t]
        base = {"source": s, "task": t, "model": m}
        for fr in res["folds"]:
            te = fr["te"]
            met = task_metrics(yi[te], res["oof"][fr["repeat"], te], cls, spec, sens, cfg.model.ece_bins)
            fold_rows.append({**base, "repeat": fr["repeat"], "fold": fr["fold"], "n_test": len(te),
                              **{k: v for k, v in met.items() if k != "confusion_matrix"}})
            param_rows.append({**base, "repeat": fr["repeat"], "fold": fr["fold"], **fr["params"]})
        for r in range(cc.outer_repeats):
            met = task_metrics(yi, res["oof"][r], cls, spec, sens, cfg.model.ece_bins)
            pooled_rows.append({**base, "repeat": r, **met})
            pr = pd.DataFrame(res["oof"][r], columns=[f"p::{c}" for c in cls])
            pr.insert(0, "image_id", lab.image_id.values[rows])
            pred_rows.append(pr.assign(**base, repeat=r, true=y))
        if t == "s1_2c" and res["extra"] is not None:
            indet_rows.append(pd.DataFrame({**base, "image_id": lab.image_id.values[T["extra"]],
                                            "subtype": lab.c10.values[T["extra"]],
                                            "p_malignant": res["extra"][:, :, 1].mean(0)}))
        imp_rows.append(importance_table(res, base, cls, qids))

    folds = pd.DataFrame(fold_rows)
    pooled = pd.DataFrame(pooled_rows)
    folds.to_csv(out / "metrics_folds.csv", index=False)
    pooled.drop(columns="confusion_matrix").to_csv(out / "metrics_pooled_by_repeat.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out / "best_params.csv", index=False)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_parquet(out / "oof_predictions.parquet", index=False)
    imp = pd.concat(imp_rows, ignore_index=True)
    imp.to_csv(out / "importance.csv", index=False)
    indet = pd.concat(indet_rows, ignore_index=True) if indet_rows else pd.DataFrame()
    indet.to_csv(out / "indeterminate_2c.csv", index=False)

    summary_tab = summarise(folds, pooled)
    summary_tab.to_csv(out / "metrics_summary.csv", index=False)
    casc = cascade(results, tasks, lab, cc, spec, sens, cfg.model.ece_bins)
    casc.drop(columns="confusion_matrix", errors="ignore").to_csv(out / "cascade.csv", index=False)
    ranking = question_ranking(imp, qk, planted)
    ranking.to_csv(out / "question_ranking.csv", index=False)
    rec = None
    if planted is not None:
        planted.to_csv(out / "planted.csv", index=False)
        rec = recovery(imp, planted, tasks, lab)
        rec.to_csv(out / "recovery.csv", index=False)

    figures(out / "figures", preds, folds, imp, indet, pooled, casc, tasks, cc, planted, lab)
    write_summary(out, cfg, lab, tasks, qk, summary_tab, casc, ranking, indet, rec, planted, preds, folds)

    s0 = cc.sources[0]
    t0 = list(tasks)[0]
    best = summary_tab[(summary_tab.source == s0) & (summary_tab.task == t0) & (summary_tab.metric == primary(t0))]
    best = best.sort_values("fold_mean").iloc[-1]
    key_metric = f"{best.task} {best.metric}={best.fold_mean:.3f} ({best.model}, {s0})"
    print("  " + key_metric)
    return {"n_images": len(lab), "n_questions": len(qids), "synthetic": bool(cc.synthetic),
            "sources": list(cc.sources), "models": list(cc.models), "tasks": list(tasks),
            "outer": f"{cc.outer_repeats}x{cc.outer_folds}", "key_metric": key_metric}


def importance_table(res, base, cls, qids) -> pd.DataFrame:
    """Long table: one row per (output, question). Binary tasks have one output (the positive class)."""
    outs = [cls[1]] if len(cls) == 2 else list(cls)
    frames = []
    shap = res["shap"]
    for j, o in enumerate(outs):
        d = {"output": o, "question_id": qids, "shap_mean_abs": shap[j]}
        if res["coef"] is not None:
            c = res["coef"][:, j, :]
            mean = c.mean(0)
            d.update({"coef_mean": mean, "coef_sd": c.std(0),
                      "coef_sign_agreement": (np.sign(c) == np.sign(mean)).mean(0),
                      "n_selected": (np.abs(c) > 1e-8).sum(0), "n_fits": res["n_fits"]})
        frames.append(pd.DataFrame(d))
    if len(outs) > 1:
        d = {"output": "overall", "question_id": qids, "shap_mean_abs": shap.sum(0)}
        if res["coef"] is not None:
            c = np.abs(res["coef"]).max(1)                           # largest |coef| over classes, per fit
            d.update({"coef_mean": c.mean(0), "coef_sd": c.std(0), "n_selected": res["selected"], "n_fits": res["n_fits"]})
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    for k, v in base.items():
        df.insert(0, k, v)
    return df


def summarise(folds: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in folds.columns if c not in ("source", "task", "model", "repeat", "fold", "n_test", "n", "n_pos")]
    f = folds.melt(id_vars=["source", "task", "model"], value_vars=metrics, var_name="metric").dropna()
    a = f.groupby(["source", "task", "model", "metric"]).value.agg(fold_mean="mean", fold_sd="std", n_folds="count")
    p = pooled.melt(id_vars=["source", "task", "model"], value_vars=[m for m in metrics if m in pooled],
                    var_name="metric").dropna()
    b = p.groupby(["source", "task", "model", "metric"]).value.mean().rename("pooled_oof")
    return a.join(b).reset_index()


def cascade(results, tasks, lab, cc, spec, sens, bins) -> pd.DataFrame:
    """Two-stage prediction from out-of-fold parts: P(subtype) = P(group | s1_3c) x P(subtype | group, s2_group)."""
    need = {"s1_3c", "flat_10c"} | {f"s2_{g}" for g in GROUPS}
    if not need <= set(tasks):
        return pd.DataFrame()
    classes = tasks["flat_10c"]["classes"]
    y10 = np.array([classes.index(v) for v in lab.c10])
    yg = np.array([GROUPS.index(v) for v in lab.c3])
    rows = []
    for s in cc.sources:
        for m in cc.models:
            for r in range(cc.outer_repeats):
                P3 = results[(s, "s1_3c", m)]["oof"][r]
                P10 = np.zeros((len(lab), len(classes)))
                sub = {}
                for gi, g in enumerate(GROUPS):
                    T, res = tasks[f"s2_{g}"], results[(s, f"s2_{g}", m)]
                    Pg = np.zeros((len(lab), len(T["classes"])))
                    Pg[T["rows"]] = res["oof"][r]
                    Pg[T["extra"]] = res["extra"][r]
                    sub[gi] = (T["classes"], Pg)
                    for j, c in enumerate(T["classes"]):
                        P10[:, classes.index(c)] = P3[:, gi] * Pg[:, j]
                route = P3.argmax(1)
                hard = np.array([classes.index(sub[route[i]][0][sub[route[i]][1][i].argmax()]) for i in range(len(lab))])
                met = task_metrics(y10, P10, classes, spec, sens, bins)
                wrong = hard != y10
                misrouted = route != yg
                from sklearn.metrics import balanced_accuracy_score, f1_score
                rows.append({"source": s, "model": m, "repeat": r,
                             "soft_balanced_accuracy": met["balanced_accuracy"], "soft_macro_f1": met["macro_f1"],
                             "soft_macro_auroc_ovr": met["macro_auroc_ovr"], "soft_log_loss": met["log_loss"],
                             "hard_balanced_accuracy": balanced_accuracy_score(y10, hard),
                             "hard_macro_f1": f1_score(y10, hard, average="macro"),
                             "stage1_routing_error": misrouted.mean(),
                             "subtype_accuracy_when_routed_correctly": (~wrong[~misrouted]).mean(),
                             "share_of_errors_from_routing": (misrouted & wrong).sum() / max(wrong.sum(), 1),
                             "confusion_matrix": pd.crosstab(pd.Categorical(y10, range(len(classes))),
                                                             pd.Categorical(hard, range(len(classes))),
                                                             dropna=False).values.tolist()})
    return pd.DataFrame(rows)


def question_ranking(imp: pd.DataFrame, qk: pd.DataFrame, planted) -> pd.DataFrame:
    """One row per (source, task, question): ridge weight, lasso / elastic-net selection counts, SHAP rank per model,
    and the mean SHAP rank over models (1 = most important)."""
    out = []
    one = imp[(imp.output == "overall") | ~imp.set_index(["source", "task", "model"]).index.isin(
        imp[imp.output == "overall"].set_index(["source", "task", "model"]).index)]
    for (s, t), g in one.groupby(["source", "task"]):
        w = qk[["question_id", "question"] + [c for c in ("topic", "scope") if c in qk]].copy()
        for m, gm in g.groupby("model"):
            gm = gm.set_index("question_id").reindex(w.question_id)
            if gm.shap_mean_abs.notna().any():
                w[f"shap_rank_{m}"] = gm.shap_mean_abs.rank(ascending=False, method="min").values
            if m == "l2":
                w["ridge_coef"] = gm.coef_mean.values
                w["ridge_coef_sd"] = gm.coef_sd.values
                w["ridge_abs_rank"] = gm.coef_mean.abs().rank(ascending=False, method="min").values
            if m in ("l1", "enet") and "n_selected" in gm:
                w[f"{m}_n_selected"] = gm.n_selected.values
                w[f"{m}_n_fits"] = gm.n_fits.values
        ranks = [c for c in w if c.startswith("shap_rank_")]
        w["mean_shap_rank"] = w[ranks].mean(1) if ranks else np.nan
        if planted is not None:
            w["planted_role"] = w.question_id.map(planted.set_index("question_id").role)
        w.insert(0, "task", t)
        w.insert(0, "source", s)
        out.append(w.sort_values("mean_shap_rank"))
    return pd.concat(out, ignore_index=True)


def recovery(imp: pd.DataFrame, planted: pd.DataFrame, tasks: dict, lab: pd.DataFrame) -> pd.DataFrame:
    """Synthetic only: how well each importance measure separates the questions that carry signal for the task."""
    from sklearn.metrics import roc_auc_score

    rows = []
    one = imp[imp.output == "overall"]
    one = pd.concat([one, imp[~imp.set_index(["source", "task", "model"]).index.isin(
        one.set_index(["source", "task", "model"]).index)]])
    for t, T in tasks.items():
        rows_t = T["rows"]
        c2s = {c: sorted(set(lab.c10[rows_t][lab[T["col"]][rows_t] == c])) for c in T["classes"]}
        rel = __import__("qaemb.synthetic", fromlist=["relevance"]).relevance(planted, c2s)
        for (s, m), g in one[one.task == t].groupby(["source", "model"]):
            g = g.set_index("question_id")
            r = rel.reindex(g.index)
            hit = (r > 0).values
            k = int(hit.sum())
            measures = {"shap": g.shap_mean_abs}
            if "coef_mean" in g and g.coef_mean.notna().any():
                measures["abs_coef"] = g.coef_mean.abs()
            if m in ("l1", "enet"):
                measures["n_selected"] = g.n_selected
            for name, v in measures.items():
                v = v.astype(float).values
                if np.isnan(v).all() or k == 0:
                    continue
                top = np.argsort(-v, kind="stable")[:k]
                rows.append({"source": s, "task": t, "model": m, "measure": name, "n_relevant": k,
                             "auroc_relevant_vs_not": roc_auc_score(hit, v),
                             "recall_at_k": hit[top].mean(),
                             "n_selected_median_relevant": float(np.median(g.n_selected[hit])) if name == "n_selected" else np.nan,
                             "n_selected_median_other": float(np.median(g.n_selected[~hit])) if name == "n_selected" else np.nan})
    return pd.DataFrame(rows)


def figures(fdir: Path, preds, folds, imp, indet, pooled, casc, tasks, cc, planted, lab):
    models, sources = list(cc.models), list(cc.sources)
    mnames = [MODEL_NAMES.get(m, m) for m in models]
    # 1. Observed vs predicted (calibration), one figure per task.
    for t, T in tasks.items():
        curves = {}
        cls = T["classes"]
        for (m, s), g in preds[preds.task == t].groupby(["model", "source"]):
            P = g[[f"p::{c}" for c in cls]].values
            yi = np.array([cls.index(v) for v in g.true])
            if len(cls) == 2:
                curves[(MODEL_NAMES[m], s)] = (P[:, 1], (yi == 1).astype(float))
            else:
                curves[(MODEL_NAMES[m], s)] = (P.max(1), (P.argmax(1) == yi).astype(float))
        if len(cls) == 2:
            lbl = ("predicted P(malignant)", "observed share malignant")
        else:
            lbl = ("predicted probability of the chosen class", "observed share correct")
        plots.calibration_grid(curves, mnames, sources, f"{t}: observed vs predicted, out-of-fold, all repeats "
                               f"(lines: {', '.join(SOURCE_NAMES.get(s, s) for s in sources)})", *lbl,
                               fdir / f"calibration_{t}.png")
    # 2. Each model against ridge, per outer fold.
    if "l2" in models:
        for s in sources:
            panels = []
            for t in tasks:
                met = primary(t)
                ref = folds[(folds.source == s) & (folds.task == t) & (folds.model == "l2")].set_index(["repeat", "fold"])[met]
                for m in models:
                    if m == "l2":
                        continue
                    o = folds[(folds.source == s) & (folds.task == t) & (folds.model == m)].set_index(["repeat", "fold"])[met]
                    panels.append((f"{t} ({met.replace('_', ' ')})\n{MODEL_NAMES[m]}", ref.values, o.reindex(ref.index).values, None))
            plots.paired_grid(panels, f"Per-fold score of each model against ridge (L2), {SOURCE_NAMES.get(s, s)}",
                              "ridge (L2) score in the same fold", "model score", fdir / f"model_vs_ridge_{s}.png",
                              ncols=max(1, len(models) - 1))
    # 3. Do the importance measures agree? Ranks, 1 = most important.
    for s in sources:
        panels = []
        for t in tasks:
            g = imp[(imp.source == s) & (imp.task == t)]
            outs = g.output.unique()
            g = g[g.output == ("overall" if "overall" in outs else outs[0])]
            def rank(m, col):
                v = g[g.model == m].set_index("question_id")[col]
                return v.abs().rank(ascending=False, method="min") if v.notna().any() else None
            hl = None
            if planted is not None:
                c2s = {c: sorted(set(lab.c10[tasks[t]["rows"]][lab[tasks[t]["col"]][tasks[t]["rows"]] == c]))
                       for c in tasks[t]["classes"]}
                from ..synthetic import relevance
                rel = relevance(planted, c2s)
            ref = rank("l2", "shap_mean_abs") if "l2" in models else None
            pairs = []
            if "l2" in models:
                pairs.append(("ridge: |coef| vs SHAP", rank("l2", "coef_mean"), ref))
            for m in models:
                if m != "l2" and ref is not None:
                    pairs.append((f"{MODEL_NAMES[m]} vs ridge (SHAP)", ref, rank(m, "shap_mean_abs")))
            for title, a, b in pairs:
                if a is None or b is None:
                    continue
                b = b.reindex(a.index)
                if planted is not None:
                    hl = (rel.reindex(a.index) > 0).values
                panels.append((f"{t}\n{title}", a.values, b.values, hl))
        if panels:
            plots.paired_grid(panels, f"Question importance ranks (1 = most important), {SOURCE_NAMES.get(s, s)}",
                              "rank (x: first named)", "rank (y: second named)", fdir / f"importance_ranks_{s}.png",
                              ncols=len(models), highlight_label="planted, relevant to the task" if planted is not None else None)
    # 4. Confusion matrices: best model per multiclass task (first source, repeat 0), and the cascade.
    s0 = sources[0]
    for t, T in tasks.items():
        if len(T["classes"]) == 2:
            continue
        sub = pooled[(pooled.source == s0) & (pooled.task == t)]
        bm = sub.groupby("model").balanced_accuracy.mean().idxmax()
        cm = sub[(sub.model == bm) & (sub.repeat == 0)].confusion_matrix.iloc[0]
        plots.confusion(cm, T["classes"], f"{t}, {MODEL_NAMES[bm]}, {SOURCE_NAMES.get(s0, s0)} (repeat 0)",
                        fdir / f"confusion_{t}.png")
    if not casc.empty:
        sub = casc[casc.source == s0]
        bm = sub.groupby("model").hard_balanced_accuracy.mean().idxmax()
        cm = sub[(sub.model == bm) & (sub.repeat == 0)].confusion_matrix.iloc[0]
        plots.confusion(cm, tasks["flat_10c"]["classes"], f"cascade, {MODEL_NAMES[bm]}, {SOURCE_NAMES.get(s0, s0)} (repeat 0)",
                        fdir / "confusion_cascade.png")
    # 5. Where do indeterminate lesions fall on the benign-malignant scale?
    if not indet.empty:
        for s in sources:
            groups = {}
            for m in models:
                g = preds[(preds.task == "s1_2c") & (preds.source == s) & (preds.model == m)]
                pm = g.groupby(["image_id", "true"])["p::malignant"].mean().reset_index()
                d = {"benign (out-of-fold)": pm[pm.true == "benign"]["p::malignant"].values,
                     "malignant (out-of-fold)": pm[pm.true == "malignant"]["p::malignant"].values}
                ind = indet[(indet.source == s) & (indet.model == m)]
                for st, gg in ind.groupby("subtype"):
                    d[st] = gg.p_malignant.values
                groups[MODEL_NAMES[m]] = d
            plots.strips(groups, [MODEL_NAMES[m] for m in models],
                         f"P(malignant) from the 2-class model, {SOURCE_NAMES.get(s, s)}", "P(malignant)",
                         fdir / f"indeterminate_2c_{s}.png")


def _pm(a, b) -> str:
    return f"{a:.3f} ± {b:.3f}" if np.isfinite(b) else f"{a:.3f}"


def write_summary(out, cfg, lab, tasks, qk, tab, casc, ranking, indet, rec, planted, preds, folds):
    cc = cfg.classify
    L = ["# Classifier results", ""]
    if cc.synthetic:
        L += [f"**Synthetic answers** (effect = {cc.synthetic.effect}): the labels are the real manifest labels, the "
              "answers are simulated with planted informative questions (qaemb/synthetic.py). Numbers here test the "
              "code, not the method.", ""]
    L += [f"{len(lab)} images, {len(qk)} questions (artefact questions left out). Each model was scored by "
          f"{cc.outer_repeats} repeats of {cc.outer_folds}-fold cross-validation, stratified on the 10 subtypes; "
          f"hyperparameters were chosen by {cc.inner_folds}-fold cross-validation inside each training set, so no score "
          "comes from an image the model or its tuning saw.", "",
          "How to read the scores:",
          "- **AUROC** (2-class): the chance that a random malignant image gets a higher P(malignant) than a random benign "
          "one. 0.5 is chance, 1 is perfect.",
          f"- **Specificity at {int(cc.sens_target * 100)}% sensitivity** (2-class): the share of benign images correctly "
          f"called benign when the threshold is set to catch {int(cc.sens_target * 100)}% of malignant ones.",
          "- **Balanced accuracy** (3- and 10-class): the average, over classes, of the share of that class predicted "
          "correctly. Chance is 1 / number of classes (0.33 for 3 classes, 0.10 for 10).",
          "- **Macro-F1**: the average over classes of the F1 score, which balances missed and wrongly assigned images.",
          "- Each cell is the mean ± standard deviation over the "
          f"{cc.outer_repeats * cc.outer_folds} test folds.", ""]

    L += ["## Task sizes", "", "| task | images fitted | classes (images per class) |", "|---|---|---|"]
    for t, T in tasks.items():
        cnt = lab[T["col"]][T["rows"]].value_counts()
        L.append(f"| {t} | {int(T['rows'].sum())} | " + "; ".join(f"{c} ({cnt.get(c, 0)})" for c in T["classes"]) + " |")
    L.append("")

    if rec is not None and rec.empty:
        L += ["## Validity check: no signal", "",
              "No question was made informative (effect = 0), so every score should sit at chance: AUROC 0.5, "
              "balanced accuracy 1 / number of classes. Anything clearly above chance would mean information leaking "
              "from test images into training or tuning.", ""]
    elif rec is not None:
        L += ["## Validity check: are the planted questions found?", "",
              "For each task, a question counts as relevant if the simulation made its answers differ between the "
              "task's classes. *AUROC relevant vs not* is the chance that a random relevant question ranks above a random "
              "uninformative one under that importance measure (0.5 = no better than chance); *recall at k* is the share "
              "of the k relevant questions found among the top k.", ""]
        s0 = cc.sources[0]
        r = rec[rec.source == s0].pivot_table(index=["task", "n_relevant"], columns=["model", "measure"],
                                               values="auroc_relevant_vs_not").round(2)
        r.columns = [f"{MODEL_NAMES[m]}: {meas}" for m, meas in r.columns]
        L += [f"AUROC relevant vs not ({SOURCE_NAMES.get(s0, s0)}):", "", r.reset_index().to_markdown(index=False), ""]

    for s in cc.sources:
        L += [f"## Results: {SOURCE_NAMES.get(s, s)} (`{s}`)", ""]
        rows = []
        for t in tasks:
            met = primary(t)
            r = {"task": t, "metric": met.replace("_", " ")}
            for m in cc.models:
                x = tab[(tab.source == s) & (tab.task == t) & (tab.model == m) & (tab.metric == met)]
                r[MODEL_NAMES[m]] = _pm(x.fold_mean.iloc[0], x.fold_sd.iloc[0]) if len(x) else "-"
            rows.append(r)
        if "s1_2c" in tasks:
            met = f"spec_at_sens{int(cc.sens_target * 100)}"
            r = {"task": "s1_2c", "metric": f"specificity at {int(cc.sens_target * 100)}% sensitivity"}
            for m in cc.models:
                x = tab[(tab.source == s) & (tab.task == "s1_2c") & (tab.model == m) & (tab.metric == met)]
                r[MODEL_NAMES[m]] = _pm(x.fold_mean.iloc[0], x.fold_sd.iloc[0]) if len(x) else "-"
            rows.append(r)
        for t in tasks:
            r = {"task": t, "metric": "macro F1"}
            for m in cc.models:
                x = tab[(tab.source == s) & (tab.task == t) & (tab.model == m) & (tab.metric == "macro_f1")]
                r[MODEL_NAMES[m]] = _pm(x.fold_mean.iloc[0], x.fold_sd.iloc[0]) if len(x) else "-"
            rows.append(r)
        L += [pd.DataFrame(rows).to_markdown(index=False), ""]
        if not casc.empty:
            c = casc[casc.source == s].groupby("model").mean(numeric_only=True).drop(columns="repeat")
            flat = tab[(tab.source == s) & (tab.task == "flat_10c") & (tab.metric == "balanced_accuracy")].set_index("model").pooled_oof
            c.insert(0, "flat_10c_balanced_accuracy", flat)
            c.index = [MODEL_NAMES[m] for m in c.index]
            L += ["Two-stage cascade against the flat 10-class model (out-of-fold, mean over repeats). *Hard* routes each "
                  "image to one stage-1 class; *soft* multiplies the probabilities. *Stage-1 routing error* is the share "
                  "of images sent to the wrong group; *share of errors from routing* is the share of wrong subtype calls "
                  "that were already misrouted at stage 1.", "",
                  c[["flat_10c_balanced_accuracy", "hard_balanced_accuracy", "soft_balanced_accuracy", "hard_macro_f1",
                     "stage1_routing_error", "subtype_accuracy_when_routed_correctly", "share_of_errors_from_routing"]]
                  .round(3).to_markdown(), ""]
        if not indet.empty:
            d = indet[indet.source == s].groupby(["model", "subtype"]).p_malignant.median().unstack().round(3)
            d.index = [MODEL_NAMES[m] for m in d.index]
            L += ["Median P(malignant) the 2-class model gives indeterminate images (never fitted on them):", "",
                  d.to_markdown(), ""]

    s0 = cc.sources[0]
    L += [f"## Most important questions ({SOURCE_NAMES.get(s0, s0)})", "",
          "Ranked by the mean SHAP rank over all models (1 = most important). *Ridge coef* is the standardised weight "
          "(positive = pushes towards the class named in the task; for multiclass tasks, the largest |weight| over "
          "classes). *Lasso kept* is the number of the outer fits in which the lasso kept the question.", ""]
    for t in tasks:
        g = ranking[(ranking.source == s0) & (ranking.task == t)].head(10)
        cols = ["question_id", "question", "ridge_coef", "l1_n_selected", "mean_shap_rank"] + \
               (["planted_role"] if planted is not None else [])
        g = g[[c for c in cols if c in g]].rename(columns={"ridge_coef": "ridge coef", "l1_n_selected": "lasso kept",
                                                             "mean_shap_rank": "mean SHAP rank", "planted_role": "planted as"})
        L += [f"**{t}**", "", g.round(3).to_markdown(index=False), ""]

    L += ["## Figures", ""]
    for p in sorted((out / "figures").glob("*.png")):
        L.append(f"- [{p.name}](figures/{p.name})")
    L += ["", "## Files", "",
          "- metrics_summary.csv: every metric, mean and SD over test folds, and the pooled out-of-fold value",
          "- metrics_folds.csv, metrics_pooled_by_repeat.csv, best_params.csv: per fold / per repeat detail",
          "- oof_predictions.parquet: out-of-fold class probabilities per image, model, source and repeat",
          "- importance.csv: per question and output class: ridge/lasso/elastic-net weights, lasso selection count, mean |SHAP|",
          "- question_ranking.csv: one row per question and task, all importance measures side by side",
          "- cascade.csv, indeterminate_2c.csv" + (", planted.csv, recovery.csv" if planted is not None else ""), ""]
    (out / "summary.md").write_text("\n".join(L))
