"""S9 Explore: unsupervised structure of the answers, as tables, figures and one interactive page (PROJECT.md, Plan
item B, answers; and the first half of item D, covariance within topics).

Questions (all questions as written except the control question; artefact questions included and flagged):
  answer clusters   Ward hierarchical clustering on correlation distance 1 - r, r = Pearson correlation of the standardised
                    scores across the 1,000 images (questions answered alike sit together; opposite answers far apart)
  text clusters     Ward clustering on 1 - cosine similarity of question text embeddings (Qwen3-Embedding-8B, DermLIP-PanDerm)
  agreement         adjusted mutual information between answer clusters, text clusters, topics, grounding and the
                    first cited source, each against shuffled labels
  topic cohesion    mean r between questions of the same topic against random groups of the same sizes (Plan item D);
                    the same for questions citing a common paper; answer r against wording similarity, same topic or not
  dimensions        PCA of the standardised scores with parallel analysis (how many components beat shuffled answers)
Images:
  map and clusters  UMAP and Ward clustering on the leading principal components of the answers
  association       adjusted mutual information of the image clusters, and variance explained in each leading
                    component, by diagnosis, group, image type, institution, body site, sex and age (ISIC metadata)
The ten-subtype model (payload key ten2; see tenclass_payload): the paper leads with S10's ten-subtype ridge, so the
page data also carry, for that model, the per-image out-of-fold probabilities and their benign-against-malignant reading
(S11's collapse), its ROC and per-fold AUROC, question importance (contrast and spread of the final model's weights, with
stability over S10's 25 outer training splits), marker questions per subtype, and the two worked cases held out by
patient. The two-class keys (eval, imp, cases, thr) are unchanged.
  ten2.classes            the ten subtypes (= subtypes, model order); ten2.mal / ten2.ben: indices of malignant / benign ones
  ten2.img.p              [1000][10] out-of-fold probabilities, mean over S10's 5 repeats (4 d.p.)
  ten2.img.pm             [1000] collapsed P(malignant), per repeat then averaged (4 d.p.); indeterminate images included
  ten2.roc.all / .derm    ROC of img.pm on the 800 benign + malignant images (all / dermoscopic): {pts, auc, op}, as
                          eval.roc; the direct question on the same images is eval.roc.ctrl_all / ctrl_derm
  ten2.folds              per outer fold (25) and subset: {subset, repeat, fold, n, n_mal, auc, spec (at 95% sensitivity,
                          read on the fold)}; ten2.fold_summary[subset]: auc_mean, auc_sd, auc_min, auc_max, spec_mean,
                          spec_sd, n_folds; ten2.pooled_auc[subset]: AUROC pooled per repeat (5 values; S11's numbers)
  ten2.imp                per question, aligned with q.id (null for questions outside the model): contrast, spread,
                          w ([10] class weights), contrast_rank (by |contrast|) and spread_rank (1 = largest), sign_share,
                          contrast_rank_med, spread_rank_med, contrast_fold_sd (over the 25 refits); C, n_folds
  ten2.markers[subtype]   the 5 largest positive weights of that subtype: {q, w, p_in, p_out} (mean P(yes) in / outside)
  ten2.cases[]            {id, i, sub, contrast_sub, p [10], p_mal, pred, logratio = log(p[sub]/p[contrast_sub]), d0,
                          sum_terms, oof_p [10], oof_p_mal, C, n_train, n_left_out, max_err, terms: [{q, x, z, dw, c, py,
                          cf}]}, logratio = d0 + sum of terms.c
  ten2.check              reproduction checks (S11 collapsed AUROC, S11 model 1, S10 fold refits, case decompositions)
  bench.models[]          the ten-subtype models of S13 (S11 model numbers 1 to 9, see bench_payload) scored per outer
                          fold: {model, no, n_folds, auroc_all, auroc_derm, auroc_nm, bacc}, each {mean, sd, min, max} over the 25
                          folds (bacc null for the direct question); bench.check: rebuilt folds against S13's fold metrics
                          and model numbers against S11. Per fold in bench_folds.csv.
Outputs: explorer.html (self-contained), narrative.html (the same data in the order of the paper; copied to
project_narrative/index.html), question_table.csv, question_clusters.csv, topic_cohesion.csv, agreement.csv,
text_vs_answers.csv, pca.csv, pc_association.csv, image_table.csv, image_cluster_association.csv, tenclass_folds.csv,
bench_folds.csv, tenclass_importance.csv, tenclass.md, figures/, summary.md.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, file_hash
from ..models import explore as ex
from ..models import plots
from .s6_classify import GROUPS, SOURCE_NAMES, load_labels
from .s11_subtypes import SECTION_ORDER, SECTIONS

UPSTREAM = ["manifest", "embed", "importance", "ridge", "subtypes", "thresholds", "qclusters", "bench"]
VERSION = 7   # 4: also writes narrative.html (project_narrative/); 5: + payload key ten2 (ten-subtype model); 6: ten-subtype worked cases chosen by that model, with their image; 7: + payload key bench (fold-level scores of the ten-subtype models of S13)
TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "explore.html"
NARRATIVE = TEMPLATE.parent / "narrative.html"   # same payload, five chapters in the order of the paper
FONTS = [  # (family, style, file): embedded so the page needs no network (templates/fonts/README.md)
    ("Garamond", "normal", "garamond.woff2"), ("Garamond", "italic", "garamond-italic.woff2")]


def font_css() -> str:
    out = []
    for fam, style, f in FONTS:
        b64 = base64.b64encode((TEMPLATE.parent / "fonts" / f).read_bytes()).decode()
        out.append(f"@font-face {{ font-family: '{fam}'; font-style: {style}; font-weight: 400 800; font-display: swap; "
                   f"src: url(data:font/woff2;base64,{b64}) format('woff2'); }}")
    return "\n".join(out)


def key(cfg, ctx) -> dict:
    e = cfg.explore
    files = [e.metadata, *e.text_embeddings.values(), e.text_questions]
    return {"explore": e, "label_columns": cfg.classify.label_columns, "seed": cfg.seed,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest),
            "outer": [cfg.classify.outer_folds, cfg.classify.outer_repeats], "sens_target": cfg.classify.sens_target,
            "reproduce_tol": cfg.thresholds.reproduce_tol,
            "files": {f: file_hash(ROOT / f) for f in files}, "template": file_hash(TEMPLATE), "narrative": file_hash(NARRATIVE),
            "fonts": {f: file_hash(TEMPLATE.parent / "fonts" / f) for _, _, f in FONTS}}


# ------------------------------------------------------------------ inputs
def short_institution(s: str) -> str:
    s = str(s).replace("\r", " ").replace("\n", " ")
    if s.startswith("Title:"):
        s = s.split("Source:")[-1]
    s = s.split(",")[0].strip()
    rename = {"Memorial Sloan Kettering Cancer Center": "Memorial Sloan Kettering",
              "Federal University of Espírito Santo (UFES)": "UFES (Brazil)",
              "Hospital Clínic de Barcelona": "Hospital Clínic Barcelona", "MILK study team": "MILK study"}
    for k, v in rename.items():
        if s.startswith(k[:25]):
            return v
    return s[:38]


def load_metadata(cfg, image_ids, min_inst: int) -> pd.DataFrame:
    m = pd.read_csv(ROOT / cfg.explore.metadata, dtype={"patient_id": str, "lesion_id": str}).set_index("isic_id")
    m = m.loc[image_ids]
    inst = m.attribution.map(short_institution)
    counts = inst.value_counts()
    small = counts[counts < min_inst].index
    inst = inst.where(~inst.isin(small), f"Other ({len(small)} institutions)")
    out = pd.DataFrame({
        "image_type": m.image_type.values, "institution": inst.values, "sex": m.sex.values,
        "site": m.anatom_site_1.values, "age": pd.to_numeric(m.age_approx, errors="coerce").values,
        "patient_id": m.patient_id.values, "diagnosis_3": m.diagnosis_3.values,
        "confirm": m.diagnosis_confirm_type.values}, index=image_ids)
    out["age_decade"] = (out.age // 10 * 10).astype("Int64").astype(str).replace("<NA>", np.nan)
    return out


def ridge_probs(inputs, image_ids) -> pd.Series:
    """S10 main model (ridge on the Directional Score, folds grouped by patient), benign vs malignant: out-of-fold
    P(malignant) averaged over the 5 repeats; the indeterminate images are scored by every outer model and averaged."""
    d = inputs["ridge"]
    o = pd.read_parquet(d / "oof_predictions.parquet")
    o = o[(o.task == "s1_2c") & (o.set == "all") & (o.grouping == "patient")]
    p = o.groupby("image_id")["p::malignant"].mean()
    ind = pd.read_csv(d / "indeterminate_2c.csv").set_index("image_id").p_malignant
    return pd.concat([p, ind]).reindex(image_ids)


def roc_points(y, s, max_pts=160):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    keep = np.unique(np.round(np.linspace(0, len(fpr) - 1, min(max_pts, len(fpr)))).astype(int))
    ok = tpr >= 0.95
    i95 = int(np.argmax(ok))
    return {"pts": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep], "auc": float(roc_auc_score(y, s)),
            "op": {"fpr": float(fpr[i95]), "tpr": float(tpr[i95]), "spec": float(1 - fpr[i95])}}


def eval_payload(inputs, ids, lab, meta, ctrl, sub_order) -> dict:
    """Everything the 'Does it classify?' chapter draws, from the S10 ridge run."""
    d = inputs["ridge"]
    summ = pd.read_csv(d / "summary.csv")
    folds = pd.read_csv(d / "metrics_folds.csv")
    o = pd.read_parquet(d / "oof_predictions.parquet")
    derm = meta.image_type.reindex(ids).eq("dermoscopic").values
    # ROC: out-of-fold P averaged over repeats (each still from a model that never saw the image)
    a = o[(o.task == "s1_2c") & (o.set == "all") & (o.grouping == "patient")].groupby("image_id").agg(
        p=("p::malignant", "mean"), true=("true", "first"))
    pos = {im: i for i, im in enumerate(ids)}
    ii = np.array([pos[i] for i in a.index])
    y = (a.true.values == "malignant").astype(int)
    roc = {"ridge_all": roc_points(y, a.p.values), "ridge_derm": roc_points(y[derm[ii]], a.p.values[derm[ii]])}
    if ctrl is not None:
        roc["ctrl_all"] = roc_points(y, ctrl[ii])
        roc["ctrl_derm"] = roc_points(y[derm[ii]], ctrl[ii][derm[ii]])
    # paired folds
    f = folds[(folds.task == "s1_2c") & (folds.grouping == "patient")]
    pf = []
    for sub in ("all images", "dermoscopic only"):
        g = f[f.subset == sub]
        r_ = g[g.set == "all"].set_index(["repeat", "fold"])
        c_ = g[g.set == "control question"].set_index(["repeat", "fold"])
        for k in r_.index:
            pf.append({"subset": sub, "ridge_auc": r_.loc[k, "auroc"], "ctrl_auc": c_.loc[k, "auroc"] if k in c_.index else None,
                       "ridge_spec": r_.loc[k, "spec_at_sens95"], "ctrl_spec": c_.loc[k, "spec_at_sens95"] if k in c_.index else None})
    # summary rows the page quotes
    def row(task, set_, grouping, subset):
        r = summ[(summ.task == task) & (summ.set == set_) & (summ.grouping == grouping) & (summ.subset == subset)]
        return r.iloc[0].to_dict() if len(r) else None
    key = {f"{t}|{s_}|{g}|{sb}": row(t, s_, g, sb) for t in ("s1_2c", "flat_10c") for s_ in ("all", "control question")
           for g in ("patient", "lesion") for sb in ("all images", "dermoscopic only")}
    trim = summ[(summ.task == "s1_2c") & (summ.grouping == "patient") & (summ.subset == "all images")]
    trim = trim[trim.set.str.match(r"^(lasso|reps|rand|top|pca)\d+$")].copy()
    trim["family"] = trim.set.str.extract(r"^([a-z]+)")[0]
    trim["k"] = trim.set.str.extract(r"(\d+)")[0].astype(int)
    groups = pd.read_csv(d / "groups.csv")
    gs = groups.group.value_counts()
    return {"roc": roc, "folds": pf, "key": key,
            "trim": trim[["family", "k", "fold_mean", "draw_min", "draw_max"]].to_dict("records"),
            "groups": {"n_groups": int(len(gs)), "multi": int((gs > 1).sum()), "largest": int(gs.max()),
                       "dups": int(len(pd.read_csv(d / "duplicates.csv")))}}


def subtypes_payload(inputs, ids, qids, sub_order, meta) -> dict:
    """The 'Ten subtypes' and 'Naevus and melanoma' chapters, from the S11 subtypes run."""
    d = inputs["subtypes"]
    per = pd.read_csv(d / "tenclass_per_class.csv").set_index("subtype").reindex(sub_order).reset_index()
    cm = pd.read_csv(d / "tenclass_confusion.csv", index_col=0).reindex(index=sub_order, columns=sub_order)
    ts = pd.read_csv(d / "tenclass_summary.csv")
    ex = json.loads((d / "pair_extra.json").read_text())
    q = pd.read_csv(d / "pair_questions.csv")
    pos, qpos = {im: i for i, im in enumerate(ids)}, {x: i for i, x in enumerate(qids)}
    sc = pd.read_csv(d / "pair_scores.csv")
    bt = pd.read_csv(d / "pair_between.csv")
    rows = []
    for r in sc.itertuples():
        if r.melanoma:
            row = {"Melanoma, NOS": "melanoma, not otherwise specified", "Melanoma Invasive": "invasive melanoma",
                   "Melanoma in situ": "melanoma in situ", "Melanoma metastasis": "melanoma metastasis"}.get(r.diagnosis, r.diagnosis.lower())
        else:
            row = "naevus, excised" if r.confirmation == "histopathology" else "naevus, not excised"
        rows.append({"i": pos[r.image_id], "row": row, "s": round(float(r.score_pair), 4), "grp": 2 if r.melanoma else 0,
                     "dx": r.diagnosis, "conf": r.confirmation})
    for r in bt.itertuples():
        rows.append({"i": pos[r.image_id], "row": r.diagnosis.lower(), "s": round(float(r.score_pair), 4), "grp": 1,
                     "dx": r.diagnosis, "conf": "histopathology"})
    qrow = lambda r: {"q": qpos[r.question_id], "auc": round(float(r.auroc_alone), 4), "w": round(float(r.final_weight), 4)}
    sep = q[q.separation > 0.25].towards.value_counts()
    return {"ten": {"per": per.to_dict("records"), "cm": np.round(cm.values, 2).tolist(),
                    "bacc": [float(ts.balanced_accuracy.mean()), float(ts.balanced_accuracy.min()), float(ts.balanced_accuracy.max())],
                    "top2": float(ts.top2.mean()), "top3": float(ts.top3.mean()), "acc": float(ts.accuracy.mean()),
                    "same_lineage": float(ts.errors_same_lineage.mean()), "same_group": float(ts.errors_same_group.mean()),
                    "collapsed_auc": float(ts.collapsed_2c_auroc.mean()), "twoclass_auc": float(np.mean(ex["twoclass_auroc_repeats"])),
                    "lineage": ex["lineage"]},
            "pair": {"metrics": pd.read_csv(d / "pair_metrics.csv").to_dict("records"),
                     "diffs": pd.read_csv(d / "pair_differences.csv").to_dict("records"),
                     "breakdown": pd.read_csv(d / "pair_breakdown.csv").to_dict("records"),
                     "curves": ex["curves"], "between": ex["between"], "final_lambda": ex["final_lambda"], "images": rows,
                     "q_mel": [qrow(r) for r in q[q.towards == "melanoma"].head(12).itertuples()],
                     "q_nev": [qrow(r) for r in q[q.towards == "naevus"].head(8).itertuples()],
                     "n_sep": {"melanoma": int(sep.get("melanoma", 0)), "naevus": int(sep.get("naevus", 0))},
                     "n_q": int(len(q))},
            "metrics": {"binary": pd.read_csv(d / "metrics_binary.csv").to_dict("records"),
                        "ten": pd.read_csv(d / "metrics_10c.csv").to_dict("records")},
            "comparison": {"rows": pd.read_csv(d / "model_comparison.csv").to_dict("records"),
                           "sections": [{"key": k, "title": SECTIONS[k][0], "note": SECTIONS[k][1]} for k in SECTION_ORDER]}}


def thresholds_payload(inputs) -> dict:
    """Table 9.10 and Fig. 9.11: P(malignant) cutoffs chosen in the training split, and the fixed-cutoff sweep (S14)."""
    d = inputs["thresholds"]
    return {"rules": pd.read_csv(d / "rules_summary.csv").to_dict("records"),
            "sweep": pd.read_csv(d / "sweep_summary.csv").to_dict("records"),
            "max_diff": float(pd.read_csv(d / "reproduction.csv").max_abs_diff.max())}


BENCH_MODELS = [  # S13 method -> the model's number in S11's model comparison (section M), checked against its AUROC
    ("questions", 1), ("skingpt_emb:img_L28", 2), ("skingpt_emb:img_L21", 3), ("skingpt_emb:img_L14", 4),
    ("skingpt_emb:img_L0", 5), ("skingpt_emb:last_L28", 6), ("skingpt_diagnosis_fitted", 7), ("skingpt_diagnosis", 8),
    ("control question", 9)]


def bench_payload(cfg, inputs, ids, lab, meta, ctrl) -> dict:
    """Fold-level scores of the ten-subtype models of Table 3.4 (S13, same patient-grouped folds as S10), for the
    benchmark figure of the 'Model selection' chapter: per outer fold, AUROC for benign against malignant (malignant
    subtypes summed against benign ones, as S11) on all and on dermoscopic test images, and balanced accuracy over the
    ten subtypes. S13 stores out-of-fold probabilities without the fold index, so the folds are rebuilt with S10's
    splitter; the rebuilt balanced accuracies must equal S13's own fold metrics, which proves they are the same folds.
    The direct question has no training, so it is scored on the same test folds as a reference (benign against
    malignant only). auroc_nm: naevus against melanoma on the test fold's naevi and melanomas, read as
    P(melanoma) / (P(melanoma) + P(naevus)) as in S11 (the direct question: its P(yes))."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    from ..models.refit import s10_splits
    from .s11_subtypes import collapse_malignant

    d = inputs["bench"]
    o = pd.read_parquet(d / "oof_predictions.parquet")
    o = o[o.task == "flat_10c"]
    classes = sorted(lab.c10.unique())
    pc = [f"p::{c}" for c in classes]
    grp3 = {c: lab.c3[lab.c10 == c].iloc[0] for c in classes}
    y10 = np.array([classes.index(v) for v in lab.c10.values])
    bm = lab.c3.isin(["benign", "malignant"]).values
    yb = (lab.c3.values == "malignant").astype(int)
    nm = lab.c10.isin(["benign melanocytic", "malignant melanocytic"]).values      # 100 naevi, 100 melanomas
    yn = (lab.c10.values == "malignant melanocytic").astype(int)
    kn, km = classes.index("benign melanocytic"), classes.index("malignant melanocytic")
    mel_vs_nev = lambda Pk: Pk[:, km] / (Pk[:, km] + Pk[:, kn])                   # S11's naevus-against-melanoma score
    derm = meta.image_type.reindex(ids).eq("dermoscopic").values
    groups = pd.read_csv(inputs["ridge"] / "groups.csv").set_index("image_id").group.reindex(ids).values
    splits = s10_splits(lab.c10.values, groups, cfg.classify, cfg.seed)
    reps = sorted(o.repeat.unique())

    P = {}                                              # method -> R x N x K, in the order of ids
    for m, _ in BENCH_MODELS:
        if m == "control question":
            continue
        g = o[o.set == m]
        P[m] = np.stack([g[g.repeat == r].set_index("image_id").reindex(ids)[pc].values for r in reps])
        if np.isnan(P[m]).any():
            raise SystemExit(f"S13's flat_10c predictions for {m} do not cover every image in every repeat")

    rows = []
    for r, f, tr, te in splits:
        k = reps.index(r)
        for m, no in BENCH_MODELS:
            row = {"model": m, "no": no, "repeat": int(r), "fold": int(f)}
            score = ctrl if m == "control question" else collapse_malignant(P[m][k], classes, grp3)
            for sub, msk in (("all", np.ones(len(ids), bool)), ("derm", derm)):
                t = te[bm[te] & msk[te]]
                row[f"auroc_{sub}"] = float(roc_auc_score(yb[t], score[t])) if len(np.unique(yb[t])) == 2 else None
            t = te[nm[te]]
            sn = ctrl if m == "control question" else mel_vs_nev(P[m][k])
            row["auroc_nm"] = float(roc_auc_score(yn[t], sn[t])) if len(np.unique(yn[t])) == 2 else None
            row["bacc"] = None if m == "control question" else float(balanced_accuracy_score(y10[te], P[m][k][te].argmax(1)))
            rows.append(row)
    F = pd.DataFrame(rows)

    # the rebuilt folds are S13's: balanced accuracy per fold must equal S13's stored fold metrics
    s13 = pd.read_csv(d / "metrics_folds.csv")
    s13 = s13[(s13.task == "flat_10c") & (s13.subset == "all images")][["set", "repeat", "fold", "balanced_accuracy"]]
    chk = F[F.model != "control question"].merge(s13, left_on=["model", "repeat", "fold"], right_on=["set", "repeat", "fold"])
    fold_diff = float((chk.bacc - chk.balanced_accuracy).abs().max())
    if len(chk) != (F.model != "control question").sum() or fold_diff > 1e-9:
        raise SystemExit(f"Rebuilt folds do not reproduce S13's fold metrics ({len(chk)} rows matched, largest difference {fold_diff:.2e})")
    # and the model numbers are S11's: pooled AUROC per repeat, averaged, against model_comparison (section M)
    mc = pd.read_csv(inputs["subtypes"] / "model_comparison.csv")
    mcn = mc[(mc.section == "M") & (mc.task == "naevus vs melanoma")].set_index("no").auroc
    mc = mc[(mc.section == "M") & (mc.task == "benign vs malignant")].set_index("no").auroc
    no_diff = 0.0
    for m, no in BENCH_MODELS:
        if m == "control question":
            a, an = roc_auc_score(yb[bm], ctrl[bm]), roc_auc_score(yn[nm], ctrl[nm])
        else:
            a = np.mean([roc_auc_score(yb[bm], collapse_malignant(P[m][k], classes, grp3)[bm]) for k in range(len(reps))])
            an = np.mean([roc_auc_score(yn[nm], mel_vs_nev(P[m][k])[nm]) for k in range(len(reps))])
        no_diff = max(no_diff, abs(a - mc[no]), abs(an - mcn[no]))
    if no_diff > 1e-6:
        raise SystemExit(f"S13 methods do not map onto S11's model numbers (largest AUROC difference {no_diff:.2e})")

    summ = []
    for (m, no), g in F.groupby(["model", "no"], sort=False):
        s = {"model": m, "no": int(no), "n_folds": int(len(g))}
        for c in ("auroc_all", "auroc_derm", "auroc_nm", "bacc"):
            v = g[c].dropna()
            s[c] = None if v.empty else {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max())}
        summ.append(s)
    return {"models": summ, "check": {"s13_fold_bacc_max_diff": fold_diff, "s11_auroc_max_diff": float(no_diff)}}, F


def importance_payload(inputs, qids, sub_order, top_n=25) -> dict:
    """The most important questions for benign vs malignant (S7 consensus over six models, Directional Score), the
    marker questions of each subtype, and the 27 questions of the final lasso model (S10)."""
    d = inputs["importance"]
    c = pd.read_csv(d / "consensus.csv")
    c = c[(c.task == "s1_2c") & (c.source == "dir")].sort_values("consensus_rank")
    c = c[c.question_id.isin(qids)].head(top_n)
    top = [{"q": qids.index(r.question_id), "rank": int(r.consensus_rank), "ridge": r.ridge_coef,
            "sign": r.ridge_sign_agreement, "lasso": r.lasso_kept_share, "auc": r.auroc_alone, "towards": r.towards,
            "doubt": bool(r.rarely_yes), "p_b": r.mean_p_yes_benign, "p_m": r.mean_p_yes_malignant,
            "y_b": r.yes_rate_benign, "y_m": r.yes_rate_malignant} for r in c.itertuples()]
    m = pd.read_csv(d / "subtype_markers.csv")
    markers = {s: [{"q": qids.index(r.question_id), "ridge": r.ridge_coef, "lasso": r.lasso_kept_share,
                    "p_in": r.mean_p_yes_in_subtype, "p_out": r.mean_p_yes_other_subtypes}
                   for r in m[m.subtype == s].sort_values("rank").itertuples() if r.question_id in qids] for s in sub_order}
    w = pd.read_csv(inputs["ridge"] / "final_models" / "s1_2c_lasso27_weights.csv")
    final = [{"q": qids.index(r.question_id), "w": float(r.weight_towards_malignant)} for r in w.itertuples() if r.question_id in qids]
    allc = pd.read_csv(d / "consensus.csv")
    allc = allc[(allc.task == "s1_2c") & (allc.source == "dir")]
    best = allc.sort_values("auroc_alone", ascending=False).iloc[0]
    # consensus rank and ridge weight of every question, so the 27-question model can be read against the ranking
    ra = allc.set_index("question_id").reindex(qids)
    rank_all = [None if np.isnan(v) else int(v) for v in ra.consensus_rank]
    ridge_all = [None if np.isnan(v) else round(float(v), 5) for v in ra.ridge_coef]
    return {"top": top, "markers": markers, "final27": final, "rank_all": rank_all, "ridge_all": ridge_all,
            "best_alone": {"q": qids.index(best.question_id) if best.question_id in qids else None, "auc": float(best.auroc_alone)},
            "n_rarely_yes": int(allc.rarely_yes.sum()), "n_questions": int(len(allc)),
            "n_top20_doubt": int(c.head(20).rarely_yes.sum())}


def pick_cases(lab, meta, rp, ids, wanted) -> list[str]:
    """Case-study images. Unless named in explore.case_images: one melanoma and one naevus ('malignant melanocytic',
    'benign melanocytic'), dermoscopic, correctly classified out of fold, each the one whose out-of-fold P(malignant)
    is closest to the median of the correctly classified images of its subtype, i.e. a typical case, not an extreme."""
    if wanted:
        return list(wanted)
    p = pd.Series(rp, index=ids)
    out = []
    for sub, mal in [("malignant melanocytic", True), ("benign melanocytic", False)]:
        ok = (lab.c10.values == sub) & (meta.image_type.values == "dermoscopic") & ((p.values >= 0.5) == mal) & ~np.isnan(p.values)
        q = p[ok]
        out.append(str((q - q.median()).abs().idxmin()))
    return out


def case_payload(cfg, inputs, lab, meta, man, ids, qids, rp, ctrl, sub_order) -> dict:
    """For each case image: its answers, and how the ridge model's prediction is built from them. The final model
    (all 446 questions) is refitted with the same lambda and the same questions on every
    labelled image except those of the case's patient, so the case is held out; the log-odds of malignancy is then
    exactly the intercept plus, for each question, weight x standardised answer."""
    import joblib
    from sklearn.base import clone

    e = cfg.explore
    case_ids = pick_cases(lab, meta, rp, ids, e.get("case_images"))
    X = pd.read_parquet(inputs["embed"] / f"X_image_{e.source}.parquet").loc[ids]
    a = pd.read_parquet(inputs["embed"] / "answers.parquet", columns=["image_id", "question_id", "original", "p_yes", "confidence"])
    a = a[(a.original.astype(int) == 1) & a.image_id.isin(case_ids)]
    groups = pd.read_csv(inputs["ridge"] / "groups.csv").set_index("image_id").group.reindex(ids).values
    rows = lab.c3.isin(["benign", "malignant"]).values
    y = (lab.c3.values == "malignant").astype(int)
    paths = man.set_index("image_id").path
    fm = inputs["ridge"] / "final_models"
    models = {"all": joblib.load(fm / "s1_2c_all.joblib")}
    out = []
    for cid in case_ids:
        i = ids.index(cid)
        train = rows & (groups != groups[i])
        ans = a[a.image_id == cid].set_index("question_id")
        case = {"id": cid, "i": i, "sub": sub_order.index(lab.c10.values[i]), "grp": GROUPS.index(lab.c3.values[i]),
                "oof": None if np.isnan(rp[i]) else float(rp[i]), "ctrl": None if np.isnan(ctrl[i]) else float(ctrl[i]),
                "n_left_out": int((rows & (groups == groups[i])).sum()), "n_train": int(train.sum()),
                "image": thumbnails([paths[cid]], 560, 1, quality=86)[0], "models": {}}
        for name, m in models.items():
            q = list(m["question_ids"])
            Xq = X[q].values.astype(float)
            scale, clf = clone(m["model"]["scale"]), clone(m["model"]["clf"])
            Zt = scale.fit_transform(Xq[train])
            clf.fit(Zt, y[train])
            z = scale.transform(Xq[[i]])[0]
            w, b = clf.coef_[0], float(clf.intercept_[0])
            logit = b + float(w @ z)
            p = float(clf.predict_proba(z[None])[0, 1])
            assert abs(1 / (1 + np.exp(-logit)) - p) < 1e-9
            case["models"][name] = {"b0": b, "logit": logit, "p": p, "C": float(m["C"]), "terms": [
                {"q": qids.index(qq), "x": float(Xq[i, k]), "z": float(z[k]), "w": float(w[k]), "c": float(w[k] * z[k]),
                 "py": float(ans.p_yes.get(qq, np.nan)), "cf": float(ans.confidence.get(qq, np.nan))}
                for k, qq in enumerate(q) if qq in qids]}
        out.append(case)
    return {"cases": out, "auto": not e.get("case_images")}


CASE_CONTRAST = {"malignant melanocytic": "benign melanocytic", "benign melanocytic": "malignant melanocytic"}


def pick_cases_tenclass(lab, meta, Pm, ids, classes) -> list[str]:
    """Worked cases for the ten-subtype model: one melanoma and one naevus, dermoscopic, assigned to their own subtype
    out of fold, each the one whose out-of-fold probability for its subtype is closest to the median of the correctly
    assigned images of that subtype (a typical case, not an extreme)."""
    out = []
    for s in ["malignant melanocytic", "benign melanocytic"]:
        k = classes.index(s)
        ok = (lab.c10.values == s) & (meta.image_type.values == "dermoscopic") & (Pm.argmax(1) == k)
        v = pd.Series(Pm[ok, k], index=np.array(ids)[ok])
        out.append(str((v - v.median()).abs().idxmin()))
    return out


def tenclass_payload(cfg, inputs, ids, lab, meta, X, P, qids, ctrl, sub_order, case_ids, ev, out) -> dict:
    """The ten-subtype ridge (S10 task flat_10c, all 446 questions, folds grouped by patient), which the paper leads
    with, in the form the 'Model selection' chapter, the cutoff analysis, the question-importance figure and the worked
    cases need. Payload key 'ten2'; nothing here changes the two-class keys (eval, imp, cases), which the explorer reads.

    Benign against malignant: S11's collapse_malignant (malignant subtypes' summed probability over their number,
    against the benign subtypes' over theirs, renormalised so 0.5 = equal prior odds), applied to each repeat's
    out-of-fold probabilities and then averaged over the repeats (as eval averages the two-class P(malignant)). Scored
    on the 800 benign and malignant images; the indeterminate images are a class of the model, so they have out-of-fold
    probabilities too, but no benign/malignant label.

    Importance, from the final model's weights on standardised answers (W, one row per subtype):
      contrast  mean of the malignant-subtype weights minus mean of the benign-subtype weights. Since
                log P_k = eta_k - log sum_j exp(eta_j), this is exactly the coefficient of the question in
                (1/3) sum_malignant eta_k - (1/5) sum_benign eta_k = log(geometric mean of the three malignant
                probabilities / geometric mean of the five benign ones): the question's push towards malignant on the
                log scale. (The collapsed score uses arithmetic means, so contrast is its log-linear counterpart.)
      spread    SD (population, ddof 0) of the question's ten class weights: how much it moves the choice among subtypes,
                whatever the direction.
    Stability: the ten-subtype ridge refitted on each of S10's 25 outer training splits with the penalty S10 chose for
    that split (qaemb/models/refit.py; the refits reproduce S10's stored test-fold probabilities); per question the share
    of splits in which contrast has the sign of the final model, and the median rank by spread (1 = largest).

    Worked cases (the same images as 'cases'): flat_10c_all refitted with its penalty on all 1,000 images except those of
    the case's patient (S10 trains the ten-subtype model on all images, indeterminate included). For the true subtype t
    and a contrast subtype c, log(P_t / P_c) = (b_t - b_c) + sum_q (w_tq - w_cq) z_q exactly (multinomial logistic)."""
    import joblib
    from scipy.stats import rankdata
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    from ..models.metrics import spec_at_sens
    from ..models.refit import fold_penalties, refit_outer, reproduction, s10_splits
    from .s11_subtypes import bin_summary, collapse_malignant

    d, cc, e = inputs["ridge"], cfg.classify, cfg.explore
    sens = float(cc.sens_target)
    m = joblib.load(d / "final_models" / "flat_10c_all.joblib")
    classes = list(m["classes"])
    if classes != list(sub_order):
        raise SystemExit(f"flat_10c class order {classes} differs from the page's subtype order {sub_order}")
    grp3 = {c: lab.c3[lab.c10 == c].iloc[0] for c in classes}
    g = np.array([grp3[c] for c in classes])
    mal, ben = np.where(g == "malignant")[0], np.where(g == "benign")[0]
    pc = [f"p::{c}" for c in classes]
    N, K = len(ids), len(classes)
    qpos = {x: k for k, x in enumerate(qids)}
    mq = list(m["question_ids"])
    Xq = X[mq].values.astype(float)
    y10 = np.array([classes.index(v) for v in lab.c10.values])
    bm = lab.c3.isin(["benign", "malignant"]).values
    yb = (lab.c3.values == "malignant").astype(int)
    derm = meta.image_type.reindex(ids).eq("dermoscopic").values
    groups = pd.read_csv(d / "groups.csv").set_index("image_id").group.reindex(ids).values
    check = {}

    # ---------------- out-of-fold probabilities, per repeat
    o = pd.read_parquet(d / "oof_predictions.parquet")
    o = o[(o.task == "flat_10c") & (o.set == "all") & (o.grouping == "patient")]
    reps = sorted(o.repeat.unique())
    Pr = np.stack([o[o.repeat == r].set_index("image_id").reindex(ids)[pc].values for r in reps])      # R x N x K
    if np.isnan(Pr).any():
        raise SystemExit("S10's flat_10c out-of-fold predictions do not cover every image in every repeat")
    Sr = np.stack([collapse_malignant(Pr[k], classes, grp3) for k in range(len(reps))])               # R x N
    Pm, s = Pr.mean(0), Sr.mean(0)
    check["collapse_of_mean_vs_mean_of_collapse_max_diff"] = float(np.abs(collapse_malignant(Pm, classes, grp3) - s).max())

    # against S11: collapsed AUROC per repeat (tenclass_summary) and model 1 of model_comparison
    ts = pd.read_csv(inputs["subtypes"] / "tenclass_summary.csv").set_index("repeat").reindex(reps)
    rep_auc = np.array([roc_auc_score(yb[bm], Sr[k][bm]) for k in range(len(reps))])
    check["s11_repeat_auroc"] = {"ours": rep_auc.round(6).tolist(), "s11": ts.collapsed_2c_auroc.round(6).tolist(),
                                 "max_abs_diff": float(np.abs(rep_auc - ts.collapsed_2c_auroc.values).max())}
    mc = pd.read_csv(inputs["subtypes"] / "model_comparison.csv")
    mc = mc[(mc.no == 1) & (mc.task == "benign vs malignant")].iloc[0]
    bs = bin_summary(yb[bm], Sr[:, bm], sens)
    check["s11_model1"] = {k: {"ours": bs[k], "s11": float(mc[k])} for k in ("auroc", "sensitivity", "specificity", "spec95")}
    check["s11_model1_max_abs_diff"] = float(max(abs(v["ours"] - v["s11"]) for v in check["s11_model1"].values()))
    if check["s11_repeat_auroc"]["max_abs_diff"] > 1e-9 or check["s11_model1_max_abs_diff"] > 1e-9:
        raise SystemExit(f"The collapsed ten-subtype score does not reproduce S11: {check}")

    # ---------------- ROC (pooled, averaged over repeats) and per outer fold
    roc = {"all": roc_points(yb[bm], s[bm]), "derm": roc_points(yb[bm & derm], s[bm & derm])}
    # the direct question on the same 800 images is eval.roc.ctrl_all / ctrl_derm; confirm rather than duplicate
    if ctrl is not None:
        mine = {"ctrl_all": roc_points(yb[bm], ctrl[bm]), "ctrl_derm": roc_points(yb[bm & derm], ctrl[bm & derm])}
        same = all(abs(mine[k]["auc"] - ev["roc"][k]["auc"]) < 1e-12 and mine[k]["op"] == ev["roc"][k]["op"]
                   and mine[k]["pts"] == ev["roc"][k]["pts"] for k in mine)
        if not same:
            raise SystemExit("eval.roc.ctrl_* is not on the 800 benign and malignant images")
    splits = s10_splits(lab.c10.values, groups, cc, cfg.seed)
    folds = []
    for r, f, tr, te in splits:
        for sub, msk in (("all images", np.ones(N, bool)), ("dermoscopic only", derm)):
            t = te[bm[te] & msk[te]]
            if len(np.unique(yb[t])) < 2:
                continue
            folds.append({"subset": sub, "repeat": int(r), "fold": int(f), "n": int(len(t)), "n_mal": int(yb[t].sum()),
                          "auc": float(roc_auc_score(yb[t], Sr[reps.index(r)][t])), "spec": spec_at_sens(yb[t], Sr[reps.index(r)][t], sens)})
    fdf = pd.DataFrame(folds)
    fdf.to_csv(out / "tenclass_folds.csv", index=False)
    fsum = {sub: {"auc_mean": float(h.auc.mean()), "auc_sd": float(h.auc.std()), "auc_min": float(h.auc.min()),
                  "auc_max": float(h.auc.max()), "spec_mean": float(h.spec.mean()), "spec_sd": float(h.spec.std()),
                  "n_folds": int(len(h))} for sub, h in fdf.groupby("subset", sort=False)}
    pooled = {sub: [float(roc_auc_score(yb[bm & msk], Sr[k][bm & msk])) for k in range(len(reps))]
              for sub, msk in (("all images", np.ones(N, bool)), ("dermoscopic only", derm))}

    # ---------------- importance: final model, then stability over the 25 outer training splits
    W = m["model"]["clf"].coef_                                                   # K x Q, standardised answers
    contrast = W[mal].mean(0) - W[ben].mean(0)
    spread = W.std(0)
    check["final_class_weight_sum_max"] = float(np.abs(W.sum(0)).max())          # L2 multinomial: weights sum to ~0
    fits = refit_outer(Xq, y10, groups, splits, fold_penalties(d, "flat_10c"), cfg.seed, e.n_jobs, keep_model=True)
    rp = reproduction(fits, o, np.array(ids), pc)
    check["refit_reproduction_max_diff"] = float(rp.max_abs_diff.max())
    if rp.max_abs_diff.max() > cfg.thresholds.reproduce_tol or rp.n_missing.any():
        raise SystemExit(f"Refitted ten-subtype fold models do not reproduce S10 (largest difference {rp.max_abs_diff.max():.3g})")
    Wf = np.stack([F["model"]["clf"].coef_ for F in fits])                       # folds x K x Q
    cf = Wf[:, mal].mean(1) - Wf[:, ben].mean(1)
    sf = Wf.std(1)
    sign_share = (np.sign(cf) == np.sign(contrast)[None]).mean(0)
    spread_rank_f = np.stack([rankdata(-x, method="min") for x in sf])
    contrast_rank_f = np.stack([rankdata(-np.abs(x), method="min") for x in cf])
    itab = pd.DataFrame({"question_id": mq, "contrast": contrast, "spread": spread,
                         "contrast_rank": rankdata(-np.abs(contrast), method="min").astype(int),
                         "spread_rank": rankdata(-spread, method="min").astype(int),
                         "sign_share": sign_share, "contrast_fold_mean": cf.mean(0), "contrast_fold_sd": cf.std(0, ddof=1),
                         "contrast_rank_median": np.median(contrast_rank_f, 0), "spread_rank_median": np.median(spread_rank_f, 0),
                         **{f"w_{c}": W[k] for k, c in enumerate(classes)}})
    itab.to_csv(out / "tenclass_importance.csv", index=False)

    def per_q(v, nd=5):
        a = dict(zip(mq, v))
        r = (lambda x: int(x)) if nd == 0 else (lambda x: round(float(x), nd) if np.isscalar(x) else np.round(x, nd).tolist())
        return [r(a[q]) if q in a else None for q in qids]
    imp = {"contrast": per_q(contrast), "spread": per_q(spread), "w": per_q(list(W.T)),
           "contrast_rank": per_q(itab.contrast_rank, 0), "spread_rank": per_q(itab.spread_rank, 0),
           "sign_share": per_q(sign_share, 3), "contrast_rank_med": per_q(itab.contrast_rank_median, 1),
           "spread_rank_med": per_q(itab.spread_rank_median, 1), "contrast_fold_sd": per_q(itab.contrast_fold_sd),
           "C": float(m["C"]), "n_folds": len(fits), "n_model_questions": len(mq)}
    Pyes = P.reindex(ids)
    markers = {}
    for k, c in enumerate(classes):
        inside = lab.c10.values == c
        top = [j for j in np.argsort(-W[k]) if W[k, j] > 0][:5]
        markers[c] = [{"q": qpos[mq[j]], "w": round(float(W[k, j]), 5), "p_in": float(Pyes[mq[j]].values[inside].mean()),
                       "p_out": float(Pyes[mq[j]].values[~inside].mean())} for j in top if mq[j] in qpos]

    # ---------------- worked cases, held out by patient
    # chosen with this model's own out-of-fold predictions unless named in explore.case_images: the two-class choice
    # (case_ids) gave a melanoma that the ten-subtype model assigns to 'inflammatory or infectious' (2026-11-17)
    if not e.get("case_images"):
        case_ids = pick_cases_tenclass(lab, meta, Pm, ids, classes)
    a = pd.read_parquet(inputs["embed"] / "answers.parquet", columns=["image_id", "question_id", "original", "p_yes", "confidence"])
    a = a[(a.original.astype(int) == 1) & a.image_id.isin(case_ids)]
    paths = pd.read_parquet(inputs["manifest"] / "manifest.parquet").set_index("image_id").path
    cases = []
    for cid in case_ids:
        i = ids.index(cid)
        train = groups != groups[i]
        scale, clf = clone(m["model"]["scale"]), clone(m["model"]["clf"])
        clf.fit(scale.fit_transform(Xq[train]), y10[train])
        z = scale.transform(Xq[[i]])[0]
        eta = clf.intercept_ + clf.coef_ @ z
        p = clf.predict_proba(z[None])[0]
        soft = np.exp(eta - eta.max()) / np.exp(eta - eta.max()).sum()
        t = int(y10[i])
        c = classes.index(CASE_CONTRAST[classes[t]]) if classes[t] in CASE_CONTRAST else \
            int(np.argsort(-p)[1] if np.argmax(p) == t else np.argmax(p))
        dw, d0 = clf.coef_[t] - clf.coef_[c], float(clf.intercept_[t] - clf.intercept_[c])
        contrib = dw * z
        lr = float(np.log(p[t] / p[c]))
        err = {"softmax": float(np.abs(soft - p).max()), "decomposition": abs(d0 + float(contrib.sum()) - lr)}
        assert err["softmax"] < 1e-9 and err["decomposition"] < 1e-9, err
        ans = a[a.image_id == cid].set_index("question_id")
        pm_case = float(collapse_malignant(p[None], classes, grp3)[0])
        cases.append({"id": cid, "i": i, "sub": t, "contrast_sub": c, "image": thumbnails([paths[cid]], 560, 1, quality=86)[0], "p": np.round(p, 6).tolist(), "p_mal": pm_case,
                      "pred": int(np.argmax(p)), "logratio": lr, "d0": d0, "sum_terms": float(contrib.sum()),
                      "oof_p": np.round(Pm[i], 6).tolist(), "oof_p_mal": float(s[i]), "C": float(m["C"]),
                      "n_train": int(train.sum()), "n_left_out": int((~train).sum()), "max_err": max(err.values()),
                      "terms": [{"q": qpos[qq], "x": float(Xq[i, k]), "z": float(z[k]), "dw": float(dw[k]), "c": float(contrib[k]),
                                 "py": float(ans.p_yes.get(qq, np.nan)), "cf": float(ans.confidence.get(qq, np.nan))}
                                for k, qq in enumerate(mq) if qq in qpos]})
    check["case_max_err"] = max(c_["max_err"] for c_ in cases)

    return {"classes": classes, "mal": mal.tolist(), "ben": ben.tolist(), "cases_auto": not e.get("case_images"),
            "img": {"p": np.round(Pm, 4).tolist(), "pm": np.round(s, 4).tolist()},
            "roc": roc, "ctrl_roc": "eval.roc.ctrl_all / eval.roc.ctrl_derm (same 800 images)",
            "folds": folds, "fold_summary": fsum, "pooled_auc": pooled,
            "imp": imp, "markers": markers, "cases": cases, "check": check}


def thumbnails(paths, px: int, n_jobs: int, quality: int = 72) -> list[str]:
    from joblib import Parallel, delayed

    def one(p):
        from PIL import Image

        im = Image.open(p)
        im.draft("RGB", (px * 2, px * 2))
        im = im.convert("RGB")
        im.thumbnail((px, px))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=quality, optimize=True)
        return base64.b64encode(b.getvalue()).decode()

    return Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in paths)


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def umap2(X, metric, seed, n_neighbors=15, min_dist=0.1):
    import umap

    u = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed, n_components=2)
    Y = u.fit_transform(X)
    # Robust scaling: the 1st-99th percentile range fills the plot, so one far-off cluster (e.g. the nail questions)
    # does not squash the rest; points beyond it are pulled to the edge.
    lo, hi = np.percentile(Y, 1, axis=0), np.percentile(Y, 99, axis=0)
    Y = np.clip((Y - lo) / (hi - lo), -0.03, 1.03)
    Y = (Y + 0.03) / 1.06
    return np.round(Y, 4)


# ------------------------------------------------------------------ main
def run(cfg, out, inputs, ctx) -> dict:
    # Plain NumPy string arrays throughout: the analysis broadcasts and indexes label arrays, which pandas 3's Arrow
    # strings do not support.
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    e = cfg.explore
    rng = np.random.default_rng(cfg.seed)
    (out / "figures").mkdir(exist_ok=True)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    src = pd.read_csv(ROOT / cfg.questions.csv, encoding="utf-8-sig")
    src = src[src.original.astype(int) == 1].set_index("Question ID")
    qt = qt[qt.kind.isin(list(e.kinds))].reset_index(drop=True)
    X = pd.read_parquet(inputs["embed"] / f"X_image_{e.source}.parquet").loc[ids]
    P = pd.read_parquet(inputs["embed"] / "X_image_orig.parquet").loc[ids]
    Z, keep = ex.standardise(X[qt.question_id].values.astype(float))
    dropped = qt.question_id[~keep].tolist()
    qt = qt[keep].reset_index(drop=True)
    qids = list(qt.question_id)
    Pq = P[qids].values
    Q, N = len(qids), len(ids)
    print(f"  {N} images x {Q} questions ({len(dropped)} without variation dropped)", flush=True)

    meta = load_metadata(cfg, ids, e.min_institution_images)
    sub_order = sorted(lab.c10.unique(), key=lambda s: (GROUPS.index(lab.c3[lab.c10 == s].iloc[0]), s))
    y2 = np.where(lab.c3 == "malignant", 1, np.where(lab.c3 == "benign", 0, np.nan))

    # ---------------- questions: answer and text similarity, clusters
    R = np.corrcoef(Z.T)
    # Ward on the standardised answers of each question (columns of Z / sqrt(N)): squared Euclidean distance between
    # two questions is then 2 (1 - r), so this is Ward clustering on correlation distance. Average linkage chained
    # into three clusters of 77-165 questions plus singletons (2026-11-15), too uneven to read.
    L_ans = ex.tree(Z.T / np.sqrt(len(Z)), "ward")
    qks = sorted(set(e.question_ks))
    cl_ans = ex.cuts(L_ans, qks)
    tq = pd.read_csv(ROOT / e.text_questions).set_index("question_id")
    T, cl_txt, xy_q = {}, {}, {"answers": umap2(np.clip(1 - R, 0, None), "precomputed", cfg.seed)}
    for name, path in e.text_embeddings.items():
        E = np.load(ROOT / path)
        E = E[[tq.index.get_loc(q) for q in qids]]
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        T[name] = E @ E.T
        cl_txt[name] = ex.cuts(ex.tree(E, "ward"), qks)          # unit vectors: squared distance = 2 (1 - cosine)
        xy_q[name] = umap2(np.clip(1 - T[name], 0, None), "precomputed", cfg.seed)
    tmain = list(T)[0]

    topics = list(dict.fromkeys(qt.topic))
    grounding = src.loc[qids, "Grounding"].fillna("").values
    sources = src.loc[qids, "Sources cited (Local ID)"].fillna("").astype(str).values
    src_sets = [set(s.strip() for s in v.split(";") if s.strip()) for v in sources]
    first_src = [sorted(s)[0] if s else np.nan for s in src_sets]

    # agreement between partitions of the questions
    n_perm_small = e.n_perm_agreement
    agr = []
    for k in qks:
        for b_name, B in [("topic", qt.topic.values), *[(f"text clusters ({n})", cl_txt[n][k]) for n in cl_txt],
                          ("grounding", grounding), ("first cited source", first_src)]:
            agr.append({"k": k, "a": "answer clusters", "b": b_name, **ex.agree(cl_ans[k], B, rng, n_perm_small)})
        for n in cl_txt:
            agr.append({"k": k, "a": f"text clusters ({n})", "b": "topic", **ex.agree(cl_txt[n][k], qt.topic.values, rng, n_perm_small)})
    agr = pd.DataFrame(agr)
    agr.to_csv(out / "agreement.csv", index=False)
    mantels = {n: ex.mantel(R, T[n], rng, e.n_perm) for n in T}

    # topic cohesion (Plan item D, first half)
    coh_all, coh_topic = ex.group_cohesion(R, qt.topic.values, rng, e.n_perm)
    coh_topic = coh_topic.rename(columns={"group": "topic"}).sort_values("within_mean_r", ascending=False)
    coh_topic.to_csv(out / "topic_cohesion.csv", index=False)
    shared = np.array([[bool(a & b) for b in src_sets] for a in src_sets])
    np.fill_diagonal(shared, False)
    coh_src = ex.pair_mask_cohesion(R, shared, rng, e.n_perm)
    same_topic = qt.topic.values[:, None] == qt.topic.values[None, :]
    tva = ex.text_vs_answer_pairs(R, T[tmain], same_topic)
    tva.to_csv(out / "text_vs_answers.csv", index=False)

    # dimensions
    pc = ex.pca(Z, rng, e.n_perm_pca, n_keep=50)
    pd.DataFrame({"component": np.arange(1, 51), "variance_share": pc["ratio"][:50], "cumulative": pc["cum"][:50],
                  "eigenvalue": pc["eigen"][:50], "shuffled_95th": pc["null95"]}).to_csv(out / "pca.csv", index=False)
    m_img = int(min(max(pc["n_parallel"], 10), 50))
    S = pc["scores"][:, :m_img]

    # per-question statistics
    auroc_alone = np.array([ex.auroc(y2, Z[:, j]) for j in range(Q)])
    cons = pd.read_csv(inputs["importance"] / "consensus.csv")
    cons = cons[(cons.task == "s1_2c") & (cons.source == e.source)].set_index("question_id")
    ridge = cons.ridge_coef.reindex(qids).values
    prof = np.array([Z[(lab.c10 == s).values].mean(0) for s in sub_order]).T          # Q x 10, mean z per subtype
    nn_txt = np.argsort(-T[tmain], axis=1)[:, 1:9]
    qtab = pd.DataFrame({"question_id": qids, "question": qt.question, "topic": qt.topic, "kind": qt.kind,
                         "scope": qt.scope, "grounding": grounding, "sources": sources,
                         "yes_rate": (Pq >= 0.5).mean(0), "mean_p_yes": Pq.mean(0),
                         "sd_score": X[qids].values.std(0), "auroc_alone": auroc_alone, "ridge_coef": ridge,
                         **{f"answer_cluster_k{k}": cl_ans[k] for k in qks}})
    for s, col in zip(sub_order, prof.T):
        qtab[f"mean_z_{s}"] = col
    qtab.to_csv(out / "question_table.csv", index=False)

    # question cluster summaries for every k
    qsum = {}
    rows_qc = []
    for k in qks:
        lst = []
        for c in range(k):
            mem = np.where(cl_ans[k] == c)[0]
            sc = Z[:, mem].mean(1)
            Rm = R[np.ix_(mem, mem)]
            central = mem[np.argsort(-(Rm.sum(1) - 1) / max(len(mem) - 1, 1))][:3]
            tc = pd.Series(qt.topic.values[mem]).value_counts()
            d = {"n": int(len(mem)), "mean_r": float((Rm.sum() - len(mem)) / (len(mem) * (len(mem) - 1))) if len(mem) > 1 else None,
                 "auroc": ex.auroc(y2, sc), "topics": [[topics.index(t), int(v)] for t, v in tc.head(3).items()],
                 "central": [int(i) for i in central],
                 "prof": [round(float(sc[(lab.c10 == s).values].mean() / (sc.std() or 1)), 3) for s in sub_order]}
            lst.append(d)
            rows_qc.append({"k": k, "cluster": c, "n_questions": d["n"], "mean_within_r": d["mean_r"],
                            "auroc_cluster_score": d["auroc"], "top_topics": "; ".join(f"{t} ({v})" for t, v in tc.head(3).items()),
                            "central_questions": " | ".join(qt.question.values[i] for i in central),
                            **{f"mean_{s}": p for s, p in zip(sub_order, d["prof"])}})
        qsum[k] = lst
    pd.DataFrame(rows_qc).to_csv(out / "question_clusters.csv", index=False)

    # ---------------- images
    iks = sorted(set(e.image_ks))
    L_img = ex.tree(S, "ward")
    cl_img = ex.cuts(L_img, iks)
    xy_i = {"umap": umap2(S, "euclidean", cfg.seed, n_neighbors=30, min_dist=0.2),
            "pca": np.round((pc["scores"][:, :2] - pc["scores"][:, :2].min(0)) / np.ptp(pc["scores"][:, :2], 0), 4)}
    variables = {"subtype": lab.c10.values, "group": lab.c3.values, "image type": meta.image_type.values,
                 "institution": meta.institution.values, "body site": meta.site.values, "sex": meta.sex.values,
                 "age (decade)": meta.age_decade.values}
    ica = []
    for k in iks:
        for vname, v in variables.items():
            row = {"k": k, "variable": vname, **ex.agree(cl_img[k], v, rng, n_perm_small)}
            for g in ("benign", "malignant"):          # within one group, so diagnosis cannot carry the association
                sel = (lab.c3 == g).values
                row[f"ami_within_{g}"] = ex.agree(cl_img[k][sel], pd.Series(v)[sel].values, rng, 1)["ami"] if vname not in ("group",) else np.nan
            ica.append(row)
    ica = pd.DataFrame(ica)
    ica.to_csv(out / "image_cluster_association.csv", index=False)
    pca_assoc = []
    for j in range(10):
        s = pc["scores"][:, j]
        r = {"component": j + 1, "variance_share": pc["ratio"][j], "auroc_malignant": ex.auroc(y2, s)}
        for vname, v in variables.items():
            if vname != "age (decade)":
                r[f"eta2_{vname}"] = ex.eta2(s, v)
        ok = ~np.isnan(meta.age.values.astype(float))
        r["r2_age"] = float(np.corrcoef(s[ok], meta.age.values[ok].astype(float))[0, 1] ** 2)
        load = pc["loadings"][:, j]
        top = np.argsort(load)
        r["top_positive"] = " | ".join(qt.question.values[i] for i in top[::-1][:5])
        r["top_negative"] = " | ".join(qt.question.values[i] for i in top[:5])
        pca_assoc.append(r)
    pca_assoc = pd.DataFrame(pca_assoc)
    pca_assoc.to_csv(out / "pc_association.csv", index=False)

    ctrl = P["DQC01"].values if "DQC01" in P.columns else np.full(N, np.nan)
    rp = ridge_probs(inputs, ids).values
    itab = pd.DataFrame({"image_id": ids, "subtype": lab.c10, "group": lab.c3, **{c: meta[c].values for c in meta},
                         "control_p_yes": ctrl, "ridge_p_malignant": rp, "umap_x": xy_i["umap"][:, 0], "umap_y": xy_i["umap"][:, 1],
                         **{f"image_cluster_k{k}": cl_img[k] for k in iks}})
    itab.to_csv(out / "image_table.csv", index=False)

    # ---------------- figures and summary
    plots.explore_figures(out / "figures", coh_topic, coh_all, tva, pc, agr, ica, qsum, sub_order, e.default_question_k,
                          [t[:60] for t in topics])
    write_summary(out, cfg, N, Q, dropped, qt, coh_all, coh_topic, coh_src, mantels, tva, pc, m_img, agr, ica, pca_assoc,
                  qsum, topics, sub_order, meta, lab, e)

    # ---------------- the ten-subtype model (payload key ten2)
    ev = eval_payload(inputs, ids, lab, meta, ctrl, sub_order)
    print("  ten-subtype model: collapsed score, 25 fold refits, worked cases ...", flush=True)
    ten2 = tenclass_payload(cfg, inputs, ids, lab, meta, X, P, qids, ctrl, sub_order,
                            pick_cases(lab, meta, rp, ids, e.get("case_images")), ev, out)
    write_tenclass_summary(out, ten2, qids, qt)
    bench, bench_folds = bench_payload(cfg, inputs, ids, lab, meta, ctrl)
    bench_folds.to_csv(out / "bench_folds.csv", index=False)

    # ---------------- page
    print("  thumbnails ...", flush=True)
    thumbs = thumbnails(man.set_index("image_id").loc[ids, "path"].tolist(), e.thumb_px, e.n_jobs)
    zq = np.clip(np.round(Z * 32), -127, 127)
    payload = {
        "meta": {"n_images": N, "n_questions": Q, "source": SOURCE_NAMES.get(e.source, e.source), "dropped": dropped,
                 "qks": qks, "iks": iks, "kq": e.default_question_k, "ki": e.default_image_k, "m_img": m_img,
                 "text_models": list(T), "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
                 "run_hash": out.name},
        "subtypes": sub_order, "sub_group": [GROUPS.index(lab.c3[lab.c10 == s].iloc[0]) for s in sub_order],
        "groups": GROUPS, "topics": topics,
        "q": {"id": qids, "text": qt.question.tolist(), "topic": [topics.index(t) for t in qt.topic],
              "kind": qt.kind.tolist(), "scope": qt.scope.tolist(), "grounding": list(grounding), "sources": list(sources),
              "yes": np.round(qtab.yes_rate, 4).tolist(), "meanp": np.round(qtab.mean_p_yes, 4).tolist(),
              "auc": [None if np.isnan(v) else round(float(v), 4) for v in auroc_alone],
              "ridge": [None if np.isnan(v) else round(float(v), 4) for v in ridge],
              "prof": np.round(prof, 3).tolist(), "nn_text": nn_txt.tolist(),
              "cl": {str(k): v.tolist() for k, v in cl_ans.items()},
              "tcl": {n: {str(k): v.tolist() for k, v in cl_txt[n].items()} for n in cl_txt},
              "xy": {n: v.tolist() for n, v in xy_q.items()},
              "dendro": ex.dendro_segments(L_ans)},
        "qsum": {str(k): v for k, v in qsum.items()},
        "R": b64(np.round(R * 127), np.int8),
        "img": {"id": ids, "sub": [sub_order.index(s) for s in lab.c10], "grp": [GROUPS.index(g) for g in lab.c3],
                "imtype": meta.image_type.fillna("").tolist(), "inst": meta.institution.fillna("").tolist(),
                "sex": meta.sex.fillna("").tolist(), "site": meta.site.fillna("").tolist(),
                "age": [None if pd.isna(a) else float(a) for a in meta.age], "patient": meta.patient_id.fillna("").tolist(),
                "dx3": meta.diagnosis_3.fillna("").tolist(), "confirm": meta.confirm.fillna("").tolist(),
                "ctrl": [None if np.isnan(v) else round(float(v), 3) for v in ctrl],
                "ridge": [None if np.isnan(v) else round(float(v), 3) for v in rp],
                "cl": {str(k): v.tolist() for k, v in cl_img.items()},
                "xy": {n: v.tolist() for n, v in xy_i.items()},
                "order": ex.dendro_segments(L_img)["leaves"], "thumb": thumbs},
        "Z": b64(zq, np.int8), "P": b64(np.round(Pq * 255), np.uint8),
        "cohesion": {"overall": coh_all, "shared_source": coh_src,
                     "topics": coh_topic.assign(topic=[topics.index(t) for t in coh_topic.topic]).round(4).to_dict("records"),
                     "tva": tva.round(4).to_dict("records")},
        "mantel": mantels,
        "agreement": agr.round(4).to_dict("records"),
        "pca": {"ratio": np.round(pc["ratio"][:50], 5).tolist(), "null95": np.round(pc["null95"] / pc["eigen"].sum(), 5).tolist(),
                "cum": np.round(pc["cum"][:50], 4).tolist(), "n_parallel": pc["n_parallel"],
                "n_for": {str(k): v for k, v in pc["n_for"].items()}, "pr": pc["participation_ratio"],
                "assoc": pca_assoc.round(4).to_dict("records")},
        "ica": ica.round(4).to_dict("records"),
        "eval": ev,
        "imp": importance_payload(inputs, qids, sub_order),
        "sub": subtypes_payload(inputs, ids, qids, sub_order, meta),
        "thr": thresholds_payload(inputs),
        "cases": case_payload(cfg, inputs, lab, meta, man, ids, qids, rp, ctrl, sub_order),
        "ten2": ten2,
        "bench": bench,
        "qcl": json.loads((inputs["qclusters"] / "payload.json").read_text()),   # S15: questions answered alike, and the checks on why
        "answers": {"yes_share": float((Pq >= 0.5).mean()),
                    "p_hist": np.histogram(Pq.ravel(), bins=20, range=(0, 1))[0].tolist(),
                    "q_yes_hist": np.histogram((Pq >= 0.5).mean(0), bins=20, range=(0, 1))[0].tolist(),
                    "n_rarely": int(((Pq >= 0.5).mean(0) < 0.05).sum())},
    }
    data = json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_json_default)
    fonts, data = font_css(), data.replace("</", "<\\/")
    for tpl, name in [(TEMPLATE, "explorer.html"), (NARRATIVE, "narrative.html")]:
        html = tpl.read_text(encoding="utf-8").replace("/*__FONTS__*/", fonts).replace("__DATA__", data)
        (out / name).write_text(html, encoding="utf-8")
    mb = (out / "explorer.html").stat().st_size / 1e6
    return {"n_images": N, "n_questions": Q, "n_parallel_components": pc["n_parallel"],
            "components_90pct": pc["n_for"][0.9], "topic_within_r": round(coh_all["within_mean_r"], 3),
            "topic_null_95": round(coh_all["null_95"], 3), "explorer_mb": round(mb, 1),
            "key_metric": f"{pc['n_parallel']} components above shuffled; topic r {coh_all['within_mean_r']:.2f} vs null {coh_all['null_95']:.2f}"}


def _clean(o):
    """NaN and infinities to None (JSON null), recursively."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------ summary
def write_tenclass_summary(out, t, qids, qt):
    """tenclass.md: the ten-subtype numbers in ten2, with the checks behind them."""
    c, fs = t["check"], t["fold_summary"]
    text = dict(zip(qids, qt.question))
    L = ["# Ten-subtype ridge read as benign against malignant (payload key ten2)", "",
         "S10 task flat_10c, all 446 questions, folds grouped by patient; malignant subtypes' summed probability over "
         "their number against the benign subtypes' over theirs (S11 collapse_malignant), on the 800 benign and malignant images.", "",
         "## Checks", "",
         f"- Collapsed AUROC pooled per repeat, against S11 tenclass_summary.csv: largest difference {c['s11_repeat_auroc']['max_abs_diff']:.1e}.",
         f"- AUROC, sensitivity and specificity at 0.5, specificity at 95% sensitivity, against S11 model_comparison.csv model 1: "
         f"largest difference {c['s11_model1_max_abs_diff']:.1e}.",
         f"- Ten-subtype ridge refitted on S10's 25 outer training splits with S10's penalties: test-fold probabilities within "
         f"{c['refit_reproduction_max_diff']:.1e} of S10's stored ones.",
         f"- Worked cases: softmax and log-ratio decomposition reproduce the model's probabilities to {c['case_max_err']:.1e}.",
         f"- Collapse of the repeat-averaged probabilities against the average of the per-repeat collapsed scores: largest "
         f"difference {c['collapse_of_mean_vs_mean_of_collapse_max_diff']:.1e} (the page uses the latter).",
         f"- Final model: largest |sum of the ten class weights| of a question {c['final_class_weight_sum_max']:.1e} "
         "(the weights are centred over the classes, so each class weight reads against the average class).", "",
         "## Benign against malignant", "", "| images | AUROC per fold, mean ± SD (range) | pooled per repeat | ROC of repeat-averaged score | spec. at 95% sens. per fold |",
         "|---|---|---|---|---|"]
    for sub, k in (("all images", "all"), ("dermoscopic only", "derm")):
        f = fs[sub]
        L.append(f"| {sub} | {f['auc_mean']:.3f} ± {f['auc_sd']:.3f} ({f['auc_min']:.3f}–{f['auc_max']:.3f}), {f['n_folds']} folds | "
                 f"{np.mean(t['pooled_auc'][sub]):.3f} | {t['roc'][k]['auc']:.3f} (spec. at 95% sens. {t['roc'][k]['op']['spec']:.3f}) | "
                 f"{f['spec_mean']:.3f} ± {f['spec_sd']:.3f} |")
    imp = t["imp"]
    rows = sorted([k for k, v in enumerate(imp["contrast"]) if v is not None], key=lambda k: -abs(imp["contrast"][k]))[:15]
    L += ["", "## Questions with the largest contrast (malignant minus benign mean weight)", "",
          "| question | contrast | sign kept in folds | spread | spread rank (median over folds) |", "|---|---|---|---|---|"]
    for k in rows:
        L.append(f"| {text[qids[k]]} | {imp['contrast'][k]:+.4f} | {imp['sign_share'][k]:.0%} | {imp['spread'][k]:.4f} | "
                 f"{imp['spread_rank'][k]} ({imp['spread_rank_med'][k]:.0f}) |")
    L += ["", "## Worked cases (held out by patient)", ""]
    cl = t["classes"]
    for cs in t["cases"]:
        L.append(f"- {cs['id']} ({cl[cs['sub']]}): held-out P({cl[cs['sub']]}) = {cs['p'][cs['sub']]:.3f}, "
                 f"P({cl[cs['contrast_sub']]}) = {cs['p'][cs['contrast_sub']]:.3f}, most probable {cl[cs['pred']]}, collapsed "
                 f"P(malignant) {cs['p_mal']:.3f} (out of fold {cs['oof_p_mal']:.3f}); log ratio {cs['logratio']:+.3f} = "
                 f"{cs['d0']:+.3f} (intercepts) {cs['sum_terms']:+.3f} (questions); {cs['n_train']} training images, "
                 f"{cs['n_left_out']} of the patient's left out.")
    L.append("")
    (out / "tenclass.md").write_text("\n".join(L), encoding="utf-8")


def _f(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_summary(out, cfg, N, Q, dropped, qt, coh_all, coh_topic, coh_src, mantels, tva, pc, m_img, agr, ica, pca_assoc,
                  qsum, topics, sub_order, meta, lab, e):
    kq, ki = e.default_question_k, e.default_image_k
    L = ["# S9 explore: structure of the answers", "",
         f"{N} images, {Q} questions ({', '.join(e.kinds)}; the control question left out"
         f"{'; ' + str(len(dropped)) + ' questions with no variation dropped: ' + ', '.join(dropped) if dropped else ''}). "
         f"Score: {SOURCE_NAMES.get(e.source, e.source)}. Open `explorer.html` in a browser for the interactive version.", ""]
    L += ["## Questions answered alike: topics", "",
          f"- Mean correlation between two questions of the same topic: {_f(coh_all['within_mean_r'])}; between questions of "
          f"different topics: {_f(coh_all['between_mean_r'])}. With the topic labels shuffled (same topic sizes, "
          f"{e.n_perm} times) the same-topic mean is {_f(coh_all['null_mean'])} on average and at most {_f(coh_all['null_max'])} "
          f"(p = {coh_all['p_perm']:.4f}).",
          f"- Questions citing a common paper ({coh_src['n_pairs']} pairs): mean r {_f(coh_src['mean_r'])}, against "
          f"{_f(coh_src['null_mean'])} for random pairs (95th percentile {_f(coh_src['null_95'])}).",
          f"- Topics more cohesive than 95% of random groups of the same size: {int((coh_topic.share_null_below >= 0.95).sum())} "
          f"of {coh_topic.share_null_below.notna().sum()} (topics with at least 2 questions).", ""]
    tb = tva.pivot_table(index="bin", columns="pairs", values="mean_r")
    if {"same topic", "different topics"} <= set(tb.columns):
        diff = (tb["same topic"] - tb["different topics"]).dropna()
        L += [f"- At equal wording similarity ({list(cfg.explore.text_embeddings)[0]}, 10 bins), same-topic pairs are more "
              f"correlated than different-topic pairs in {int((diff > 0).sum())} of {len(diff)} bins (median difference "
              f"{_f(diff.median())}).", ""]
    L += ["## Answers against wording", ""]
    for n, m in mantels.items():
        L.append(f"- Rank correlation between answer correlation and text similarity ({n}), over all question pairs: "
                 f"{_f(m['spearman'], 3)} (shuffled 95th percentile {_f(m['null_95'], 3)}).")
    a = agr[agr.k == kq]
    L += ["", f"Agreement of the {kq} answer clusters with other groupings (adjusted mutual information; 0 = chance, 1 = identical; "
          "in brackets the 95th percentile with one grouping shuffled):", ""]
    for r in a.itertuples():
        L.append(f"- {r.a} vs {r.b}: {_f(r.ami)} ({_f(r.ami_null95)})")
    L += ["", "## How many independent dimensions", "",
          f"- Components needed for 50 / 80 / 90 / 95% of the variance: {pc['n_for'][0.5]} / {pc['n_for'][0.8]} / "
          f"{pc['n_for'][0.9]} / {pc['n_for'][0.95]}.",
          f"- Components whose variance beats the same answers shuffled across images (parallel analysis): {pc['n_parallel']}.",
          f"- Participation ratio (effective number of equally strong dimensions): {_f(pc['participation_ratio'], 1)}.",
          f"- The first component carries {pc['ratio'][0]:.0%} of the variance; separates malignant from benign with AUROC "
          f"{_f(pca_assoc.auroc_malignant.iloc[0])}.", "",
          "| component | variance | AUROC malignant | subtype | image type | institution | body site | sex | age |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in pca_assoc.to_dict("records"):
        L.append(f"| {r['component']} | {r['variance_share']:.1%} | {_f(r['auroc_malignant'])} | {_f(r['eta2_subtype'])} | "
                 f"{_f(r['eta2_image type'])} | {_f(r['eta2_institution'])} | {_f(r['eta2_body site'])} | {_f(r['eta2_sex'])} | "
                 f"{_f(r['r2_age'])} |")
    L += ["", "(Share of each component's variance explained by each variable: eta squared, or r squared for age.)", "",
          "## Images: what the answer profiles are organised by", "",
          f"Ward clustering of the images on the first {m_img} principal components. Adjusted mutual information between "
          f"the {ki} image clusters and each variable, overall and within benign or malignant images only:", "",
          "| variable | AMI | shuffled 95th | within benign | within malignant |", "|---|---|---|---|---|"]
    for r in ica[ica.k == ki].itertuples():
        L.append(f"| {r.variable} | {_f(r.ami)} | {_f(r.ami_null95)} | {_f(r.ami_within_benign)} | {_f(r.ami_within_malignant)} |")
    L += ["", f"## The {kq} question clusters", "",
          "| cluster | questions | mean r within | AUROC of cluster score | main topics | most central question |",
          "|---|---|---|---|---|---|"]
    for c, d in enumerate(qsum[kq]):
        tops = "; ".join(f"{topics[t][:40]} ({v})" for t, v in d["topics"])
        L.append(f"| {c + 1} | {d['n']} | {_f(d['mean_r'])} | {_f(d['auroc'])} | {tops} | {qt.question.values[d['central'][0]]} |")
    L.append("")
    (out / "summary.md").write_text("\n".join(L))
