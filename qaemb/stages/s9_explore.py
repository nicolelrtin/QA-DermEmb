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
Outputs: explorer.html (self-contained), question_table.csv, question_clusters.csv, topic_cohesion.csv, agreement.csv,
text_vs_answers.csv, pca.csv, pc_association.csv, image_table.csv, image_cluster_association.csv, figures/, summary.md.
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

UPSTREAM = ["manifest", "embed", "importance", "ridge"]
VERSION = 2
TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "explore.html"
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
            "files": {f: file_hash(ROOT / f) for f in files}, "template": file_hash(TEMPLATE),
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
    # 10-class confusion, row-normalised, averaged over repeats
    c10 = o[(o.task == "flat_10c") & (o.set == "all") & (o.grouping == "patient")]
    pcols = [f"p::{c}" for c in sub_order]
    M = np.zeros((len(sub_order), len(sub_order)))
    pred = c10[pcols].values.argmax(1)
    for t, pr in zip(c10.true.values, pred):
        M[sub_order.index(t), pr] += 1
    M = M / M.sum(1, keepdims=True)
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
    return {"roc": roc, "folds": pf, "confusion": np.round(M, 3).tolist(), "key": key,
            "trim": trim[["family", "k", "fold_mean", "draw_min", "draw_max"]].to_dict("records"),
            "groups": {"n_groups": int(len(gs)), "multi": int((gs > 1).sum()), "largest": int(gs.max()),
                       "dups": int(len(pd.read_csv(d / "duplicates.csv")))}}


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
    return {"top": top, "markers": markers, "final27": final,
            "best_alone": {"q": qids.index(best.question_id) if best.question_id in qids else None, "auc": float(best.auroc_alone)},
            "n_rarely_yes": int(allc.rarely_yes.sum()), "n_questions": int(len(allc)),
            "n_top20_doubt": int(c.head(20).rarely_yes.sum())}


def thumbnails(paths, px: int, n_jobs: int) -> list[str]:
    from joblib import Parallel, delayed

    def one(p):
        from PIL import Image

        im = Image.open(p)
        im.draft("RGB", (px * 2, px * 2))
        im = im.convert("RGB")
        im.thumbnail((px, px))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=72, optimize=True)
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
        "eval": eval_payload(inputs, ids, lab, meta, ctrl, sub_order),
        "imp": importance_payload(inputs, qids, sub_order),
        "answers": {"yes_share": float((Pq >= 0.5).mean()),
                    "p_hist": np.histogram(Pq.ravel(), bins=20, range=(0, 1))[0].tolist(),
                    "q_yes_hist": np.histogram((Pq >= 0.5).mean(0), bins=20, range=(0, 1))[0].tolist(),
                    "n_rarely": int(((Pq >= 0.5).mean(0) < 0.05).sum())},
    }
    data = json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_json_default)
    html = (TEMPLATE.read_text(encoding="utf-8").replace("/*__FONTS__*/", font_css())
            .replace("__DATA__", data.replace("</", "<\\/")))
    (out / "explorer.html").write_text(html, encoding="utf-8")
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
