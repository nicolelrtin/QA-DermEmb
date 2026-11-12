"""S7 Importance: which questions the S6 classifiers relied on, and what Sonnet answered to them.

Reads the S6 importance tables (no refitting) and the answers, and for each task and feature source works out:
  consensus      each model's questions ranked by mean |SHAP| on the test folds; the consensus rank is the mean of
                 the six ranks. Ridge weight (sign = direction), the share of the 25 outer fits in which the lasso kept
                 the question, and the share in which ridge gave it the same sign.
  agreement      how far the six models agree (rank correlation of their SHAP values; overlap of their top N with the
                 consensus top N), and how far the consensus agrees between feature sources.
  answers        the share of images in each class that Sonnet answered "yes" (P(yes) >= 0.5); for the 2-class task
                 also each question's own AUROC (its score alone, no model, all 800 images).
  markers        for the 10 subtypes: the questions with the largest positive ridge weight for each subtype (flat_10c).
  topics/scopes  share of the importance carried by each topic and scope, against its share of the questions.
  redundancy     pairs among the top N whose answers correlate strongly (near-duplicates the models split weight over).
Outputs: consensus.csv, agreement.csv, source_agreement.csv, subtype_markers.csv, topics.csv, scopes.csv,
redundancy.csv, figures/, summary.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..models import plots
from .s5_model import feature_mask
from .s6_classify import MODEL_NAMES, SOURCE_NAMES, define_tasks, load_labels

UPSTREAM = ["manifest", "embed", "classify"]
VERSION = 2


def key(cfg, ctx) -> dict:
    return {"importance": cfg.importance}


def shap_ranks(g: pd.DataFrame) -> pd.DataFrame:
    """[question x model] rank of mean |SHAP| (1 = most important) for one task, source and output."""
    w = g.pivot_table(index="question_id", columns="model", values="shap_mean_abs")
    return w.rank(ascending=False, method="average")


def one_output(imp: pd.DataFrame) -> pd.DataFrame:
    """'overall' rows for multiclass tasks, the single positive-class rows for binary ones."""
    has = imp.groupby(["source", "task"]).output.transform(lambda o: (o == "overall").any())
    return imp[(imp.output == "overall") | ~has]


def yes_rates(P: pd.DataFrame, labels: pd.Series, classes) -> pd.DataFrame:
    """Share of images in each class answered yes (P(yes) >= 0.5), one column per class."""
    Y = (P >= 0.5).astype(float)
    return pd.DataFrame({c: Y[labels.values == c].mean(0) for c in classes})


def consensus_table(imp1, imp_all, task, source, qt, tasks, lab, Xorig, Xsrc) -> pd.DataFrame:
    g = imp1[(imp1.task == task) & (imp1.source == source)]
    R = shap_ranks(g)
    T = tasks[task]
    cls = T["classes"]
    out = qt.set_index("question_id").loc[R.index, ["question", "topic", "scope"]].copy()
    out["consensus_rank_score"] = R.mean(1)
    for m in R.columns:
        out[f"shap_rank_{m}"] = R[m]
    l2 = g[g.model == "l2"].set_index("question_id")
    l1 = g[g.model == "l1"].set_index("question_id")
    out["ridge_coef"] = l2.coef_mean
    if len(cls) == 2:
        out["ridge_sign_agreement"] = l2.coef_sign_agreement
    out["lasso_kept_share"] = l1.n_selected / l1.n_fits
    rows = T["rows"]
    lbl = lab[T["col"]][rows]
    yr = yes_rates(Xorig.loc[lab.image_id[rows], out.index], lbl, cls)
    yr.index = out.index
    P = Xorig.loc[lab.image_id[rows], out.index]
    for c in cls:
        out[f"yes_rate_{c}"] = yr[c]
        out[f"mean_p_yes_{c}"] = P[lbl.values == c].mean(0).values
    out["rarely_yes"] = (yr.max(1) < 0.05).values          # under 5% yes in every class: the signal is in the doubt
    if len(cls) == 2:
        y = (lbl.values == cls[1]).astype(int)
        S = Xsrc.loc[lab.image_id[rows], out.index]
        out["auroc_alone"] = [roc_auc_score(y, S[q].values) for q in out.index]
        out["towards"] = np.where(out.ridge_coef >= 0, cls[1], cls[0])               # in the model (ridge sign)
        out["towards_alone"] = np.where(out.auroc_alone >= 0.5, cls[1], cls[0])     # the question by itself
    else:
        per = imp_all[(imp_all.task == task) & (imp_all.source == source) & (imp_all.model == "l2")
                      & (imp_all.output.isin(cls))].pivot_table(index="question_id", columns="output", values="coef_mean")
        out["towards"] = per.reindex(out.index).idxmax(1)
    out = out.sort_values("consensus_rank_score")
    out.insert(0, "consensus_rank", np.arange(1, len(out) + 1))
    out.insert(0, "source", source)
    out.insert(0, "task", task)
    return out.reset_index()


def agreement_rows(imp1, task, source, top_n) -> list[dict]:
    g = imp1[(imp1.task == task) & (imp1.source == source)]
    W = g.pivot_table(index="question_id", columns="model", values="shap_mean_abs")
    R = W.rank(ascending=False, method="average")
    top = set(R.mean(1).nsmallest(top_n).index)
    rows = []
    for m in W.columns:
        others = [o for o in W.columns if o != m]
        rho = np.median([spearmanr(W[m], W[o]).statistic for o in others])
        rows.append({"task": task, "source": source, "model": m, "median_rank_corr_with_other_models": rho,
                     f"top{top_n}_overlap_with_consensus": len(set(R[m].nsmallest(top_n).index) & top) / top_n})
    return rows


def topic_shares(imp1, task, source, qt, col) -> pd.DataFrame:
    """Each model's |SHAP| normalised to sum to 1, averaged over models, summed by `col`."""
    g = imp1[(imp1.task == task) & (imp1.source == source)]
    W = g.pivot_table(index="question_id", columns="model", values="shap_mean_abs")
    share = (W / W.sum(0)).mean(1)
    grp = qt.set_index("question_id").loc[share.index, col]
    t = pd.DataFrame({"importance_share": share.groupby(grp).sum(), "question_share": grp.value_counts(normalize=True),
                      "n_questions": grp.value_counts()})
    t["enrichment"] = t.importance_share / t.question_share
    t = t.sort_values("importance_share", ascending=False).reset_index().rename(columns={"index": col})
    t.insert(0, "source", source)
    t.insert(0, "task", task)
    return t


def redundancy(cons: pd.DataFrame, Xsrc: pd.DataFrame, top_n: int, r_min: float) -> pd.DataFrame:
    top = cons.head(top_n)
    C = Xsrc[top.question_id].corr()
    rows = []
    ids = list(top.question_id)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if abs(C.loc[a, b]) >= r_min:
                rows.append({"task": top.task.iloc[0], "source": top.source.iloc[0], "question_a": a, "question_b": b,
                             "r": C.loc[a, b]})
    return pd.DataFrame(rows, columns=["task", "source", "question_a", "question_b", "r"])


def subtype_markers(imp_all, source, tasks, lab, Xorig, qt, k) -> pd.DataFrame:
    if "flat_10c" not in tasks:
        return pd.DataFrame()
    cls = tasks["flat_10c"]["classes"]
    per = imp_all[(imp_all.task == "flat_10c") & (imp_all.source == source) & (imp_all.model == "l2")
                  & (imp_all.output.isin(cls))].pivot_table(index="question_id", columns="output", values="coef_mean")
    l1 = imp_all[(imp_all.task == "flat_10c") & (imp_all.source == source) & (imp_all.model == "l1")
                 & (imp_all.output.isin(cls))].pivot_table(index="question_id", columns="output", values="n_selected")
    nf = imp_all[(imp_all.task == "flat_10c") & (imp_all.model == "l1")].n_fits.max()
    Pm = Xorig.loc[lab.image_id]
    Y = (Pm >= 0.5).astype(float)
    text = qt.set_index("question_id").question
    rows = []
    for c in cls:
        inside = (lab.c10 == c).values
        for rank, q in enumerate(per[c].nlargest(k).index, 1):
            rows.append({"subtype": c, "rank": rank, "question_id": q, "question": text[q], "ridge_coef": per.loc[q, c],
                         "lasso_kept_share": l1.loc[q, c] / nf if q in l1.index else np.nan,
                         "yes_rate_in_subtype": Y[q].values[inside].mean(),
                         "yes_rate_other_subtypes": Y[q].values[~inside].mean(),
                         "mean_p_yes_in_subtype": Pm[q].values[inside].mean(),
                         "mean_p_yes_other_subtypes": Pm[q].values[~inside].mean()})
    return pd.DataFrame(rows)


def run(cfg, out, inputs, ctx) -> dict:
    ic = cfg.importance
    top_n = int(ic.top_n)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    tasks = define_tasks(lab, list(cfg.classify.tasks))
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qt = qt[feature_mask(qt, cfg.model, patch=False)].reset_index(drop=True)
    imp_all = pd.read_csv(inputs["classify"] / "importance.csv")
    imp1 = one_output(imp_all)
    sources = [s for s in cfg.classify.sources if (imp_all.source == s).any()]
    main = ic.source if ic.source in sources else sources[0]
    X = {s: pd.read_parquet(inputs["embed"] / f"X_image_{s}.parquet") for s in set(sources) | {"orig"}}
    (out / "figures").mkdir(exist_ok=True)

    cons, agree, topics, scopes, red = [], [], [], [], []
    for s in sources:
        for t in tasks:
            c = consensus_table(imp1, imp_all, t, s, qt, tasks, lab, X["orig"], X[s])
            cons.append(c)
            agree += agreement_rows(imp1, t, s, top_n)
            topics.append(topic_shares(imp1, t, s, qt, "topic"))
            scopes.append(topic_shares(imp1, t, s, qt, "scope"))
            red.append(redundancy(c, X[s], top_n, float(ic.redundancy_r)))
    cons = pd.concat(cons, ignore_index=True)
    agree = pd.DataFrame(agree)
    topics, scopes = pd.concat(topics, ignore_index=True), pd.concat(scopes, ignore_index=True)
    red = pd.concat(red, ignore_index=True)

    src_rows = []
    for t in tasks:
        for i, a in enumerate(sources):
            for b in sources[i + 1:]:
                ra = cons[(cons.task == t) & (cons.source == a)].set_index("question_id").consensus_rank
                rb = cons[(cons.task == t) & (cons.source == b)].set_index("question_id").consensus_rank.reindex(ra.index)
                src_rows.append({"task": t, "source_a": a, "source_b": b, "rank_corr": spearmanr(ra, rb).statistic,
                                 f"top{top_n}_overlap": len(set(ra.nsmallest(top_n).index) & set(rb.nsmallest(top_n).index)) / top_n})
    src_agree = pd.DataFrame(src_rows)
    markers = subtype_markers(imp_all, main, tasks, lab, X["orig"], qt, int(ic.per_class_top))

    cons.to_csv(out / "consensus.csv", index=False)
    agree.to_csv(out / "agreement.csv", index=False)
    src_agree.to_csv(out / "source_agreement.csv", index=False)
    topics.to_csv(out / "topics.csv", index=False)
    scopes.to_csv(out / "scopes.csv", index=False)
    red.to_csv(out / "redundancy.csv", index=False)
    markers.to_csv(out / "subtype_markers.csv", index=False)

    figures(out / "figures", cons, topics, markers, tasks, lab, X["orig"], main, top_n)
    write_summary(out, cfg, tasks, cons, agree, src_agree, topics, scopes, red, markers, main, sources, top_n)

    top1 = cons[(cons.task == list(tasks)[0]) & (cons.source == main)].iloc[0]
    return {"n_questions": int(qt.shape[0]), "sources": sources, "main_source": main, "tasks": list(tasks),
            "key_metric": f"{top1.task} top question {top1.question_id} ({main})"}


def _short(q: str, n: int = 70) -> str:
    return q if len(q) <= n else q[: n - 1].rstrip() + "…"


def figures(fdir: Path, cons, topics, markers, tasks, lab, Xorig, main, top_n):
    # 1. Two-class task: share answered yes in each class for the top questions (dumbbell), ordered by consensus rank.
    if "s1_2c" in tasks:
        c = cons[(cons.task == "s1_2c") & (cons.source == main)].head(top_n)
        labels = [f"{r.question_id} {_short(r.question)}" for r in c.itertuples()]
        plots.dumbbell(labels, c.mean_p_yes_benign.values, c.mean_p_yes_malignant.values, ("benign", "malignant"),
                       f"Benign vs malignant: top {top_n} questions by consensus rank ({SOURCE_NAMES.get(main, main)})",
                       "Sonnet's mean P(yes) over the images of each class", fdir / "top_s1_2c.png")
    # 2. Importance share by topic against the topic's share of the questions.
    for t in [t for t in ("s1_2c", "flat_10c") if t in tasks]:
        g = topics[(topics.task == t) & (topics.source == main)].head(15)
        plots.share_bars(list(g.topic), g.importance_share.values, g.question_share.values,
                         f"{t}: share of importance by topic (15 largest; {SOURCE_NAMES.get(main, main)})",
                         "share of total importance", fdir / f"topics_{t}.png")
    # 3. Subtype markers: share answered yes to each marker question, in every subtype.
    if not markers.empty:
        cls = tasks["flat_10c"]["classes"]
        qs = list(dict.fromkeys(markers.question_id))
        Pq = Xorig.loc[lab.image_id, qs]
        M = np.array([[Pq[q].values[(lab.c10 == c).values].mean() for c in cls] for q in qs])
        text = markers.drop_duplicates("question_id").set_index("question_id").question
        chosen = {(qs.index(r.question_id), cls.index(r.subtype)) for r in markers.itertuples()}
        plots.heat([f"{q} {_short(text[q], 60)}" for q in qs], cls, M, chosen,
                   "Mean P(yes) by subtype for each subtype's marker questions (boxed: the subtype it marks)",
                   fdir / "subtype_markers.png")


def _pct(x) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.0f}%"


def write_summary(out, cfg, tasks, cons, agree, src_agree, topics, scopes, red, markers, main, sources, top_n):
    mn = SOURCE_NAMES.get(main, main)
    L = ["# Which questions the classifiers relied on", "",
         f"Main feature source: {mn} (`{main}`). Every table is also in the CSV files for "
         f"{', '.join(SOURCE_NAMES.get(s, s) for s in sources)}. Nothing is refitted here; the importance values are the "
         "S6 ones, measured on the test folds.", "",
         "How to read the columns:",
         "- **Consensus rank**: each of the six models ranks the questions by how much they moved its predictions on "
         "the test folds (mean |SHAP|); the consensus is the mean of the six ranks, re-ranked. 1 = most important.",
         "- **Ridge weight**: the standardised ridge regression weight. For benign vs malignant, positive pushes towards "
         "malignant. Its size depends on the other questions in the model, so read it alongside the others.",
         "- **Lasso kept**: the share of the 25 outer fits in which the lasso kept the question. Close to 100% means "
         "the question is reliably useful, not picked by chance in one split.",
         "- **Mean P(yes) in <class>**: Sonnet's P(yes) averaged over the images of that class. Many questions are "
         "almost never answered yes; for those the difference lies in how doubtful the \"no\" is (say 0.05 against "
         "0.2), which the share answered yes cannot show. That share is in consensus.csv (`yes_rate_*`).",
         "- **Rarely yes**: fewer than 5% of the images in every class were answered yes.",
         "- **AUROC alone** (2-class only): how well this one question's score separates malignant from benign by "
         "itself, with no model: 0.5 is no better than chance, above 0.5 means higher scores go with malignant, "
         "below 0.5 with benign. Computed once on all 800 images; it describes the answers, it is not a test score.",
         "- **Towards**: the class the question pushes towards inside the model: for 2-class tasks the sign of the "
         "ridge weight; for multiclass tasks the class with the largest ridge weight. For 2-class tasks, a † marks "
         "questions whose direction inside the model is the opposite of their direction alone (AUROC alone). Such a "
         "question corrects for others it is correlated with, and should not be read on its own.", ""]

    L += ["## Validity checks", "",
          "These show whether the ranking is a property of the answers or an accident of one model or one split. "
          "If the models disagreed, or the lasso kept the top questions only in a few fits, the ranking would not be "
          "worth interpreting.", "",
          f"| task | median rank correlation between models | models' top {top_n} shared with the consensus top {top_n} "
          f"(median) | consensus top {top_n} kept by the lasso in at least 80% of fits |",
          "|---|---|---|---|"]
    for t in tasks:
        a = agree[(agree.task == t) & (agree.source == main)]
        c = cons[(cons.task == t) & (cons.source == main)].head(top_n)
        L.append(f"| {t} | {a.median_rank_corr_with_other_models.median():.2f} | "
                 f"{_pct(a[f'top{top_n}_overlap_with_consensus'].median())} | {int((c.lasso_kept_share >= 0.8).sum())} of {top_n} |")
    L += ["", "Rank correlation of 1 means the models order the questions identically; 0 means unrelated. Per model:",
          "", agree[agree.source == main].assign(model=lambda d: d.model.map(MODEL_NAMES)).pivot_table(
              index="model", columns="task", values="median_rank_corr_with_other_models").round(2).to_markdown(), "",
          f"Agreement between feature sources (rank correlation of the consensus ranks; share of the top {top_n} in common):",
          "", src_agree.round(2).to_markdown(index=False), ""]

    for t, T in tasks.items():
        c = cons[(cons.task == t) & (cons.source == main)].head(top_n)
        L += [f"## {t}: top {top_n} questions ({mn})", ""]
        cols = {"consensus_rank": "rank", "question_id": "ID", "question": "question", "scope": "scope",
                "towards": "towards", "ridge_coef": "ridge weight", "lasso_kept_share": "lasso kept"}
        d = c[list(cols)].rename(columns=cols)
        d["question"] = d.question.map(lambda q: _short(q, 80))
        d["lasso kept"] = d["lasso kept"].map(_pct)
        d["ridge weight"] = d["ridge weight"].round(3)
        for k in (T["classes"] if len(T["classes"]) <= 3 else []):
            d[f"mean P(yes) {k}"] = c[f"mean_p_yes_{k}"].round(2).values
        if len(T["classes"]) == 2:
            d["AUROC alone"] = c.auroc_alone.round(2).values
            d["towards"] = [f"{a} †" if a != b else a for a, b in zip(c.towards, c.towards_alone)]
        d["rarely yes"] = np.where(c.rarely_yes, "yes", "")
        L += [d.to_markdown(index=False), "",
              f"{int(c.rarely_yes.sum())} of these {top_n} questions were rarely answered yes in any class." +
              (f" {int((c.towards != c.towards_alone).sum())} point the other way inside the model than alone (†)."
               if len(T["classes"]) == 2 else ""), ""]
        if t in ("s1_2c", "flat_10c"):
            L += [f"Figure: [topics_{t}.png](figures/topics_{t}.png)" + (
                f"; [top_s1_2c.png](figures/top_s1_2c.png)" if t == "s1_2c" else ""), ""]

    if not markers.empty:
        L += ["## Marker questions for each subtype (10-class model)", "",
              "For each subtype, the questions with the largest positive ridge weight for that subtype in the 10-class "
              "model, with Sonnet's answers in that subtype and in all the others (share answered yes; mean P(yes)). A good "
              "marker has a clearly higher P(yes) in its subtype than elsewhere.", ""]
        d = markers.copy()
        d["question"] = d.question.map(lambda q: _short(q, 70))
        for col in ("lasso_kept_share", "yes_rate_in_subtype", "yes_rate_other_subtypes"):
            d[col] = d[col].map(_pct)
        d = d.rename(columns={"question_id": "ID", "ridge_coef": "ridge weight", "lasso_kept_share": "lasso kept",
                              "yes_rate_in_subtype": "yes in subtype", "yes_rate_other_subtypes": "yes elsewhere",
                              "mean_p_yes_in_subtype": "mean P(yes) in subtype",
                              "mean_p_yes_other_subtypes": "mean P(yes) elsewhere"})
        L += [d.drop(columns="rank").round(3).to_markdown(index=False), "",
              "Figure: [subtype_markers.png](figures/subtype_markers.png)", ""]

    L += ["## Topics and scopes", "",
          "*Importance share*: each model's |SHAP| scaled to sum to 1, averaged over models, summed over the topic. "
          "*Enrichment* is that share divided by the topic's share of the questions: above 1, the topic carries more "
          "than its size would suggest.", ""]
    for t in tasks:
        g = topics[(topics.task == t) & (topics.source == main)].head(10)
        L += [f"**{t}**, 10 largest topics", "",
              g[["topic", "n_questions", "importance_share", "question_share", "enrichment"]].round(3).to_markdown(index=False), ""]
    sc = scopes[scopes.source == main].pivot_table(index="scope", columns="task", values="enrichment").round(2)
    L += ["Enrichment by scope (local = a part of the lesion; global = the whole lesion or image; metadata = facts "
          "about the patient or site that the image may show):", "", sc.to_markdown(), ""]

    L += [f"## Near-duplicate questions among each task's top {top_n}", "",
          f"Pairs whose scores correlate at |r| ≥ {cfg.importance.redundancy_r} over the 1,000 images. The models "
          "can split weight between such questions, so either one may stand for both.", ""]
    rr = red[red.source == main]
    L += [rr.round(2).to_markdown(index=False) if len(rr) else "None.", ""]
    L += ["## Files", "",
          "- consensus.csv: every question, task and source; all ranks, weights, yes shares, AUROC alone",
          "- agreement.csv, source_agreement.csv: the validity checks above",
          "- subtype_markers.csv, topics.csv, scopes.csv, redundancy.csv", ""]
    (out / "summary.md").write_text("\n".join(L))
