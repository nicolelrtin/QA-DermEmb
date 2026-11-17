"""S15 Qclusters: which questions are answered alike, at a stated minimum correlation, and whether that reflects the
images or the way the answers were produced.

The answers come from one model (Claude Sonnet 5.5), one replicate, and questions posed one topic per prompt. Two
questions can covary across images because the features they describe occur together in lesions (what we want), or
because of how they were answered: the questions of one prompt share a context and, as it turns out, partly share
their stated confidence. This stage clusters the questions and then measures each of these explanations.

Answer components:
  judgement     P(yes): the yes/no judgement without the stated confidence
  directional   the Directional Score (2 P(yes) - 1) x confidence, as used by the classifiers
  confidence    the stated confidence alone
Correlation: Spearman's rho across images (Pearson r of ranks). Most questions are answered "no" on nearly every image;
Pearson r between two such questions is decided by whether their rare "yes" images coincide, and reached 0.70 between
shuffled answers on logit P(yes) (first run of this stage, 2026-11-17), while rho stays below 0.3. Below, r means rho.
Clustering: complete linkage on the distance 1 - r, cut at 1 - r_min, so that every pair of
questions in a cluster correlates at r >= r_min. Leaves in optimal order for the heatmap. All 500 questions as written
(clinical and artefact); the control question is never included.

Checks, per component unless stated:
  prompt        mean r of pairs in the same topic (same prompt) and in different topics (separate requests), overall
                and at matched wording similarity (deciles of the cosine between question embeddings)
  confidence    share of the stated confidence's variance shared by the questions of one prompt (image x topic mean),
                against topics reassigned at random with the same sizes
  rewordings    r of near-identical questions (top 0.5% of wording similarity) asked in the same or in different
                prompts: a floor on how consistently the model answers the same thing twice
  diagnosis     correlations recomputed within subtype (residuals of the ranks after the ten subtype means), and under
                strict controls: subtype, image type and the image's mean stated confidence
  graded "no"   (judgement) the same pairs with only the gradation among "no" answers kept (P(yes) capped at
                graded_cap) and with only the yes/no split kept (phi on P(yes) >= 0.5, for pairs in which both questions
                have at least min_yes "yes" images): which of the two carries the agreement
  impression    per cluster, mean rho after removing the n_pcs leading principal components of the answers to all
                questions from other topics: an upper bound on how much an image-wide impression (the model's own
                reading of the lesion, which the true subtype cannot remove) could explain. It over-controls.
  chance        largest |r| after shuffling each question across images
  stability     per cluster, the share of 200 bootstrap resamples of the images (stratified on subtype) in which its
                pairs stay together; adjusted Rand index between the clusterings of 50 pairs of disjoint halves
  per cluster   size, topics, share of pairs from different prompts, min and mean r, the same pairs' r in the other
                components and under the controls, mean wording cosine (and its percentile among all pairs), share of
                pairs citing a common source, AUROC of the cluster mean for malignant against benign, mean z by subtype
Map: one node per cluster, placed by metric multidimensional scaling on 1 - (mean rho between the two clusters' questions);
each cluster linked to its map_links most correlated clusters where that mean is at least map_min_r.

Revision 3 (2026-11-17), after an independent review of version 2: (1) near-identical wordings include opposites
(embeddings ignore negation; "hazy border" and "sharply demarcated border"), so their agreement is now summarised as the
median |rho|; (2) the strict controls no longer include the image's mean rank, which forced the mean correlation
between separate prompts to -1/(Q-1) by construction; (3) the share of P(yes) shared within a prompt is reported as
well as the confidence's; (4) the graded-"no" and image-impression checks above were added; (5) the count of clusters
that hold under every check is given over a grid of criteria, not at one setting; (6) the question-level network
(one tangle of 378 questions) is replaced by the cluster map.

Outputs: pairs_summary.csv, wording_matched.csv, rewordings.csv, confidence_by_prompt.csv, clusters_<component>.csv,
cluster_members_<component>.csv, thresholds.csv, stability.csv, payload.json (for S9), summary.md.
"""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list

from ..config import ROOT, file_hash
from ..models import explore as ex
from ..models import qclusters as qc
from .s6_classify import GROUPS, load_labels

UPSTREAM = ["manifest", "embed"]
VERSION = 3                                       # 3: after independent review (2026-11-17): see the docstring's last paragraph
COMPONENTS = ["judgement", "directional", "confidence"]


def key(cfg, ctx) -> dict:
    e = cfg.explore
    return {"qclusters": cfg.qclusters, "kinds": e.kinds, "seed": cfg.seed, "label_columns": cfg.classify.label_columns,
            "labels_sha": file_hash(ROOT / cfg.data.root / cfg.data.manifest), "metadata_sha": file_hash(ROOT / e.metadata),
            "text": {n: file_hash(ROOT / p) for n, p in e.text_embeddings.items()}, "questions_sha": file_hash(ROOT / cfg.questions.csv)}


def run(cfg, out, inputs, ctx) -> dict:
    with pd.option_context("future.infer_string", False):
        return _run(cfg, out, inputs, ctx)


def _run(cfg, out, inputs, ctx) -> dict:
    from .s9_explore import load_metadata
    c, e = cfg.qclusters, cfg.explore
    rng = np.random.default_rng(cfg.seed)
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    lab = load_labels(cfg, man)
    ids = list(lab.image_id)
    qt = pd.read_parquet(inputs["embed"] / "questions.parquet")
    qt = qt[qt.kind.isin(list(e.kinds))].reset_index(drop=True)
    qt["topic"] = qt.topic.astype(object)
    a = pd.read_parquet(inputs["embed"] / "answers.parquet", columns=["image_id", "question_id", "original", "p_yes", "confidence"])
    a = a[a.original.astype(int) == 1]
    P = a.pivot(index="image_id", columns="question_id", values="p_yes").loc[ids]
    C = a.pivot(index="image_id", columns="question_id", values="confidence").loc[ids]
    Xd = pd.read_parquet(inputs["embed"] / f"X_image_{e.source}.parquet").loc[ids]
    keep = [q for q in qt.question_id if Xd[q].std() > 0 and P[q].std() > 0 and C[q].std() > 0]
    dropped = sorted(set(qt.question_id) - set(keep))
    qt = qt[qt.question_id.isin(keep)].reset_index(drop=True)
    qids = list(qt.question_id)
    Q, N = len(qids), len(ids)
    F = {"judgement": P[qids].values.astype(float), "directional": Xd[qids].values.astype(float), "confidence": C[qids].values.astype(float)}
    Fr = {m: qc.zscore(qc.ranks(F[m])) for m in COMPONENTS}
    print(f"  {N} images x {Q} questions ({len(dropped)} without variation in some component dropped)", flush=True)

    meta = load_metadata(cfg, ids, e.min_institution_images)
    sub = lab.c10.values
    sub_order = sorted(lab.c10.unique(), key=lambda s: (GROUPS.index(lab.c3[lab.c10 == s].iloc[0]), s))
    y2 = np.where(lab.c3 == "malignant", 1, np.where(lab.c3 == "benign", 0, np.nan))
    mconf = C[qids].values.mean(1)
    design_sub = [qc.dummies(sub)]
    design_base = [qc.dummies(sub), qc.dummies(meta.image_type.fillna("unknown").values), mconf[:, None]]

    topics = qt.topic.values
    iu = np.triu_indices(Q, 1)
    same = (topics[:, None] == topics[None, :])[iu]
    tq = pd.read_csv(ROOT / e.text_questions).set_index("question_id")
    tname, tpath = next(iter(e.text_embeddings.items()))
    Em = np.load(ROOT / tpath)[[tq.index.get_loc(q) for q in qids]]
    Em = Em / np.linalg.norm(Em, axis=1, keepdims=True)
    T = Em @ Em.T
    tpair = T[iu]
    src = pd.read_csv(ROOT / cfg.questions.csv, encoding="utf-8-sig")
    src = src[src.original.astype(int) == 1].set_index("Question ID")
    src_sets = [set(s.strip() for s in str(v).split(";") if s.strip() and str(v) != "nan") for v in src.loc[qids, "Sources cited (Local ID)"]]
    shared_src = np.array([[bool(x & y) for y in src_sets] for x in src_sets])

    # ---------------- correlations per component, raw and under the controls
    R, Rsub, Rctrl = {}, {}, {}
    for m in COMPONENTS:
        R[m] = qc.corr(Fr[m])
        Rsub[m] = qc.corr(qc.residualise(Fr[m], design_sub))
        Rctrl[m] = qc.corr(qc.residualise(Fr[m], design_base))

    # pairs: same prompt against separate prompts
    rows = []
    for m in COMPONENTS:
        for ctl, M in [("none", R[m]), ("within subtype", Rsub[m]), ("strict", Rctrl[m])]:
            r = M[iu]
            for t in c.r_thresholds:
                hit = r >= t
                rows.append({"component": m, "control": ctl, "r_min": t, "mean_r_same_prompt": r[same].mean(), "mean_r_separate_prompts": r[~same].mean(),
                             "pairs_at_or_above": int(hit.sum()), "same_prompt_share_of_those": float(same[hit].mean()) if hit.any() else np.nan,
                             "pairs_separate_prompts_at_or_above": int((hit & ~same).sum()), "same_prompt_base_rate": float(same.mean())})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out / "pairs_summary.csv", index=False)

    # at matched wording similarity (within-subtype correlations, so shared diagnosis is not the explanation)
    edges = np.quantile(tpair, np.linspace(0, 1, 11))
    wm = []
    for m in COMPONENTS:
        r = Rsub[m][iu]
        for i in range(10):
            sel = (tpair >= edges[i]) & (tpair <= edges[i + 1])
            wm.append({"component": m, "decile": i + 1, "cos_lo": edges[i], "cos_hi": edges[i + 1],
                       "mean_r_same_prompt": r[sel & same].mean(), "n_same": int((sel & same).sum()),
                       "mean_r_separate_prompts": r[sel & ~same].mean(), "n_separate": int((sel & ~same).sum())})
    wm = pd.DataFrame(wm)
    wm.to_csv(out / "wording_matched.csv", index=False)

    # rewordings: the most similar wordings, asked together or apart
    hi = tpair >= np.quantile(tpair, 1 - c.reword_top_share)
    rw = pd.DataFrame([{"component": m, "asked": where, "n_pairs": int((hi & msk).sum()),
                        "median_abs_r": float(np.median(np.abs(R[m][iu][hi & msk]))), "q25_abs": float(np.quantile(np.abs(R[m][iu][hi & msk]), .25)),
                        "q75_abs": float(np.quantile(np.abs(R[m][iu][hi & msk]), .75)), "n_negative": int((R[m][iu][hi & msk] < 0).sum()),
                        "median_r_positive_only": float(np.median(R[m][iu][hi & msk & (R[m][iu] > 0)]))}
                       for m in COMPONENTS for where, msk in [("same prompt", same), ("separate prompts", ~same)]])
    rw["cos_min"] = float(np.quantile(tpair, 1 - c.reword_top_share))
    rw.to_csv(out / "rewordings.csv", index=False)

    # confidence shared within a prompt, against topics reassigned at random (same sizes)
    topic_of = dict(zip(qids, topics))
    vp = qc.variance_by_prompt(C[qids], topic_of)
    null = []
    for _ in range(c.n_perm_prompt):
        perm = rng.permutation(topics)
        null.append(qc.variance_by_prompt(C[qids], dict(zip(qids, perm)))["prompt_share_of_residual"])
    vp.update({"null_mean": float(np.mean(null)), "null_95": float(np.quantile(null, .95)), "n_perm": int(c.n_perm_prompt)})
    pd.DataFrame([vp]).to_csv(out / "confidence_by_prompt.csv", index=False)
    # the same for the judgement (ranks of P(yes))
    Jr = pd.DataFrame(qc.ranks(F["judgement"]), columns=qids)
    vj = qc.variance_by_prompt(Jr, topic_of)
    nj = [qc.variance_by_prompt(Jr, dict(zip(qids, rng.permutation(topics))))["prompt_share_of_residual"] for _ in range(c.n_perm_prompt)]
    vj.update({"null_mean": float(np.mean(nj)), "null_95": float(np.quantile(nj, .95)), "n_perm": int(c.n_perm_prompt)})
    pd.DataFrame([{"component": "confidence", **vp}, {"component": "judgement (ranks of P(yes))", **vj}]).to_csv(out / "shared_by_prompt.csv", index=False)

    # graded "no" against the yes/no split (judgement)
    Pj = F["judgement"]
    Rcap = qc.corr(qc.ranks(np.minimum(Pj, c.graded_cap)))
    Rbin = qc.corr((Pj >= 0.5).astype(float))
    nyes = (Pj >= 0.5).sum(0)
    both = (nyes[iu[0]] >= c.min_yes) & (nyes[iu[1]] >= c.min_yes)
    hiJ = (R["judgement"][iu] >= float(c.r_main)) & ~same
    graded = {"cross_pairs_at_r_min": int(hiJ.sum()), "kept_with_graded_no_only": float((Rcap[iu][hiJ] >= c.r_main).mean()),
              "kept_with_yes_no_only": float((Rbin[iu][hiJ] >= c.r_main).mean()), "graded_cap": float(c.graded_cap),
              "pairs_both_min_yes": int((hiJ & both).sum()), "median_phi_those": float(np.median(Rbin[iu][hiJ & both])) if (hiJ & both).any() else None,
              "median_rho_those": float(np.median(R["judgement"][iu][hiJ & both])) if (hiJ & both).any() else None,
              "min_yes": int(c.min_yes), "never_yes": int((nyes == 0).sum()), "share_answers_below_half": float((Pj < 0.5).mean())}

    # chance
    null_r = {m: qc.null_max_r(F[m], rng, c.n_perm_null) for m in COMPONENTS}

    # ---------------- clustering
    r0 = float(c.r_main)
    trees = {m: qc.complete_tree(R[m]) for m in COMPONENTS}
    thr_rows = []
    for m in COMPONENTS:
        for t in c.r_thresholds:
            labs = qc.cut(trees[m], t)
            vc = pd.Series(labs).value_counts()
            multi = vc[vc >= 2]
            ntop = [len(set(topics[labs == k])) for k in multi.index]
            thr_rows.append({"component": m, "r_min": t, "clusters_2plus": int(len(multi)), "questions_in_them": int(multi.sum()),
                             "clusters_3plus": int((vc >= 3).sum()), "largest": int(vc.max()),
                             "clusters_spanning_prompts": int(sum(n > 1 for n in ntop))})
    pd.DataFrame(thr_rows).to_csv(out / "thresholds.csv", index=False)

    payload = {"meta": {"r_min": r0, "r_thresholds": list(c.r_thresholds), "components": COMPONENTS, "main": c.main_component,
                        "n_images": N, "n_questions": Q, "qids": qids, "dropped": dropped, "text_model": tname,
                        "n_boot": int(c.n_boot), "n_split": int(c.n_split), "reword_cos_min": float(rw.cos_min.iloc[0])},
               "pairs": pairs.round(4).to_dict("records"), "wording": wm.round(4).to_dict("records"),
               "rewordings": rw.round(4).to_dict("records"), "conf_prompt": {k: round(v, 4) if isinstance(v, float) else v for k, v in vp.items()},
               "judgement_prompt": {k: round(v, 4) if isinstance(v, float) else v for k, v in vj.items()}, "graded": graded,
               "null_max_r": {m: round(float(np.max(v)), 3) for m, v in null_r.items()},
               "thresholds": thr_rows, "comp": {}}
    stab_rows = []
    pct = lambda v: float((tpair < v).mean())
    for m in COMPONENTS:
        labs = qc.cut(trees[m], r0)
        order = [int(i) for i in leaves_list(trees[m])]
        print(f"  {m}: bootstrap and split halves", flush=True)
        Cc = qc.coclustering(F[m], r0, rng, c.n_boot, sub)
        sh = qc.split_half(F[m], r0, rng, c.n_split, sub)
        stab_rows.append({"component": m, "split_half_ari_mean": float(np.mean(sh)), "split_half_ari_min": float(np.min(sh)),
                          "split_half_ari_max": float(np.max(sh))})
        Zm = qc.zscore(F[m])
        cl_rows, mem_rows, cls = [], [], []
        for k in sorted(set(labs)):
            mem = np.where(labs == k)[0]
            if len(mem) < 2:
                continue
            mem = np.array([i for i in order if labs[i] == k])          # leaf order
            tp = topics[mem][:, None] != topics[mem][None, :]
            cross = float(tp[np.triu_indices(len(mem), 1)].mean())
            score = Zm[:, mem].mean(1)
            tc = pd.Series(topics[mem]).value_counts()
            Rm = R[m][np.ix_(mem, mem)]
            central = int(mem[np.argmax(Rm.sum(1))])
            pi, pj = np.triu_indices(len(mem), 1)
            xp = topics[mem][pi] != topics[mem][pj]
            rew = T[mem[pi], mem[pj]] >= float(rw.cos_min.iloc[0])
            # image-wide impression (upper bound): remove the leading components of the other topics' answers
            others = np.where(~np.isin(topics, topics[mem]))[0]
            U = np.linalg.svd(Fr[m][:, others] - Fr[m][:, others].mean(0), full_matrices=False)[0][:, :c.n_pcs_impression]
            r_imp = qc.pair_mean(qc.corr(qc.residualise(Fr[m][:, mem], [U])), np.arange(len(mem)))
            elig = (nyes[mem[pi]] >= c.min_yes) & (nyes[mem[pj]] >= c.min_yes)
            d = {"k": int(k), "n": int(len(mem)), "n_pairs": int(len(pi)), "cross_pairs": int(xp.sum()),
                 "mean_r_impression_removed": round(r_imp, 3),
                 "mean_r_graded_no_only": round(qc.pair_mean(Rcap, mem), 3) if m == "judgement" else None,
                 "yes_pairs": int(elig.sum()), "mean_phi_yes_pairs": round(float(Rbin[mem[pi], mem[pj]][elig].mean()), 3) if elig.any() else None,
                 "cross_pairs_distinct": int((xp & ~rew).sum()), "reworded_pairs": int(rew.sum()), "members": [int(i) for i in mem], "central": central,
                 "topics": [[t, int(v)] for t, v in tc.items()], "n_topics": int(len(tc)), "cross_prompt_share": round(cross, 3),
                 "min_r": round(qc.pair_min(R[m], mem), 3), "mean_r": round(qc.pair_mean(R[m], mem), 3),
                 "mean_r_within_subtype": round(qc.pair_mean(Rsub[m], mem), 3), "mean_r_strict": round(qc.pair_mean(Rctrl[m], mem), 3),
                 "min_r_strict": round(qc.pair_min(Rctrl[m], mem), 3),
                 "mean_r_other": {o: round(qc.pair_mean(R[o], mem), 3) for o in COMPONENTS if o != m},
                 "mean_cos": round(qc.pair_mean(T, mem), 3), "cos_pct": round(pct(qc.pair_mean(T, mem)), 3),
                 "shared_source_share": round(qc.pair_mean(shared_src.astype(float), mem), 3),
                 "stability": round(qc.pair_mean(Cc, mem), 3), "auroc": round(float(ex.auroc(y2, score)), 3),
                 "artefact_share": round(float((qt.kind.values[mem] == "artefact").mean()), 3),
                 "prof": [round(float(score[sub == s].mean() / (score.std() or 1)), 3) for s in sub_order]}
            cls.append(d)
            cl_rows.append({kk: (json.dumps(v) if isinstance(v, (list, dict)) else v) for kk, v in d.items() if kk not in ("members", "prof")}
                           | {"central_question": qt.question.values[central], "top_topics": "; ".join(f"{t} ({v})" for t, v in tc.head(3).items())})
            mem_rows += [{"cluster": int(k), "question_id": qids[i], "question": qt.question.values[i], "topic": topics[i]} for i in mem]
        pd.DataFrame(cl_rows).to_csv(out / f"clusters_{m}.csv", index=False)
        pd.DataFrame(mem_rows).to_csv(out / f"cluster_members_{m}.csv", index=False)
        # map of the clusters: mean rho between clusters, metric MDS on 1 - that, links to the closest clusters
        from sklearn.manifold import MDS
        K = len(cls)
        Mb = np.eye(K)
        Sb = np.zeros((K, K))
        for a_ in range(K):
            for b_ in range(a_ + 1, K):
                A_, B_ = cls[a_]["members"], cls[b_]["members"]
                Mb[a_, b_] = Mb[b_, a_] = R[m][np.ix_(A_, B_)].mean()
                Sb[a_, b_] = Sb[b_, a_] = (topics[A_][:, None] != topics[B_][None, :]).mean()
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=cfg.seed, n_init=4, normalized_stress="auto").fit_transform(np.clip(1 - Mb, 0, 2))
        xy = (xy - xy.min(0)) / np.where(np.ptp(xy, 0) > 0, np.ptp(xy, 0), 1)
        links = set()
        for a_ in range(K):
            nb = [b_ for b_ in np.argsort(-Mb[a_]) if b_ != a_][: c.map_links]
            links |= {(min(a_, b_), max(a_, b_)) for b_ in nb if Mb[a_, b_] >= c.map_min_r}
        mirror = int(sum(Mb[a_, b_] <= -0.5 for a_ in range(K) for b_ in range(a_ + 1, K)))
        payload["comp"][m] = {"labels": [int(x) for x in labs], "order": order, "clusters": cls,
                              "R": ex_b64(R[m]), "Rsub": ex_b64(Rsub[m]), "Rstrict": ex_b64(Rctrl[m]),
                              "map": {"xy": np.round(xy, 4).tolist(), "links": [[int(a_), int(b_), round(float(Mb[a_, b_]), 3), round(float(Sb[a_, b_]), 3)] for a_, b_ in sorted(links)],
                                      "mirror_pairs": mirror},
                              "split_half": [round(v, 3) for v in sh]}
        # how many clusters hold under every check, over a grid of criteria
        grid = []
        for st in c.robust_grid.stability:
            for rs in c.robust_grid.strict:
                ok = [d_ for d_ in cls if d_["n_topics"] > 1 and d_["stability"] >= st and d_["mean_r_strict"] >= rs]
                grid.append({"stability_min": st, "strict_min": rs, "n": len(ok), "n_clinical": sum(d_["artefact_share"] == 0 for d_ in ok),
                             "n_3plus": sum(d_["n"] >= 3 for d_ in ok)})
        payload["comp"][m]["robust_grid"] = grid
    pd.DataFrame(stab_rows).to_csv(out / "stability.csv", index=False)
    payload["stability"] = stab_rows
    payload["questions"] = {"text": qt.question.tolist(), "topic": list(topics), "kind": qt.kind.tolist(),
                            "auroc": [None if np.isnan(v) else round(float(v), 3) for v in [ex.auroc(y2, F["judgement"][:, j]) for j in range(Q)]],
                            "id": qids}
    payload["subtypes"] = sub_order
    (out / "payload.json").write_text(json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")))
    write_summary(out, payload, pairs, wm, rw, vp, thr_rows, stab_rows, null_r, vj, graded)
    main = payload["comp"][c.main_component]
    span = sum(1 for d in main["clusters"] if d["n_topics"] > 1)
    sm = pairs[(pairs.component == c.main_component) & (pairs.control == "none") & (pairs.r_min == r0)].iloc[0]
    return {"n_questions": Q, "clusters_main": len(main["clusters"]), "clusters_spanning_prompts": span,
            "same_prompt_share_main": round(float(sm.same_prompt_share_of_those), 3),
            "key_metric": f"{c.main_component} r>={r0}: {len(main['clusters'])} clusters, {span} span prompts; "
                          f"{sm.same_prompt_share_of_those:.0%} of pairs same-prompt (base {sm.same_prompt_base_rate:.0%})"}


def ex_b64(R: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.round(R * 127).astype(np.int8).tobytes()).decode()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_summary(out, payload, pairs, wm, rw, vp, thr_rows, stab_rows, null_r, vj, graded):
    m0, r0 = payload["meta"]["main"], payload["meta"]["r_min"]
    f = lambda v, d=2: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    L = [f"# S15 Qclusters: questions answered alike at r >= {r0}, and why", "",
         f"{payload['meta']['n_questions']} questions (clinical and artefact) on {payload['meta']['n_images']} images. Complete linkage on 1 - r, "
         f"cut at r = {r0}: every pair inside a cluster correlates at r >= {r0}.", "",
         "## Same prompt or separate prompts", "",
         "Questions of one topic are asked in one prompt; different topics are separate requests, so a correlation between "
         "questions of different topics cannot come from a shared prompt.", "",
         "| component | control | mean r, same prompt | mean r, separate prompts | pairs at r >= r_min | of which same prompt (base rate) |", "|---|---|---|---|---|---|"]
    for r in pairs[pairs.r_min == r0].itertuples():
        L.append(f"| {r.component} | {r.control} | {f(r.mean_r_same_prompt, 3)} | {f(r.mean_r_separate_prompts, 3)} | {r.pairs_at_or_above} | "
                 f"{f(100 * r.same_prompt_share_of_those, 0)}% ({f(100 * r.same_prompt_base_rate, 0)}%) |")
    L += ["", f"Stated confidence shared within a prompt: after removing image and question means, the prompt (image x topic) mean "
          f"explains {f(100 * vp['prompt_share_of_residual'], 0)}% of the remaining variance, against {f(100 * vp['null_mean'], 0)}% "
          f"(95th percentile {f(100 * vp['null_95'], 0)}%) with topics reassigned at random. The same for the judgement (ranks of "
          f"P(yes)): {f(100 * vj['prompt_share_of_residual'], 0)}%, against {f(100 * vj['null_mean'], 0)}%.", "",
          f"Graded 'no' or yes/no: of the {graded['cross_pairs_at_r_min']} pairs from separate prompts at rho >= {r0} (judgement), "
          f"{f(100 * graded['kept_with_graded_no_only'], 0)}% stay at that level when only the gradation among 'no' answers is kept "
          f"(P(yes) capped at {graded['graded_cap']}) and {f(100 * graded['kept_with_yes_no_only'], 0)}% when only the yes/no split is kept. "
          f"For the {graded['pairs_both_min_yes']} of them in which both questions have >= {graded['min_yes']} 'yes' images, median phi "
          f"{f(graded['median_phi_those'])} against median rho {f(graded['median_rho_those'])}. {f(100 * graded['share_answers_below_half'], 0)}% "
          f"of answers are below 0.5; {graded['never_yes']} questions are never answered yes.", "",
          f"Near-identical wordings (cosine >= {f(payload['meta']['reword_cos_min'])}, top share of pairs; these include opposites, so |rho|):", "",
          "| component | asked | pairs | median abs rho [IQR] | negative | median rho, positive pairs |", "|---|---|---|---|---|---|"]
    for r in rw.itertuples():
        L.append(f"| {r.component} | {r.asked} | {r.n_pairs} | {f(r.median_abs_r)} [{f(r.q25_abs)}, {f(r.q75_abs)}] | {r.n_negative} | {f(r.median_r_positive_only)} |")
    L += ["", "Largest |r| after shuffling each question across images: " + ", ".join(f"{m} {max(v):.2f}" for m, v in null_r.items()) + ".", "",
          "## Clusters", "", "| component | r_min | clusters (2+) | questions in them | 3+ | spanning prompts | largest |", "|---|---|---|---|---|---|---|"]
    for t in thr_rows:
        L.append(f"| {t['component']} | {t['r_min']} | {t['clusters_2plus']} | {t['questions_in_them']} | {t['clusters_3plus']} | {t['clusters_spanning_prompts']} | {t['largest']} |")
    g = payload["comp"][m0]["robust_grid"]
    L += ["", f"Clusters spanning prompts that hold under every check ({m0}), over the grid of criteria: {min(x['n'] for x in g)} to {max(x['n'] for x in g)} "
          f"({min(x['n_clinical'] for x in g)} to {max(x['n_clinical'] for x in g)} of clinical questions only). Pairs of clusters that mirror each other "
          f"(mean rho <= -0.5): {payload['comp'][m0]['map']['mirror_pairs']}."]
    L += ["", "Split-half agreement (adjusted Rand index, disjoint halves of the images): " +
          "; ".join(f"{s['component']} {s['split_half_ari_mean']:.2f} ({s['split_half_ari_min']:.2f} to {s['split_half_ari_max']:.2f})" for s in stab_rows) + ".", "",
          f"## Largest clusters spanning prompts ({m0})", ""]
    qtext, topics = payload["questions"]["text"], payload["questions"]["topic"]
    cl = sorted([d for d in payload["comp"][m0]["clusters"] if d["n_topics"] > 1], key=lambda d: -d["n"])[:15]
    for d in cl:
        L.append(f"- **{d['n']} questions, {d['n_topics']} topics** (min r {d['min_r']}, within subtype {d['mean_r_within_subtype']}, "
                 f"graded no only {d['mean_r_graded_no_only']}, impression removed {d['mean_r_impression_removed']}, "
                 f"strict {d['mean_r_strict']}, stability {d['stability']}, wording cosine {d['mean_cos']}, "
                 f"{d['cross_pairs_distinct']} of {d['n_pairs']} pairs from separate prompts and not rewordings): " + "; ".join(qtext[i] for i in d["members"][:6]) + ("; ..." if d["n"] > 6 else ""))
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
