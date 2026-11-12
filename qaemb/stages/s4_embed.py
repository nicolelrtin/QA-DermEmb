"""S4 Embedding store.

answers.parquet: the long table (handoff §5.1) with one row per (unit, Question ID, original), plus the
pair-level scores on both rows of a pair (Po = P(yes | original), Pc = P(yes | complement)):
  Net_P             = Po - Pc, in [-1, 1]. Near +1 / -1 = confident yes / no; near 0 = the same answer to a
                      question and its negation, or uncertain.
  incoherence       = Po + Pc - 1; 0 for a perfectly complementary pair, > 0 = yes to both, < 0 = no to both.
  P_w               = Po x |Net_P|: the original answer, discounted when the pair is inconsistent.
  Directional_Score = (2 Po - 1) x |Net_P|, in [-1, 1]: direction and confidence from the original question,
                      shrunk by the gap to the complement (about sign(Net_P) x Net_P^2 for coherent pairs).
  Coherence_Score   = (2 Po - 1) x (1 - |incoherence|), in [-1, 1]: direction and confidence from the original
                      question, shrunk only by contradiction (coherent 0.7 / 0.3 keeps its full 0.4).
  confidence  = the answerer's stated confidence in its answer (Claude; NaN for logit answerers).

Without negations (questions.complements = false; the Claude pipeline from 2026-11-05), the stated confidence c
stands in for Net_P, signed by the direction of the answer: Net_P = sign(2 Po - 1) x c. The implied complement
Pc = Po - Net_P then gives every score through the same formulas, which reduce to (d = 2 Po - 1):
  Directional_Score = d x c                    -- the answer, shrunk when Claude says it is unsure;
  Coherence_Score   = d x (1 - | |d| - c |)    -- the answer, shrunk when the stated probability and the stated
                                                  confidence disagree (e.g. P(yes) = 0.95 with confidence 0.3);
  P_w = Po x c; incoherence = d - sign(d) x c. summary['scores_from'] records which route was taken.
Matrices per source s in {orig, neg, adj (Net_P), w (P_w), dir (Directional_Score), coh (Coherence_Score), conf (stated
confidence; confidence route only)}: X_image_<s>.parquet [N_img x Q] and, for patch runs,
X_patch_<s>.npy [N_img x n x n x Q] (n = cells per side; 2G-1 with 50% overlap) with patch_index.json.
questions.parquet: one row per Question ID (original text, complement text, topic/scope/kind).
answers_readable.csv: one row per (unit, Question ID) with label, question texts, P(yes) to each version,
Net_P, P_w, both scores and a plain-language verdict -- the file to open when reviewing what the model answered.
"""
import json

import numpy as np
import pandas as pd

from ..questions import question_table

UPSTREAM = ["manifest", "answer"]
VERSION = 8

SCHEMA = ["run_id", "image_id", "patch_id", "grid_G", "bbox", "question_id", "original", "qset_hash", "p_yes", "answer_bin",
          "threshold", "Net_P", "answer_adj_bin", "adj_threshold", "incoherence", "mode", "batch_k", "batch_id",
          "position_in_batch", "prompt_hash", "answerer_id", "yesno_mass", "P_w", "Directional_Score", "Coherence_Score",
          "confidence"]
SCORES = ["Net_P", "P_w", "Directional_Score", "Coherence_Score"]


def pair_scores(po, pc) -> dict:
    """The pair-level scores from P(yes | original) and P(yes | complement) (see module docstring)."""
    net, inc = po - pc, po + pc - 1
    return {"Net_P": net, "incoherence": inc, "P_w": po * net.abs(),
            "Directional_Score": (2 * po - 1) * net.abs(), "Coherence_Score": (2 * po - 1) * (1 - inc.abs())}


def confidence_scores(po, conf) -> dict:
    """The same scores without a negated question: Net_P = sign(2 Po - 1) x stated confidence."""
    net = np.sign(2 * po - 1) * conf
    return pair_scores(po, po - net)


UNIT = ["image_id", "patch_id"]


def key(cfg, ctx) -> dict:
    return {"threshold": cfg.answer.threshold, "adj_threshold": cfg.answer.adj_threshold}


def run(cfg, out, inputs, ctx) -> dict:
    raw = pd.read_parquet(inputs["answer"] / "answers_raw.parquet")
    thr, adj_thr = cfg.answer.threshold, cfg.answer.adj_threshold
    raw["threshold"], raw["adj_threshold"] = thr, adj_thr
    raw["answer_bin"] = (raw.p_yes >= thr).astype("int8")
    if raw.p_yes.isna().any():
        raise RuntimeError(f"{int(raw.p_yes.isna().sum())} answers are missing; every question must be answered")
    if "confidence" not in raw:
        raw["confidence"] = np.nan

    wide = raw.pivot(index=UNIT + ["question_id"], columns="original", values="p_yes")
    has_neg = 0 in wide.columns
    sources = {"orig": wide[1]}
    if has_neg:
        sources["neg"] = wide[0]
        ps = pair_scores(wide[1], wide[0])
        sources.update({"adj": ps["Net_P"], "w": ps["P_w"], "dir": ps["Directional_Score"], "coh": ps["Coherence_Score"]})
        pair = pd.DataFrame(ps).reset_index()
        raw = raw.merge(pair, on=UNIT + ["question_id"], how="left")
        raw["answer_adj_bin"] = (raw.Net_P >= adj_thr).astype("int8")
    elif raw.confidence.notna().all():
        conf = raw.pivot(index=UNIT + ["question_id"], columns="original", values="confidence")[1]
        ps = confidence_scores(wide[1], conf)
        sources.update({"adj": ps["Net_P"], "w": ps["P_w"], "dir": ps["Directional_Score"], "coh": ps["Coherence_Score"],
                        "conf": conf})
        raw = raw.merge(pd.DataFrame(ps).reset_index(), on=UNIT + ["question_id"], how="left")
        raw["answer_adj_bin"] = (raw.Net_P >= adj_thr).astype("int8")
    else:
        for c in SCORES + ["incoherence"]:
            raw[c] = np.nan
        raw["answer_adj_bin"] = -1
    scores_from = "negation" if has_neg else "confidence" if "dir" in sources else "none"
    raw[SCHEMA].to_parquet(out / "answers.parquet", index=False)

    qt = question_table(ctx.questions)
    qt.to_parquet(out / "questions.parquet", index=False)
    qids = list(qt.question_id)

    summary = {"n_images": int(raw.image_id.nunique()), "n_question_ids": len(qids), "complements": has_neg,
               "sources": list(sources), "scores_from": scores_from}
    if raw.confidence.notna().any():
        summary["confidence_mean"] = float(raw.confidence.mean())
    if scores_from == "confidence":
        d, c = 2 * wide[1] - 1, sources["conf"]
        summary.update({
            "mean_Directional_Score": float(sources["dir"].mean()),
            "mean_Coherence_Score": float(sources["coh"].mean()),
            "corr_abs_d_confidence": float(np.corrcoef(d.abs(), c)[0, 1]),  # how far confidence tracks |2 P(yes) - 1|
            "corr_dir_coh": float(np.corrcoef(sources["dir"], sources["coh"])[0, 1]),
        })
    if has_neg:
        p1, p0 = wide[1], wide[0]
        summary.update({
            "mean_abs_Net_P": float(sources["adj"].abs().mean()),
            "mean_incoherence": float((p1 + p0 - 1).mean()),
            "mean_abs_incoherence": float((p1 + p0 - 1).abs().mean()),
            "contradiction_rate": float(((p1 >= thr) == (p0 >= thr)).mean()),  # same answer to Q and not-Q
            "mean_P_w": float(sources["w"].mean()),
            "mean_Directional_Score": float(sources["dir"].mean()),
            "mean_Coherence_Score": float(sources["coh"].mean()),
        })

    for name, s in sources.items():
        s = s.rename("v").reset_index()
        whole = s[s.patch_id < 0].pivot(index="image_id", columns="question_id", values="v")
        if not whole.empty:
            whole[qids].to_parquet(out / f"X_image_{name}.parquet")
        patches = s[s.patch_id >= 0]
        if patches.empty:
            continue
        n = int(np.sqrt(patches.patch_id.max() + 1))
        images = sorted(patches.image_id.unique())
        X = np.full((len(images), n, n, len(qids)), np.nan, dtype=np.float32)
        ii = patches.image_id.map({k: i for i, k in enumerate(images)}).values
        qi = patches.question_id.map({k: i for i, k in enumerate(qids)}).values
        X[ii, patches.patch_id.values // n, patches.patch_id.values % n, qi] = patches.v.values
        if np.isnan(X).any():
            raise RuntimeError(f"X_patch_{name} has missing (image, patch, question) entries")
        np.save(out / f"X_patch_{name}.npy", X)
        summary["X_patch_shape"] = list(X.shape)
        if name == "orig":
            p = raw[raw.patch_id >= 0]
            bboxes = p.drop_duplicates("patch_id").sort_values("patch_id").bbox.map(list).tolist()
            (out / "patch_index.json").write_text(json.dumps(
                {"image_ids": images, "question_ids": qids, "grid_n": n, "grid_G": int(p.grid_G.iloc[0]),
                 "bbox_by_patch_id": [[float(v) for v in b] for b in bboxes]}, indent=1))
    write_readable(out, raw, qt, inputs["manifest"], thr, has_neg)
    if has_neg:
        summary["key_metric"] = f"contradiction={summary['contradiction_rate']:.3f}"
    return summary


def verdict(p1: float, p0: float, thr: float) -> str:
    y1, y0 = p1 >= thr, p0 >= thr
    if y1 != y0:
        return "yes" if y1 else "no"
    return "contradiction: yes to both" if y1 else "contradiction: no to both"


def write_readable(out, raw: pd.DataFrame, qt: pd.DataFrame, manifest_dir, thr: float, has_neg: bool) -> None:
    man = pd.read_parquet(manifest_dir / "manifest.parquet")[["image_id", "label", "label_raw", "split"]]
    man["label_raw"] = man.label_raw.mask(man.label < 0, "indeterminate (holdout)")
    wide = raw.pivot(index=UNIT + ["question_id"], columns="original", values="p_yes").rename(
        columns={1: "p_yes_original", 0: "p_yes_complement"})
    conf = raw.pivot(index=UNIT + ["question_id"], columns="original", values="confidence").rename(
        columns={1: "confidence_original", 0: "confidence_complement"})
    wide = wide.join(conf).reset_index()
    wide.columns.name = None
    cols = ["question_id", "topic", "scope", "kind", "question", "question_neg"]
    df = man.merge(wide, on="image_id").merge(qt[[c for c in cols if c in qt]], on="question_id", how="left")
    if has_neg:
        for c, v in pair_scores(df.p_yes_original, df.p_yes_complement).items():
            df[c] = v
        df["verdict"] = [verdict(a, b, thr) for a, b in zip(df.p_yes_original, df.p_yes_complement)]
    elif "confidence_original" in df and df.confidence_original.notna().all():
        for c, v in confidence_scores(df.p_yes_original, df.confidence_original).items():
            df[c] = v
        df["verdict"] = np.where(df.p_yes_original >= thr, "yes", "no")
    else:
        df["verdict"] = np.where(df.p_yes_original >= thr, "yes", "no")
    front = ["image_id", "label_raw", "split", "patch_id", "question_id", "topic", "scope", "kind", "question",
             "p_yes_original", "confidence_original", "question_neg", "p_yes_complement", "confidence_complement",
             "Net_P", "P_w", "Directional_Score", "Coherence_Score", "incoherence", "verdict"]
    df = df[[c for c in front if c in df]].sort_values(["image_id", "patch_id", "question_id"])
    df = df.drop(columns=[c for c in ("confidence_original", "confidence_complement") if c in df and df[c].isna().all()])
    df.round(4).to_csv(out / "answers_readable.csv", index=False)
