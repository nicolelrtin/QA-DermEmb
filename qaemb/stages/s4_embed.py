"""S4 Embedding store.

answers.parquet: the long table (handoff §5.1) with one row per (unit, Question ID, original), plus the
pair-level adjusted answer on both rows of a pair:
  p_adj       = P(yes | original) - P(yes | complement), in [-1, 1]. Near +1 / -1 = confident yes / no;
                near 0 = the model said the same thing to a question and its negation (not confident).
  incoherence = P(yes | original) + P(yes | complement) - 1; 0 for a perfectly complementary pair,
                > 0 = says yes to both, < 0 = says no to both.
Matrices per source s in {orig, neg, adj}: X_image_<s>.parquet [N_img x Q] and, for patch runs,
X_patch_<s>.npy [N_img x n x n x Q] (n = cells per side; 2G-1 with 50% overlap) with patch_index.json.
questions.parquet: one row per Question ID (original text, complement text, topic/scope/kind).
answers_readable.csv: one row per (unit, Question ID) with label, question texts, P(yes) to each version,
p_adj and a plain-language verdict -- the file to open when reviewing what the model answered.
"""
import json

import numpy as np
import pandas as pd

from ..questions import question_table

UPSTREAM = ["manifest", "answer"]
VERSION = 4

SCHEMA = ["run_id", "image_id", "patch_id", "grid_G", "bbox", "question_id", "original", "qset_hash", "p_yes", "answer_bin",
          "threshold", "p_adj", "answer_adj_bin", "adj_threshold", "incoherence", "mode", "batch_k", "batch_id",
          "position_in_batch", "prompt_hash", "answerer_id", "yesno_mass"]
UNIT = ["image_id", "patch_id"]


def key(cfg, ctx) -> dict:
    return {"threshold": cfg.answer.threshold, "adj_threshold": cfg.answer.adj_threshold}


def run(cfg, out, inputs, ctx) -> dict:
    raw = pd.read_parquet(inputs["answer"] / "answers_raw.parquet")
    thr, adj_thr = cfg.answer.threshold, cfg.answer.adj_threshold
    raw["threshold"], raw["adj_threshold"] = thr, adj_thr
    raw["answer_bin"] = (raw.p_yes >= thr).astype("int8")

    wide = raw.pivot_table(index=UNIT + ["question_id"], columns="original", values="p_yes")
    has_neg = 0 in wide.columns
    sources = {"orig": wide[1]}
    if has_neg:
        if wide[0].isna().any():
            raise RuntimeError("Some questions were asked without their complement")
        sources["neg"] = wide[0]
        sources["adj"] = wide[1] - wide[0]
        pair = pd.DataFrame({"p_adj": sources["adj"], "incoherence": wide[1] + wide[0] - 1}).reset_index()
        raw = raw.merge(pair, on=UNIT + ["question_id"], how="left")
        raw["answer_adj_bin"] = (raw.p_adj >= adj_thr).astype("int8")
    else:
        raw["p_adj"] = raw["incoherence"] = np.nan
        raw["answer_adj_bin"] = -1
    raw[SCHEMA].to_parquet(out / "answers.parquet", index=False)

    qt = question_table(ctx.questions)
    qt.to_parquet(out / "questions.parquet", index=False)
    qids = list(qt.question_id)

    summary = {"n_images": int(raw.image_id.nunique()), "n_question_ids": len(qids), "complements": has_neg,
               "sources": list(sources)}
    if has_neg:
        p1, p0 = wide[1], wide[0]
        summary.update({
            "mean_abs_p_adj": float(sources["adj"].abs().mean()),
            "mean_incoherence": float((p1 + p0 - 1).mean()),
            "mean_abs_incoherence": float((p1 + p0 - 1).abs().mean()),
            "contradiction_rate": float(((p1 >= thr) == (p0 >= thr)).mean()),  # same binary answer to Q and not-Q
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
    wide = raw.pivot_table(index=UNIT + ["question_id"], columns="original", values="p_yes").reset_index()
    wide = wide.rename(columns={1: "p_yes_original", 0: "p_yes_complement"})
    cols = ["question_id", "topic", "scope", "kind", "question", "question_neg"]
    df = man.merge(wide, on="image_id").merge(qt[[c for c in cols if c in qt]], on="question_id", how="left")
    if has_neg:
        df["p_adj"] = df.p_yes_original - df.p_yes_complement
        df["incoherence"] = df.p_yes_original + df.p_yes_complement - 1
        df["verdict"] = [verdict(a, b, thr) for a, b in zip(df.p_yes_original, df.p_yes_complement)]
    else:
        df["verdict"] = np.where(df.p_yes_original >= thr, "yes", "no")
    front = ["image_id", "label_raw", "split", "patch_id", "question_id", "topic", "scope", "kind", "question",
             "p_yes_original", "question_neg", "p_yes_complement", "p_adj", "incoherence", "verdict"]
    df = df[[c for c in front if c in df]].sort_values(["image_id", "patch_id", "question_id"])
    df.round(4).to_csv(out / "answers_readable.csv", index=False)
