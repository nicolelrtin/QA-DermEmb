"""S5 Classifiers on the embedding store: W0/W1 (whole image) and P0-P3 (patch MIL).

Images with label -1 (indeterminate holdout) are never fitted or scored in metrics; they get
predictions in predictions.parquet under split 'holdout'. Questions of kind in model.exclude_kinds
(artefact by default) are left out; patch models also use scope == local only.
Features come from feat.source: adj (default; p_adj = P(yes|original) - P(yes|complement)), orig or neg.
split 'loo': leave-one-out over labelled images (pooled predictions -> metrics['loo']); the final
model for coefficients/holdout is refit on all labelled images.
"""
import json

import numpy as np
import pandas as pd

from ..models.metrics import binary_metrics

UPSTREAM = ["manifest", "embed"]
VERSION = 2


def key(cfg, ctx) -> dict:
    if not cfg.model.name:
        raise SystemExit("stage=model needs a model experiment (W0, W1, P0, ...)")
    thr = cfg.answer.adj_threshold if cfg.feat.source == "adj" else cfg.answer.threshold
    return {"feat": cfg.feat, "bin_threshold": thr if cfg.feat.kind == "bin" else None, "model": cfg.model, "seed": cfg.seed}


def feature_mask(qs: pd.DataFrame, mcfg, patch: bool) -> np.ndarray:
    keep = np.ones(len(qs), bool)
    if "kind" in qs:
        keep &= ~qs.kind.isin(list(mcfg.exclude_kinds)).values
    if "scope" in qs:
        keep &= ~qs.scope.isin(list(mcfg.exclude_scopes)).values
        if patch:
            keep &= (qs.scope == "local").values
    if not keep.any():
        raise SystemExit("No questions left after kind/scope filtering")
    return keep


def load_features(cfg, embed_dir, image_ids, keep) -> np.ndarray:
    src = cfg.feat.source
    name = f"X_image_{src}.parquet" if cfg.model.name.startswith("W") else f"X_patch_{src}.npy"
    if not (embed_dir / name).exists():
        raise SystemExit(f"{name} not in the embedding store (feat.source={src}; adj/neg need complement questions)")
    if cfg.model.name.startswith("W"):
        X = pd.read_parquet(embed_dir / name).loc[image_ids].values[:, keep]
    else:
        idx = json.loads((embed_dir / "patch_index.json").read_text())
        pos = {k: i for i, k in enumerate(idx["image_ids"])}
        Xp = np.load(embed_dir / name)[[pos[i] for i in image_ids]]
        X = Xp.reshape(len(image_ids), -1, Xp.shape[-1])[..., keep]
    if cfg.feat.kind == "bin":
        X = (X >= (cfg.answer.adj_threshold if src == "adj" else cfg.answer.threshold)).astype(np.float32)
    return X.astype(np.float32)


def build(cfg):
    if cfg.model.name in ("W0", "W1"):
        from ..models.linear import LinearModel

        return LinearModel(cfg.model, cfg.seed)
    if cfg.model.name in ("P0", "P1", "P2", "P3"):
        from ..models.mil import MILModel

        return MILModel(cfg.model, cfg.seed)
    raise SystemExit(f"Model {cfg.model.name!r} is not implemented yet")


def run(cfg, out, inputs, ctx) -> dict:
    m = cfg.model
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    qs = pd.read_parquet(inputs["embed"] / "questions.parquet")
    keep = feature_mask(qs, m, patch=m.name.startswith("P"))
    qkeep = qs[keep].reset_index(drop=True)
    X = load_features(cfg, inputs["embed"], list(man.image_id), keep)
    y, groups, split = man.label.values, man.patient_id.values, man.split.values
    lab = y >= 0
    pred = np.full(len(man), np.nan)
    metrics = {}

    if cfg.split.strategy == "loo":
        idx = np.where(lab)[0]
        for i in idx:
            tr = idx[idx != i]
            pred[i] = build(cfg).fit(X[tr], y[tr], groups[tr]).predict(X[[i]])[0]
        metrics["loo"] = binary_metrics(y[idx], pred[idx], m.sens_at_spec, m.ece_bins)
        model = build(cfg).fit(X[idx], y[idx], groups[idx])
        metrics["train_refit"] = binary_metrics(y[idx], model.predict(X[idx]), m.sens_at_spec, m.ece_bins)
        if (~lab).any():
            pred[~lab] = model.predict(X[~lab])
        key_split = "loo"
    else:
        tr, va = split == "train", split == "val"
        model = build(cfg).fit(X[tr], y[tr], groups[tr], X[va], y[va])
        pred = model.predict(X)
        for s in ("train", "val", "test"):
            sel = split == s
            if sel.any():
                metrics[s] = binary_metrics(y[sel], pred[sel], m.sens_at_spec, m.ece_bins)
        key_split = "test"

    preds = man[["image_id", "split", "label", "label_raw"]].assign(prob=pred)
    preds.to_parquet(out / "predictions.parquet", index=False)
    if (~lab).any():
        metrics["holdout_mean_prob"] = float(pred[~lab].mean())

    coefs = qkeep.assign(coef=model.coef).sort_values("coef", key=np.abs, ascending=False)
    if hasattr(model, "stability"):
        coefs["stability"] = pd.Series(model.stability, index=qkeep.index)
        coefs["selected"] = coefs.stability >= m.stability_threshold
        sel = coefs[coefs.selected][["question_id", "question", "stability", "coef"]]
        sel.to_csv(out / "selected_questions.csv", index=False)
        metrics["n_selected"] = int(len(sel))
    coefs.to_csv(out / "coefficients.csv", index=False)
    curves = pd.DataFrame(getattr(model, "path", None) or getattr(model, "curves", []))
    curves.to_csv(out / "curves.csv", index=False)

    if m.name.startswith("P"):
        s, a = model.patch_outputs(X)
        idx = json.loads((inputs["embed"] / "patch_index.json").read_text())
        n = idx["grid_n"]
        rows = [{"image_id": img, "patch_id": k, "row": k // n, "col": k % n, "bbox": idx["bbox_by_patch_id"][k],
                 "score": float(s[i, k]), "attention": float(a[i, k]) if a is not None else None}
                for i, img in enumerate(man.image_id) for k in range(s.shape[1])]
        pd.DataFrame(rows).to_parquet(out / "patch_scores.parquet", index=False)
        if model.coef_by_position is not None:
            np.save(out / "coef_by_position.npy", model.coef_by_position)

    (out / "metrics.json").write_text(json.dumps({"params": model.params, "n_features": int(keep.sum()), **metrics},
                                                 indent=2, default=float))
    auroc = metrics.get(key_split, {}).get("auroc", float("nan"))
    print(f"  {m.name}: {key_split} AUROC={auroc:.3f}  features={int(keep.sum())}  params={model.params}")
    return {"n_features": int(keep.sum()), "params": model.params,
            f"{key_split}_auroc": auroc, f"{key_split}_auprc": metrics.get(key_split, {}).get("auprc"),
            "key_metric": f"{key_split}_AUROC={auroc:.3f}"}
