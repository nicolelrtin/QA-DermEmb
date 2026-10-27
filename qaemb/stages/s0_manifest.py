"""S0 Manifest: existing manifest -> image_id, path, label, patient_id, split, provenance.

label: 1 = positive (malignant), 0 = negative (benign), -1 = indeterminate kept as holdout.
split: train / val / test (grouped, stratified), 'loo' (leave-one-out over labelled images), or
'holdout' for label -1. The source manifest and images are read-only.
"""
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from ..config import ROOT, file_hash, stable_hash

UPSTREAM: list[str] = []

VERSION = 1  # bump when the stage's output changes for the same config


def key(cfg, ctx) -> dict:
    manifest = ROOT / cfg.data.root / cfg.data.manifest
    return {"data": cfg.data, "label": cfg.label, "split": cfg.split, "seed": cfg.seed,
            "manifest_sha": file_hash(manifest)}


def map_labels(raw: pd.Series, lcfg) -> pd.Series:
    raw = raw.fillna("").astype(str).str.strip().str.lower()
    pos, neg = {v.lower() for v in lcfg.positive}, {v.lower() for v in lcfg.negative}
    ind = {v.lower() for v in lcfg.indeterminate_values}
    unknown = sorted(set(raw) - pos - neg - ind)
    if unknown:
        raise SystemExit(f"Label values not covered by label.positive/negative/indeterminate_values: {unknown}")
    ind_label = {"holdout": -1, "drop": -2, "benign": 0, "malignant": 1}[lcfg.indeterminate]
    return raw.map(lambda v: 1 if v in pos else 0 if v in neg else ind_label)


def grouped_split(df: pd.DataFrame, fractions, seed: int) -> pd.Series:
    """Assign train/val/test with no group (patient proxy) crossing splits, stratified by label."""
    split = pd.Series("holdout", index=df.index)
    lab = df[df.label >= 0]
    n_test = max(2, round(1 / fractions.test))
    test_idx = next(StratifiedGroupKFold(n_test, shuffle=True, random_state=seed).split(lab, lab.label, lab.patient_id))[1]
    split[lab.index[test_idx]] = "test"
    rest = lab.drop(lab.index[test_idx])
    n_val = max(2, round((1 - fractions.test) / fractions.val))
    val_idx = next(StratifiedGroupKFold(n_val, shuffle=True, random_state=seed).split(rest, rest.label, rest.patient_id))[1]
    split[rest.index] = "train"
    split[rest.index[val_idx]] = "val"
    return split


def run(cfg, out, inputs, ctx) -> dict:
    d = cfg.data
    root = ROOT / d.root
    src = pd.read_csv(root / d.manifest, dtype=str, keep_default_na=False)
    if d.image_ids:
        missing = set(d.image_ids) - set(src[d.id_column])
        if missing:
            raise SystemExit(f"data.image_ids not in manifest: {sorted(missing)}")
        src = src[src[d.id_column].isin(set(d.image_ids))]

    df = pd.DataFrame({
        "image_id": src[d.id_column].values,
        "path": [str(root / p) for p in src[d.path_column]],
        "label_raw": src[cfg.label.column].values,
        "label": map_labels(src[cfg.label.column], cfg.label).values,
        "patient_id": src[cfg.split.group_column].values,
    })
    df = df[df.label != -2].reset_index(drop=True)

    missing = [p for p in df.path if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(df)} manifest images are missing on disk, e.g. {missing[:3]}")

    if cfg.split.strategy == "loo":
        df["split"] = df.label.map(lambda v: "loo" if v >= 0 else "holdout")
    elif cfg.split.strategy == "grouped":
        df["split"] = grouped_split(df, cfg.split.fractions, cfg.seed)
    else:
        raise SystemExit(f"Unknown split.strategy {cfg.split.strategy!r}")

    prov = dict(cfg.data.provenance)
    df["provenance"] = f"{d.name}@{prov['version']}#{stable_hash(prov, 8)}"
    sizes = [Image.open(p).size for p in df.path]
    df["orig_w"], df["orig_h"] = [s[0] for s in sizes], [s[1] for s in sizes]
    df["file_sha256"] = [file_hash(p) for p in df.path]
    df["path"] = [str((ROOT / p).relative_to(ROOT)) for p in df.path]
    df.to_parquet(out / "manifest.parquet", index=False)

    counts = df.groupby(["split", "label"]).size()
    print(counts.to_string())
    return {"n_images": len(df), "counts": {f"{s}/{l}": int(n) for (s, l), n in counts.items()},
            "label_mapping": {"column": cfg.label.column, "indeterminate": cfg.label.indeterminate},
            "provenance": prov}
