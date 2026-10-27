"""S1 Prepare: aspect-preserving resize to the answerer pixel budget, plus an optional G x G grid.

Crops are cut from the original-resolution image and resized to the patch budget, so the VLM sees
a zoom. Output: units.parquet (one row per image or patch) and the resized PNGs.
bbox is (x0, y0, x1, y1) as fractions of the original image.
"""
import math

import numpy as np
import pandas as pd
from PIL import Image

from ..config import ROOT

UPSTREAM = ["manifest"]
FACTOR = 28  # Qwen2.5-VL: 14 px patches merged 2x2

VERSION = 1  # bump when the stage's output changes for the same config


def key(cfg, ctx) -> dict:
    return {"prepare": cfg.prepare, "grid": cfg.grid}


def budget_size(w: int, h: int, max_pixels: int, min_pixels: int) -> tuple[int, int]:
    """Scale (up or down) to fill max_pixels, keeping aspect; sides are multiples of 28 so the
    Qwen processor does not resize again. Small crops are therefore upscaled (a zoom)."""
    beta = math.sqrt(h * w / max_pixels)
    hb, wb = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR), max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    if hb * wb < min_pixels:
        raise ValueError(f"crop {w}x{h} too small for min_pixels={min_pixels}")
    return wb, hb


def grid_boxes(G: int, overlap: float) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    cell = 1.0 / G
    stride = cell * (1.0 - overlap)
    starts = np.round(np.arange(0, 1 - cell + 1e-9, stride), 6)
    return [(r, c, (float(x), float(y), float(min(1.0, x + cell)), float(min(1.0, y + cell))))
            for r, y in enumerate(starts) for c, x in enumerate(starts)]


def run(cfg, out, inputs, ctx) -> dict:
    manifest = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    p, g = cfg.prepare, cfg.grid
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    boxes = grid_boxes(g.G, g.overlap) if g.G > 1 else []
    n_side = int(math.isqrt(len(boxes))) if boxes else 1

    rows = []
    for rec in manifest.itertuples():
        img = Image.open(ROOT / rec.path).convert("RGB")
        W, H = img.size
        units = [(-1, -1, -1, (0.0, 0.0, 1.0, 1.0), p.max_pixels)] if (g.G == 1 or g.include_whole) else []
        units += [(r * n_side + c, r, c, box, p.patch_max_pixels) for r, c, box in boxes]
        for patch_id, row, col, box, budget in units:
            crop = img if patch_id < 0 else img.crop((round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)))
            w, h = budget_size(*crop.size, budget, p.min_pixels)
            name = f"{rec.image_id}.png" if patch_id < 0 else f"{rec.image_id}_G{g.G}_p{patch_id:03d}.png"
            crop.resize((w, h), Image.BICUBIC).save(img_dir / name)
            rows.append({"unit_id": name[:-4], "image_id": rec.image_id, "patch_id": patch_id,
                         "grid_G": g.G if patch_id >= 0 else 1, "grid_n": n_side if patch_id >= 0 else 1,
                         "row": row, "col": col, "bbox": list(box), "path": f"images/{name}", "width": w, "height": h})
    units = pd.DataFrame(rows)
    units.to_parquet(out / "units.parquet", index=False)
    return {"n_units": len(units), "n_images": manifest.shape[0], "grid_n": n_side,
            "vision_tokens_mean": float((units.width * units.height / FACTOR**2).mean())}
