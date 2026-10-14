"""PanDerm image analysis via DermLIP-PanDerm (PanDerm-base vision encoder, CLIP-aligned text).

For a folder of skin images this script:
  1. embeds every image with PanDerm      -> outputs/image_embeddings.npy + images.csv
  2. (optional) zero-shot classifies each image against --labels, using the model card's
     prompt "This is a skin image of {label}"  -> outputs/zero_shot.csv
  3. (optional) ranks the generated questions by image-text similarity, i.e. which questions
     PanDerm considers most relevant to each image -> outputs/question_relevance.csv
     This is relevance, not an answer: CLIP-style models cannot answer yes/no questions.

Usage (from the project root, inside `poetry shell`, after ./setup_derm1m.sh):
  python "Image Analysis/panderm_image_analysis.py" path/to/images
  python "Image Analysis/panderm_image_analysis.py" path/to/images \\
      --labels melanoma nevus "basal cell carcinoma" "seborrheic keratosis" \\
      --questions "2026-10-08 Generated Questions/2026-10-13_binary_dermal_image_questions_A.csv"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dermlip import encode_texts, load_dermlip, pick_device  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@torch.no_grad()
def embed_images(model, preprocess, paths: list[Path], device: str, batch_size: int) -> torch.Tensor:
    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc="PanDerm images"):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]])
        feats = model.encode_image(batch.to(device)).float()
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.cat(out)


def read_questions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    return pd.DataFrame({"qid": df[cols.get("id") or cols["question id"]], "question": df[cols["question"]].astype(str)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, help="image file or folder (searched recursively)")
    parser.add_argument("--labels", nargs="+", help="diagnosis labels for zero-shot classification")
    parser.add_argument("--questions", type=Path, help="question CSV to rank by relevance per image")
    parser.add_argument("--top-k", type=int, default=10, help="questions kept per image")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    if args.images.is_file():
        paths = [args.images]
    else:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} images; device={device}")
    model, preprocess, tokenizer = load_dermlip(device)

    img = embed_images(model, preprocess, paths, device, args.batch_size)
    np.save(args.out / "image_embeddings.npy", img.numpy())
    pd.DataFrame({"image": [str(p) for p in paths]}).to_csv(args.out / "images.csv", index_label="row")
    print(f"saved image_embeddings.npy {tuple(img.shape)}")

    # CLIP logit scale turns cosine similarities into softmax-ready logits.
    scale = model.logit_scale.exp().item()

    if args.labels:
        txt = encode_texts(model, tokenizer, [f"This is a skin image of {c}" for c in args.labels], device)
        probs = (scale * img @ txt.T).softmax(dim=-1).numpy()
        zs = pd.DataFrame(probs, columns=args.labels)
        zs.insert(0, "image", [str(p) for p in paths])
        zs.insert(1, "predicted", np.array(args.labels)[probs.argmax(1)])
        zs.to_csv(args.out / "zero_shot.csv", index=False)
        print("saved zero_shot.csv")

    if args.questions:
        qs = read_questions(args.questions)
        sims = (img @ encode_texts(model, tokenizer, qs["question"].tolist(), device).T).numpy()
        rows = []
        for i, p in enumerate(paths):
            for rank, j in enumerate(np.argsort(-sims[i])[: args.top_k], start=1):
                rows.append({"image": str(p), "rank": rank, "qid": qs.qid[j], "question": qs.question[j], "cosine": round(float(sims[i, j]), 4)})
        pd.DataFrame(rows).to_csv(args.out / "question_relevance.csv", index=False)
        print("saved question_relevance.csv")


if __name__ == "__main__":
    main()
