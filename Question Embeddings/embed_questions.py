"""Embed the four generated question sets with three models, for coverage plots.

Models
  qwen3      general state-of-the-art text embedder (Qwen3-Embedding, sentence-transformers)
  panderm    DermLIP-PanDerm text tower, aligned to the PanDerm image encoder
  skingpt    SkinGPT-R1 language model (Qwen2.5-VL-7B + derm LoRA), mean-pooled hidden states

Usage (from the project root, inside `poetry shell`):
  python "Question Embeddings/embed_questions.py"                 # all three models
  python "Question Embeddings/embed_questions.py" --models qwen3  # just one

Writes outputs/questions.csv (one row per question) and outputs/emb_<model>.npy
(row-aligned with questions.csv, L2-normalised).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dermlip import encode_texts, load_dermlip, pick_device  # noqa: E402

QUESTIONS_DIR = ROOT / "2026-10-08 Generated Questions"
# file -> short label used in plots
QUESTION_SETS = {
    "2026-10-08_binary_dermal_image_questions.csv": "Binary (10-08)",
    "2026-10-08_nonbinary_dermal_image_questions_B.csv": "Non-binary B (10-08)",
    "2026-10-13_binary_dermal_image_questions_A.csv": "Binary A (10-13)",
    "2026-10-13_dermal_yes_no_questions_B.csv": "Yes/no B (10-13)",
}
QWEN3_SIZES = {"0.6B": "Qwen/Qwen3-Embedding-0.6B", "4B": "Qwen/Qwen3-Embedding-4B", "8B": "Qwen/Qwen3-Embedding-8B"}
SKINGPT_REPO = "yuhos16/SkinGPT-R1"


def load_questions() -> pd.DataFrame:
    frames = []
    for fname, label in QUESTION_SETS.items():
        df = pd.read_csv(QUESTIONS_DIR / fname, encoding="utf-8-sig")
        cols = {c.lower(): c for c in df.columns}
        frames.append(
            pd.DataFrame(
                {
                    "set": label,
                    "source_file": fname,
                    "qid": df[cols.get("id") or cols["question id"]],
                    "section": df[cols.get("section") or cols["topic"]],
                    "question": df[cols["question"]].astype(str).str.strip(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def embed_qwen3(texts: list[str], device: str, size: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(QWEN3_SIZES[size], device=device, model_kwargs={"dtype": torch.bfloat16})
    # No query instruction: questions are compared with each other (symmetric), not used as search queries.
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def embed_panderm(texts: list[str], device: str) -> np.ndarray:
    model, _, tokenizer = load_dermlip(device)
    return encode_texts(model, tokenizer, texts, device).numpy()


@torch.no_grad()
def embed_skingpt(texts: list[str], device: str, batch_size: int = 16) -> np.ndarray:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    repo_dir = snapshot_download(SKINGPT_REPO, allow_patterns=["checkpoint/*"], ignore_patterns=["*.ipynb_checkpoints*"])
    ckpt = Path(repo_dir) / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # The checkpoint also holds SkinGPT-R1's visual adapter (distill_head.*); it is unused for text and
    # transformers reports those keys as unexpected, which is fine.
    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None, output_loading_info=True
    )
    # Guard against silently embedding with randomly initialised layers (e.g. a key-naming mismatch
    # between the checkpoint and this transformers version).
    missing = [k for k in info["missing_keys"] if "language_model" in k or k.startswith("model.layers")]
    if missing:
        raise RuntimeError(f"{len(missing)} language-model weights not loaded from {ckpt}, e.g. {missing[:3]}")
    if device != "cuda":
        model = model.to(device)
    model.eval()

    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SkinGPT-R1"):
        batch = tokenizer(texts[i : i + batch_size], padding=True, return_tensors="pt").to(model.device)
        hidden = model.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1).to(device=hidden.device, dtype=hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean over real tokens
        out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3", "panderm", "skingpt"], choices=["qwen3", "panderm", "skingpt"])
    parser.add_argument("--qwen3-size", default="8B", choices=list(QWEN3_SIZES))
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    questions.to_csv(args.out / "questions.csv", index=False)
    texts = questions["question"].tolist()
    print(f"{len(texts)} questions from {questions['set'].nunique()} sets; device={device}")

    for name in args.models:
        print(f"\n== {name}")
        if name == "qwen3":
            emb = embed_qwen3(texts, device, args.qwen3_size)
        elif name == "panderm":
            emb = embed_panderm(texts, device)
        else:
            emb = embed_skingpt(texts, device)
        np.save(args.out / f"emb_{name}.npy", emb)
        print(f"saved emb_{name}.npy {emb.shape}")
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
