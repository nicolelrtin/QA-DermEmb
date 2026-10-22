"""Embed the generated question sets with three models, for coverage plots.

Models
  qwen3      general state-of-the-art text embedder (Qwen3-Embedding, sentence-transformers)
  panderm    DermLIP-PanDerm text tower, aligned to the PanDerm image encoder
  skingpt    SkinGPT-R1 language model (Qwen2.5-VL-7B + derm LoRA), mean-pooled hidden states

Usage (from the project root, inside `poetry shell`):
  python "Question Embeddings/embed_questions.py"                 # all three models
  python "Question Embeddings/embed_questions.py" --models qwen3  # just one

Writes outputs/questions.csv (one row per question) and outputs/emb_<model>.npy
(row-aligned with questions.csv, L2-normalised).

Checkpointing: embeddings are cached per model in outputs/cache/<model>/, keyed by the
exact question text and saved every --chunk-size questions. Re-running after an
interruption, or after adding/editing question files, only embeds questions not yet in
the cache (and skips loading a model entirely when nothing is new). --rebuild clears the
cache for the selected models.
"""
# poetry run python Question\ Embeddings/group_topics.py
# poetry run python Question\ Embeddings/embed_questions.py --rebuild
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dermlip import encode_texts, load_dermlip, pick_device  # noqa: E402

QUESTIONS_DIR = ROOT / "2026-10-15 Generated Questions"
# file -> short label used in plots
QUESTION_SETS = {
    "DermQs_Source-1015A_Prompt-C_N-500.csv" : "Abstract - N500",
    "DermQs_Source-1015A_Prompt-C_N-1000.csv" : "Abstract - N1000",
    "DermQs_Source-1015A_Prompt-C_N-2500.csv" : "Abstract - N2500",
    "DermQs_Source-1015A_Prompt-C_N-5000.csv" : "Abstract - N5000",
    "DermQs_Source-1015FT_Prompt-C_N-100.csv" : "Full Text - N100",
    "DermQs_Source-1015FT_Prompt-C_N-2500.csv" : "Full Text - N2500",
    "DermQs_Source-1015AC_Prompt-C_N-2500.csv" : "Abstract - N2500 - Sci",
    # "2026-10-08_nonbinary_dermal_image_questions_A.csv": "Non-binary A (10-08)",
    # "2026-10-08_binary_dermal_image_questions_B.csv": "Binary B (10-08)",
    # "2026-10-13_binary_dermal_image_questions_A.csv": "Binary A (10-13)",
    # "2026-10-13_dermal_yes_no_questions_B.csv": "Binary B (10-13)",
}
QWEN3_SIZES = {"0.6B": "Qwen/Qwen3-Embedding-0.6B", "4B": "Qwen/Qwen3-Embedding-4B", "8B": "Qwen/Qwen3-Embedding-8B"}
SKINGPT_REPO = "yuhos16/SkinGPT-R1"
Encoder = Callable[[list[str]], np.ndarray]


def load_questions() -> pd.DataFrame:
    frames = []
    for fname, label in QUESTION_SETS.items():
        df = pd.read_csv(QUESTIONS_DIR / fname, encoding="utf-8-sig")
        cols = {c.lower(): c for c in df.columns}
        # Topic comes straight from the CSV. Older sets had a separate Section column; when a
        # file has only Topic, section repeats it so downstream scripts always find both.
        topic = df[cols.get("topic") or cols["section"]]
        frames.append(
            pd.DataFrame(
                {
                    "set": label,
                    "source_file": fname,
                    "qid": df[cols.get("id") or cols["question id"]],
                    "topic": topic,
                    "section": df[cols["section"]] if "section" in cols else topic,
                    "question": df[cols["question"]].astype(str).str.strip(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- checkpoint cache

def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Append-only cache of question embeddings for one model, stored as numbered .npz shards.

    Each shard holds `keys` (sha1 of the question text), `emb` (float32 rows) and `model_id`.
    Shards are written to a temp file and renamed, so an interrupted run never leaves a
    half-written shard behind.
    """

    def __init__(self, directory: Path, model_id: str):
        self.dir, self.model_id = directory, model_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vectors: dict[str, np.ndarray] = {}
        self.n_shards = 0
        for shard in sorted(self.dir.glob("shard_*.npz")):
            with np.load(shard) as z:
                if str(z["model_id"]) != model_id:
                    raise SystemExit(f"{shard} was made with {z['model_id']}, not {model_id}; use --rebuild.")
                self.vectors.update(zip(z["keys"].tolist(), z["emb"]))
            self.n_shards += 1

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def add(self, keys: list[str], emb: np.ndarray):
        emb = np.asarray(emb, dtype=np.float32)
        final = self.dir / f"shard_{self.n_shards:05d}.npz"
        tmp = final.with_suffix(".partial")
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=np.array(keys), emb=emb, model_id=np.array(self.model_id))
        os.replace(tmp, final)  # atomic: the shard either fully exists or not at all
        self.n_shards += 1
        self.vectors.update(zip(keys, emb))

    def matrix(self, keys: list[str]) -> np.ndarray:
        return np.stack([self.vectors[k] for k in keys])


# ---------------------------------------------------------------- model loaders
# Each loader loads its model once and returns a function: list of texts -> (N, dim) array.

def load_qwen3(device: str, size: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(QWEN3_SIZES[size], device=device, model_kwargs={"dtype": torch.bfloat16})
    # No query instruction: questions are compared with each other (symmetric), not used as search queries.
    return lambda texts: np.asarray(
        model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
    )


def load_panderm(device: str) -> Encoder:
    model, _, tokenizer = load_dermlip(device)
    return lambda texts: encode_texts(model, tokenizer, texts, device).numpy()


def load_skingpt(device: str, batch_size: int = 16) -> Encoder:
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

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i : i + batch_size], padding=True, return_tensors="pt").to(model.device)
            hidden = model.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
            mask = batch.attention_mask.unsqueeze(-1).to(device=hidden.device, dtype=hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean over real tokens
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    return encode


def model_spec(name: str, args) -> tuple[str, str, Callable[[], Encoder]]:
    """(cache folder name, model id stored in the cache, loader) for one model."""
    if name == "qwen3":
        return f"qwen3-{args.qwen3_size}", QWEN3_SIZES[args.qwen3_size], lambda: load_qwen3(args.device, args.qwen3_size)
    if name == "panderm":
        return "panderm", "redlessone/DermLIP_PanDerm-base-w-PubMed-256", lambda: load_panderm(args.device)
    return "skingpt", SKINGPT_REPO, lambda: load_skingpt(args.device)


def embed_with_cache(name: str, texts: list[str], args) -> np.ndarray:
    tag, model_id, loader = model_spec(name, args)
    cache_dir = args.out / "cache" / tag
    if args.rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache = EmbeddingCache(cache_dir, model_id)

    keys = [text_key(t) for t in texts]
    todo = list(dict.fromkeys(k for k in keys if k not in cache))  # unique, in question order
    n_cached = sum(k in cache for k in keys)
    print(f"{n_cached} of {len(texts)} questions already cached; {len(todo)} unique to embed")
    if todo:
        text_of = dict(zip(keys, texts))
        encode = loader()  # only load the model when there is something new to embed
        with tqdm(total=len(todo), desc=name, unit="q") as bar:
            for i in range(0, len(todo), args.chunk_size):
                chunk = todo[i : i + args.chunk_size]
                cache.add(chunk, encode([text_of[k] for k in chunk]))
                bar.update(len(chunk))
        del encode
    return cache.matrix(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3", "panderm", "skingpt"], choices=["qwen3", "panderm", "skingpt"])
    parser.add_argument("--qwen3-size", default="8B", choices=list(QWEN3_SIZES))
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=256, help="questions embedded between checkpoints")
    parser.add_argument("--rebuild", action="store_true", help="discard cached embeddings for the selected models")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    args.device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    questions.to_csv(args.out / "questions.csv", index=False)
    texts = questions["question"].tolist()
    print(f"{len(texts)} questions from {questions['set'].nunique()} sets; device={args.device}")

    for name in args.models:
        print(f"\n== {name}")
        emb = embed_with_cache(name, texts, args)
        np.save(args.out / f"emb_{name}.npy", emb)
        print(f"saved emb_{name}.npy {emb.shape}")
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
