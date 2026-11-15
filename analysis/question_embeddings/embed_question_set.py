"""Embed one question CSV (the questions as written, one row per Question ID) with the text models of embed_questions.py.

  python analysis/question_embeddings/embed_question_set.py \
      --csv inputs/questions/2026-10-15/DermQs_Source-1015FT_Prompt-C_N-500_Negated.csv \
      --dest analysis/question_embeddings/2026-11-15_N500

Writes <dest>/questions.csv (question_id, question) and <dest>/emb_<model>.npy, row-aligned and L2-normalised. Reuses
the October embedding cache (2026-10-15_outputs/cache), so questions embedded then are not embedded again. Read by the
S9 explore stage (explore.text_embeddings in conf/base.yaml) to cluster the questions by wording.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from embed_questions import QWEN3_SIZES, embed_with_cache  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["qwen3", "panderm"], choices=["qwen3", "panderm", "skingpt"])
    p.add_argument("--cache-root", type=Path, default=HERE / "2026-10-15_outputs")
    p.add_argument("--qwen3-size", default="8B", choices=list(QWEN3_SIZES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-size", type=int, default=256)
    a = p.parse_args()

    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    if "original" in df.columns:
        df = df[df.original.astype(int) == 1]
    q = pd.DataFrame({"question_id": df["Question ID"].str.strip(), "question": df["Question"].astype(str).str.strip()})
    a.dest.mkdir(parents=True, exist_ok=True)
    q.to_csv(a.dest / "questions.csv", index=False)
    args = argparse.Namespace(out=a.cache_root, rebuild=False, chunk_size=a.chunk_size, qwen3_size=a.qwen3_size,
                              device=a.device)
    for m in a.models:
        E = embed_with_cache(m, q.question.tolist(), args)
        np.save(a.dest / f"emb_{m}.npy", E.astype(np.float32))
        print(f"{m}: {E.shape} -> {a.dest / f'emb_{m}.npy'}")


if __name__ == "__main__":
    main()
