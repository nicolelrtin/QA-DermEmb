"""S12 VLM features: image embeddings and a candidate-label diagnosis from a local vision-language model (SkinGPT-R1), for
the benchmarks in S12 (PROJECT.md, Plan item E).

One forward pass per image over SkinGPT-R1's own candidate-label classification prompt (its prompts/classification_user.txt,
no system message, labels one per line; saved as inputs/prompts/<vlmfeat.prompt>.txt), followed by vlmfeat.answer_prefix,
which closes the reasoning block empty and opens the answer, as our S2 answerer does for SkinGPT-R1. From that pass:

  Embeddings  for each layer in vlmfeat.layers, the hidden state averaged over the image tokens (img_L<l>) and the hidden
              state at the last position, where the answer is about to start (last_L<l>). Layer 0 is the token-embedding
              layer: at the image positions it holds the vision encoder's output after the merger (vis = img_L0). The image
              tokens come before the instruction, so the img_L embeddings do not depend on the candidate list; the last_L
              ones do. SkinGPT-R1's vision encoder is almost unchanged from Qwen2.5-VL-7B (weights differ by about 0.15%,
              against 2-4% in the language model, checked 2026-11-16), so vis is in effect Qwen2.5-VL's image embedding.
  Diagnosis   log P(label + vlmfeat.label_suffix | prompt) for every candidate label, the probability that the model writes
              exactly that label as its answer. Softmax over the candidates gives SkinGPT-R1's diagnosis as a distribution;
              S13 sums it into P(malignant) and into the 10 subtypes with the candidate map in vlmfeat.candidates.
              Without reasoning the choice depends on where a label sits in the list (smoke test 2026-11-16: reversing
              the list moved a melanoma from "Melanoma" to "Pyogenic granuloma"), so each image is scored with the list
              in the given order and reversed (vlmfeat.orders), and the diagnosis is the mean of the two log-probabilities.
              Embeddings come from the pass with the given order.

Images are read from the manifest at original resolution and resized here (aspect kept, sides multiples of 28) to
vlmfeat.max_pixels, SkinGPT-R1's own evaluation bound, independently of the run's prepare settings: the benchmark runs
alongside the Sonnet answers, whose S1 images are prepared for Claude.

Outputs: embeddings.npz (one [n_images x hidden] float32 array per embedding, plus image_ids), diagnosis.parquet
(log-probabilities per candidate), images.parquet (size, image tokens, timings), shards/ (restartable).
"""
import io
import time

import numpy as np
import pandas as pd
from PIL import Image

from ..config import ROOT
from .s1_prepare import budget_size

UPSTREAM = ["manifest"]
VERSION = 1
PROMPTS = ROOT / "inputs" / "prompts"


def prompt_text(cfg) -> str:
    return (PROMPTS / f"{cfg.vlmfeat.prompt}.txt").read_text()


def key(cfg, ctx) -> dict:
    v = cfg.vlmfeat
    return {"vlmfeat": {k: v[k] for k in v if k not in ("device", "shard_images")},
            "revision": revision(v), "prompt": prompt_text(cfg)}


def revision(v) -> str:
    from huggingface_hub import snapshot_download

    kw = {"allow_patterns": [f"{v.subfolder}/*"], "ignore_patterns": ["*.ipynb_checkpoints*"]} if v.subfolder else {}
    try:
        path = snapshot_download(v.checkpoint, revision=v.revision, local_files_only=True, **kw)
    except Exception:
        path = snapshot_download(v.checkpoint, revision=v.revision, **kw)
    from pathlib import Path

    return Path(path).name


def labels(v) -> list[str]:
    return [c.label for c in v.candidates]


def run(cfg, out, inputs, ctx) -> dict:
    import torch
    from omegaconf import OmegaConf

    from ..answerers.qwen_vl import QwenVL

    v = cfg.vlmfeat
    man = pd.read_parquet(inputs["manifest"] / "manifest.parquet")
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    acfg = OmegaConf.merge(cfg.answerer, {k: v[k] for k in ("checkpoint", "subfolder", "revision", "dtype", "attn", "device")})
    model = QwenVL(acfg, revision(v), v.max_pixels, v.min_pixels)
    cands = labels(v)
    def full_prompt(cs):
        user = prompt_text(cfg).replace("{candidate_labels}", "\n".join(cs)).rstrip("\n")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        return model.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + v.answer_prefix

    orders = {o: (cands if o == "forward" else cands[::-1]) for o in v.orders}
    texts = {o: full_prompt(cs) for o, cs in orders.items()}
    seqs = {o: [model.encode(c + v.label_suffix) for c in cs] for o, cs in orders.items()}
    (out / "prompt_example.txt").write_text(texts[v.orders[0]])
    layers = list(v.layers)

    n = len(man)
    for s0 in range(0, n, v.shard_images):
        path = shard_dir / f"{s0:05d}.npz"
        if path.exists():
            continue
        embs, diag, info = {}, [], []
        for rec in man.iloc[s0 : s0 + v.shard_images].itertuples():
            im = Image.open(ROOT / rec.path).convert("RGB")
            w, h = budget_size(*im.size, v.max_pixels, v.min_pixels)
            im = im.resize((w, h), Image.BICUBIC)
            t = time.perf_counter()
            lp = {}
            for k, o in enumerate(v.orders):
                prefix, last, hid, n_tok = model.prefill_hidden(im, texts[o], layers if k == 0 else [])
                if k == 0:
                    hs, n_img, t_pre = hid, n_tok, time.perf_counter() - t
                sc = dict(zip(orders[o], model.sequence_logprobs(prefix, last, seqs[o])))
                lp[o] = [sc[c] for c in cands]                     # always stored in the given order
                del prefix
            for l, (m_img, m_last) in hs.items():
                embs.setdefault(f"img_L{l}", []).append(m_img.numpy())
                embs.setdefault(f"last_L{l}", []).append(m_last.numpy())
            diag.append([lp[o] for o in v.orders])
            info.append({"image_id": rec.image_id, "width": w, "height": h, "image_tokens": n_img, "prefill_s": t_pre, "total_s": time.perf_counter() - t,
                         **{f"top_label_{o}": cands[int(np.argmax(lp[o]))] for o in v.orders}})
        np.savez(path.with_suffix(".tmp.npz"), **{k: np.stack(a) for k, a in embs.items()},
                 diag=np.array(diag), info=np.array(pd.DataFrame(info).to_json(orient="records"), dtype=object))
        path.with_suffix(".tmp.npz").rename(path)
        print(f"  images {s0 + 1}-{min(n, s0 + v.shard_images)} of {n}: "
              f"{np.mean([i['total_s'] for i in info]):.2f} s per image", flush=True)
        torch.cuda.empty_cache()

    parts = [np.load(p, allow_pickle=True) for p in sorted(shard_dir.glob("[0-9]*.npz"))]
    names = [k for k in parts[0].files if k.startswith(("img_L", "last_L"))]
    info = pd.concat([pd.read_json(io.StringIO(str(p["info"])), orient="records") for p in parts], ignore_index=True)
    ids = info.image_id.astype(str).values
    if list(ids) != list(man.image_id):
        raise RuntimeError("feature shards are out of order or incomplete")
    E = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in names}
    np.savez(out / "embeddings.npz", image_ids=ids, **E)
    Do = np.concatenate([p["diag"] for p in parts])                  # [images x orders x candidates]
    for k, o in enumerate(v.orders):
        d = pd.DataFrame(Do[:, k], columns=cands)
        d.insert(0, "image_id", ids)
        d.to_parquet(out / f"diagnosis_{o}.parquet", index=False)
    D = Do.mean(1)
    diag = pd.DataFrame(D, columns=cands)
    diag.insert(0, "image_id", ids)
    diag.to_parquet(out / "diagnosis.parquet", index=False)
    first = Do.argmax(2)
    info.to_parquet(out / "images.parquet", index=False)
    P = np.exp(D - D.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return {"n_images": n, "embeddings": {k: list(a.shape) for k, a in E.items()},
            "sec_per_image": round(float(info.total_s.mean()), 3), "image_tokens_mean": float(info.image_tokens.mean()),
            "candidate_mass_mean": float(np.exp(np.logaddexp.reduce(D, axis=1)).mean()),
            "top_label_counts": {o: info[f"top_label_{o}"].value_counts().to_dict() for o in v.orders},
            "orders_agree_on_first_choice": float((first == first[:, :1]).all(1).mean()),
            "key_metric": f"{n} images, {info.total_s.mean():.2f} s/image, mean max candidate prob {P.max(1).mean():.2f}"}
