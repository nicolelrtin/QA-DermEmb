"""S2 Answer (+ S3 timing): P(yes) for every (unit, question).

Each item is a (Question ID, original) pair: originals and their negated complements are asked
independently and never share a prompt.
Modes: stateless (one sequence per item, image-prefix KV reused, `compute_batch` sequences per forward
pass), topic (one prompt per original x topic) and all (one prompt per original value). The batched
modes force a JSON skeleton and read P(yes) at each answer slot; `shuffle` permutes order within a prompt.
Stateless results are written in shards of ~shard_questions items, batched results one shard per prompt;
a restart skips finished shards.
Outputs: answers_raw.parquet (no threshold applied; S4 adds answer_bin) and timing.parquet.
"""
import hashlib
import random
import zlib
from pathlib import Path

import pandas as pd
from PIL import Image

from ..config import ROOT, stable_hash

UPSTREAM = ["prepare"]
VERSION = 2
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def prompt_text(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def prompt_hash(name: str) -> str:
    return hashlib.sha256(prompt_text(name).encode()).hexdigest()[:12]


def used_prompts(cfg) -> dict[str, str]:
    a = cfg.answer
    if a.mode not in ("stateless", "topic", "all"):
        raise SystemExit(f"answer.mode must be stateless, topic or all; got {a.mode!r}")
    names = {"whole": a.prompts.whole, "patch": a.prompts.patch} if a.mode == "stateless" else \
            {"whole": a.prompts.batched_whole, "patch": a.prompts.batched_patch}
    if cfg.grid.G == 1:
        names.pop("patch")
    return names


def key(cfg, ctx) -> dict:
    a = cfg.answer
    acfg = {k: v for k, v in cfg.answerer.items() if k != "device"}
    return {"qset_hash": ctx.qset_hash, "answerer": {**acfg, "revision": ctx.answerer_revision},
            "mode": a.mode, "shuffle": a.shuffle,
            "shuffle_seed": cfg.seed if a.shuffle else None,
            "prompts": {k: prompt_hash(v) for k, v in used_prompts(cfg).items()}}


def plan_unit(cfg, unit_id: str, qs: pd.DataFrame) -> list[list[list[str]]]:
    """Shards -> batches -> item ids. Stateless batches are single items; batched shards hold one prompt."""
    a = cfg.answer
    items = list(qs.item_id)
    if a.mode == "stateless":
        return [[[q] for q in items[i : i + a.shard_questions]] for i in range(0, len(items), a.shard_questions)]
    keys = ["original", "topic"] if a.mode == "topic" else ["original"]
    if "topic" in keys and "topic" not in qs:
        raise SystemExit("answer.mode=topic needs a Topic column in the question CSV")
    batches = [list(g.item_id) for _, g in qs.groupby(keys, sort=False)]
    if a.shuffle:
        rng = random.Random(cfg.seed * 1_000_003 + zlib.crc32(unit_id.encode()))
        for b in batches:
            rng.shuffle(b)
    return [[b] for b in batches]


def run(cfg, out, inputs, ctx) -> dict:
    from ..answerers.qwen_vl import SENTINEL, QwenVL
    from ..run import gpu_info

    a = cfg.answer
    units = pd.read_parquet(inputs["prepare"] / "units.parquet")
    qs = ctx.questions
    qtext = dict(zip(qs.item_id, qs.question))
    qkey = {i: (q, o) for i, q, o in zip(qs.item_id, qs.question_id, qs.original)}
    items = list(qs.item_id)
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    prompts = used_prompts(cfg)
    phash = {k: prompt_hash(v) for k, v in prompts.items()}
    gpu = gpu_info()["gpu_name"]
    qset_h = ctx.qset_hash

    answerer = None
    verify = {}
    for n_unit, u in enumerate(units.itertuples()):
        plan = plan_unit(cfg, u.unit_id, qs)
        pending = [j for j in range(len(plan)) if not (shard_dir / f"{u.unit_id}__{j:03d}.ans.parquet").exists()]
        if not pending:
            continue
        if answerer is None:
            answerer = QwenVL(cfg.answerer, ctx.answerer_revision, max(cfg.prepare.max_pixels, cfg.prepare.patch_max_pixels),
                              cfg.prepare.min_pixels)
        print(f"  unit {n_unit + 1}/{len(units)} {u.unit_id}: {len(pending)} shard(s)")
        kind = "whole" if u.patch_id < 0 else "patch"
        tmpl = prompt_text(prompts[kind])
        var = "{question}" if a.mode == "stateless" else "{questions}"
        pre, post = answerer.split_template(tmpl.replace(var, SENTINEL))
        image = Image.open(inputs["prepare"] / u.path).convert("RGB")
        prefix = answerer.prefill(image, pre)
        base = {"run_id": ctx.run_id, "image_id": u.image_id, "patch_id": u.patch_id, "unit_id": u.unit_id}
        trow = {**base, "mode": a.mode, "gpu_name": gpu, "dtype": cfg.answerer.dtype,
                "compute_batch": a.compute_batch if a.mode == "stateless" else 1}
        first_shard = True

        if not verify and a.mode == "stateless":
            q0 = items[0]
            cached = answerer.answer_stateless(prefix, post, [qtext[q0]])[0][0]
            verify = {"item_id": q0, "cached": cached, "reference": answerer.reference_p_yes(image, pre + qtext[q0] + post)}
            verify["abs_diff"] = abs(verify["cached"] - verify["reference"])
            print(f"  prefix-cache check {q0}: cached={cached:.4f} ref={verify['reference']:.4f}")

        for j in pending:
            ans, tim = [], []
            if first_shard:
                tim.append({**trow, "question_id": "__image_prefix__", "original": -1, "batch_id": -1, "batch_k": 0, "wall_ms": prefix.ms,
                            "prefill_ms": prefix.ms, "decode_ms": 0.0, "n_input_tokens": prefix.length,
                            "n_output_tokens": 0, "kv_cache_hit": False})
                first_shard = False
            batches = plan[j]
            if a.mode == "stateless":
                flat = [b[0] for b in batches]
                for i in range(0, len(flat), a.compute_batch):
                    chunk = flat[i : i + a.compute_batch]
                    p, m, ntok, ms = answerer.answer_stateless(prefix, post, [qtext[q] for q in chunk])
                    for q, pi, mi, ni in zip(chunk, p, m, ntok):
                        qid, orig = qkey[q]
                        ans.append({**base, "question_id": qid, "original": orig, "p_yes": pi, "yesno_mass": mi,
                                    "batch_id": -1, "batch_k": 1, "position_in_batch": 0})
                        tim.append({**trow, "question_id": qid, "original": orig, "batch_id": -1, "batch_k": 1, "wall_ms": ms / len(chunk),
                                    "prefill_ms": ms / len(chunk), "decode_ms": 0.0, "n_input_tokens": ni,
                                    "n_output_tokens": 1, "kv_cache_hit": True})
            else:
                for b_local, batch in enumerate(batches):
                    batch_id = sum(len(s) for s in plan[:j]) + b_local
                    block = "\n".join(f"{i + 1}. {qtext[q]}" for i, q in enumerate(batch))
                    p, m, n_in, n_out, pre_ms, dec_ms = answerer.answer_batched(prefix, block + post, len(batch))
                    for pos, (q, pi, mi) in enumerate(zip(batch, p, m)):
                        qid, orig = qkey[q]
                        ans.append({**base, "question_id": qid, "original": orig, "p_yes": pi, "yesno_mass": mi,
                                    "batch_id": batch_id, "batch_k": len(batch), "position_in_batch": pos})
                    tim.append({**trow, "question_id": f"batch:{batch_id}", "original": qkey[batch[0]][1], "batch_id": batch_id,
                                "batch_k": len(batch), "wall_ms": pre_ms + dec_ms,
                                "prefill_ms": pre_ms, "decode_ms": dec_ms, "n_input_tokens": n_in,
                                "n_output_tokens": n_out, "kv_cache_hit": True})
            stem = shard_dir / f"{u.unit_id}__{j:03d}"
            pd.DataFrame(tim).to_parquet(f"{stem}.time.parquet", index=False)
            pd.DataFrame(ans).to_parquet(f"{stem}.tmp", index=False)
            Path(f"{stem}.tmp").rename(f"{stem}.ans.parquet")  # answers file marks the shard done

    answers = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*.ans.parquet"))], ignore_index=True)
    timing = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*.time.parquet"))], ignore_index=True)
    meta = units.set_index("unit_id")[["grid_G", "bbox"]]
    answers = answers.join(meta, on="unit_id")
    answers["qset_hash"] = qset_h
    answers["mode"] = a.mode
    answers["prompt_hash"] = answers.patch_id.map(lambda p: phash["whole" if p < 0 else "patch"])
    answers["answerer_id"] = f"{cfg.answerer.id}@{ctx.answerer_revision[:12]}"
    answers.to_parquet(out / "answers_raw.parquet", index=False)
    timing.to_parquet(out / "timing.parquet", index=False)

    n = len(answers)
    expected = len(units) * len(items)
    if n != expected:
        raise RuntimeError(f"answers has {n} rows, expected {expected}")
    total_s = timing.wall_ms.sum() / 1e3
    spq = total_s / n
    return {
        "n_answers": n, "n_units": len(units), "n_items": len(items), "n_question_ids": int(qs.question_id.nunique()),
        "total_answer_s": round(total_s, 2), "sec_per_query": round(spq, 5),
        "prefix_ms_mean": round(timing[timing.question_id == "__image_prefix__"].wall_ms.mean(), 1),
        "yesno_mass_min": float(answers.yesno_mass.min()), "yesno_mass_mean": float(answers.yesno_mass.mean()),
        "p_yes_mean": float(answers.p_yes.mean()), "frac_yes": float((answers.p_yes >= a.threshold).mean()),
        "prefix_cache_check": verify,
        "prompts": {k: {"name": v, "hash": phash[k], "text": prompt_text(v)} for k, v in prompts.items()},
        "key_metric": f"s/q={spq:.4f}",
    }
