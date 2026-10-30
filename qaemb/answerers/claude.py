"""Claude answerer: stated P(yes) and confidence, returned as forced JSON (structured outputs).

The Messages API returns no token probabilities, so P(yes) here is the model's stated probability, not a
logit readout as for Qwen. Every prompt is an independent request (stateless by construction). Images are
uploaded once with the Files API and referenced by file_id. With claude.use_batch, all prompts of a stage
run go into one Message Batch whose id is saved, so a restart polls the same batch instead of resubmitting.
A request that does not return a complete, numeric answer for every question is resent (claude.max_attempts);
if any is still incomplete the stage stops, so missing answers never reach the embedding.
Results are written in the same shard format as the Qwen path (one shard per unit).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

NUM = {"type": "number"}
SCHEMA_ONE = {"type": "object", "properties": {"p_yes": NUM, "confidence": NUM},
              "required": ["p_yes", "confidence"], "additionalProperties": False}
SCHEMA_MANY = {"type": "object", "properties": {"answers": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "p_yes": NUM, "confidence": NUM},
    "required": ["id", "p_yes", "confidence"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False}


def upload_images(client, units: pd.DataFrame, prepare_dir: Path, cache: Path) -> dict[str, str]:
    ids = json.loads(cache.read_text()) if cache.exists() else {}
    for u in units.itertuples():
        if u.unit_id not in ids:
            ids[u.unit_id] = client.files.upload(file=prepare_dir / u.path).id
            cache.write_text(json.dumps(ids, indent=1))
    return ids


def build_params(cfg, file_id: str, text: str) -> dict:
    mode = cfg.answer.mode
    return {
        "model": cfg.claude.model,
        "max_tokens": int(cfg.claude.max_tokens[mode]),
        "output_config": {"effort": cfg.claude.effort,
                          "format": {"type": "json_schema", "schema": SCHEMA_ONE if mode == "stateless" else SCHEMA_MANY}},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "file", "file_id": file_id}},
            {"type": "text", "text": text}]}],
    }


def _clip(v):
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return float("nan")


def parse(msg, n_items: int) -> tuple[list[tuple[float, float]], str]:
    """[(p_yes, confidence)] per item in prompt order, and a status string."""
    nan = [(float("nan"), float("nan"))] * n_items
    if msg.stop_reason == "refusal":
        return nan, "refusal"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return nan, "parse_error" if msg.stop_reason != "max_tokens" else "max_tokens"
    return parse_data(data, n_items)


def parse_data(data: dict, n_items: int) -> tuple[list[tuple[float, float]], str]:
    nan = [(float("nan"), float("nan"))] * n_items
    if "answers" not in data:
        out = [(_clip(data.get("p_yes")), _clip(data.get("confidence")))]
    else:
        by_id = {a.get("id"): a for a in data["answers"]}
        if any(i not in by_id for i in range(1, n_items + 1)):
            return nan, f"missing_{sum(i not in by_id for i in range(1, n_items + 1))}"
        out = [(_clip(by_id[i]["p_yes"]), _clip(by_id[i]["confidence"])) for i in range(1, n_items + 1)]
    return (out, "ok") if all(v == v for pair in out for v in pair) else (nan, "non_numeric")


def usage_tokens(msg) -> dict:
    u = msg.usage
    cr, cw = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
    return {"n_input_tokens": u.input_tokens + cr + cw, "n_output_tokens": u.output_tokens,
            "n_cache_read_tokens": cr, "n_cache_write_tokens": cw, "kv_cache_hit": cr > 0}


TOK0 = {"n_input_tokens": 0, "n_output_tokens": 0, "n_cache_read_tokens": 0, "n_cache_write_tokens": 0, "kv_cache_hit": False}


def _submit(client, c, reqs, run_dir: Path, attempt: int) -> dict[str, tuple]:
    """One round of requests. Returns custom_id -> (message or None, error, wall_ms)."""
    import anthropic

    out = {}
    if c.use_batch:
        id_file = run_dir / f"batch_id_{attempt}.txt"  # saved so a restart polls the same batch
        if not id_file.exists():
            batch = client.messages.batches.create(requests=[{"custom_id": r[0], "params": r[4]} for r in reqs])
            id_file.write_text(batch.id)
        batch_id = id_file.read_text().strip()
        while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            rc = b.request_counts
            print(f"  batch {batch_id}: {b.processing_status}, processing {rc.processing}, succeeded {rc.succeeded}, errored {rc.errored}")
            time.sleep(c.poll_seconds)
        for r in client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                out[r.custom_id] = (r.result.message, None, float("nan"))
            else:
                err = getattr(getattr(r.result, "error", None), "type", "") if r.result.type == "errored" else ""
                out[r.custom_id] = (None, f"{r.result.type}{':' + err if err else ''}", float("nan"))
        return out

    def call(req):
        t = time.perf_counter()
        try:
            return req[0], client.messages.create(**req[4]), None, (time.perf_counter() - t) * 1e3
        except anthropic.APIStatusError as e:
            return req[0], None, f"api_error:{e.status_code}", (time.perf_counter() - t) * 1e3
    with ThreadPoolExecutor(c.concurrency) as pool:
        for cid, msg, err, ms in pool.map(call, reqs):
            out[cid] = (msg, err, ms)
    return out


def run_with_retries(client, c, reqs, run_dir: Path) -> dict[str, tuple]:
    """Resend every request that did not return a complete answer (API error, refusal, truncated or unparseable
    JSON, missing questions) up to claude.max_attempts times. Raises if any request is still incomplete, so the
    stage never completes with missing answers. Tokens of failed attempts are added to the request's total.
    Returns custom_id -> (message, [(p_yes, confidence)], attempts, wall_ms, tokens)."""
    done, spent, failures = {}, {}, {}
    todo = list(reqs)
    for attempt in range(1, int(c.max_attempts) + 1):
        res = _submit(client, c, todo, run_dir, attempt)
        failures = {}
        for r in todo:
            cid, n = r[0], len(r[3])
            msg, err, ms = res.get(cid, (None, "no_result", float("nan")))
            tok = usage_tokens(msg) if msg is not None else dict(TOK0)
            prev = spent.get(cid, dict(TOK0))
            spent[cid] = {k: (prev[k] + v if k != "kv_cache_hit" else v) for k, v in tok.items()}
            vals, why = parse(msg, n) if msg is not None else (None, err)
            if why == "ok":
                done[cid] = (msg, vals, attempt, ms, spent[cid])
            else:
                failures[cid] = why
        todo = [r for r in todo if r[0] in failures]
        if not todo:
            return done
        print(f"  attempt {attempt}: {len(todo)} incomplete requests ({sorted(set(failures.values()))}); resending")
    raise RuntimeError(f"{len(failures)} Claude requests still incomplete after {c.max_attempts} attempts: "
                       f"{dict(list(failures.items())[:10])}. No answers were written for these units.")


def answer_units(cfg, ctx, inputs, units, qs, plan_unit, template: str, shard_dir: Path) -> None:
    import anthropic

    c = cfg.claude
    if (units.patch_id >= 0).any():
        raise SystemExit("The Claude answerer supports whole images only (grid G1)")
    done = {p.name.split("__")[0] for p in shard_dir.glob("*.ans.parquet")}
    todo = units[~units.unit_id.isin(done)]
    if todo.empty:
        return
    client = anthropic.Anthropic(max_retries=8)  # SDK backs off on 429/5xx
    files = upload_images(client, todo, inputs["prepare"], shard_dir.parent / "files.json")
    reqs = [(f"u{n}_b{b_id}", u, b_id, items, build_params(cfg, files[u.unit_id], text))
            for n, (u, b_id, items, text) in enumerate(build_requests(cfg, todo, qs, plan_unit, template))]
    print(f"  {len(reqs)} Claude requests ({c.model}, effort {c.effort}, {'batch' if c.use_batch else 'direct'})")

    results = run_with_retries(client, c, reqs, shard_dir.parent)
    write_shards(cfg, ctx, qs, [(u, b_id, items, results[cid][1:]) for cid, u, b_id, items, _ in reqs], shard_dir)


def build_requests(cfg, units, qs, plan_unit, template: str) -> list[tuple]:
    """(unit row, batch_id, item ids, prompt text) for every prompt; batch_id is -1 in stateless mode."""
    qtext = dict(zip(qs.item_id, qs.question))
    out = []
    for u in units.itertuples():
        for b_id, items in enumerate(b for shard in plan_unit(cfg, u.unit_id, qs) for b in shard):
            if cfg.answer.mode == "stateless":
                out.append((u, -1, items, template.replace("{question}", qtext[items[0]])))
            else:
                out.append((u, b_id, items, template.replace(
                    "{questions}", "\n".join(f"{i + 1}. {qtext[q]}" for i, q in enumerate(items)))))
    return out


def write_shards(cfg, ctx, qs, done: list[tuple], shard_dir: Path) -> None:
    """done: (unit row, batch_id, item ids, (values, attempts, wall_ms, tokens)) for every prompt of each unit."""
    mode = cfg.answer.mode
    qkey = {i: (q, o) for i, q, o in zip(qs.item_id, qs.question_id, qs.original)}
    nan = float("nan")
    per_unit: dict[str, tuple[list, list]] = {}
    for u, b_id, items, (vals, attempts, ms, tok) in done:
        ans, tim = per_unit.setdefault(u.unit_id, ([], []))
        base = {"run_id": ctx.run_id, "image_id": u.image_id, "patch_id": u.patch_id, "unit_id": u.unit_id}
        for pos, (q, (p, conf)) in enumerate(zip(items, vals)):
            qid, orig = qkey[q]
            ans.append({**base, "question_id": qid, "original": orig, "p_yes": p, "confidence": conf, "yesno_mass": nan,
                        "batch_id": b_id, "batch_k": len(items), "position_in_batch": pos})
        qid0, orig0 = qkey[items[0]]
        tim.append({**base, "question_id": qid0 if mode == "stateless" else f"batch:{b_id}", "original": orig0,
                    "batch_id": b_id, "batch_k": len(items), "mode": mode, "gpu_name": None, "dtype": None,
                    "compute_batch": 1, "wall_ms": ms, "prefill_ms": nan, "decode_ms": nan, "attempts": attempts, **tok})
    for unit_id, (ans, tim) in per_unit.items():
        stem = shard_dir / f"{unit_id}__000"
        pd.DataFrame(tim).to_parquet(f"{stem}.time.parquet", index=False)
        pd.DataFrame(ans).to_parquet(f"{stem}.tmp", index=False)
        Path(f"{stem}.tmp").rename(f"{stem}.ans.parquet")
