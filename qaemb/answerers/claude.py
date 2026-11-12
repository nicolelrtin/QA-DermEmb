"""Claude answerer: stated P(yes) and confidence, returned as forced JSON (structured outputs).

The Messages API returns no token probabilities, so P(yes) here is the model's stated probability, not a
logit readout as for Qwen. Every prompt is an independent request (stateless by construction). Images are
uploaded once with the Files API and referenced by file_id. With claude.use_batch, all prompts of a stage
run go into one Message Batch whose id is saved, so a restart polls the same batch instead of resubmitting.
A request that does not return a complete, numeric answer for every question is resent (claude.max_attempts);
if any is still incomplete the stage stops, so missing answers never reach the embedding.
Results are written in the same shard format as the Qwen path (one shard per unit).

Spend: every billed response is priced from its usage and appended to a ledger shared by all runs (claude.budget).
Before each submission (and each resend round) the cost is projected; a submission that would take the total past
checkpoint_usd stops until claude.budget.approved_usd covers it, and none may pass limit_usd.
"""
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import ROOT

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


class Budget:
    """Running Claude API spend, one ledger row per billed response, kept across runs and deduplicated by message id."""

    COLS = ["time", "run_id", "message_id", "model", "batch", "n_questions", "input_tokens", "cache_read_tokens",
            "cache_write_tokens", "output_tokens", "usd"]

    def __init__(self, cfg, run_id: str):
        self.b, self.prices, self.run_id, self.model = cfg.claude.budget, cfg.claude.prices, run_id, cfg.claude.model
        self.path = ROOT / self.b.ledger
        self.df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=self.COLS)
        self.seen = set(self.df.message_id)
        self.lock = threading.Lock()

    def _rate(self, model: str, batch: bool):
        if model not in self.prices:
            raise SystemExit(f"No price for {model} in claude.prices; add it before running")
        return self.prices[model], (float(self.prices.batch_factor) if batch else 1.0)

    def usd(self, model: str, batch: bool, inp: float, out: float, cr: float = 0, cw: float = 0) -> float:
        p, f = self._rate(model, batch)
        return f * (inp * p.input + cr * p.cache_read + cw * p.cache_write + out * p.output) / 1e6

    @property
    def spent(self) -> float:
        return float(self.df.usd.sum()) if len(self.df) else 0.0

    def record(self, msg, batch: bool, n_questions: int) -> None:
        if msg is None or msg.id in self.seen:
            return
        u = msg.usage
        cr, cw = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
        row = {"time": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), "run_id": self.run_id, "message_id": msg.id,
               "model": self.model, "batch": batch, "n_questions": n_questions, "input_tokens": u.input_tokens,
               "cache_read_tokens": cr, "cache_write_tokens": cw, "output_tokens": u.output_tokens,
               "usd": self.usd(self.model, batch, u.input_tokens, u.output_tokens, cr, cw)}
        with self.lock:
            self.seen.add(msg.id)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row], columns=self.COLS).to_csv(self.path, mode="a", index=False, header=not self.path.exists())
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True) if len(self.df) else pd.DataFrame([row])

    def ceiling(self) -> float:
        return min(float(self.b.limit_usd), max(float(self.b.checkpoint_usd), float(self.b.approved_usd)))

    def estimate(self, client, model: str, reqs: list, batch: bool) -> tuple[float, float, float]:
        """(usd, input tokens, output tokens) for reqs. Input: count_tokens (free) on a random sample of up to 40
        requests, scaled to all. Output: a + b per question from the config, rescaled by what this model actually
        produced so far in the ledger."""
        # count_tokens takes no Files API images, so the text is counted there and the image estimated from its
        # pixels (about pixels / 750, at most ~4,784; measured with count_tokens on 2026-11-11).
        sample = random.Random(len(reqs)).sample(reqs, min(40, len(reqs)))
        text = []
        for r in sample:
            body = {k: v for k, v in r[4].items() if k in ("model", "messages", "output_config", "system")}
            body["messages"] = [{**m, "content": [b for b in m["content"] if b.get("type") != "image"]}
                                for m in body["messages"]]
            text.append(client.messages.count_tokens(**body).input_tokens)
        image = sum(min(r[1].width * r[1].height / 750, 4784) for r in reqs)
        inp = sum(text) / len(text) * len(reqs) + image
        a, b = float(self.b.output_tokens_per_prompt), float(self.b.output_tokens_per_question)
        prior = self.df[self.df.model == model] if len(self.df) else self.df
        scale = 1.0
        if len(prior) >= 20:
            scale = prior.output_tokens.sum() / (a * len(prior) + b * prior.n_questions.sum())
        out = scale * sum(a + b * len(r[3]) for r in reqs)
        return self.usd(model, batch, inp, out), inp, out

    def check(self, client, model: str, reqs: list, batch: bool, label: str) -> None:
        est, inp, out = self.estimate(client, model, reqs, batch)
        est *= 1.1  # margin for the sampling error of the input count and resends
        total = self.spent + est
        msg = (f"Claude API spend so far ${self.spent:.2f}; {label} ({len(reqs)} requests, ~{inp / 1e6:.2f} M input and "
               f"~{out / 1e6:.2f} M output tokens{', batch price' if batch else ''}) is estimated at ${est:.2f} "
               f"including a 10% margin; total ${total:.2f}")
        print(f"  {msg}")
        if total > float(self.b.limit_usd):
            raise SystemExit(f"{msg}. That passes the ${self.b.limit_usd} limit (claude.budget.limit_usd); nothing was submitted.")
        if total > self.ceiling():
            raise SystemExit(f"{msg}. That passes the ${self.b.checkpoint_usd} checkpoint; nothing was submitted. To go on, "
                             f"rerun with claude.budget.approved_usd={math.ceil(total)} (at most {self.b.limit_usd}).")


def _submit(client, c, reqs, run_dir: Path, attempt: int, budget: "Budget") -> dict[str, tuple]:
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
        nq = {r[0]: len(r[3]) for r in reqs}
        for r in client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                budget.record(r.result.message, True, nq.get(r.custom_id, 0))
                out[r.custom_id] = (r.result.message, None, float("nan"))
            else:
                err = getattr(getattr(r.result, "error", None), "type", "") if r.result.type == "errored" else ""
                out[r.custom_id] = (None, f"{r.result.type}{':' + err if err else ''}", float("nan"))
        return out

    def call(req):
        if budget.spent > budget.ceiling():  # the projection was too low: stop sending
            return req[0], None, "budget_stop", float("nan")
        t = time.perf_counter()
        try:
            msg = client.messages.create(**req[4])
            budget.record(msg, False, len(req[3]))
            return req[0], msg, None, (time.perf_counter() - t) * 1e3
        except anthropic.APIStatusError as e:
            return req[0], None, f"api_error:{e.status_code}", (time.perf_counter() - t) * 1e3
    with ThreadPoolExecutor(c.concurrency) as pool:
        for cid, msg, err, ms in pool.map(call, reqs):
            out[cid] = (msg, err, ms)
    return out


def run_with_retries(client, c, reqs, run_dir: Path, budget: "Budget") -> dict[str, tuple]:
    """Resend every request that did not return a complete answer (API error, refusal, truncated or unparseable
    JSON, missing questions) up to claude.max_attempts times. Raises if any request is still incomplete, so the
    stage never completes with missing answers. Tokens of failed attempts are added to the request's total.
    Returns custom_id -> (message, [(p_yes, confidence)], attempts, wall_ms, tokens)."""
    done, spent, failures = {}, {}, {}
    todo = list(reqs)
    for attempt in range(1, int(c.max_attempts) + 1):
        if not (c.use_batch and (run_dir / f"batch_id_{attempt}.txt").exists()):  # an already submitted batch is paid for
            budget.check(client, c.model, todo, bool(c.use_batch), "this submission" if attempt == 1 else f"resend round {attempt}")
        res = _submit(client, c, todo, run_dir, attempt, budget)
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

    results = run_with_retries(client, c, reqs, shard_dir.parent, Budget(cfg, ctx.run_id))
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
