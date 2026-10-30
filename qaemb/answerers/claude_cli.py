"""Claude through the Claude Code CLI (`claude -p`), on the user's Claude subscription instead of an API key.

Same prompts, JSON schemas and shard format as answerers/claude.py. Each prompt is one headless call:
the image goes in base64 through --input-format stream-json, the answer comes back forced to the schema
(--json-schema), with tools off, a fixed minimal system prompt, no settings or memory loaded and an empty
working directory, so every call sees the same context.

Subscriptions have 5-hour and weekly usage windows. Every call reports how much of each is used; when
either passes claude_cli.stop_at, or a call is rate-limited, no new calls start, finished units are written
and the process exits with code 75 (PAUSED), recording the reset time in outputs/qaemb/queue/paused_until.
Each finished prompt is appended to cli_results.jsonl in the stage dir, so the next run (e.g. from cron,
see qaemb/queue.py) continues where this one stopped. A prompt without a complete, numeric answer is retried
up to claude.max_attempts times; after that the stage fails rather than keep missing answers. Whatever an
incomplete attempt did return is kept in cli_partial.jsonl for review (reported separately, never embedded).
"""
import base64
import json
import mimetypes
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..config import ROOT
from .claude import SCHEMA_MANY, SCHEMA_ONE, TOK0, build_requests, parse_data, write_shards

PAUSED = 75
PAUSE_FILE = ROOT / "outputs" / "qaemb" / "queue" / "paused_until"


def schema(cfg, n_items: int) -> dict:
    """Batched prompts must return exactly one answer per question: without a fixed length Claude often stopped
    after ~150 of 500 (pilot 2026-10-30). The CLI enforces minItems/maxItems on the structured output."""
    if cfg.answer.mode == "stateless":
        return SCHEMA_ONE
    arr = {**SCHEMA_MANY["properties"]["answers"], "minItems": n_items, "maxItems": n_items}
    return {**SCHEMA_MANY, "properties": {"answers": arr}}


def command(cfg, n_items: int) -> list[str]:
    c, cli = cfg.claude, cfg.claude_cli
    return [cli.binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", c.model, "--effort", c.effort, "--tools", "", "--system-prompt", cli.system_prompt,
            "--setting-sources", "", "--no-session-persistence", "--json-schema", json.dumps(schema(cfg, n_items))]


def call(cfg, cwd: Path, image: dict, text: str, n_items: int) -> dict:
    """One headless call. Returns {data, error, rate_limited, rate, tokens, usd, wall_ms}."""
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_API_KEY")}
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(cfg.claude.max_tokens[cfg.answer.mode])
    msg = {"type": "user", "message": {"role": "user", "content": [image, {"type": "text", "text": text}]}}
    t = time.perf_counter()
    try:
        p = subprocess.run(command(cfg, n_items), input=json.dumps(msg) + "\n", capture_output=True, text=True, cwd=cwd, env=env,
                           timeout=cfg.claude_cli.timeout_s)
        lines = p.stdout.splitlines()
    except subprocess.TimeoutExpired:
        lines = []
    out = {"data": None, "error": "timeout", "rate_limited": False, "rate": None, "tokens": dict(TOK0), "usd": 0.0,
           "wall_ms": (time.perf_counter() - t) * 1e3}
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "rate_limit_event":
            out["rate"] = e["rate_limit_info"]
        elif e.get("type") == "result":
            u = e.get("usage") or {}
            cr, cw = u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0)
            out["tokens"] = {"n_input_tokens": u.get("input_tokens", 0) + cr + cw, "n_output_tokens": u.get("output_tokens", 0),
                             "n_cache_read_tokens": cr, "n_cache_write_tokens": cw, "kv_cache_hit": cr > 0}
            out["usd"] = e.get("total_cost_usd") or 0.0
            out["data"] = e.get("structured_output")
            out["error"] = None if out["data"] is not None and not e.get("is_error") else \
                f"{e.get('subtype')}:{e.get('api_error_status')}:{str(e.get('result'))[:200]}"
            out["rate_limited"] = e.get("api_error_status") == 429
    if out["rate"] and out["rate"].get("status") not in (None, "allowed", "allowed_warning"):
        out["rate_limited"] = True
    return out


def over_budget(rate: dict | None, stop_at) -> tuple[bool, int | None]:
    """(stop?, unix time when the window that triggered it resets)."""
    if not rate:
        return False, None
    for name, w in (rate.get("unifiedWindows") or {}).items():
        if name in stop_at and (w.get("utilization") or 0) >= stop_at[name]:
            return True, w.get("resetsAt")
    return False, None


def answer_units(cfg, ctx, inputs, units, qs, plan_unit, template: str, shard_dir: Path) -> None:
    cli = cfg.claude_cli
    if (units.patch_id >= 0).any():
        raise SystemExit("The Claude answerer supports whole images only (grid G1)")
    done_units = {p.name.split("__")[0] for p in shard_dir.glob("*.ans.parquet")}
    todo = units[~units.unit_id.isin(done_units)]
    if todo.empty:
        return
    stage_dir = shard_dir.parent
    cwd = stage_dir / "cli_cwd"  # empty: nothing from the repo (CLAUDE.md, git status) reaches the prompt
    cwd.mkdir(exist_ok=True)
    cache_file, fail_file = stage_dir / "cli_results.jsonl", stage_dir / "cli_failures.json"
    partial_file = stage_dir / "cli_partial.jsonl"  # answers from incomplete attempts: kept for review, never embedded
    cache = {r["key"]: r for r in map(json.loads, cache_file.read_text().splitlines())} if cache_file.exists() else {}
    fails = json.loads(fail_file.read_text()) if fail_file.exists() else {}

    reqs = build_requests(cfg, todo, qs, plan_unit, template)
    keyed = [(f"{u.unit_id}|{b_id if b_id >= 0 else items[0]}", u, b_id, items, text) for u, b_id, items, text in reqs]
    pending = [r for r in keyed if r[0] not in cache]
    images = {}
    for u in todo.itertuples():
        path = inputs["prepare"] / u.path
        images[u.unit_id] = {"type": "image", "source": {"type": "base64", "media_type": mimetypes.guess_type(path)[0],
                                                          "data": base64.b64encode(path.read_bytes()).decode()}}
    print(f"  {len(keyed)} prompts, {len(keyed) - len(pending)} already answered; {cfg.claude.model} via claude -p, "
          f"effort {cfg.claude.effort}, {cli.concurrency} at a time")

    lock = threading.Lock()
    state = {"stop": False, "reset": None, "rate": None, "n": 0, "usd": 0.0}

    def work(req):
        key, u, b_id, items, text = req
        while not state["stop"]:
            f = fails.get(key, {"attempts": 0, "tokens": dict(TOK0), "errors": []})
            if f["attempts"] >= cfg.claude.max_attempts:
                return
            r = call(cfg, cwd, images[u.unit_id], text, len(items))
            with lock:
                state["rate"] = r["rate"] or state["rate"]
                stop, reset = over_budget(r["rate"], cli.stop_at)
                if r["rate_limited"] or stop:
                    state["stop"], state["reset"] = True, reset or (r["rate"] or {}).get("resetsAt")
                if r["rate_limited"]:
                    return  # not counted as an attempt: the same prompt runs after the reset
                tok = {k: (f["tokens"][k] + v if k != "kv_cache_hit" else v) for k, v in r["tokens"].items()}
                vals, why = parse_data(r["data"], len(items)) if r["data"] is not None else (None, r["error"])
                state["n"] += 1
                state["usd"] += r["usd"]
                if why == "ok":
                    rec = {"key": key, "vals": vals, "attempts": f["attempts"] + 1, "wall_ms": r["wall_ms"], "tokens": tok}
                    cache[key] = rec
                    with open(cache_file, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    fails.pop(key, None)
                    if state["n"] % 50 == 0:
                        w = (state["rate"] or {}).get("unifiedWindows", {})
                        print(f"  {len(cache)}/{len(keyed)} answered; usage 5h {w.get('five_hour', {}).get('utilization')}, "
                              f"7d {w.get('seven_day', {}).get('utilization')}", flush=True)
                    return
                fails[key] = {"attempts": f["attempts"] + 1, "tokens": tok, "errors": (f["errors"] + [why])[-3:]}
                fail_file.write_text(json.dumps(fails, indent=1))
                with open(partial_file, "a") as fh:
                    fh.write(json.dumps({"key": key, "attempt": f["attempts"] + 1, "error": why, "data": r["data"]}) + "\n")

    with ThreadPoolExecutor(cli.concurrency) as pool:
        list(pool.map(work, pending))

    w = (state["rate"] or {}).get("unifiedWindows", {})
    print(f"  this run: {state['n']} calls, list-price equivalent ${state['usd']:.2f}; usage now "
          f"5h {w.get('five_hour', {}).get('utilization')}, 7d {w.get('seven_day', {}).get('utilization')}")
    complete = [(u, b_id, items, (cache[k]["vals"], cache[k]["attempts"], cache[k]["wall_ms"], cache[k]["tokens"]))
                for k, u, b_id, items, _ in keyed if k in cache]
    by_unit = {}
    for row in keyed:
        by_unit.setdefault(row[1].unit_id, []).append(row[0] in cache)
    finished = {uid for uid, ok in by_unit.items() if all(ok)}
    write_shards(cfg, ctx, qs, [d for d in complete if d[0].unit_id in finished], shard_dir)

    if state["stop"]:
        reset = state["reset"]
        PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FILE.write_text(str(reset or int(time.time()) + 3600))
        when = datetime.fromtimestamp(reset).strftime("%Y-%m-%d-%H:%M:%S") if reset else "in about an hour"
        print(f"PAUSED: usage limit reached ({len(finished)}/{len(by_unit)} images complete); resumes after {when}")
        raise SystemExit(PAUSED)
    if fails:
        raise RuntimeError(f"{len(fails)} prompts still have no complete answer after {cfg.claude.max_attempts} attempts "
                           f"(details in {fail_file.relative_to(ROOT)})")
