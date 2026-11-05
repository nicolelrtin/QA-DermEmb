"""Token counts, time and cost of every answer stage run so far, and the projection for 500 questions
(no negations) on 1,000 images. Writes token_usage_by_stage.csv and projection.csv next to this file.

    poetry run python reports/2026-11-04_token_usage/token_usage.py

Sources: outputs/qaemb/answer/<hash>/timing.parquet (one row per prompt; image-prefix rows for local models),
stage_meta.json (answerer, mode), every run_meta.json (which runs used the stage), and the Claude list-price
cost that claude -p reports per run, logged in outputs/qaemb/queue/queue.log and pilot_topic_R1.log.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ANS = ROOT / "outputs" / "qaemb" / "answer"
QUEUE = ROOT / "outputs" / "qaemb" / "queue"
IMAGES, QUESTIONS = 1000, 500


def runs_by_stage() -> dict:
    """answer dir name -> [(run_id, experiments)] for every run that used it."""
    out = {}
    for p in sorted((ROOT / "outputs" / "qaemb" / "runs").glob("*/run_meta.json")):
        m = json.loads(p.read_text())
        if "answer" in m["stage_dirs"]:
            out.setdefault(Path(m["stage_dirs"]["answer"]).name, []).append((p.parent.name, m["experiments"]))
    return out


def claude_costs(by_stage: dict) -> dict:
    """answer dir -> list-price USD summed over its runs (a paused run logs two parts)."""
    exp_to_dir = {}
    for d, runs in by_stage.items():
        for _, e in runs:
            exp_to_dir[",".join(e.values())] = d
    cost = {}
    for log in (QUEUE / "pilot_topic_R1.log", QUEUE / "queue.log"):
        exp = "D0_poc2,Q0_all,ANS_claude_opus_cli,M_topic,G1,R1" if log.name.startswith("pilot") else None
        for line in log.read_text().splitlines():
            m = re.search(r"start: qaemb\.run .*\+exp=\[([^\]]+)\]", line)
            if m:
                exp = m.group(1)
            m = re.search(r"this run: (\d+) calls, list-price equivalent \$([\d.]+)", line)
            if m and exp in exp_to_dir:
                d = exp_to_dir[exp]
                cost[d] = cost.get(d, 0.0) + float(m.group(2))
    return cost


def stage_table() -> pd.DataFrame:
    by_stage = runs_by_stage()
    cost = claude_costs(by_stage)
    rows = []
    for d in sorted(ANS.iterdir()):
        if not (d / "timing.parquet").exists():
            continue
        meta = json.loads((d / "stage_meta.json").read_text())
        k = meta["key"]
        t = pd.read_parquet(d / "timing.parquet")
        prompts = t[~t.question_id.astype(str).str.startswith("__")]
        n_items = len(pd.read_parquet(d / "answers_raw.parquet", columns=["p_yes"]))  # every image x item answered
        runs = by_stage.get(d.name, [])
        exps = runs[0][1] if runs else {}
        cached = t.n_cache_read_tokens.fillna(0).sum() if "n_cache_read_tokens" in t else 0
        written = t.n_cache_write_tokens.fillna(0).sum() if "n_cache_write_tokens" in t else 0
        usd = cost.get(d.name, np.nan)
        rows.append({
            "answer_stage": d.name,
            "runs": " ".join(r for r, _ in runs) or "(no finished run; stopped M_all pilot)",
            "answerer": k["answerer"]["id"],
            "mode": k["mode"] + (", shuffled" if k.get("shuffle") else ""),
            "replicate": k.get("replicate", "-"),
            "data": exps.get("data", "D0_smoke10"), "qset": exps.get("qset", "Q0_all"), "grid": exps.get("grid", "G1"),
            "images": int(t.image_id.nunique()),
            "prompts": len(prompts), "items_answered": int(n_items) if n_items else np.nan,
            "input_tokens": int(t.n_input_tokens.fillna(0).sum()),
            "of_which_cache_read": int(cached), "of_which_cache_write": int(written),
            "output_tokens": int(t.n_output_tokens.fillna(0).sum()),
            "answer_hours": round(t.wall_ms.sum() / 3.6e6, 3),
            "list_price_usd": round(usd, 2) if usd == usd else np.nan,
        })
    df = pd.DataFrame(rows)
    df["input_per_item"] = (df.input_tokens / df.items_answered).round(1)
    df["output_per_item"] = (df.output_tokens / df.items_answered).round(1)
    df["usd_per_prompt"] = (df.list_price_usd / df.prompts).round(4)
    return df


def originals_per_prompt(d: str) -> dict:
    """Mean tokens per prompt for the original questions only (what a no-negation run would send)."""
    t = pd.read_parquet(ANS / d / "timing.parquet")
    t = t[(t.original == 1) & ~t.question_id.astype(str).str.startswith("__")]
    return {"input": t.n_input_tokens.mean(), "output": t.n_output_tokens.mean(),
            "cache_read": t.n_cache_read_tokens.mean(), "prompts_per_image": len(t) / t.image_id.nunique()}


def projection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    claude = df[(df.answerer == "claude-opus-cli") & df.list_price_usd.notna() & (df.data == "D0_poc2")]
    for mode in ("stateless", "topic"):
        sub = claude[claude["mode"] == mode]
        per = pd.DataFrame([originals_per_prompt(d) for d in sub.answer_stage])
        n = IMAGES * QUESTIONS if mode == "stateless" else IMAGES * per.prompts_per_image.mean()
        usd = sub.usd_per_prompt
        rows.append({"answerer": "Claude Opus 5.5 (claude -p)", "mode": mode, "prompts": int(round(n)),
                     "input_tokens_M": round(n * per.input.mean() / 1e6, 1),
                     "output_tokens_M": round(n * per.output.mean() / 1e6, 1),
                     "list_price_usd_low": round(n * usd.min()), "list_price_usd_median": round(n * usd.median()),
                     "list_price_usd_high": round(n * usd.max()), "basis": f"{len(sub)} runs, 2 images"})
    for answerer in ("qwen2.5-vl-7b", "skingpt-r1"):
        for mode in ("stateless", "topic"):
            sub = df[(df.answerer == answerer) & (df["mode"] == mode) & (df.data == "D0_smoke10") & (df.qset == "Q0_all")
                     & (df.grid == "G1")]
            if sub.empty:
                continue
            s = sub.iloc[-1]
            items = IMAGES * QUESTIONS
            rows.append({"answerer": answerer, "mode": mode,
                         "prompts": items if mode == "stateless" else int(round(s.prompts / s.images / 2 * IMAGES)),
                         "input_tokens_M": round(items * s.input_per_item / 1e6, 1),
                         "output_tokens_M": round(items * s.output_per_item / 1e6, 1),
                         "gpu_hours": round(s.answer_hours / s.items_answered * items, 1),
                         "basis": f"{s.answer_stage}, 10 images"})
    return pd.DataFrame(rows)


def main():
    df = stage_table()
    df.to_csv(HERE / "token_usage_by_stage.csv", index=False)
    pr = projection(df)
    pr.to_csv(HERE / "projection.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["runs"]).to_string(index=False))
        print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
