"""Human-readable run report, written for every run next to run_meta.json:
outputs/qaemb/runs/<run_id>/report.md. Links are relative to the report so they open from the IDE."""
import json
import os
from pathlib import Path

import pandas as pd

from .config import ROOT

SKIP = {"prompts", "counts", "provenance", "label_mapping", "prefix_cache_check", "params", "key_metric"}


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix().replace(" ", "%20")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _metrics_table(metrics: dict) -> list[str]:
    splits = [k for k, v in metrics.items() if isinstance(v, dict) and "auroc" in v]
    if not splits:
        return []
    cols = [c for c in metrics[splits[0]] if c != "confusion_matrix"]
    lines = ["| split | " + " | ".join(cols) + " | confusion [[TN, FP], [FN, TP]] |",
             "|---" * (len(cols) + 2) + "|"]
    for s in splits:
        m = metrics[s]
        lines.append(f"| {s} | " + " | ".join(_fmt(m[c]) for c in cols) + f" | {m['confusion_matrix']} |")
    return lines


def write_run_report(meta: dict, status: dict, run_dir: Path) -> Path:
    cfg = meta["config"]
    L = [f"# Run {meta['run_id']}", ""]
    L += [f"- **Experiments:** {', '.join(f'`{v}`' for v in meta['experiments'].values()) or '(base config only)'}",
          f"- **Stage requested:** `{cfg['stage']}`   **Tag:** {cfg['tag'] or '-'}",
          f"- **Data:** {cfg['data']['name']} ({len(cfg['data']['image_ids']) if cfg['data']['image_ids'] else 'all'} images), "
          f"split `{cfg['split']['strategy']}`, label `{cfg['label']['column']}`, indeterminate `{cfg['label']['indeterminate']}`",
          f"- **Git:** `{meta['git']['commit'][:10]}`{' (uncommitted changes)' if meta['git']['dirty'] else ''}   "
          f"**GPU:** {meta.get('gpu_name')}   **Wall time:** {meta['wall_s']} s"]
    if "question_set" in meta:
        q = meta["question_set"]
        L.append(f"- **Questions:** {q['count']} items, hash `{q['hash']}` from `{q['csv']}`")
    if "answerer" in meta:
        a = meta["answerer"]
        L.append(f"- **Answerer:** `{a['checkpoint']}` @ `{a['revision'][:10]}` ({a['dtype']}, {a['attn']}), "
                 f"mode `{cfg['answer']['mode']}`")
    if "model" in meta["stages"]:
        L.append(f"- **Model:** `{cfg['model']['name']}`, features `{cfg['feat']['source']}` / `{cfg['feat']['kind']}`")

    L += ["", "## Stages", "", "| stage | hash | this run | output |", "|---|---|---|---|"]
    for s in meta["stages"]:
        d = ROOT / meta["stage_dirs"][s]
        L.append(f"| {s} | `{meta['stage_hashes'][s]}` | {'reused (cached)' if status[s] == 'cached' else 'ran'} | "
                 f"[{meta['stage_dirs'][s]}]({_rel(d, run_dir)}) |")

    L += ["", "## Results", ""]
    for s in meta["stages"]:
        summ = {k: v for k, v in meta["summaries"].get(s, {}).items() if k not in SKIP and not isinstance(v, (dict, list))}
        if summ:
            L.append(f"**{s}:** " + ", ".join(f"{k} = {_fmt(v)}" for k, v in summ.items()))
            L.append("")
    check = meta["summaries"].get("answer", {}).get("prefix_cache_check")
    if check:
        L += [f"Prefix-cache check ({check.get('item_id') or check.get('question_id')}): cached {check['cached']:.4f} "
              f"vs uncached {check['reference']:.4f} (|diff| {check['abs_diff']:.4f})", ""]

    if "embed" in meta["stages"]:
        d = ROOT / meta["stage_dirs"]["embed"]
        readable = d / "answers_readable.csv"
        if readable.exists():
            L += [f"**Answers to review:** [answers_readable.csv]({_rel(readable, run_dir)}) "
                  "(one row per image x question pair: question texts, P(yes) to each version, p_adj, verdict)", ""]
            df = pd.read_csv(readable)
            if "verdict" in df and (df.patch_id < 0).any():
                v = df[df.patch_id < 0].pivot_table(index="image_id", columns="verdict", values="question_id",
                                                    aggfunc="count", fill_value=0)
                v.insert(0, "label", df.drop_duplicates("image_id").set_index("image_id").label_raw.reindex(v.index))
                L += ["Verdicts per image (whole image):", "", v.to_markdown(), ""]

    if "model" in meta["stages"]:
        d = ROOT / meta["stage_dirs"]["model"]
        metrics = json.loads((d / "metrics.json").read_text())
        L += ["## Model metrics", ""] + _metrics_table(metrics) + [""]
        if cfg["split"]["strategy"] == "loo":
            L += ["*Leave-one-out on the smoke set: plumbing only, the numbers are not meaningful.*", ""]
        coefs = pd.read_csv(d / "coefficients.csv")
        cols = [c for c in ["question_id", "question", "kind", "scope", "coef", "stability"] if c in coefs]
        L += ["Top 15 questions by |coefficient| (positive = towards malignant):", "",
              coefs[cols].head(15).round(4).to_markdown(index=False), "",
              f"All files: [{meta['stage_dirs']['model']}]({_rel(d, run_dir)}) "
              "(metrics.json, predictions.parquet, coefficients.csv, curves.csv)", ""]

    if "answer" in meta["stages"]:
        L += token_section(ROOT / meta["stage_dirs"]["answer"], run_dir)

    L += ["## Files", "", "- [run_meta.json](run_meta.json): full resolved config, versions, hashes", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(L))
    return path


TOK = ["n_input_tokens", "n_output_tokens", "n_cache_read_tokens", "n_cache_write_tokens"]


def token_tables(answer_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run totals, tokens per image, and tokens per item (Question ID x original) per image.

    Per item: a stateless item's own tokens; in topic/all modes, the prompt's tokens split evenly over its
    questions. A Qwen-style image prefix (processed once, then reused) is shared evenly over the image's items.
    """
    t = pd.read_parquet(answer_dir / "timing.parquet")
    ans = pd.read_parquet(answer_dir / "answers_raw.parquet")[["unit_id", "image_id", "question_id", "original", "batch_id"]]
    for c in TOK:
        t[c] = t[c].fillna(0) if c in t else 0
    totals = {c.replace("n_", "").replace("_tokens", ""): int(t[c].sum()) for c in TOK}
    totals["requests_or_passes"] = int((t.question_id != "__image_prefix__").sum())

    prefix = t[t.question_id == "__image_prefix__"].groupby("unit_id")[TOK].sum()
    single = t[(t.batch_id < 0) & (t.question_id != "__image_prefix__")][["unit_id", "question_id", "original"] + TOK]
    batched = t[t.batch_id >= 0][["unit_id", "batch_id", "batch_k"] + TOK]
    if not batched.empty:
        batched = ans.merge(batched, on=["unit_id", "batch_id"])
        batched[TOK] = batched[TOK].div(batched.batch_k, axis=0)
    items = pd.concat([single, batched[["unit_id", "question_id", "original"] + TOK] if not batched.empty else None])
    items = items.merge(ans[["unit_id", "image_id", "question_id", "original"]], on=["unit_id", "question_id", "original"])
    if not prefix.empty:
        n_items = items.groupby("unit_id").size()
        share = prefix.div(n_items, axis=0).reindex(items.unit_id).fillna(0).values
        items[TOK] = items[TOK].values + share
    by_image = items.groupby("image_id")[TOK].sum().round(0).astype(int)
    return totals, by_image, items


def token_section(answer_dir: Path, run_dir: Path) -> list[str]:
    totals, by_image, items = token_tables(answer_dir)
    by_q = items.groupby(["question_id", "original"])[TOK].mean().round(1)
    by_image.to_csv(run_dir / "tokens_by_image.csv")
    by_q.to_csv(run_dir / "tokens_by_question.csv")
    stats = items[["n_input_tokens", "n_output_tokens"]].describe().loc[["mean", "50%", "min", "max"]].round(1)
    stats.index = ["mean", "median", "min", "max"]
    return ["## Tokens", "",
            "Per run: " + ", ".join(f"{k.replace('_', ' ')} = {v:,}" for k, v in totals.items()), "",
            "Per image (all of its questions):", "", by_image.to_markdown(), "",
            "Per question, per image (input includes its share of the image tokens; in topic/all modes, "
            "an even share of the prompt):", "", stats.to_markdown(), "",
            "Files: [tokens_by_image.csv](tokens_by_image.csv), [tokens_by_question.csv](tokens_by_question.csv)", ""]
