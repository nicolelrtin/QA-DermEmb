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

    L += ["## Files", "", "- [run_meta.json](run_meta.json): full resolved config, versions, hashes", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(L))
    return path
