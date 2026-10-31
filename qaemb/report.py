"""Reports over finished stage outputs (read-only).

  python -m qaemb.report modes <answer_dir> <answer_dir> ...    # timing + agreement vs the first dir
  python -m qaemb.report compare <run_id> <run_id> ... --out reports/<name>
      # side-by-side answers of several runs (e.g. stateless / topic / all) for the same images:
      # compare_answers.csv (image x question), compare_by_question.csv, summary.md
      # a run_id of the form A+B+C+D averages those replicate runs into one column set
  python -m qaemb.report replicates <run_id> <run_id> ... --out reports/<name>
      # replicate runs of one answerer and mode: replicates_raw.csv (every replicate's values),
      # replicates_summary.csv (mean, SD and verdict agreement per image x question), summary.md
  python -m qaemb.report cost <answer_dir> --images 1000 --G 4 --questions 1000 [--local 466]   # count items: originals + complements
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(d: Path):
    meta = json.loads((d / "stage_meta.json").read_text())
    return pd.read_parquet(d / "answers_raw.parquet"), pd.read_parquet(d / "timing.parquet"), meta


def modes(dirs: list[Path], threshold: float = 0.5, images: set | None = None) -> pd.DataFrame:
    """images: restrict every run to these image_ids (runs over different image sets)."""
    ref, _, _ = load(dirs[0])
    if images is not None:
        ref = ref[ref.image_id.isin(images)]
    keys = ["unit_id", "question_id", "original"]
    rows = []
    for d in dirs:
        ans, tim, meta = load(d)
        if images is not None:
            ans, tim = ans[ans.image_id.isin(images)], tim[tim.image_id.isin(images)]
        m = ref[keys + ["p_yes"]].merge(ans[keys + ["p_yes"]], on=keys, suffixes=("_ref", ""))
        rows.append({
            "dir": d.name, "mode": meta["key"]["mode"], "prompts": ans.batch_id.nunique() if meta["key"]["mode"] != "stateless" else 0,
            "shuffle": meta["key"]["shuffle"],
            "n": len(ans), "sec_per_query": tim.wall_ms.sum() / 1e3 / len(ans),
            "prefix_ms": tim[tim.question_id == "__image_prefix__"].wall_ms.mean(),
            "yesno_mass_min": ans.yesno_mass.min(), "frac_yes": (ans.p_yes >= threshold).mean(),
            "pearson_vs_ref": m.p_yes_ref.corr(m.p_yes),
            "bin_agree_vs_ref": ((m.p_yes_ref >= threshold) == (m.p_yes >= threshold)).mean(),
            "mean_abs_dp_vs_ref": (m.p_yes_ref - m.p_yes).abs().mean(),
        })
        if meta["key"]["mode"] != "stateless":
            pos = ans.groupby("position_in_batch").p_yes.mean()
            rows[-1]["p_yes_first_vs_last_pos"] = f"{pos.iloc[0]:.3f} / {pos.iloc[-1]:.3f}"
        w = ans.pivot(index=["unit_id", "question_id"], columns="original", values="p_yes")
        if 0 in w.columns:
            rows[-1]["contradiction_rate"] = ((w[1] >= threshold) == (w[0] >= threshold)).mean()
            rows[-1]["mean_abs_p_adj"] = (w[1] - w[0]).abs().mean()
            rows[-1]["mean_p_w"] = (w[1] * (w[1] - w[0]).abs()).mean()
        if "confidence" in ans:
            rows[-1]["mean_confidence"] = ans.confidence.mean()
        rows[-1]["tokens_in"] = int(tim.n_input_tokens.fillna(0).sum())
        rows[-1]["tokens_out"] = int(tim.n_output_tokens.fillna(0).sum())
    return pd.DataFrame(rows)


def cost(d: Path, images: int, G: int, questions: int, local: int | None) -> dict:
    ans, tim, _ = load(d)
    spq = tim.wall_ms.sum() / 1e3 / len(ans)
    q_patch = local if local is not None else questions
    n_queries = images * questions + images * G * G * q_patch
    return {"sec_per_query": round(spq, 5), "n_queries": n_queries, "hours": round(n_queries * spq / 3600, 1),
            "formula": f"{images}x{questions} + {images}x{G}^2x{q_patch} queries x {spq:.4f} s"}


ROOT = Path(__file__).resolve().parent.parent


VALS = ["p_yes_original", "p_yes_complement", "p_adj", "p_w", "confidence_original", "confidence_complement"]
SD_VALS = ["p_yes_original", "p_yes_complement", "p_adj", "p_w"]


def run_dirs(run_id: str) -> tuple[str, dict]:
    meta = json.loads((ROOT / "outputs/qaemb/runs" / run_id / "run_meta.json").read_text())
    c = meta["config"]
    name = f"{c['answerer']['id']}/{c['answer']['mode']}" + ("_shuf" if c["answer"]["shuffle"] else "")
    if c["answerer"].get("backend") in ("anthropic", "claude_cli") or c["answer"]["shuffle"]:
        name += f"/R{c['answer'].get('replicate', 1)}"
    return name, {k: ROOT / v for k, v in meta["stage_dirs"].items()}


def glossary() -> dict:
    import yaml

    g = yaml.safe_load((ROOT / "conf/exp/glossary.yaml").read_text())
    return {k: v for group in g.values() for k, v in group.items()}


def runs_covered(run_ids: list[str]) -> list[str]:
    """Markdown table: every run behind a report, its experiment IDs spelled out, prompt and images."""
    words = glossary()
    rows = []
    for group in run_ids:
        for r in group.split("+"):
            meta = json.loads((ROOT / "outputs/qaemb/runs" / r / "run_meta.json").read_text())
            exps = meta["experiments"]
            prompts = meta.get("prompts") or json.loads((ROOT / meta["stage_dirs"]["answer"] / "stage_meta.json").read_text())["summary"].get("prompts", {})
            ids = meta["config"]["data"]["image_ids"]
            rows.append({"run": f"`{r}`", "label": run_dirs(r)[0],
                         "experiments": "<br>".join(f"`{e}`: {words.get(e, '?')}" for e in exps.values()),
                         "prompt": ", ".join(f"`{v['name']}` ({v['hash']})" for v in prompts.values()),
                         "images": f"{len(ids)}" if ids else "all"})
    return ["## Runs covered", "", pd.DataFrame(rows).to_markdown(index=False), ""]


def readable(run_id: str) -> tuple[str, pd.DataFrame, list[Path]]:
    """Name, answers_readable rows (P_w added for older runs), and answer dirs; A+B+... averages replicates."""
    from .stages.s4_embed import verdict

    parts = run_id.split("+")
    frames, answer_dirs, names = [], [], []
    for r in parts:
        name, dirs = run_dirs(r)
        df = pd.read_csv(dirs["embed"] / "answers_readable.csv")
        if "p_w" not in df:
            df["p_w"] = df.p_yes_original * df.p_adj.abs()
        frames.append(df)
        answer_dirs.append(dirs["answer"])
        names.append(name)
    if len(frames) == 1:
        return names[0], frames[0], answer_dirs
    idx = ["image_id", "label_raw", "patch_id", "question_id"]
    vals = [c for c in VALS if c in frames[0]]
    g = pd.concat(frames).groupby(idx)
    mean = g[vals].mean().join(g[SD_VALS].std().add_suffix("_sd"))
    mean["verdict_agreement"] = g.verdict.agg(lambda v: v.value_counts().iloc[0] / len(v))
    df = frames[0].drop(columns=vals + ["verdict"], errors="ignore").merge(mean.reset_index(), on=idx)
    df["verdict"] = [verdict(a, b, 0.5) for a, b in zip(df.p_yes_original, df.p_yes_complement)]
    return names[0].rsplit("/R", 1)[0] + f"/mean{len(parts)}", df, answer_dirs


def compare(run_ids: list[str], out: Path, threshold: float = 0.5) -> Path:
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    runs = [readable(r) for r in run_ids]
    names = [n for n, _, _ in runs]
    if len(set(names)) != len(names):
        names = [f"{m}_{i}" for i, m in enumerate(names)]
    idx = ["image_id", "label_raw", "patch_id", "question_id"]
    text = ["topic", "scope", "kind", "question", "question_neg"]
    merged = None
    for name, (_, df, _) in zip(names, runs):
        cols = [c for c in VALS + ["verdict"] + [f"{v}_sd" for v in SD_VALS] + ["verdict_agreement"] if c in df]
        vals = df[idx + cols].rename(columns={c: f"{c}__{name}" for c in cols})
        merged = df[idx + [c for c in text if c in df]].merge(vals, on=idx) if merged is None else merged.merge(vals, on=idx)
    merged.round(4).to_csv(out / "compare_answers.csv", index=False)

    whole = merged[merged.patch_id < 0]
    byq = whole.groupby("question_id").agg(
        question=("question", "first"), topic=("topic", "first"),
        **{f"mean_p_adj__{n}": (f"p_adj__{n}", "mean") for n in names},
        **{f"mean_p_w__{n}": (f"p_w__{n}", "mean") for n in names},
        **{f"contradiction_rate__{n}": (f"verdict__{n}", lambda v: v.str.startswith("contradiction").mean()) for n in names})
    byq.round(4).to_csv(out / "compare_by_question.csv")

    answer_dirs = [dirs[0] for _, _, dirs in runs]  # first replicate of an averaged group
    table = modes(answer_dirs, threshold, images=set(merged.image_id))
    table.insert(0, "run", [f"{n} ({r.split('+')[0]}{' +' + str(r.count('+')) + ' more' if '+' in r else ''})"
                            for n, r in zip(names, run_ids)])
    L = [f"# Answer comparison: {', '.join(names)}", "",
         f"Images: {', '.join(sorted(merged.image_id.unique()))} (those every run answered). "
         f"Reference for the agreement columns: `{names[0]}`.", ""] + runs_covered(run_ids) + ["## Summary", "",
         table.drop(columns=["dir"]).round(4).to_markdown(index=False), "",
         "- [compare_answers.csv](compare_answers.csv): one row per image x question; P(yes) to the original and "
         "complement, p_adj, P_w, stated confidence (Claude) and verdict for each run side by side. "
         "Averaged replicate groups (`/meanN`) give the mean of each value over the N replicates, its SD "
         "(`_sd` columns) and `verdict_agreement` (share of replicates giving the most common verdict); "
         "the table's timing and token columns describe their first replicate.",
         "- [compare_by_question.csv](compare_by_question.csv): per question, mean p_adj and contradiction rate per run.", ""]
    verdicts = pd.concat({n: whole[f"verdict__{n}"].value_counts(normalize=True) for n in names}, axis=1).fillna(0)
    L += ["Share of verdicts (whole image, all images x questions):", "", verdicts.round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(L))
    return out / "summary.md"


def replicates(run_ids: list[str], out: Path) -> Path:
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    idx = ["image_id", "label_raw", "patch_id", "question_id"]
    frames = []
    for r in run_ids:
        name, df, _ = readable(r)
        frames.append(df.assign(replicate=name.rsplit("/", 1)[-1], run_id=r))
    raw = pd.concat(frames, ignore_index=True)
    raw.round(4).to_csv(out / "replicates_raw.csv", index=False)
    vals = [c for c in VALS if c in raw]
    g = raw.groupby(idx)
    agg = g[vals].agg(["mean", "std"])
    agg.columns = [f"{v}_{s}" for v, s in agg.columns]
    agg["verdict_modal"] = g.verdict.agg(lambda v: v.mode().iloc[0])
    agg["verdict_agreement"] = g.verdict.agg(lambda v: v.value_counts().iloc[0] / len(v))
    agg = agg.reset_index().merge(frames[0][idx + ["question", "question_neg"]], on=idx)
    agg.round(4).to_csv(out / "replicates_summary.csv", index=False)

    wide = raw.pivot_table(index=idx, columns="replicate", values="p_adj")
    corr = wide.corr().values[np.triu_indices(wide.shape[1], 1)]
    rows = [{"value": v, "mean": raw[v].mean(), "mean SD across replicates": agg[f"{v}_std"].mean()} for v in vals]
    L = [f"# Replicates: {run_dirs(run_ids[0])[0].rsplit('/R', 1)[0]} ({len(run_ids)} runs)", ""] + runs_covered(run_ids) + [
         "## Summary", "",
         pd.DataFrame(rows).round(4).to_markdown(index=False), "",
         f"- Image × question pairs with the same verdict in every replicate: **{(agg.verdict_agreement == 1).mean():.1%}**",
         f"- Mean pairwise Pearson r of p_adj between replicates: **{np.nanmean(corr):.3f}**",
         "",
         "- [replicates_raw.csv](replicates_raw.csv): every replicate's values, one row per replicate x image x question.",
         "- [replicates_summary.csv](replicates_summary.csv): mean and SD per image x question, modal verdict and its share.", ""]
    (out / "summary.md").write_text("\n".join(L))
    return out / "summary.md"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("modes")
    pm.add_argument("dirs", nargs="+", type=Path)
    pc = sub.add_parser("cost")
    pc.add_argument("dir", type=Path)
    pc.add_argument("--images", type=int, required=True)
    pc.add_argument("--G", type=int, default=1)
    pc.add_argument("--questions", type=int, required=True)
    pc.add_argument("--local", type=int, default=None, help="questions asked per patch (default: --questions)")
    pr = sub.add_parser("compare")
    pr.add_argument("run_ids", nargs="+")
    pr.add_argument("--out", type=Path, required=True)
    pp = sub.add_parser("replicates")
    pp.add_argument("run_ids", nargs="+")
    pp.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "replicates":
        print(replicates(a.run_ids, a.out))
    elif a.cmd == "compare":
        print(compare(a.run_ids, a.out))
    elif a.cmd == "modes":
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(modes(a.dirs).round(4).to_string(index=False))
    else:
        print(json.dumps(cost(a.dir, a.images, a.G, a.questions, a.local), indent=2))


if __name__ == "__main__":
    main()
