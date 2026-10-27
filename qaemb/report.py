"""Reports over finished stage outputs (read-only).

  python -m qaemb.report modes <answer_dir> <answer_dir> ...    # timing + agreement vs the first dir
  python -m qaemb.report compare <run_id> <run_id> ... --out reports/<name>
      # side-by-side answers of several runs (e.g. stateless / topic / all) for the same images:
      # compare_answers.csv (image x question), compare_by_question.csv, summary.md
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


def modes(dirs: list[Path], threshold: float = 0.5) -> pd.DataFrame:
    ref, _, _ = load(dirs[0])
    keys = ["unit_id", "question_id", "original"]
    rows = []
    for d in dirs:
        ans, tim, meta = load(d)
        m = ref[keys + ["p_yes"]].merge(ans[keys + ["p_yes"]], on=keys, suffixes=("_ref", ""))
        rows.append({
            "dir": d.name, "mode": meta["key"]["mode"], "prompts": ans.batch_id.nunique() if meta["key"]["mode"] != "stateless" else 0,
            "shuffle": meta["key"]["shuffle"],
            "n": len(ans), "sec_per_query": tim.wall_ms.sum() / 1e3 / len(ans),
            "prefix_ms": tim[tim.question_id == "__image_prefix__"].wall_ms.mean(),
            "yesno_mass_min": ans.yesno_mass.min(), "frac_yes": (ans.p_yes >= threshold).mean(),
            "pearson_vs_ref": np.corrcoef(m.p_yes_ref, m.p_yes)[0, 1],
            "bin_agree_vs_ref": ((m.p_yes_ref >= threshold) == (m.p_yes >= threshold)).mean(),
            "mean_abs_dp_vs_ref": (m.p_yes_ref - m.p_yes).abs().mean(),
        })
        if meta["key"]["mode"] != "stateless":
            pos = ans.groupby("position_in_batch").p_yes.mean()
            rows[-1]["p_yes_first_vs_last_pos"] = f"{pos.iloc[0]:.3f} / {pos.iloc[-1]:.3f}"
        w = ans.pivot_table(index=["unit_id", "question_id"], columns="original", values="p_yes")
        if 0 in w.columns:
            rows[-1]["contradiction_rate"] = ((w[1] >= threshold) == (w[0] >= threshold)).mean()
            rows[-1]["mean_abs_p_adj"] = (w[1] - w[0]).abs().mean()
    return pd.DataFrame(rows)


def cost(d: Path, images: int, G: int, questions: int, local: int | None) -> dict:
    ans, tim, _ = load(d)
    spq = tim.wall_ms.sum() / 1e3 / len(ans)
    q_patch = local if local is not None else questions
    n_queries = images * questions + images * G * G * q_patch
    return {"sec_per_query": round(spq, 5), "n_queries": n_queries, "hours": round(n_queries * spq / 3600, 1),
            "formula": f"{images}x{questions} + {images}x{G}^2x{q_patch} queries x {spq:.4f} s"}


ROOT = Path(__file__).resolve().parent.parent


def run_dirs(run_id: str) -> tuple[str, dict]:
    meta = json.loads((ROOT / "outputs/qaemb/runs" / run_id / "run_meta.json").read_text())
    mode = meta["config"]["answer"]["mode"] + ("_shuf" if meta["config"]["answer"]["shuffle"] else "")
    return mode, {k: ROOT / v for k, v in meta["stage_dirs"].items()}


def compare(run_ids: list[str], out: Path, threshold: float = 0.5) -> Path:
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    runs = [run_dirs(r) for r in run_ids]
    names = [m for m, _ in runs]
    if len(set(names)) != len(names):
        names = [f"{m}_{i}" for i, m in enumerate(names)]
    idx = ["image_id", "label_raw", "patch_id", "question_id"]
    text = ["topic", "scope", "kind", "question", "question_neg"]
    merged = None
    for name, (_, dirs) in zip(names, runs):
        df = pd.read_csv(dirs["embed"] / "answers_readable.csv")
        vals = df[idx + ["p_yes_original", "p_yes_complement", "p_adj", "verdict"]].rename(
            columns={c: f"{c}__{name}" for c in ["p_yes_original", "p_yes_complement", "p_adj", "verdict"]})
        merged = df[idx + [c for c in text if c in df]].merge(vals, on=idx) if merged is None else merged.merge(vals, on=idx)
    merged.round(4).to_csv(out / "compare_answers.csv", index=False)

    whole = merged[merged.patch_id < 0]
    byq = whole.groupby("question_id").agg(
        question=("question", "first"), topic=("topic", "first"),
        **{f"mean_p_adj__{n}": (f"p_adj__{n}", "mean") for n in names},
        **{f"contradiction_rate__{n}": (f"verdict__{n}", lambda v: v.str.startswith("contradiction").mean()) for n in names})
    byq.round(4).to_csv(out / "compare_by_question.csv")

    answer_dirs = [d["answer"] for _, d in runs]
    table = modes(answer_dirs, threshold)
    table.insert(0, "run", [f"{n} ({r})" for n, r in zip(names, run_ids)])
    L = [f"# Answer-mode comparison: {', '.join(names)}", "",
         f"Runs: {', '.join(f'`{r}`' for r in run_ids)}. Reference for agreement columns: `{names[0]}`.", "",
         table.drop(columns=["dir"]).round(4).to_markdown(index=False), "",
         "- [compare_answers.csv](compare_answers.csv): one row per image x question; P(yes) to the original and "
         "complement, p_adj and verdict for each run side by side.",
         "- [compare_by_question.csv](compare_by_question.csv): per question, mean p_adj and contradiction rate per run.", ""]
    verdicts = pd.concat({n: whole[f"verdict__{n}"].value_counts(normalize=True) for n in names}, axis=1).fillna(0)
    L += ["Share of verdicts (whole image, all images x questions):", "", verdicts.round(3).to_markdown(), ""]
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
    a = p.parse_args()
    if a.cmd == "compare":
        print(compare(a.run_ids, a.out))
    elif a.cmd == "modes":
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(modes(a.dirs).round(4).to_string(index=False))
    else:
        print(json.dumps(cost(a.dir, a.images, a.G, a.questions, a.local), indent=2))


if __name__ == "__main__":
    main()
