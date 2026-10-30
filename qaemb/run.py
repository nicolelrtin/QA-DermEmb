"""QA-Emb pipeline CLI.

  python -m qaemb.run stage=<stage|all> +exp=[<ID>,<ID>,...] tag=<free_text> [key=value ...]
  python -m qaemb.run stage=all +exp=[...] dry_run=true      # resolved config + cache hits, runs nothing

Stages: manifest -> prepare -> answer -> embed -> model. Asking for a stage runs its upstream chain;
stages whose content-addressed output dir is already complete are skipped.
"""
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .config import ROOT, compose, parse_cli, stable_hash

ORDER = ["manifest", "prepare", "answer", "embed", "model"]
MODULES = {
    "manifest": "qaemb.stages.s0_manifest",
    "prepare": "qaemb.stages.s1_prepare",
    "answer": "qaemb.stages.s2_answer",
    "embed": "qaemb.stages.s4_embed",
    "model": "qaemb.stages.s5_model",
}
DONE = "_DONE"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class Context:
    """Lazily resolved values shared by stages (question set, model revision, run id)."""

    def __init__(self, cfg, run_id: str):
        self.cfg, self.run_id = cfg, run_id
        self._questions = None
        self._revision = None
        self.stage_dirs: dict[str, Path] = {}
        self.stage_hashes: dict[str, str] = {}

    @property
    def questions(self):
        if self._questions is None:
            from .questions import load_questions

            self._questions = load_questions(self.cfg.questions)
        return self._questions

    @property
    def qset_hash(self) -> str:
        from .questions import qset_hash

        return qset_hash(self.questions)

    @property
    def answerer_revision(self) -> str:
        if self._revision is None:
            from huggingface_hub import snapshot_download

            a = self.cfg.answerer
            if a.backend in ("anthropic", "claude_cli"):  # API models are versioned by their ID
                self._revision = self.cfg.claude.model
                return self._revision
            kw = {"allow_patterns": [f"{a.subfolder}/*"], "ignore_patterns": ["*.ipynb_checkpoints*"]} if a.subfolder else {}
            try:
                path = snapshot_download(a.checkpoint, revision=a.revision, local_files_only=True, **kw)
            except Exception:
                path = snapshot_download(a.checkpoint, revision=a.revision, **kw)
            self._revision = Path(path).name
        return self._revision


def stage_chain(target: str) -> list[str]:
    if target == "all":
        return list(ORDER)
    if target not in ORDER:
        raise SystemExit(f"Unknown stage {target!r}; choose from {ORDER + ['all']}")
    return ORDER[: ORDER.index(target) + 1]


def resolve(cfg, ctx: Context, chain: list[str]) -> dict:
    out_root = ROOT / cfg.paths.out_root
    modules = {}
    for name in chain:
        mod = importlib.import_module(MODULES[name])
        up = {u: ctx.stage_hashes[u] for u in mod.UPSTREAM}
        h = stable_hash({"stage": name, "version": getattr(mod, "VERSION", 1), "key": mod.key(cfg, ctx), "upstream": up})
        ctx.stage_hashes[name] = h
        ctx.stage_dirs[name] = out_root / name / h
        modules[name] = mod
    return modules


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in ["torch", "transformers", "numpy", "pandas", "sklearn", "omegaconf", "mlflow", "PIL", "pyarrow"]:
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            pass
    return out


def gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    except Exception:
        pass
    return {"gpu_name": None}


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def main(argv: list[str] | None = None) -> None:
    exp_ids, overrides = parse_cli(sys.argv[1:] if argv is None else argv)
    cfg = compose(exp_ids, overrides)
    chain = stage_chain(cfg.stage)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{cfg.tag}" if cfg.tag else "")
    ctx = Context(cfg, run_id)
    modules = resolve(cfg, ctx, chain)

    status = {n: ("cached" if (ctx.stage_dirs[n] / DONE).exists() else "to run") for n in chain}
    print(f"run {run_id}  exp={dict(cfg.exp)}")
    for n in chain:
        print(f"  {n:9s} {ctx.stage_hashes[n]}  {status[n]:7s}  {ctx.stage_dirs[n].relative_to(ROOT)}")
    if cfg.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return

    import random

    import numpy as np

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        import torch

        torch.manual_seed(cfg.seed)
    except ImportError:
        pass

    t0 = time.time()
    summaries = {}
    for name in chain:
        out = ctx.stage_dirs[name]
        if (out / DONE).exists():
            meta = out / "stage_meta.json"
            summaries[name] = json.loads(meta.read_text()).get("summary", {}) if meta.exists() else {}
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] -> {out.relative_to(ROOT)}")
        ts = time.time()
        inputs = {u: ctx.stage_dirs[u] for u in modules[name].UPSTREAM}
        summary = modules[name].run(cfg, out, inputs, ctx) or {}
        stage_meta = {
            "stage": name, "hash": ctx.stage_hashes[name], "key": modules[name].key(cfg, ctx),
            "upstream": {u: ctx.stage_hashes[u] for u in modules[name].UPSTREAM},
            "run_id": run_id, "wall_s": round(time.time() - ts, 2), "summary": summary,
        }
        (out / "stage_meta.json").write_text(json.dumps(stage_meta, indent=2, default=str))
        (out / DONE).touch()
        summaries[name] = summary

    write_run_meta(cfg, ctx, chain, summaries, time.time() - t0, status)


def write_run_meta(cfg, ctx: Context, chain: list[str], summaries: dict, wall_s: float, status: dict) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    meta = {
        "run_id": ctx.run_id,
        "experiments": dict(cfg.exp),
        "stages": chain,
        "stage_hashes": ctx.stage_hashes,
        "stage_dirs": {k: str(v.relative_to(ROOT)) for k, v in ctx.stage_dirs.items()},
        "config": resolved,
        "data_provenance": resolved["data"]["provenance"],
        "seed": cfg.seed,
        "versions": versions(),
        **gpu_info(),
        "git": git_state(),
        "wall_s": round(wall_s, 2),
        "summaries": summaries,
    }
    if "answer" in chain:
        meta["answerer"] = {"checkpoint": cfg.answerer.checkpoint, "revision": ctx.answerer_revision,
                            "dtype": cfg.answerer.dtype, "attn": cfg.answerer.attn}
        meta["prompts"] = summaries.get("answer", {}).get("prompts")
    if any(s in chain for s in ("answer", "embed", "model")):
        meta["question_set"] = {"hash": ctx.qset_hash, "count": len(ctx.questions), "csv": cfg.questions.csv}

    run_dir = ROOT / cfg.paths.out_root / "runs" / ctx.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    from .docs import write_run_report

    report = write_run_report(meta, status, run_dir)
    log_mlflow(cfg, meta, summaries)
    log_markdown(cfg, meta, summaries, report)
    print(f"report   -> {report.relative_to(ROOT)}")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def log_mlflow(cfg, meta: dict, summaries: dict) -> None:
    import mlflow

    (ROOT / cfg.paths.mlflow_artifacts).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.paths.mlflow_uri.replace("sqlite:///", f"sqlite:///{ROOT}/"))
    name = "qaemb"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / cfg.paths.mlflow_artifacts).as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=meta["run_id"]):
        mlflow.set_tags({"experiments": ",".join(meta["experiments"].values()), "tag": cfg.tag,
                         **{f"hash.{k}": v for k, v in meta["stage_hashes"].items()}})
        params = {k: str(v)[:500] for k, v in _flat(meta["config"]).items()}
        mlflow.log_params(params)
        metrics = {f"{s}.{k}": v for s, summ in summaries.items() for k, v in _flat(summ).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if metrics:
            mlflow.log_metrics(metrics)


def log_markdown(cfg, meta: dict, summaries: dict, report: Path) -> None:
    path = ROOT / cfg.paths.experiments_log
    if not path.exists():
        path.write_text("# Experiments log\n\n| run_id | experiments | stage hashes | key metric | notes |\n|---|---|---|---|---|\n")
    hashes = " ".join(f"{k[:3]}:{v[:6]}" for k, v in meta["stage_hashes"].items())
    key = summaries.get("model", {}).get("key_metric") or summaries.get("answer", {}).get("key_metric") or ""
    exps = ",".join(meta["experiments"].values())
    with open(path, "a") as f:
        link = report.relative_to(path.parent).as_posix().replace(" ", "%20")
        f.write(f"| [{meta['run_id']}]({link}) | {exps} | {hashes} | {key} | {cfg.tag} |\n")


if __name__ == "__main__":
    main()
