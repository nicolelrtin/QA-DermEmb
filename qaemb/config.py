"""Config composition: conf/base.yaml + one experiment per group + CLI dotlist overrides.

Experiments live in conf/exp/<group>/<ID>.yaml; the group is the folder name. Two IDs from the
same group conflict and are rejected.
"""
import hashlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "conf"


def known_experiments() -> dict[str, str]:
    return {p.stem: p.parent.name for p in sorted((CONF / "exp").glob("*/*.yaml"))}


def parse_cli(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `+exp=[A,B] key=value ...` into (experiment IDs, dotlist overrides)."""
    exp_ids, overrides = [], []
    for arg in argv:
        if arg.startswith("+exp=") or arg.startswith("exp="):
            value = arg.split("=", 1)[1].strip().strip("[]")
            exp_ids += [e.strip() for e in value.split(",") if e.strip()]
        elif "=" in arg:
            overrides.append(arg)
        else:
            raise SystemExit(f"Cannot parse argument {arg!r}; expected key=value or +exp=[...]")
    return exp_ids, overrides


def compose(exp_ids: list[str], overrides: list[str]) -> DictConfig:
    known = known_experiments()
    cfg = OmegaConf.load(CONF / "base.yaml")
    groups: dict[str, str] = {}
    for exp_id in exp_ids:
        if exp_id not in known:
            raise SystemExit(f"Unknown experiment {exp_id!r}. Known: {', '.join(sorted(known))}")
        group = known[exp_id]
        if group in groups:
            raise SystemExit(f"{exp_id} and {groups[group]} are both in group '{group}'; pick one.")
        groups[group] = exp_id
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONF / "exp" / group / f"{exp_id}.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.exp = groups
    return cfg


def stable_hash(obj, n: int = 12) -> str:
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
