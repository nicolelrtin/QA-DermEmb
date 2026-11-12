"""Answer variability per original question ID (2026-10-30 Claude PoC): Claude replicate to replicate, and
stateless vs topic for each model, shown separately for each image.

variability_by_question.csv: per image x question, each model's Net_P per mode (Claude: mean of 4) and the changes
(|difference| averaged over run pairs) that the boxes show.
Every box is one value per question ID, |difference in Net_P| (0 = identical answer, 2 = certain and opposite).
Claude comparisons average over every pair of individual replicates, so they compare single runs, like the
one-run local models.

    poetry run python reports/2026-10-30_claude_poc/plot_variability.py
"""
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# local models are deterministic: one run each, read from compare_answers.csv (runs listed in README.md)
LOCAL = {"Qwen": "Qwen2.5-VL-7B", "SkinGPT-R1": "SkinGPT-R1"}
IMAGES = {"ISIC_0000147": "ISIC_0000147 (malignant)", "ISIC_0000215": "ISIC_0000215 (benign)"}
# per model: (first image, second image) = darker and lighter step of one hue (reference palette ramps)
COLORS = {"Qwen": ("#256abf", "#9ec5f4"), "SkinGPT-R1": ("#178f63", "#8fd9bc"), "Claude": ("#d95926", "#f5b596")}
INK, INK2, GRID, BG = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#ffffff"


def p_adj_table() -> pd.DataFrame:
    """Rows: image x question ID; columns: (model, mode, replicate)."""
    cols = {}
    for mode in ("stateless", "topic"):
        raw = pd.read_csv(HERE / f"replicates_claude_{mode}" / "replicates_raw.csv")
        for rep, g in raw.groupby("replicate"):
            cols[("Claude", mode, rep)] = g.set_index(["image_id", "question_id"]).p_adj
    d = pd.read_csv(HERE / "compare" / "compare_answers.csv").set_index(["image_id", "question_id"])
    for (model, name), mode in product(LOCAL.items(), ("stateless", "topic")):
        cols[(model, mode, "-")] = d[f"{name} | {mode} | Net_P"]
    return pd.DataFrame(cols)


def mean_abs_diff(t, a_cols, b_cols=None) -> pd.Series:
    """Per image x question ID: |p_adj difference| averaged over the given column pairs."""
    pairs = list(combinations(a_cols, 2)) if b_cols is None else list(product(a_cols, b_cols))
    return sum((t[a] - t[b]).abs() for a, b in pairs) / len(pairs)


def boxes(ax, groups):
    """groups: [(x label, model, per-image-x-question Series)]; two adjacent boxes per group, one per image."""
    for i, (label, model, v) in enumerate(groups):
        for j, img in enumerate(IMAGES):
            c = COLORS[model][j]
            ax.boxplot(v.xs(img, level="image_id"), positions=[i + (j - 0.5) * 0.34], widths=0.3, whis=(5, 95), showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=c, edgecolor=c, lw=0),
                       medianprops=dict(color="white", lw=2), whiskerprops=dict(color=c, lw=1.4),
                       capprops=dict(color=c, lw=1.4))
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK2, labelsize=10)
    ax.tick_params(axis="x", colors=INK, labelsize=10.5, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def readable_columns(score: str, t: pd.DataFrame) -> dict:
    """CSV columns for one score: each model's value per mode (Claude: mean of its replicates) next to the
    changes, named so that values and changes cannot be confused."""
    runs = lambda model, mode: [c for c in t.columns if c[:2] == (model, mode)]
    out = {}
    for model in ("Claude", "Qwen", "SkinGPT-R1"):
        n = len(runs(model, "stateless"))
        for mode in ("stateless", "topic"):
            out[f"{score} | {model} {mode} | value" + (f" (mean of {n})" if n > 1 else "")] = t[runs(model, mode)].mean(axis=1)
        out[f"{score} | {model} | change, stateless vs topic"] = mean_abs_diff(t, runs(model, "stateless"), runs(model, "topic"))
        if n > 1:
            for mode in ("stateless", "topic"):
                out[f"{score} | {model} {mode} | change between replicates"] = mean_abs_diff(t, runs(model, mode))
    return out


def highlight_changes(ax, v: pd.Series, x: float) -> None:
    """Overlay Claude's flagged questions (claude_stateless_vs_topic.csv, from claude_mode_changes.py) on its
    stateless-vs-topic boxes: filled = verdict changed, open = large shift with the same verdict."""
    flags = pd.read_csv(HERE / "claude_stateless_vs_topic.csv").set_index(["image_id", "question_id"]).flag
    rng = np.random.default_rng(0)
    for j, img in enumerate(IMAGES):
        for flag, style in (("large shift, same verdict", dict(facecolor="white", edgecolor="#c0392b", lw=0.8, s=12)),
                            ("verdict changed", dict(color="#c0392b", lw=0, s=12))):
            keys = flags[(flags == flag)].xs(img, level="image_id").index
            y = v.xs(img, level="image_id").reindex(keys)
            xs = x + (j - 0.5) * 0.34 + rng.uniform(-0.12, 0.12, len(y))
            ax.scatter(xs, y, zorder=4, alpha=0.85, label=f"Claude: {flag}" if j == 0 else None, **style)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)


def main():
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK})
    t = p_adj_table()
    runs = lambda model, mode: [c for c in t.columns if c[:2] == (model, mode)]
    reps = [(f"Claude\n{mode}", "Claude", mean_abs_diff(t, runs("Claude", mode))) for mode in ("stateless", "topic")]
    modes = [(m, m, mean_abs_diff(t, runs(m, "stateless"), runs(m, "topic"))) for m in ("Qwen", "SkinGPT-R1", "Claude")]
    pd.DataFrame(readable_columns("Net_P", t)).round(4).to_csv(HERE / "variability_by_question.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True, gridspec_kw={"width_ratios": [2, 3], "wspace": 0.08})
    fig.patch.set_facecolor(BG)
    boxes(a, reps)
    boxes(b, modes)
    highlight_changes(b, modes[2][2], x=2)
    a.set_ylim(0, 1.6)
    a.set_ylabel("Change in Net_P per question ID\n0 = same answer, 2 = certain and opposite", color=INK2, fontsize=10)
    a.set_title("Claude: replicate to replicate", loc="left", fontsize=11.5, color=INK, pad=10)
    b.set_title("Stateless vs topic mode", loc="left", fontsize=11.5, color=INK, pad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in ("#5f5e5a", "#c3c2b7")]
    a.legend(handles, [f"darker: {IMAGES[i]}" if j == 0 else f"lighter: {IMAGES[i]}" for j, i in enumerate(IMAGES)],
             title="Each pair of boxes, per image", alignment="left", loc="upper right", frameon=False, fontsize=9.5,
             title_fontsize=9.5, labelcolor=INK2)
    fig.suptitle("How much do answers change between Claude replicates and between answer modes?",
                 x=0.075, y=1.06, ha="left", fontsize=13.5, weight="bold", color=INK)
    fig.text(0.075, 0.975, "500 question pairs (Q0_all), whole image (G1). Box = middle half of the 500 question IDs, "
             "white line = median, whiskers = 5th to 95th percentile.\n"
             "Red points: Claude questions whose answer changes between stateless and topic "
             "(list: claude_stateless_vs_topic.xlsx).", fontsize=9.5, color=INK2)
    fig.savefig(HERE / "variability_by_question.png", dpi=170, bbox_inches="tight", facecolor=BG)


if __name__ == "__main__":
    main()
