"""2D UMAP of question embeddings, one panel per model, coloured by question set.

Run after embed_questions.py:
  python "Question Embeddings/plot_umap.py"

UMAP is fitted on all four sets together (per model), so within a panel the sets share
one map and gaps/overlaps are comparable. Axes are unitless; distances between panels
are not comparable. Writes, per model, outputs/umap_<model>.png (all sets overlaid) and
outputs/umap_<model>_by_set.png (one panel per set highlighted over the others in gray,
the clearer view of each set's coverage), plus outputs/umap_all.png (models side by side)
and outputs/umap_coords.csv (the same points as a table).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_TITLES = {
    "qwen3": "Qwen3-Embedding (general SOTA)",
    "panderm": "DermLIP-PanDerm (text tower)",
    "skingpt": "SkinGPT-R1 (LLM hidden states)",
}
# Categorical slots 1-4 (validated for CVD separation); marker shape is the secondary encoding.
SET_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]
SURFACE, TEXT, TEXT_2, CONTEXT_GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"


def style_axes(ax, title: str):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(CONTEXT_GRAY)


def draw_panel(ax, coords: np.ndarray, sets: pd.Series, title: str):
    for (label, (color, marker)) in zip(sets.unique(), SET_STYLE):
        m = (sets == label).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=16, c=color, marker=marker, alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.4, label=f"{label} (n={m.sum()})")
    style_axes(ax, title)


def draw_by_set(coords: np.ndarray, sets: pd.Series, title: str, path: Path):
    labels = sets.unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4.4), facecolor=SURFACE, squeeze=False)
    for ax, label, (color, marker) in zip(axes[0], labels, SET_STYLE):
        m = (sets == label).to_numpy()
        ax.scatter(coords[~m, 0], coords[~m, 1], s=8, c=CONTEXT_GRAY, linewidths=0)
        ax.scatter(coords[m, 0], coords[m, 1], s=16, c=color, marker=marker, alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.4)
        style_axes(ax, f"{label} (n={m.sum()})")
    fig.suptitle(f"{title}: each set highlighted over the others", color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = pd.read_csv(args.out / "questions.csv")
    models = [m for m in MODEL_TITLES if (args.out / f"emb_{m}.npy").exists()]
    if not models:
        raise SystemExit(f"No emb_*.npy in {args.out}; run embed_questions.py first.")

    table = questions[["set", "qid", "section", "question"]].copy()
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5), facecolor=SURFACE, squeeze=False)
    for ax, name in zip(axes[0], models):
        emb = np.load(args.out / f"emb_{name}.npy")
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="cosine", random_state=args.seed)
        coords = reducer.fit_transform(emb)
        table[f"{name}_x"], table[f"{name}_y"] = coords[:, 0], coords[:, 1]

        draw_panel(ax, coords, questions["set"], MODEL_TITLES[name])
        solo, solo_ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
        draw_panel(solo_ax, coords, questions["set"], MODEL_TITLES[name])
        solo_ax.legend(frameon=False, labelcolor=TEXT_2, fontsize=9, loc="best")
        solo.tight_layout()
        solo.savefig(args.out / f"umap_{name}.png", dpi=200)
        plt.close(solo)
        draw_by_set(coords, questions["set"], MODEL_TITLES[name], args.out / f"umap_{name}_by_set.png")
        print(f"saved umap_{name}.png and umap_{name}_by_set.png")

    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = len(labels) if len(models) >= 3 else 2
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, labelcolor=TEXT_2, fontsize=9)
    fig.suptitle("Question-set coverage (UMAP)", color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.06 if ncol == len(labels) else 0.1, 1, 0.97))
    fig.savefig(args.out / "umap_all.png", dpi=200)
    table.to_csv(args.out / "umap_coords.csv", index=False)
    print("saved umap_all.png and umap_coords.csv")


if __name__ == "__main__":
    main()
