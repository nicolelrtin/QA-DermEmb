"""Figures for S6. Every comparison of two quantities on the same scale carries the y = x line.

Colours: the first three slots of the validated reference palette (safe for all pairs under colour-vision
deficiency), one per feature source; neutrals for text, grid and the y = x line. Small multiples instead of more hues.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 9, "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def _identity(ax, lo=0.0, hi=1.0):
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=REF, zorder=0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def reliability(p, hit, bins: int = 10):
    """Binned (mean predicted, observed rate, n); empty bins dropped."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [(p[idx == b].mean(), hit[idx == b].mean(), (idx == b).sum()) for b in range(bins) if (idx == b).any()]
    return np.array(rows).T if rows else np.zeros((3, 0))


def calibration_grid(curves: dict, models: list, sources: list, title: str, xlabel: str, ylabel: str, path):
    """curves[(model, source)] = (pred, observed) arrays pooled over repeats. One panel per model, one line per source."""
    n = len(models)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.5 * rows + 0.4), squeeze=False)
    for ax, m in zip(axes.flat, models):
        _identity(ax)
        for s, color in zip(sources, SERIES):
            if (m, s) not in curves:
                continue
            x, y, cnt = reliability(*curves[(m, s)])
            ax.plot(x, y, color=color, lw=2, marker="o", ms=4, label=s)
        ax.set_title(m)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paired_grid(panels: list, title: str, xlabel: str, ylabel: str, path, lo=None, hi=None, ncols: int = 5,
                highlight_label: str | None = None):
    """panels: [(panel title, x, y, highlight mask or None)]; y = x in every panel."""
    n = len(panels)
    cols = min(ncols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows + 0.5), squeeze=False)
    for ax, (t, x, y, hl) in zip(axes.flat, panels):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        a = lo if lo is not None else min(x[ok].min(), y[ok].min()) if ok.any() else 0
        b = hi if hi is not None else max(x[ok].max(), y[ok].max()) if ok.any() else 1
        pad = 0.03 * (b - a or 1)
        _identity(ax, a - pad, b + pad)
        base = ~hl if hl is not None else np.ones(len(x), bool)
        ax.scatter(x[base], y[base], s=10, color=SERIES[0], alpha=0.55, linewidths=0)
        if hl is not None and hl.any():
            ax.scatter(x[hl], y[hl], s=14, color=SERIES[1], alpha=0.9, linewidths=0, label=highlight_label)
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if highlight_label:
        axes.flat[0].legend(loc="upper left", fontsize=7)
    fig.supxlabel(xlabel, color=INK, fontsize=9)
    fig.supylabel(ylabel, color=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion(cm, classes: list, title: str, path):
    """Row-normalised confusion matrix (rows = true class), counts annotated."""
    cm = np.asarray(cm, float)
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    k = len(classes)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * k + 2.2, 0.8 + 0.5 * k + 1.2))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(k), classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(k), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strips(groups: dict, models: list, title: str, ylabel: str, path):
    """groups[model] = {group name: values}; one panel per model, one box per group."""
    n = len(models)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows + 0.4), squeeze=False)
    for ax, m in zip(axes.flat, models):
        g = groups[m]
        ax.boxplot(list(g.values()), widths=0.55, showfliers=False,
                   medianprops={"color": SERIES[1], "lw": 2}, boxprops={"color": INK2}, whiskerprops={"color": INK2},
                   capprops={"color": INK2})
        for i, v in enumerate(g.values(), 1):
            jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(i + jit, v, s=5, color=SERIES[0], alpha=0.35, linewidths=0)
        ax.set_xticks(range(1, len(g) + 1), list(g), rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(m)
        ax.set_ylabel(ylabel)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# S7 (question importance) figures.
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # blue ramp, light->dark


def dumbbell(labels: list, a, b, names: tuple, title: str, xlabel: str, path):
    """One row per item (top row first): value a and value b joined by a line, e.g. share yes in two classes."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 0.32 * n + 1.4))
    y = np.arange(n)[::-1]
    for yi, ai, bi in zip(y, a, b):
        ax.plot([ai, bi], [yi, yi], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter(a, y, s=40, color=SERIES[0], zorder=2, label=names[0], edgecolors="white", linewidths=1)
    ax.scatter(b, y, s=40, color=SERIES[1], zorder=3, label=names[1], edgecolors="white", linewidths=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, max(1e-3, float(np.nanmax(np.r_[a, b]))) * 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right", ncols=2)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def share_bars(labels: list, share, expected, title: str, xlabel: str, path):
    """Horizontal bars (largest first) with a tick at the share expected from the group's size."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 0.32 * n + 1.4))
    y = np.arange(n)[::-1]
    ax.barh(y, share, height=0.62, color=SERIES[0], label="share of importance")
    ax.scatter(expected, y, marker="|", s=180, color=INK, linewidths=2, zorder=3, label="share of the questions")
    ax.set_yticks(y, labels, fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heat(rows: list, cols: list, M, boxed: set, title: str, path):
    """Values from 0 to the largest on a one-hue sequential scale, each cell labelled; `boxed` = {(row, col)} outlined."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 6, 0.3 * len(rows) + 2.6))
    vmax = max(float(np.nanmax(M)), 1e-6)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.55 * vmax else INK)
    for i, j in boxed:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SERIES[1], lw=2))
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
