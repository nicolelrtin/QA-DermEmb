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


# S8 tune --------------------------------------------------------------------------------------------------------
SOURCE_COLOURS = {"orig": SERIES[0], "dir": SERIES[1], "coh": SERIES[2]}   # same slots as the S6 figures


def penalty_curves(curves, sel, tasks, sources, primary, names, title, path):
    """Columns = tasks. Rows: inner log loss, inner primary score, outer primary score (reference only), against
    lambda = 1/C. Line = mean over the outer folds, band = 10th to 90th percentile over folds. Ticks at the bottom of
    each panel: the lambda each outer fold chose by inner log loss (S6's rule), one row of ticks per source."""
    rows = [("ll_mean", "inner log loss (lower is better)"), ("prim_mean", "inner {}"), ("outer_{}", "outer {} (reference)")]
    fig, axes = plt.subplots(3, len(tasks), figsize=(4.2 * len(tasks), 8.4), squeeze=False, sharex=True)
    for j, t in enumerate(tasks):
        pm = primary(t)
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            col = col.format(pm)
            for k, s in enumerate(sources):
                g = curves[(curves.task == t) & (curves.source == s)]
                if g.empty:
                    continue
                q = g.groupby("lambda")[col].describe(percentiles=[0.1, 0.9]).sort_index()
                c = SOURCE_COLOURS.get(s, SERIES[k % 3])
                ax.fill_between(q.index, q["10%"], q["90%"], color=c, alpha=0.15, lw=0)
                ax.plot(q.index, q["mean"], color=c, lw=2, label=names.get(s, s))
                ib = q["mean"].idxmin() if col == "ll_mean" else q["mean"].idxmax()
                ax.plot([ib], [q["mean"][ib]], marker="o", ms=5, color=c)
                if i == 2:
                    ch = sel[(sel.task == t) & (sel.source == s) & (sel.rule == "logloss")]["lambda"]
                    lo, hi = ax.get_ylim()
                    ax.plot(ch, np.full(len(ch), lo + (0.02 + 0.04 * k) * (hi - lo)), "|", ms=8, color=c, alpha=0.6)
            ax.set_xscale("log")
            ax.set_ylabel(lab.format(pm.replace("_", " ")))
            if i == 0:
                ax.set_title(t)
            if i == 2:
                ax.set_xlabel("lambda = 1/C (stronger penalty to the right)")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chosen_lambda(sel, tasks, sources, rules, names, title, path):
    """One panel per (task, source): for each selection rule, how many of the outer folds chose each lambda."""
    fig, axes = plt.subplots(len(tasks), len(sources), figsize=(4.3 * len(sources), 2.6 * len(tasks) + 0.4),
                             squeeze=False, sharex=True)
    for i, t in enumerate(tasks):
        for j, s in enumerate(sources):
            ax = axes[i, j]
            c = SOURCE_COLOURS.get(s, SERIES[j % 3])
            for k, r in enumerate(rules):
                g = sel[(sel.task == t) & (sel.source == s) & (sel.rule == r)]["lambda"].round(6).value_counts()
                ax.scatter(g.index, np.full(len(g), k), s=12 * g.values, color=c, alpha=0.8, edgecolor="none")
                for lam, n in g.items():
                    ax.annotate(str(n), (lam, k), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
                                color=INK2)
            ax.set_yticks(range(len(rules)))
            ax.set_yticklabels(rules)
            ax.set_ylim(-0.6, len(rules) - 0.3)
            ax.set_xscale("log")
            ax.set_title(f"{t}, {names.get(s, s)}")
            if i == len(tasks) - 1:
                ax.set_xlabel("lambda chosen (number = outer folds)")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def repeat_spread(reps, tasks, sources, rules, primary, names, title, path):
    """One panel per (task, source): the pooled score of each of the repeats (dots) and their mean (bar), by rule."""
    fig, axes = plt.subplots(len(tasks), len(sources), figsize=(4.3 * len(sources), 2.6 * len(tasks) + 0.4),
                             squeeze=False)
    for i, t in enumerate(tasks):
        pm = primary(t)
        for j, s in enumerate(sources):
            ax = axes[i, j]
            c = SOURCE_COLOURS.get(s, SERIES[j % 3])
            for k, r in enumerate(rules):
                v = reps[(reps.task == t) & (reps.source == s) & (reps.rule == r)][pm].values
                ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=c, edgecolor="none")
                ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=INK, lw=1.5)
            ax.set_xticks(range(len(rules)))
            ax.set_xticklabels(rules)
            ax.set_ylabel(pm.replace("_", " "))
            ax.set_title(f"{t}, {names.get(s, s)}")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# S9 explore -------------------------------------------------------------------------------------------------------
DIV_NEG, DIV_POS, DIV_MID = "#2a78d6", "#e34948", "#f0efec"


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bluegrayred", [DIV_NEG, DIV_MID, DIV_POS])


def explore_figures(fdir, coh_topic, coh_all, tva, pc, agr, ica, qsum, sub_order, kq, topic_names):
    import pandas as pd

    # 1. cohesion of each topic against random groups of the same size
    t = coh_topic.dropna(subset=["within_mean_r"]).sort_values("within_mean_r")
    fig, ax = plt.subplots(figsize=(8.5, 0.22 * len(t) + 1.4))
    yy = np.arange(len(t))
    ax.hlines(yy, t.null_05, t.null_95, color=GRID, lw=5, label="random groups of the same size (5th to 95th percentile)")
    ax.scatter(t.within_mean_r, yy, s=22, color=SERIES[0], zorder=3, label="mean r within the topic")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{str(n)[:55]} ({k})" for n, k in zip(t.topic, t.n_questions)], fontsize=7)
    ax.axvline(coh_all["between_mean_r"], color=REF, ls="--", lw=1, label="mean r between topics")
    ax.set_xlabel("mean correlation between questions (Directional Score, 1,000 images)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("How alike the questions of one topic are answered")
    fig.tight_layout()
    fig.savefig(fdir / "topic_cohesion.png", dpi=150)
    plt.close(fig)

    # 2. answer correlation against wording similarity
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (name, g), c in zip(tva.groupby("pairs", sort=False), SERIES):
        ax.errorbar(g.text_sim_mid, g.mean_r, yerr=1.96 * g.se, color=c, lw=2, marker="o", ms=4, capsize=2, label=name)
    ax.set_xlabel("wording similarity of the pair (cosine, bin median)")
    ax.set_ylabel("mean answer correlation")
    ax.legend(fontsize=8)
    ax.set_title("Question pairs: answered alike vs worded alike")
    fig.tight_layout()
    fig.savefig(fdir / "text_vs_answers.png", dpi=150)
    plt.close(fig)

    # 3. scree with parallel analysis
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    k = np.arange(1, 51)
    tot = pc["eigen"].sum()
    ax.plot(k, pc["ratio"][:50], color=SERIES[0], lw=2, marker="o", ms=3, label="answers")
    ax.plot(k, pc["null95"] / tot, color=REF, lw=1.5, ls="--", label="answers shuffled across images (95th pct)")
    ax.axvline(pc["n_parallel"] + 0.5, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("principal component")
    ax.set_ylabel("share of variance")
    ax.legend(fontsize=8)
    ax.set_title(f"{pc['n_parallel']} components beat shuffled answers; {pc['n_for'][0.9]} needed for 90%")
    fig.tight_layout()
    fig.savefig(fdir / "scree.png", dpi=150)
    plt.close(fig)

    # 4. agreement of question groupings against k
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    pairs = [("answer clusters", "topic"), ("answer clusters", agr.b[agr.b.str.startswith("text")].iloc[0]),
             (agr.a[agr.a.str.startswith("text")].iloc[0], "topic")]
    for (a, b), c in zip(pairs, SERIES):
        g = agr[(agr.a == a) & (agr.b == b)]
        ax.plot(g.k, g.ami, color=c, lw=2, marker="o", ms=4, label=f"{a} vs {b}")
        ax.plot(g.k, g.ami_null95, color=c, lw=1, ls=":")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("adjusted mutual information")
    ax.legend(fontsize=7)
    ax.set_title("Agreement between question groupings (dotted: shuffled 95th pct)")
    fig.tight_layout()
    fig.savefig(fdir / "question_grouping_agreement.png", dpi=150)
    plt.close(fig)

    # 5. image clusters against variables
    h = ica.pivot_table(index="variable", columns="k", values="ami", sort=False)
    fig, ax = plt.subplots(figsize=(0.55 * h.shape[1] + 2.6, 0.42 * h.shape[0] + 1.2))
    im = ax.imshow(h.values, cmap="Blues", vmin=0, vmax=max(0.3, np.nanmax(h.values)), aspect="auto")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.2 else INK)
    ax.set_xticks(range(h.shape[1]))
    ax.set_xticklabels(h.columns)
    ax.set_yticks(range(h.shape[0]))
    ax.set_yticklabels(h.index)
    ax.set_xlabel("number of image clusters")
    ax.grid(False)
    ax.set_title("Image clusters vs each variable (adjusted mutual information)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fdir / "image_clusters_vs_variables.png", dpi=150)
    plt.close(fig)

    # 6. question clusters by subtype
    M = np.array([d["prof"] for d in qsum[kq]])
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(M) + 2.6))
    lim = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=_div_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(sub_order)))
    ax.set_xticklabels(sub_order, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{c + 1} ({d['n']} q, AUROC {d['auroc']:.2f})" for c, d in enumerate(qsum[kq])], fontsize=7)
    ax.grid(False)
    ax.set_title(f"{kq} question clusters: mean cluster score by subtype (red above average, blue below)")
    fig.tight_layout()
    fig.savefig(fdir / "question_clusters_by_subtype.png", dpi=150)
    plt.close(fig)


# S10 ridge --------------------------------------------------------------------------------------------------------
def ridge_sets_figure(summ, path, n_all):
    """Score against number of questions: cluster representatives, principal components, random questions (mean of
    draws, bar = range over draws), all questions. Panels: benign vs malignant (all images, dermoscopic only), 10-class."""
    panels = [("s1_2c", "all images", "AUROC, benign vs malignant"), ("s1_2c", "dermoscopic only", "AUROC, dermoscopic images only"),
              ("flat_10c", "all images", "balanced accuracy, 10 subtypes")]
    panels = [p for p in panels if ((summ.task == p[0]) & (summ.subset == p[1])).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.8), squeeze=False)
    fams = [("lasso", "chosen jointly by the lasso (in-fold)", SERIES[0]), ("reps", "most typical question per cluster", SERIES[1]),
            ("rand", "random questions", SERIES[2]), ("top", "best-separating one by one (in-fold)", INK2), ("pca", "principal components (all asked)", REF)]
    for ax, (t, sub, title) in zip(axes[0], panels):
        s = summ[(summ.task == t) & (summ.subset == sub) & (summ.grouping == "patient")]
        for fam, lab, c in fams:
            g = s[s.set.str.startswith(fam)].copy()
            if g.empty:
                continue
            g["k"] = g.set.str.extract(r"(\d+)").astype(int)
            g = g.sort_values("k")
            ax.errorbar(g.k, g.fold_mean, yerr=[g.fold_mean - g.draw_min, g.draw_max - g.fold_mean] if fam == "rand" else None,
                        color=c, lw=2, marker="o", ms=5, capsize=3, label=lab, ls=":" if fam in ("pca", "top") else "-")
        a = s[s.set == "all"]
        if len(a):
            ax.axhline(a.fold_mean.iloc[0], color=INK2, ls="--", lw=1.2, label=f"all {n_all} questions")
        cq = s[s.set == "control question"]
        if len(cq):
            ax.axhline(cq.fold_mean.iloc[0], color=REF, ls=":", lw=1.2, label='Sonnet asked "malignant?"')
        ax.set_xlabel("number of features")
        ax.set_title(title)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Ridge, nested CV grouped by patient (mean over 25 outer folds)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pair_roc(curves: dict, names: dict, aucs: dict, sens: float, title: str, path):
    """ROC curves of several scores on the same images; dashed line at the target sensitivity."""
    cols = {"pair": SERIES[0], "tenclass": SERIES[1], "twoclass": SERIES[2], "direct": REF}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    _identity(ax)
    for k, pts in curves.items():
        a = np.array(pts)
        ax.plot(a[:, 0], a[:, 1], color=cols.get(k, INK2), lw=2, ls=":" if k == "direct" else "-",
                label=f"{names[k]} (AUROC {aucs[k]:.2f})")
    ax.axhline(sens, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("1 − specificity (naevi called melanoma)")
    ax.set_ylabel("sensitivity (melanomas caught)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# S12 (benchmarks) figures. One colour per kind of method: questions orange (the Directional Score slot of S6), SkinGPT-R1
# embeddings blue, SkinGPT-R1 diagnosis green, Sonnet's direct answer grey.
def _bench_colour(m: str) -> str:
    if m == "questions":
        return SERIES[1]
    if m.startswith("skingpt_emb"):
        return SERIES[0]
    if m.startswith("skingpt_diagnosis"):
        return SERIES[2]
    return REF


def bench_figure(summ, methods: list, name, path):
    """Fold mean ± sd over the 25 outer folds for each method; panels: benign vs malignant (all images, dermoscopic
    only), 10 subtypes. The questions' mean is drawn as a vertical line across each panel."""
    panels = [("s1_2c", "all images", "AUROC, benign vs malignant"), ("s1_2c", "dermoscopic only", "AUROC, dermoscopic images only"),
              ("flat_10c", "all images", "balanced accuracy, 10 subtypes")]
    panels = [p for p in panels if ((summ.task == p[0]) & (summ.subset == p[1])).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 2.6, 0.42 * len(methods) + 1.3), squeeze=False,
                             sharey=True)
    ys = np.arange(len(methods))[::-1]
    for i, (ax, (t, sub, title)) in enumerate(zip(axes[0], panels)):
        s = summ[(summ.task == t) & (summ.subset == sub)].set_index("set")
        for y, m in zip(ys, methods):
            if m not in s.index:
                continue
            x = s.loc[m]
            c = _bench_colour(m)
            ax.errorbar(x.fold_mean, y, xerr=x.fold_sd, color=c, lw=2, marker="o", ms=6, capsize=3)
            ax.annotate(f"{x.fold_mean:.3f}", (x.fold_mean, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        if "questions" in s.index:
            ax.axvline(s.loc["questions"].fold_mean, color=SERIES[1], ls="--", lw=1, zorder=0)
        if t != "s1_2c":
            K = 10
            ax.axvline(1 / K, color=REF, ls=":", lw=1, zorder=0)
        else:
            ax.axvline(0.5, color=REF, ls=":", lw=1, zorder=0)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        if i == 0:
            ax.set_yticks(ys, [name(m) for m in methods], fontsize=8)
    fig.suptitle("Same 25 outer folds (patient-grouped); mean ± sd over folds; dotted line = chance", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bench_roc(roc: dict, name, sens: float, path):
    """Pooled out-of-fold ROC curves (first repeat) for benign vs malignant."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    _identity(ax)
    for m, (y, p) in roc.items():
        fpr, tpr, _ = roc_curve(y, p)
        c = INK2 if m == "skingpt_diagnosis" else _bench_colour(m)     # its own colour next to the fitted readout
        ax.plot(fpr, tpr, color=c, lw=2, ls=":" if m == "control question" else "-", label=f"{name(m)} ({roc_auc_score(y, p):.3f})")
    ax.axhline(sens, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("1 − specificity (benign lesions called malignant)")
    ax.set_ylabel("sensitivity (malignant lesions caught)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(fontsize=6.5, loc="lower right", title="pooled AUROC, repeat 1", title_fontsize=7)
    ax.set_title("Benign vs malignant, out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
