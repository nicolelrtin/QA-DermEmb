"""Figures and summary table for the 2026-10-30 letter (letter.md): coherence, variability, order and confidence
of yes/no answers from Qwen2.5-VL-7B, SkinGPT-R1 and Claude Opus 5.5 on ISIC_0000147 and ISIC_0000215.

    poetry run python reports/2026-10-30_claude_poc/make_figures.py

Reads each run's embed/answers.parquet (one row per image x question x original/negated), writes
figures/fig{1,2,3}.{pdf,png}, table1.csv and numbers.json (every number quoted in the letter).
"""
import json
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RUNS = ROOT / "outputs" / "qaemb" / "runs"
FIG = HERE / "figures"
IMAGES = ["ISIC_0000147", "ISIC_0000215"]
IMG_LABEL = {"ISIC_0000147": "ISIC_0000147 (melanoma)", "ISIC_0000215": "ISIC_0000215 (benign)"}
MODELS = ["Qwen", "SkinGPT-R1", "Claude"]
ANSWERER = {"ANS_qwen7b": "Qwen", "ANS_skingpt_r1": "SkinGPT-R1", "ANS_claude_opus_cli": "Claude"}
CONDS = ["stateless", "topic", "topic, shuffled"]
MODE = {"M_stateless": "stateless", "M_topic": "topic", "M_topic_shuf": "topic, shuffled"}
# one hue per model; darker step = first image, lighter step = second image
COL = {"Qwen": ("#256abf", "#9ec5f4"), "SkinGPT-R1": ("#178f63", "#8fd9bc"), "Claude": ("#d95926", "#f5b596")}
VERDICTS = ["yes", "no", "contradiction: no to both", "contradiction: yes to both"]
VCOL = {"yes": "#3d3d3a", "no": "#bcbbb4", "contradiction: no to both": "#d13c3b", "contradiction: yes to both": "#f2a09f"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e1"
FIXED = {  # deterministic local models, one run each (10-image runs; only the two images above are used)
    ("Qwen", "stateless"): ["20261029-152402_answerers-oct29"], ("Qwen", "topic"): ["20261029-152411_answerers-oct29"],
    ("SkinGPT-R1", "stateless"): ["20261029-152430_answerers-oct29"], ("SkinGPT-R1", "topic"): ["20261029-152656_answerers-oct29"]}
TAGS = ("claude-cli-poc", "local-shuffle", "claude-shuffle")

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42, "savefig.dpi": 300})


def run_list() -> dict:
    """(model, condition) -> run ids: fixed local runs + every replicate run tagged for this experiment."""
    runs = {k: list(v) for k, v in FIXED.items()}
    found = {}
    for d in sorted(RUNS.glob("2026103*")):
        if not d.name.endswith(TAGS) or not (d / "run_meta.json").exists():
            continue
        e = json.loads((d / "run_meta.json").read_text())["experiments"]
        if e["data"] == "D0_poc2" and e.get("mode") in MODE:
            found[(ANSWERER[e["answerer"]], MODE[e["mode"]], e.get("rep", "R1"))] = d.name  # latest run per replicate
    for (m, c, _), r in sorted(found.items()):
        runs.setdefault((m, c), []).append(r)
    return runs


def load(run_id: str) -> pd.DataFrame:
    """One row per image x question: P(yes) to original and negation, confidence, position in prompt, verdict."""
    meta = json.loads((RUNS / run_id / "run_meta.json").read_text())
    a = pd.read_parquet(ROOT / meta["stage_dirs"]["embed"] / "answers.parquet")
    a = a[(a.patch_id == -1) & a.image_id.isin(IMAGES)]
    o, c = (a[a.original == v].set_index(["image_id", "question_id"]) for v in (1, 0))
    d = pd.DataFrame({"p_o": o.p_yes, "p_c": c.p_yes, "conf_o": o.confidence, "conf_c": c.confidence,
                      "pos": o.position_in_batch / (o.batch_k - 1).clip(lower=1), "mass": np.minimum(o.yesno_mass, c.yesno_mass)})
    d["p_adj"] = d.p_o - d.p_c
    yo, yc = d.p_o >= 0.5, d.p_c >= 0.5
    d["verdict"] = np.select([yo & ~yc, ~yo & yc, ~yo & ~yc], VERDICTS[:3], VERDICTS[3])
    topics = pd.read_parquet(ROOT / meta["stage_dirs"]["embed"] / "questions.parquet").set_index("question_id").topic
    return d.reset_index().assign(run=run_id, topic=lambda x: x.question_id.map(topics))


def pair_mad(frames: list, other: list | None = None) -> pd.Series:
    """Per image x question: |difference in p_adj| averaged over run pairs (within frames, or frames x other)."""
    pairs = list(combinations(frames, 2)) if other is None else list(product(frames, other))
    return sum((a.p_adj - b.p_adj).abs() for a, b in pairs) / len(pairs)


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def letter(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom", ha="left")


def paired_boxes(ax, groups, ylab):
    """groups: [(label, model, Series indexed by image x question)] -> two adjacent boxes per group (one per image)."""
    for i, (_, model, v) in enumerate(groups):
        for j, img in enumerate(IMAGES):
            c = COL[model][j]
            ax.boxplot(v.xs(img, level="image_id"), positions=[i + (j - 0.5) * 0.36], widths=0.32, whis=(5, 95),
                       showfliers=False, patch_artist=True, boxprops=dict(facecolor=c, lw=0),
                       medianprops=dict(color="white", lw=1.3), whiskerprops=dict(color=c, lw=0.9), capprops=dict(color=c, lw=0.9))
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylab)
    ax.tick_params(axis="x", length=0)
    style(ax)


def image_legend(ax, loc="upper right", **kw):
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in ("#5f5e5a", "#c3c2b7")]
    ax.legend(h, [f"{IMG_LABEL[i]}" for i in IMAGES], loc=loc, frameon=False, handlelength=1, handleheight=0.8, **kw)


def fig1(data, runs, num):
    """Coherence: P(yes | question) vs P(yes | negation), and verdict shares per model x condition."""
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.55, wspace=0.32, left=0.075, right=0.98, top=0.9, bottom=0.17)
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i])
        ax.add_patch(plt.Rectangle((0, 0), 0.5, 0.5, fc=VCOL[VERDICTS[2]], alpha=0.10, lw=0))
        ax.add_patch(plt.Rectangle((0.5, 0.5), 0.5, 0.5, fc=VCOL[VERDICTS[3]], alpha=0.16, lw=0))
        for cond, mk, z in (("stateless", "o", 3), ("topic", "^", 2)):
            d = pd.concat(data[(m, cond)])
            ax.scatter(d.p_c, d.p_o, s=3.5, marker=mk, lw=0, alpha=0.35 if cond == "stateless" else 0.25,
                       color=COL[m][0] if cond == "stateless" else COL[m][1], zorder=z, rasterized=True)
        ax.plot([0, 1], [1, 0], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=4)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        ax.set_xlabel("P(yes | negated question)")
        if i == 0:
            ax.set_ylabel("P(yes | question)")
        n = len(data[(m, "stateless")])
        ax.set_title(f"{m}" + (f" ({n} replicates pooled)" if n > 1 else ""), loc="left", color=INK)
        for x, y, t in ((0.25, 0.25, "no to both"), (0.75, 0.75, "yes to both"), (0.25, 0.75, "yes"), (0.75, 0.25, "no")):
            ax.text(x, y, t, ha="center", va="center", fontsize=6, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        style(ax, grid=None)
        if i == 0:
            letter(ax, "a", x=-0.38)
    h = [plt.Line2D([], [], ls="", marker=mk, ms=3.5, color=c) for mk, c in (("o", "#5f5e5a"), ("^", "#c3c2b7"))]
    fig.legend(h, ["stateless prompts (darker)", "topic prompts (lighter)"], loc="upper left", bbox_to_anchor=(0.075, 1.0),
               ncol=2, frameon=False, handletextpad=0.2)

    ax = fig.add_subplot(gs[1, :3])
    rows = [(m, c) for m in MODELS for c in CONDS if (m, c) in data]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for v in VERDICTS:
        share = np.array([(pd.concat(data[r]).verdict == v).mean() for r in rows])
        ax.bar(x, share, bottom=bottom, width=0.72, color=VCOL[v], label=v, edgecolor="white", lw=0.5)
        bottom += share
    for xi, r in zip(x, rows):
        contra = (pd.concat(data[r]).verdict.str.startswith("contradiction")).mean()
        num.setdefault("contradiction_rate", {})[f"{r[0]} | {r[1]}"] = round(contra, 4)
        ax.text(xi, 1.015, f"{contra:.0%}", ha="center", va="bottom", fontsize=6.5, color=VCOL[VERDICTS[2]])
    ax.set_xticks(x, [c.replace(", ", ",\n") for _, c in rows])
    for m in MODELS:
        xm = np.mean([xi for xi, r in zip(x, rows) if r[0] == m])
        ax.text(xm, -0.25, m, ha="center", va="top", fontsize=7.5, weight="bold", color=COL[m][0],
                transform=ax.get_xaxis_transform())
    ax.set(ylim=(0, 1.12), xlim=(-0.6, len(rows) - 0.4), yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=["0", "25", "50", "75", "100"])
    ax.set_ylabel("Question pairs (%)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("Verdict per question pair (number above bar = contradiction rate)", loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=4, frameon=False, handlelength=1, handleheight=0.8)
    style(ax)
    letter(ax, "b", x=-0.075, y=1.08)
    fig.savefig(FIG / "fig1.pdf")
    fig.savefig(FIG / "fig1.png")


def fig2(data, runs, num):
    """Variability: between replicates (sampling and/or question order), stateless vs topic, verdict stability,
    and the effect of position within the prompt."""
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.62, wspace=0.28, left=0.08, right=0.98, top=0.95, bottom=0.1)
    rep_groups = [("Claude\nstateless\n[sampling]", "Claude", ("Claude", "stateless")),
                  ("Claude\ntopic\n[sampling]", "Claude", ("Claude", "topic")),
                  ("Claude\ntopic, shuffled\n[sampling + order]", "Claude", ("Claude", "topic, shuffled")),
                  ("Qwen\ntopic, shuffled\n[order]", "Qwen", ("Qwen", "topic, shuffled")),
                  ("SkinGPT-R1\ntopic, shuffled\n[order]", "SkinGPT-R1", ("SkinGPT-R1", "topic, shuffled"))]
    rep_groups = [g for g in rep_groups if len(data.get(g[2], [])) > 1]
    keyed = lambda frames: [f.set_index(["image_id", "question_id"]) for f in frames]
    ax = fig.add_subplot(gs[0, 0])
    groups = [(lab, m, pair_mad(keyed(data[k]))) for lab, m, k in rep_groups]
    paired_boxes(ax, groups, "|Δ p_adj| between replicates")
    ax.set_ylim(0, 1.0)
    ax.set_title("Replicate to replicate [source of variation]", loc="left")
    for (lab, _, v), (_, _, k) in zip(groups, rep_groups):
        num.setdefault("replicate_mad", {})[" | ".join(k)] = {i: round(v.xs(i, level="image_id").mean(), 4) for i in IMAGES}
    image_legend(ax, loc="upper left")
    letter(ax, "a", x=-0.13)

    ax = fig.add_subplot(gs[0, 1])
    groups = [(m, m, pair_mad(keyed(data[(m, "stateless")]), keyed(data[(m, "topic")]))) for m in MODELS]
    paired_boxes(ax, groups, "|Δ p_adj|, stateless vs topic")
    ax.set_ylim(0, 1.6)
    ax.set_title("Stateless vs topic prompt (fixed order)", loc="left")
    for lab, _, v in groups:
        num.setdefault("stateless_vs_topic_mad", {})[lab] = {i: round(v.xs(i, level="image_id").mean(), 4) for i in IMAGES}
    letter(ax, "b", x=-0.16)

    ax = fig.add_subplot(gs[1, 0])
    for i, (lab, m, k) in enumerate(rep_groups):
        fr = keyed(data[k])
        same = pd.concat([f.verdict for f in fr], axis=1).nunique(axis=1).eq(1)
        for j, img in enumerate(IMAGES):
            s = same.xs(img, level="image_id").mean()
            ax.bar(i + (j - 0.5) * 0.36, s, width=0.32, color=COL[m][j], lw=0)
            ax.text(i + (j - 0.5) * 0.36, s + 0.015, f"{s:.0%}", ha="center", va="bottom", fontsize=5.8, color=INK2)
            num.setdefault("verdict_stable_all_reps", {}).setdefault(" | ".join(k), {})[img] = round(s, 4)
    ax.set_xticks(range(len(rep_groups)), [g[0] for g in rep_groups])
    ax.set(ylim=(0, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=["0", "25", "50", "75", "100"], xlim=(-0.6, len(rep_groups) - 0.4))
    ax.set_ylabel("Pairs with the same verdict\nin all 4 replicates (%)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("Verdict stability across replicates", loc="left")
    style(ax)
    letter(ax, "c", x=-0.13)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 6)
    for m in MODELS:
        fr = data.get((m, "topic, shuffled"), [])
        if len(fr) < 2:
            continue
        d = pd.concat(fr)
        d["dev"] = d.p_o - d.groupby(["image_id", "question_id"]).p_o.transform("mean")
        d["bin"] = pd.cut(d.pos, bins, include_lowest=True, labels=False)
        g = d.groupby("bin").dev.agg(["mean", "sem"])
        xs = (bins[:-1] + bins[1:]) / 2
        ax.errorbar(xs[g.index], g["mean"], yerr=1.96 * g["sem"], color=COL[m][0], lw=1.2, marker="o", ms=3,
                    capsize=1.5, elinewidth=0.7, label=m)
        num.setdefault("position_effect_first_minus_last_bin", {})[m] = round(g["mean"].iloc[0] - g["mean"].iloc[-1], 4)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set(xlim=(0, 1), xticks=[0, 0.5, 1], xticklabels=["first", "middle", "last"])
    ax.set_xlabel("Position of the question within its topic prompt (shuffled runs)")
    ax.set_ylabel("P(yes | question) minus that\nquestion's mean over replicates")
    ax.set_title("Order effect within a prompt (mean ± 95% CI)", loc="left")
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    letter(ax, "d", x=-0.16)
    fig.savefig(FIG / "fig2.pdf")
    fig.savefig(FIG / "fig2.png")


def fig3(data, runs, num):
    """Confidence: Claude's stated confidence vs stated P(yes), vs replicate variability and vs contradictions;
    the local models' implicit confidence (decisiveness |p_adj|) vs variability under shuffling."""
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.5, left=0.08, right=0.98, top=0.95, bottom=0.1)
    claude = pd.concat([pd.concat(data[("Claude", c)]).assign(cond=c) for c in CONDS if ("Claude", c) in data])

    ax = fig.add_subplot(gs[0, 0])
    p = np.r_[claude.p_o, claude.p_c]
    cf = np.r_[claude.conf_o, claude.conf_c]
    h, xe, ye = np.histogram2d(p, cf, bins=[np.linspace(-0.025, 1.025, 22), np.linspace(-0.025, 1.025, 22)])
    ax.imshow(np.log1p(h.T), origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Oranges", aspect="auto")
    ax.set(xlabel="Stated P(yes)", ylabel="Stated confidence", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    ax.set_title("Claude: stated confidence vs stated P(yes) (log count)", loc="left")
    num["claude_conf_mean"] = round(float(np.nanmean(cf)), 4)
    num["claude_conf_corr_with_extremeness"] = round(float(pd.Series(cf).corr(pd.Series(np.abs(p - 0.5)))), 4)
    style(ax, grid=None)
    letter(ax, "a", x=-0.14)

    ax = fig.add_subplot(gs[0, 1])
    for c, ls in (("stateless", "-"), ("topic", "--"), ("topic, shuffled", ":")):
        if len(data.get(("Claude", c), [])) < 2:
            continue
        fr = [f.set_index(["image_id", "question_id"]) for f in data[("Claude", c)]]
        conf = sum((f.conf_o + f.conf_c) / 2 for f in fr) / len(fr)
        d = pd.DataFrame({"conf": conf, "mad": pair_mad(fr)}).dropna()
        d["bin"] = pd.cut(d.conf, [0, 0.55, 0.65, 0.75, 0.85, 1.0])
        g = d.groupby("bin", observed=True).mad.agg(["mean", "sem", "size"])
        xs = [b.mid for b in g.index]
        ax.errorbar(xs, g["mean"], yerr=1.96 * g["sem"], color=COL["Claude"][0], ls=ls, lw=1.1, marker="o", ms=2.5,
                    capsize=1.5, elinewidth=0.6, label=c)
        num.setdefault("claude_conf_vs_replicate_mad_spearman", {})[c] = round(d.conf.corr(d.mad, method="spearman"), 4)
    ax.set(xlabel="Mean stated confidence (pair, binned)", ylabel="|Δ p_adj| between replicates", ylim=(0, None))
    ax.set_title("Claude: confidence vs replicate variability (mean ± 95% CI)", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=2)
    style(ax)
    letter(ax, "b", x=-0.16)

    ax = fig.add_subplot(gs[1, 0])
    claude["pair_conf"] = (claude.conf_o + claude.conf_c) / 2
    claude["kind"] = np.where(claude.verdict.str.startswith("contradiction"), "contradiction", "coherent")
    pos = 0
    ticks = []
    for c in [c for c in CONDS if c in set(claude.cond)]:
        for j, kd in enumerate(("coherent", "contradiction")):
            v = claude[(claude.cond == c) & (claude.kind == kd)].pair_conf.dropna()
            colr = COL["Claude"][0] if kd == "coherent" else VCOL[VERDICTS[2]]
            ax.boxplot(v, positions=[pos + j * 0.42], widths=0.36, whis=(5, 95), showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=colr, lw=0), medianprops=dict(color="white", lw=1.2),
                       whiskerprops=dict(color=colr, lw=0.8), capprops=dict(color=colr, lw=0.8))
            num.setdefault("claude_conf_by_kind", {}).setdefault(c, {})[kd] = {"median": round(v.median(), 3), "n": int(len(v))}
        ticks.append((pos + 0.21, c.replace(", ", ",\n")))
        pos += 1.2
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Mean stated confidence (pair)")
    ax.set_title("Claude: confidence for coherent vs contradictory pairs", loc="left")
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in (COL["Claude"][0], VCOL[VERDICTS[2]])]
    ax.legend(h, ["coherent", "contradiction"], frameon=False, loc="lower left", handlelength=1, handleheight=0.8)
    ax.set_ylim(0, 1.02)
    style(ax)
    letter(ax, "c", x=-0.14)

    ax = fig.add_subplot(gs[1, 1])
    for m in MODELS:
        fr = data.get((m, "topic, shuffled"), [])
        if len(fr) < 2:
            continue
        fr = [f.set_index(["image_id", "question_id"]) for f in fr]
        dec = (sum(f.p_adj for f in fr) / len(fr)).abs()
        d = pd.DataFrame({"dec": dec, "mad": pair_mad(fr)})
        d["bin"] = pd.cut(d.dec, np.linspace(0, 1, 6), include_lowest=True)
        g = d.groupby("bin", observed=True).mad.agg(["mean", "sem"])
        ax.errorbar([b.mid for b in g.index], g["mean"], yerr=1.96 * g["sem"], color=COL[m][0], lw=1.1, marker="o",
                    ms=2.5, capsize=1.5, elinewidth=0.6, label=m)
        num.setdefault("coherence_weight_vs_shuffle_mad_spearman", {})[m] = round(d.dec.corr(d.mad, method="spearman"), 4)
    ax.set(xlabel="Coherence weight w = |p_adj| (mean over replicates, binned)", ylabel="|Δ p_adj| between\nshuffled replicates",
           xlim=(0, 1), ylim=(0, 0.72))
    ax.set_title("Coherence weight vs variability under shuffling (mean ± 95% CI)", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=3)
    style(ax)
    letter(ax, "d", x=-0.16)
    fig.savefig(FIG / "fig3.pdf")
    fig.savefig(FIG / "fig3.png")


def claude_reference(data) -> pd.DataFrame:
    """Claude, stateless, mean of its 4 replicates: the reference answers for the agreement figures."""
    fr = [f.set_index(["image_id", "question_id"]) for f in data[("Claude", "stateless")]]
    ref = sum(f[["p_o", "p_c", "conf_o", "conf_c"]] for f in fr) / len(fr)
    ref["p_adj"] = ref.p_o - ref.p_c
    ref["verdict"] = verdicts(ref.p_o, ref.p_c)
    ref["conf"] = (ref.conf_o + ref.conf_c) / 2
    ref["topic"] = data[("Claude", "stateless")][0].set_index(["image_id", "question_id"]).topic
    return ref


def verdicts(p_o, p_c):
    yo, yc = p_o >= 0.5, p_c >= 0.5
    return pd.Series(np.select([yo & ~yc, ~yo & yc, ~yo & ~yc], VERDICTS[:3], VERDICTS[3]), index=p_o.index)


def kappa(x: pd.Series, y: pd.Series) -> float:
    """Cohen's kappa for two yes/no series: agreement beyond what the two yes-rates give by chance."""
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe)


def agree(a: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per image x question: same yes/no answer as Claude (P(yes) >= 0.5) to the question and to its negation."""
    return pd.DataFrame({"same_q": (a.p_o >= 0.5) == (ref.p_o >= 0.5), "same_neg": (a.p_c >= 0.5) == (ref.p_c >= 0.5)})


def yes_no_table(ax, x, ref_x, title, label, num_key, num):
    m = pd.crosstab(ref_x >= 0.5, x >= 0.5, normalize=True).reindex(index=[True, False], columns=[True, False], fill_value=0)
    ax.imshow(m.values, cmap="Greens", vmin=0, vmax=0.8)
    for (i, j), v in np.ndenumerate(m.values):
        ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7, color="white" if v > 0.45 else INK)
    ax.set_xticks([0, 1], ["yes", "no"])
    ax.set_yticks([0, 1], ["yes", "no"])
    ax.set(xlabel=f"SkinGPT-R1 (stateless)", ylabel="Claude")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    agree_ = np.trace(m.values)
    k = kappa(x >= 0.5, ref_x >= 0.5)
    ax.text(0.5, -0.5, f"agree on {agree_:.0%}, κ = {k:.2f}", transform=ax.transAxes, ha="center", fontsize=6.5, color=INK2)
    num[num_key] = {"table_pct_claude_rows_yes_no": (100 * m.values).round(1).tolist(), "agree": round(agree_, 4), "kappa": round(k, 4)}


def fig4(data, num):
    """SkinGPT-R1 against Claude (stateless, mean of 4 replicates), on P(yes | question) first; negations shown apart."""
    ref = claude_reference(data)
    sk = {c: [f.set_index(["image_id", "question_id"]) for f in data[("SkinGPT-R1", c)]] for c in CONDS}
    f = sk["stateless"][0]
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.1, 0.8, 0.8], hspace=0.75, wspace=0.55, left=0.07, right=0.98,
                          top=0.94, bottom=0.1)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    ax.hist(ref.p_o, bins, histtype="stepfilled", color=COL["Claude"][1], label="Claude (mean of 4)")
    for c, ls in (("stateless", "-"), ("topic", "--")):
        ax.hist(sk[c][0].p_o, bins, histtype="step", color=COL["SkinGPT-R1"][0], ls=ls, lw=1.1, label=f"SkinGPT-R1 {c}")
    ax.set(xlabel="P(yes | question)", ylabel="Question × image", xlim=(0, 1))
    ax.set_title("Answers to the questions", loc="left")
    ax.legend(frameon=False, loc="upper center", fontsize=6)
    for c in ("stateless", "topic"):
        p = sk[c][0].p_o
        num.setdefault("skingpt_extreme_share_q", {})[c] = round(float(((p < 0.05) | (p > 0.95)).mean()), 4)
    num["claude_extreme_share_q"] = round(float(((ref.p_o < 0.05) | (ref.p_o > 0.95)).mean()), 4)
    style(ax)
    letter(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[0, 1])
    for j, img in enumerate(IMAGES):
        ax.scatter(ref.p_o.xs(img, level="image_id"), f.p_o.xs(img, level="image_id"), s=3, lw=0, alpha=0.45,
                   color=COL["SkinGPT-R1"][j], rasterized=True)
    rho = ref.p_o.corr(f.p_o, method="spearman")
    num["claude_vs_skingpt_stateless_spearman_p_q"] = round(rho, 4)
    num["claude_vs_skingpt_stateless_spearman_p_neg"] = round(ref.p_c.corr(f.p_c, method="spearman"), 4)
    num["claude_vs_skingpt_stateless_spearman_p_w"] = round((ref.p_o * ref.p_adj.abs()).corr(f.p_o * f.p_adj.abs(), method="spearman"), 4)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal",
           xlabel="Claude P(yes | question)", ylabel="SkinGPT-R1 P(yes | question)")
    ax.text(0.97, 0.5, f"Spearman\nρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=5)
    ax.set_title("Per question (stateless)", loc="left")
    style(ax, grid=None)
    letter(ax, "b", x=-0.36)

    yes_no_table(fig.add_subplot(gs[0, 2]), f.p_o, ref.p_o, "Questions", "q", "yes_no_table_q", num)
    yes_no_table(fig.add_subplot(gs[0, 3]), f.p_c, ref.p_c, "Negated\nquestions", "neg", "yes_no_table_neg", num)
    fig.axes[-2].text(-0.55, 1.2, "c", transform=fig.axes[-2].transAxes, fontsize=9, weight="bold")

    ax = fig.add_subplot(gs[1, :])
    cats = [("questions", "p_o"), ("negated questions", "p_c")]
    shades = [COL["SkinGPT-R1"][0], COL["SkinGPT-R1"][1], "#c9ecdd"]
    for k, c in enumerate(CONDS):
        for i, (_, col) in enumerate(cats):
            per = [kappa(x[col] >= 0.5, ref[col] >= 0.5) for x in sk[c]]
            raw = np.mean([((x[col] >= 0.5) == (ref[col] >= 0.5)).mean() for x in sk[c]])
            x = i + (k - 1) * 0.26
            ax.bar(x, np.mean(per), width=0.24, color=shades[k], edgecolor=COL["SkinGPT-R1"][0], lw=0.6, label=c if i == 0 else None)
            if len(per) > 1:
                ax.scatter([x] * len(per), per, s=6, color=INK, zorder=3, lw=0)
            ax.text(x, max(max(per), 0) + 0.02, f"κ {np.mean(per):.2f}\n({raw:.0%} same)", ha="center", fontsize=5.8, color=INK2)
            num.setdefault("skingpt_agreement_with_claude", {}).setdefault(c, {})[col] = {
                "kappa": [round(p_, 4) for p_ in per], "raw_agree_mean": round(raw, 4)}
    qw = data[("Qwen", "stateless")][0].set_index(["image_id", "question_id"])
    for i, (_, col) in enumerate(cats):
        kq = kappa(qw[col] >= 0.5, ref[col] >= 0.5)
        ax.plot([i - 0.42, i + 0.42], [kq] * 2, color=COL["Qwen"][0], lw=0.9, ls=(0, (3, 2)))
        num.setdefault("qwen_stateless_agreement_with_claude", {})[col] = {
            "kappa": round(kq, 4), "raw_agree": round(((qw[col] >= 0.5) == (ref[col] >= 0.5)).mean(), 4)}
    ax.plot([], [], color=COL["Qwen"][0], lw=0.9, ls=(0, (3, 2)), label="Qwen stateless, for scale")
    ax.set_xticks(range(len(cats)), [c[0] for c in cats])
    ax.set(ylim=(min(0, ax.get_ylim()[0]), 0.75), xlim=(-0.6, 1.6))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("Agreement with Claude\n(Cohen's κ; 0 = chance)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("SkinGPT-R1 agreement with Claude's yes/no answers, beyond chance (dots: the 4 shuffled replicates)", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style(ax)
    letter(ax, "d", x=-0.075)
    fig.savefig(FIG / "fig4.pdf")
    fig.savefig(FIG / "fig4.png")


def fig5(data, num):
    """Where SkinGPT-R1 agrees with Claude on the questions: by topic, Claude's confidence, SkinGPT-R1's stability."""
    ref = claude_reference(data)
    sk = {c: [f.set_index(["image_id", "question_id"]) for f in data[("SkinGPT-R1", c)]] for c in CONDS}
    a = agree(sk["stateless"][0], ref)
    shift = (sk["stateless"][0].p_o - sk["topic"][0].p_o).abs()
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.5, wspace=0.12, left=0.33, right=0.98, top=0.93, bottom=0.08)

    ax = fig.add_subplot(gs[:, 0])
    t = pd.DataFrame({"agree": a.same_q, "shift": shift, "topic": ref.topic}).groupby("topic").agg(
        agree=("agree", "mean"), shift=("shift", "mean"), n=("agree", "size")).sort_values("agree")
    y = np.arange(len(t))
    ax.hlines(y, 0, 1, color=GRID, lw=0.5)
    ax.scatter(t.agree, y, s=14, color=COL["SkinGPT-R1"][0], zorder=3, label="same answer as Claude (stateless)")
    ax.scatter(t["shift"].clip(upper=1), y, s=14, marker="D", facecolor="white", edgecolor=COL["SkinGPT-R1"][0], lw=0.8,
               zorder=3, label="SkinGPT-R1 change, stateless vs topic")
    ax.set_yticks(y, [f"{n[:52] + '…' if len(n) > 53 else n} ({k})" for n, k in zip(t.index, t.n)], fontsize=5.6)
    ax.set(xlim=(0, 1.03), ylim=(-0.8, len(t) - 0.2), xlabel="Share of questions (filled) or mean |Δ P(yes | question)| (open)")
    ax.tick_params(axis="y", length=0)
    ax.set_title("By topic (n = questions × 2 images)", loc="left")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, loc="lower right", fontsize=6, handletextpad=0.2)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    num["topic_agreement_q"] = {k: round(v, 3) for k, v in t.agree.items()}
    num["topic_agreement_vs_shift_spearman"] = round(t.agree.corr(t["shift"], method="spearman"), 4)
    letter(ax, "a", x=-0.62, y=1.01)

    ax = fig.add_subplot(gs[0, 1])
    d = pd.DataFrame({"conf": ref.conf_o, "agree": a.same_q.astype(float)})
    d["bin"] = pd.cut(d.conf, [0, 0.55, 0.65, 0.75, 0.85, 1.0])
    g = d.groupby("bin", observed=True).agree.agg(["mean", "sem", "size"])
    ax.errorbar([b.mid for b in g.index], g["mean"], yerr=1.96 * g["sem"], color=COL["SkinGPT-R1"][0], marker="o", ms=3,
                lw=1.1, capsize=1.5, elinewidth=0.6)
    for b, n_ in zip(g.index, g["size"]):
        ax.text(b.mid, 0.04, f"n={n_}", ha="center", fontsize=5.5, color=INK2)
    ax.set(ylim=(0, 1), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"], xlabel="Claude stated confidence (question, binned)",
           ylabel="Same answer (%)")
    ax.set_title("By Claude's confidence", loc="left")
    num["agreement_q_by_claude_conf"] = {str(k): round(v, 4) for k, v in g["mean"].items()}
    style(ax)
    letter(ax, "b", x=-0.3)

    ax = fig.add_subplot(gs[1, 1])
    stable = pd.concat([x.p_o >= 0.5 for x in sk["topic, shuffled"]], axis=1).nunique(axis=1).eq(1)
    rows = []
    for lab, mask in (("stable\n(same answer in\nall 4 shuffles)", stable), ("unstable", ~stable)):
        v = a.same_q[mask]
        rows.append((lab, v.mean(), v.sem(), int(mask.sum())))
    for i, (lab, mn, se, n_) in enumerate(rows):
        ax.bar(i, mn, width=0.55, color=COL["SkinGPT-R1"][0] if i == 0 else COL["SkinGPT-R1"][1], lw=0)
        ax.errorbar(i, mn, yerr=1.96 * se, color=INK, capsize=2, lw=0.8)
        ax.text(i, mn + 1.96 * se + 0.03, f"{mn:.0%}\n(n={n_})", ha="center", fontsize=6, color=INK2)
        num.setdefault("agreement_q_by_skingpt_shuffle_stability", {})[lab.split(chr(10))[0]] = {"agree": round(mn, 4), "n": n_}
    ax.set_xticks([0, 1], [r[0] for r in rows], fontsize=6)
    ax.set(ylim=(0, 1.1), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"], xlim=(-0.6, 1.6), ylabel="Same answer as Claude (%)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("By SkinGPT-R1 stability\nunder shuffling", loc="left")
    style(ax)
    letter(ax, "c", x=-0.3)
    fig.savefig(FIG / "fig5.pdf")
    fig.savefig(FIG / "fig5.png")



def claude_order_check(data, num):
    """Order effect for Claude beyond sampling: mean of 2 fixed-order replicates vs mean of 2 others (sampling only)
    against mean of 2 fixed-order vs mean of 2 shuffled replicates (sampling + order), over all splits."""
    fx = [f.set_index(["image_id", "question_id"]).p_adj for f in data.get(("Claude", "topic"), [])]
    sh = [f.set_index(["image_id", "question_id"]).p_adj for f in data.get(("Claude", "topic, shuffled"), [])]
    if len(fx) < 4 or len(sh) < 2:
        return
    halves = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    null = np.mean([((fx[a] + fx[b]) / 2 - (fx[c] + fx[d]) / 2).abs().mean() for (a, b), (c, d) in halves])
    test = np.mean([((fx[a] + fx[b]) / 2 - (sh[c] + sh[d]) / 2).abs().mean()
                    for a, b in combinations(range(len(fx)), 2) for c, d in combinations(range(len(sh)), 2)])
    num["claude_order_check"] = {"fixed_vs_fixed_2rep_means": round(null, 4), "fixed_vs_shuffled_2rep_means": round(test, 4)}


def table1(data, runs) -> pd.DataFrame:
    rows = []
    for (m, c), frames in sorted(data.items(), key=lambda kv: (MODELS.index(kv[0][0]), CONDS.index(kv[0][1]))):
        d = pd.concat(frames)
        fr = [f.set_index(["image_id", "question_id"]) for f in frames]
        rows.append({"model": m, "condition": c, "runs": len(frames),
                     "said yes (%)": 100 * (d.verdict == "yes").mean(),
                     "contradiction (%)": 100 * d.verdict.str.startswith("contradiction").mean(),
                     "  no to both (%)": 100 * (d.verdict == VERDICTS[2]).mean(),
                     "  yes to both (%)": 100 * (d.verdict == VERDICTS[3]).mean(),
                     "mean |p_adj|": d.p_adj.abs().mean(),
                     "replicate |Δ p_adj|": pair_mad(fr).mean() if len(fr) > 1 else np.nan,
                     "same verdict in all replicates (%)": 100 * pd.concat([f.verdict for f in fr], axis=1).nunique(axis=1).eq(1).mean() if len(fr) > 1 else np.nan,
                     "mean stated confidence": ((d.conf_o + d.conf_c) / 2).mean(),
                     "answers with P(Yes)+P(No) < 0.9 (%)": 100 * (d.mass < 0.9).mean()})
    t = pd.DataFrame(rows).round(3)
    t.to_csv(HERE / "table1.csv", index=False)
    return t


def main():
    FIG.mkdir(exist_ok=True)
    runs = run_list()
    data = {k: [load(r) for r in v] for k, v in runs.items()}
    num = {"runs": {" | ".join(k): v for k, v in runs.items()}}
    fig1(data, runs, num)
    fig2(data, runs, num)
    fig3(data, runs, num)
    fig4(data, num)
    fig5(data, num)
    claude_order_check(data, num)
    t = table1(data, runs)
    (HERE / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
