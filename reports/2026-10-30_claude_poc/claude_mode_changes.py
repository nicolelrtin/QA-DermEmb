"""Which of Claude's answers change between stateless and topic prompts (2026-10-30 PoC, 2 images x 500 questions).

Both modes are the mean of Claude's 4 replicates. Each image x question is flagged:
  verdict changed            the verdict from the replicate means differs (yes / no / contradiction)
  large shift, same verdict  |Δ Directional_Score| >= 0.3 between the modes' means, verdict unchanged
  -                          neither
'beyond replicate noise' is True when the 4 stateless and the 4 topic Directional_Score values do not overlap at all.

Writes claude_stateless_vs_topic.csv / .xlsx (flagged rows first, largest change first) and
claude_stateless_vs_topic.png (Directional_Score stateless vs topic per question, flagged questions labelled).

    poetry run python reports/2026-10-30_claude_poc/claude_mode_changes.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from plot_variability import BG, COLORS, IMAGES, INK, INK2, GRID  # noqa: E402
from qaemb.stages.s4_embed import pair_scores, verdict  # noqa: E402

IDX = ["image_id", "question_id"]
TEXT = ["label_raw", "topic", "scope", "kind", "question", "question_neg"]
SCORE = "Directional_Score"  # the score used for the shift flag, the noise check, the sort order and the plot
LARGE = 0.3
FLAGS = ["verdict changed", "large shift, same verdict", "-"]


def mode_table(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate means per image x question (scores averaged per replicate), and the per-replicate SCORE."""
    raw = pd.read_csv(HERE / f"replicates_claude_{mode}" / "replicates_raw.csv")
    for c, v in pair_scores(raw.p_yes_original, raw.p_yes_complement).items():
        raw[c] = v
    vals = ["p_yes_original", "p_yes_complement", "Net_P", "Directional_Score", "Coherence_Score",
            "confidence_original", "confidence_complement"]
    g = raw.groupby(IDX)
    m = g[vals].mean()
    m[f"SD {SCORE}"] = g[SCORE].std()
    m["verdict"] = [verdict(a, b, 0.5) for a, b in zip(m.p_yes_original, m.p_yes_complement)]
    return m, raw.pivot_table(index=IDX, columns="replicate", values=SCORE), g[TEXT].first()


def build() -> pd.DataFrame:
    (s, s_reps, text), (t, t_reps, _) = mode_table("stateless"), mode_table("topic")
    show = {"p_yes_original": "P(Y|Qo)", "p_yes_complement": "P(Y|Qc)", "Net_P": "Net_P",
            "Directional_Score": "Directional_Score", "Coherence_Score": "Coherence_Score", "verdict": "verdict",
            "confidence_original": "confidence(Qo)", f"SD {SCORE}": f"SD {SCORE}"}
    d = text.copy()
    for name, m in (("stateless", s), ("topic", t)):
        for c, n in show.items():
            d[f"{name} | {n}"] = m[c]
    for c in ("Net_P", "Directional_Score", "Coherence_Score"):
        d[f"Δ {c} (topic − stateless)"] = t[c] - s[c]
    d["beyond replicate noise"] = (s_reps.max(axis=1) < t_reps.min(axis=1)) | (t_reps.max(axis=1) < s_reps.min(axis=1))
    changed = s.verdict != t.verdict
    large = (t[SCORE] - s[SCORE]).abs() >= LARGE
    d.insert(0, "flag", np.select([changed, large], FLAGS[:2], FLAGS[2]))
    d["_order"] = d.flag.map({f: i for i, f in enumerate(FLAGS)})
    d["_size"] = -d[f"Δ {SCORE} (topic − stateless)"].abs()
    return d.sort_values(["_order", "_size"]).drop(columns=["_order", "_size"]).reset_index()


def write_xlsx(d: pd.DataFrame, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fills = {"verdict changed": PatternFill("solid", fgColor="F8C9C9"), "large shift, same verdict": PatternFill("solid", fgColor="FBEBB5")}
    wb = Workbook()
    ws = wb.active
    ws.title = "stateless vs topic"
    for k, c in enumerate(d.columns, 1):
        cell = ws.cell(1, k, c)
        cell.font, cell.alignment = Font(bold=True), Alignment(wrap_text=True, vertical="top")
    for r, row in enumerate(d.itertuples(index=False), 2):
        fill = fills.get(row[d.columns.get_loc("flag")])
        for k, v in enumerate(row, 1):
            v = v.item() if hasattr(v, "item") else v
            cell = ws.cell(r, k, None if isinstance(v, float) and np.isnan(v) else v)
            if isinstance(v, float):
                cell.number_format = "0.00"
            if fill:
                cell.fill = fill
    widths = {"question": 55, "question_neg": 40, "topic": 26, "flag": 22}
    for k, c in enumerate(d.columns, 1):
        ws.column_dimensions[get_column_letter(k)].width = widths.get(c, 11)
    ws.freeze_panes = "I2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(d.columns))}{len(d) + 1}"
    wb.save(path)


def scatter(d: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(BG)
    for ax, img in zip(axes, IMAGES):
        x = d[d.image_id == img]
        rest, big, chg = (x[x.flag == f] for f in ("-", FLAGS[1], FLAGS[0]))
        ax.scatter(rest[f"stateless | {SCORE}"], rest[f"topic | {SCORE}"], s=8, color="#c3c2b7", lw=0, label="same verdict, |Δ| < 0.3")
        ax.scatter(big[f"stateless | {SCORE}"], big[f"topic | {SCORE}"], s=14, facecolor="white", edgecolor=COLORS["Claude"][0],
                   lw=0.9, label=f"large shift, same verdict ({len(big)})")
        ax.scatter(chg[f"stateless | {SCORE}"], chg[f"topic | {SCORE}"], s=16, color="#c0392b", lw=0, label=f"verdict changed ({len(chg)})")
        for _, r in chg.head(12).iterrows():
            ax.annotate(r.question_id, (r[f"stateless | {SCORE}"], r[f"topic | {SCORE}"]), fontsize=7, color=INK2,
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([-1, 1], [-1, 1], color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal", xlabel=f"{SCORE}, stateless (mean of 4)")
        ax.set_title(IMAGES[img], loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"{SCORE}, topic (mean of 4)")
    fig.suptitle("Claude: answers that change between stateless and topic prompts (12 largest verdict changes labelled)",
                 x=0.06, ha="left", fontsize=12.5, weight="bold")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)


def main():
    d = build()
    d.round(4).to_csv(HERE / "claude_stateless_vs_topic.csv", index=False)
    write_xlsx(d.round(4), HERE / "claude_stateless_vs_topic.xlsx")
    scatter(d, HERE / "claude_stateless_vs_topic.png")
    print(d.groupby(["image_id", "flag"]).size().unstack(fill_value=0))
    print(pd.crosstab(d.flag, d["beyond replicate noise"]))


if __name__ == "__main__":
    main()
