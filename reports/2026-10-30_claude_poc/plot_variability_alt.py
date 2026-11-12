"""Figure 1 Alternate: the variability chart of plot_variability.py with Directional_Score and Coherence_Score
in place of Net_P (one row per score). Writes variability_by_question_alt.png and variability_by_question_alt.csv.

Every box is one value per question ID: |difference in the score| (0 = identical answer, 2 = certain and opposite).
Claude comparisons average over every pair of individual replicates (stateless vs topic: all 16 cross pairs).

    poetry run python reports/2026-10-30_claude_poc/plot_variability_alt.py
"""
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from plot_variability import BG, IMAGES, INK, INK2, LOCAL, boxes, mean_abs_diff, readable_columns  # noqa: E402
from qaemb.stages.s4_embed import pair_scores  # noqa: E402

SCORES = {"Directional_Score": "Directional_Score = (2 P(Y|Qo) − 1) × |Net_P|",
          "Coherence_Score": "Coherence_Score = (2 P(Y|Qo) − 1) × (1 − |P(Y|Qo) + P(Y|Qc) − 1|)"}


def score_table(score: str) -> pd.DataFrame:
    """Rows: image x question ID; columns: (model, mode, replicate)."""
    cols = {}
    for mode in ("stateless", "topic"):
        raw = pd.read_csv(HERE / f"replicates_claude_{mode}" / "replicates_raw.csv")
        for rep, g in raw.groupby("replicate"):
            g = g.set_index(["image_id", "question_id"])
            cols[("Claude", mode, rep)] = pair_scores(g.p_yes_original, g.p_yes_complement)[score]
    d = pd.read_csv(HERE / "compare" / "compare_answers.csv").set_index(["image_id", "question_id"])
    for (model, name), mode in product(LOCAL.items(), ("stateless", "topic")):
        cols[(model, mode, "-")] = d[f"{name} | {mode} | {score}"]
    return pd.DataFrame(cols)


def main():
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.6), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3], "wspace": 0.08, "hspace": 0.42})
    fig.patch.set_facecolor(BG)
    out = {}
    for row, (score, formula) in enumerate(SCORES.items()):
        t = score_table(score)
        runs = lambda model, mode: [c for c in t.columns if c[:2] == (model, mode)]
        reps = [(f"Claude\n{mode}", "Claude", mean_abs_diff(t, runs("Claude", mode))) for mode in ("stateless", "topic")]
        modes = [(m, m, mean_abs_diff(t, runs(m, "stateless"), runs(m, "topic"))) for m in ("Qwen", "SkinGPT-R1", "Claude")]
        out.update(readable_columns(score, t))
        a, b = axes[row]
        boxes(a, reps)
        boxes(b, modes)
        a.set_ylim(0, 1.6)
        a.set_ylabel(f"Change in {score} per question ID\n0 = same answer, 2 = certain and opposite", color=INK2, fontsize=10)
        a.set_title("Claude: replicate to replicate", loc="left", fontsize=11.5, color=INK, pad=10)
        b.set_title("Stateless vs topic mode", loc="left", fontsize=11.5, color=INK, pad=10)
        a.text(0, 1.22, f"{'ab'[row]}   {formula}", transform=a.transAxes, fontsize=11, weight="bold", color=INK)
        if row == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in ("#5f5e5a", "#c3c2b7")]
            a.legend(handles, [f"darker: {IMAGES[i]}" if j == 0 else f"lighter: {IMAGES[i]}" for j, i in enumerate(IMAGES)],
                     title="Each pair of boxes, per image", alignment="left", loc="upper right", frameon=False, fontsize=9.5,
                     title_fontsize=9.5, labelcolor=INK2)
    pd.DataFrame(out).round(4).to_csv(HERE / "variability_by_question_alt.csv")
    fig.suptitle("Figure 1 Alternate: how much do the scores change between Claude replicates and between answer modes?",
                 x=0.075, y=1.0, ha="left", fontsize=13.5, weight="bold", color=INK)
    fig.text(0.075, 0.965, "500 question pairs (Q0_all), whole image (G1). Box = middle half of the 500 question IDs, "
             "white line = median, whiskers = 5th to 95th percentile.", fontsize=9.5, color=INK2)
    fig.savefig(HERE / "variability_by_question_alt.png", dpi=170, bbox_inches="tight", facecolor=BG)


if __name__ == "__main__":
    main()
