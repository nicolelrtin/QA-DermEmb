"""Claude Sonnet 5.5 vs Opus 5.5 on one image (ISIC_0000147), topic prompts, 500 questions + negations.

    poetry run python reports/2026-11-04_sonnet_test/compare_sonnet.py <sonnet_run_id>

Reference: the four Opus topic replicates. Their agreement with each other (each replicate against the mean
of the other three) is the ceiling: Sonnet cannot be expected to agree with Opus better than Opus agrees with itself.
Writes agreement.csv, sonnet_vs_opus.csv (per question) and numbers.json next to this file.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from qaemb.report import readable  # noqa: E402  (recomputes Net_P, P_w and both scores from P(yes))

IMAGE = "ISIC_0000147"
OPUS_TOPIC = ["20261030-002157_claude-cli-poc", "20261030-003848_claude-cli-poc",
              "20261030-004312_claude-cli-poc", "20261030-004738_claude-cli-poc"]
OPUS_TOPIC_USD = [4.91, 4.41, 2.83, 2.46]  # list price per run (2 images, 164 prompts), outputs/qaemb/queue/*.log
MEASURES = {"p_yes_original": "P(Y|Qo)", "Net_P": "Net_P", "Directional_Score": "Directional_Score",
            "Coherence_Score": "Coherence_Score", "confidence_original": "confidence(Qo)"}
DEADBAND = 0.1


def load(run_id: str) -> pd.DataFrame:
    _, df, _ = readable(run_id)
    return df[(df.image_id == IMAGE) & (df.patch_id == -1)].set_index("question_id")


def three_way(x):
    return np.sign(x).where(x.abs() >= DEADBAND, 0)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for col, name in MEASURES.items():
        out[f"rho {name}"] = a[col].corr(b[col], method="spearman")
    out["same yes/no to Qo (%)"] = 100 * ((a.p_yes_original >= 0.5) == (b.p_yes_original >= 0.5)).mean()
    out["same verdict (%)"] = 100 * (a.verdict == b.verdict).mean()
    for col in ("Directional_Score", "Coherence_Score"):
        out[f"same 3-way {col} (%)"] = 100 * (three_way(a[col]) == three_way(b[col])).mean()
    out["mean |Δ Net_P|"] = (a.Net_P - b.Net_P).abs().mean()
    return out


def mean_of(frames):
    cols = list(MEASURES) + ["p_yes_complement", "P_w"]
    m = sum(f[cols] for f in frames) / len(frames)
    yo, yc = m.p_yes_original >= 0.5, m.p_yes_complement >= 0.5
    m["verdict"] = np.select([yo & ~yc, ~yo & yc, ~yo & ~yc], ["yes", "no", "contradiction: no to both"], "contradiction: yes to both")
    return m


def run_cost(run_id: str) -> tuple[int, float]:
    log = (ROOT / "outputs/qaemb/queue/sonnet_test.log").read_text()
    m = re.findall(r"this run: (\d+) calls, list-price equivalent \$([\d.]+)", log)
    return int(m[-1][0]), float(m[-1][1])


def main(sonnet_run: str):
    son = load(sonnet_run)
    opus = [load(r) for r in OPUS_TOPIC]
    rows = []
    for i, o in enumerate(opus):  # Opus ceiling: one replicate against the mean of the other three
        rows.append({"comparison": f"Opus R{i + 1} vs mean of other 3 Opus", **compare(o, mean_of([x for j, x in enumerate(opus) if j != i]))})
    for i, o in enumerate(opus):
        rows.append({"comparison": f"Sonnet vs Opus R{i + 1}", **compare(son, o)})
    rows.append({"comparison": "Sonnet vs mean of 4 Opus", **compare(son, mean_of(opus))})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(HERE / "agreement.csv", index=False)

    summary = pd.DataFrame({
        "Opus self-agreement (mean of 4)": t[t.comparison.str.startswith("Opus")].drop(columns="comparison").mean(),
        "Sonnet vs single Opus replicate (mean of 4)": t[t.comparison.str.match(r"Sonnet vs Opus R")].drop(columns="comparison").mean(),
        "Sonnet vs mean of 4 Opus": t[t.comparison == "Sonnet vs mean of 4 Opus"].drop(columns="comparison").iloc[0]})

    om = mean_of(opus)
    side = pd.DataFrame({"question": son.question, "topic": son.topic,
                         **{f"Opus mean {n}": om[c] for c, n in MEASURES.items()},
                         **{f"Sonnet {n}": son[c] for c, n in MEASURES.items()},
                         "Opus verdict": om.verdict, "Sonnet verdict": son.verdict})
    side.round(4).to_csv(HERE / "sonnet_vs_opus.csv")

    contra = lambda d: 100 * d.verdict.str.startswith("contradiction").mean()
    calls, usd = run_cost(sonnet_run)
    opus_pp = np.array(OPUS_TOPIC_USD) / 164
    num = {"sonnet_run": sonnet_run, "sonnet_calls": calls, "sonnet_usd": usd, "sonnet_usd_per_prompt": usd / calls,
           "opus_topic_usd_per_prompt": {"min": opus_pp.min(), "median": float(np.median(opus_pp)), "max": opus_pp.max()},
           "contradiction_pct": {"sonnet": contra(son), "opus_replicates": [contra(o) for o in opus]},
           "yes_to_Qo_pct": {"sonnet": 100 * (son.p_yes_original >= 0.5).mean(),
                             "opus_replicates": [100 * (o.p_yes_original >= 0.5).mean() for o in opus]},
           "mean_confidence_Qo": {"sonnet": son.confidence_original.mean(), "opus": om.confidence_original.mean()},
           "summary": summary.round(3).to_dict()}
    (HERE / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string())
        print(json.dumps({k: v for k, v in num.items() if k != "summary"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
