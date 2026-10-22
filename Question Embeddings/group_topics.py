"""Draft meta-topics: cluster the CSV topics into K broader groups for colouring plots.

The questions themselves (their embeddings from embed_questions.py) are clustered into K
groups with k-means, and each topic joins the group holding most of its questions. Working
on questions rather than one point per topic weights topics by size, so large topics split
the dense regions into several groups and one-question topics cannot form groups of their
own. Run after embed_questions.py:
  python "Question Embeddings/group_topics.py"            # 12 groups from Qwen3 embeddings
  python "Question Embeddings/group_topics.py" --k 8 --model panderm

Writes a draft into topic_map.csv (section -> topic, the file the plot scripts read) and
prints each group's members with their purity (share of the topic's questions that
fell in that group; low purity means the topic straddles groups). Group names start as placeholders taken from the group's
largest topic: rename them in topic_map.csv (the `topic` column), then re-run the plot
scripts. Rows in topic_map.csv for sections not in the current data are kept, and an
existing entry is only replaced with --overwrite, so hand edits survive re-runs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from topics import TOPIC_MAP

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=12, help="number of meta-topics")
    parser.add_argument("--model", default="qwen3", help="which emb_<model>.npy to cluster on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace existing topic_map.csv entries for these topics")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    questions = pd.read_csv(args.out / "questions.csv")
    emb = np.load(args.out / f"emb_{args.model}.npy")
    if len(emb) != len(questions):
        raise SystemExit(f"emb_{args.model}.npy has {len(emb)} rows but questions.csv has {len(questions)}; re-run embed_questions.py.")

    topics = questions["section"].value_counts()
    if args.k > len(topics):
        raise SystemExit(f"--k {args.k} is more than the {len(topics)} topics present.")
    # Embeddings are unit vectors, so k-means' Euclidean distance ranks pairs like cosine distance.
    question_group = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(emb)
    votes = pd.crosstab(questions["section"], question_group)
    majority = votes.idxmax(axis=1)
    purity = votes.max(axis=1) / votes.sum(axis=1)  # share of a topic's questions in its group

    groups = pd.DataFrame({"section": topics.index, "n": topics.to_numpy()})
    groups["group"] = groups["section"].map(majority)
    groups["purity"] = groups["section"].map(purity)
    # Groups can end up empty once topics move as a whole; keep the ones that are used.
    groups["group"] = groups["group"].rank(method="dense").astype(int) - 1
    # Placeholder name: the group's largest topic, so the draft is readable before renaming.
    groups["topic"] = groups.groupby("group")["section"].transform("first")
    groups["note"] = "auto-grouped; rename topic"

    existing = pd.read_csv(TOPIC_MAP) if TOPIC_MAP.exists() else pd.DataFrame(columns=["section", "topic", "note"])
    keep_existing = existing["section"].isin(groups["section"]) & (not args.overwrite)
    new_rows = groups[~groups["section"].isin(existing.loc[keep_existing, "section"])]
    merged = pd.concat([existing[keep_existing | ~existing["section"].isin(groups["section"])],
                        new_rows[["section", "topic", "note"]]], ignore_index=True)
    merged.to_csv(TOPIC_MAP, index=False)

    kept = int(keep_existing.sum())
    print(f"{len(topics)} topics -> {groups['group'].nunique()} groups; wrote {len(new_rows)} entries to {TOPIC_MAP.name}"
          + (f" (kept {kept} existing; use --overwrite to replace them)" if kept else ""))
    for g, members in groups.sort_values("n", ascending=False).groupby("group", sort=False):
        print(f"\n[{members['topic'].iloc[0]}]  {members['n'].sum()} questions, {len(members)} topics")
        for _, r in members.iterrows():
            print(f"  {r.n:5d}  {r.purity:4.0%}  {r.section}")


if __name__ == "__main__":
    main()
