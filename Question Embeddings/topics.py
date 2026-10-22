"""Topics for plotting.

Question CSVs carry their own `topic` (written into questions.csv by embed_questions.py).
topic_map.csv can optionally regroup labels into broader topics: any row whose `section`
appears in the map gets the mapped topic; every other row keeps the topic from its CSV.
The map is applied at plot time, so editing it only needs the (fast) plot scripts re-run.
"""
from pathlib import Path

import pandas as pd

TOPIC_MAP = Path(__file__).resolve().parent / "topic_map.csv"


def _mapping() -> pd.Series:
    if not TOPIC_MAP.exists():
        return pd.Series(dtype=str)
    mapping = pd.read_csv(TOPIC_MAP).set_index("section")["topic"]
    if mapping.index.duplicated().any():
        dupes = mapping.index[mapping.index.duplicated()].unique().tolist()
        raise SystemExit(f"{TOPIC_MAP.name} lists these sections more than once: {dupes}")
    return mapping


def topic_order(topics: pd.Series) -> list[str]:
    """Mapped topics first, in map order, then the rest by frequency; this fixes colour slots."""
    present = topics.value_counts()
    mapped = [t for t in dict.fromkeys(_mapping()) if t in present.index]
    return mapped + [t for t in present.index if t not in mapped]


def attach_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` whose `topic` column applies topic_map.csv where it has an entry."""
    out = df.copy()
    mapped = out["section"].map(_mapping())
    if "topic" in out.columns:
        out["topic"] = mapped.fillna(out["topic"])
    else:
        out["topic"] = mapped
    missing = out.loc[out["topic"].isna(), "section"].unique().tolist()
    if missing:
        raise SystemExit(f"No topic for these sections (not in the CSV or {TOPIC_MAP.name}): {missing}")
    return out
