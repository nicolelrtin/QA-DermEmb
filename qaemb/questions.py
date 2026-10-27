"""Question-set loading. Everything downstream is keyed by (Question ID, original) and the question-set
hash, which covers (id, original, text) only, so annotation changes do not change it.

Each Question ID can appear twice: original = 1 (as written) and original = 0 (its negated complement).
scope/kind come from an annotation sidecar keyed by Question ID and apply to both versions.
"""
from pathlib import Path

import pandas as pd

from .config import ROOT, stable_hash


def _col(df: pd.DataFrame, name: str) -> str:
    cols = {c.strip().lower(): c for c in df.columns}
    if name.lower() not in cols:
        raise SystemExit(f"Column {name!r} not found; have {list(df.columns)}")
    return cols[name.lower()]


def _has(df: pd.DataFrame, name: str) -> bool:
    return name.lower() in {c.strip().lower() for c in df.columns}


def load_questions(qcfg) -> pd.DataFrame:
    """One row per asked item: item_id, question_id, original, question, plus topic/scope/kind when present."""
    path = ROOT / qcfg.csv
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = pd.DataFrame({"question_id": raw[_col(raw, qcfg.id_column)], "question": raw[_col(raw, qcfg.text_column)]})
    df["original"] = raw[_col(raw, qcfg.original_column)].astype(int) if _has(raw, qcfg.original_column) else 1
    if _has(raw, "topic"):
        df["topic"] = raw[_col(raw, "topic")]

    if qcfg.annotations:
        ann = pd.read_csv(ROOT / qcfg.annotations, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        ann = ann.rename(columns={_col(ann, qcfg.id_column): "question_id"})
        ann = ann[["question_id"] + [c for c in ("scope", "kind") if c in ann]]
        df = df.merge(ann, on="question_id", how="left", validate="many_to_one")
        unannotated = sorted(df.question_id[df.scope.isna()].unique()) if "scope" in df else []
        if unannotated:
            raise SystemExit(f"{len(unannotated)} question IDs missing from {qcfg.annotations}, e.g. {unannotated[:5]}")
    else:
        for extra in ("scope", "kind"):
            if _has(raw, extra):
                df[extra] = raw[_col(raw, extra)].values

    if df.duplicated(["question_id", "original"]).any():
        raise SystemExit(f"Duplicate (Question ID, original) pairs in {path}")
    orphans = set(df.question_id[df.original == 0]) - set(df.question_id[df.original == 1])
    if orphans:
        raise SystemExit(f"Complement questions without an original: {sorted(orphans)[:5]}")
    if not qcfg.complements:
        df = df[df.original == 1]

    if qcfg.selected_from:
        keep = pd.read_csv(ROOT / qcfg.selected_from, dtype=str).question_id
        df = df[df.question_id.isin(set(keep))]
    if qcfg.ids:
        missing = set(qcfg.ids) - set(df.question_id)
        if missing:
            raise SystemExit(f"Question IDs not in {path.name}: {sorted(missing)}")
        df = df[df.question_id.isin(set(qcfg.ids))]
    if qcfg.scope:
        if "scope" not in df:
            raise SystemExit(f"questions.scope={list(qcfg.scope)} needs scope annotations")
        df = df[df.scope.isin(set(qcfg.scope))]
    if df.empty:
        raise SystemExit("Question selection is empty")
    df["item_id"] = df.question_id + "#" + df.original.astype(str)
    return df.reset_index(drop=True)


def question_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Question ID: original text, complement text (if asked), topic/scope/kind."""
    orig = df[df.original == 1].drop(columns=["original", "item_id"])
    neg = df[df.original == 0][["question_id", "question"]].rename(columns={"question": "question_neg"})
    return orig.merge(neg, on="question_id", how="left").reset_index(drop=True)


def qset_hash(df: pd.DataFrame) -> str:
    return stable_hash(sorted(zip(df.question_id, df.original.astype(int), df.question)))


def write_qset(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
