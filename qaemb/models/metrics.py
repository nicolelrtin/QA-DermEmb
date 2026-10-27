"""Model metrics (handoff §5.5). Undefined metrics (e.g. AUROC with one class) are NaN, not errors."""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)


def ece(y, p, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def sens_at_spec(y, p, spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_metrics(y, p, spec: float = 0.9, bins: int = 10) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    both = len(np.unique(y)) == 2
    nan = float("nan")
    return {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if both else nan,
        "auprc": float(average_precision_score(y, p)) if both else nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)) if both else nan,
        f"sens_at_spec{int(spec * 100)}": sens_at_spec(y, p, spec) if both else nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else nan,
        "ece": ece(y, p, bins) if len(y) else nan,
        "confusion_matrix": confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist(),
    }
