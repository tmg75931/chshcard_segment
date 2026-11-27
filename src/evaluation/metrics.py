"""대회 공식 평가지표(Macro F1-Score) 및 클래스별 상세 진단 리포터."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.config import CLASS_NAMES


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """정수 인코딩된 정답(0..4)과 예측(0..4)으로부터 전체 Macro F1 및 클래스별 지표 산출."""
    labels = list(range(len(CLASS_NAMES)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": macro_f1,
        "class_f1": {CLASS_NAMES[i]: float(per_f1[i]) for i in labels},
        "class_precision": {CLASS_NAMES[i]: float(per_prec[i]) for i in labels},
        "class_recall": {CLASS_NAMES[i]: float(per_rec[i]) for i in labels},
        "pred_counts": {CLASS_NAMES[i]: int((y_pred == i).sum()) for i in labels},
        "confusion_matrix": cm,
    }


def format_metrics_summary(metrics: dict[str, Any], prefix: str = "") -> str:
    f1s = metrics["class_f1"]
    f1_str = " / ".join(f"{k}:{f1s[k]:.4f}" for k in CLASS_NAMES)
    cnts = metrics["pred_counts"]
    cnt_str = " / ".join(f"{k}:{cnts[k]:,}" for k in CLASS_NAMES)
    title = f"[{prefix}] " if prefix else ""
    return (
        f"{title}Macro F1 = {metrics['macro_f1']:.5f} | "
        f"Class F1 ({f1_str}) | Pred Counts ({cnt_str})"
    )
