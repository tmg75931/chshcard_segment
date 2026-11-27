"""Macro F1 극대화를 위한 다중 클래스 확률 승수(Multiplier) 사후 최적화기.

초소수 클래스(B=24명, A=162명) 과적합을 막기 위해 반드시 Nested Out-Of-Fold 방식으로
일반화 성능을 평가하며, 로그 스케일 정규화가 결합된 Coordinate Ascent를 수행합니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.metrics import evaluate_predictions


def apply_multipliers(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """확률 행렬 (N, K)에 클래스별 승수 (K,)를 곱해 argmax 클래스 인덱스 반환."""
    scaled = probs * multipliers.reshape(1, -1)
    return np.argmax(scaled, axis=1)


def _fast_macro_f1_over_candidates(
    y_true: np.ndarray,
    other_best_cls: np.ndarray,
    other_max_val: np.ndarray,
    p_k: np.ndarray,
    k: int,
    cand_weights: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """단일 클래스 k 승수 후보군 전체에 대해 정확히 동일한 Macro F1을 O(N log N + M) 벡터 연산으로 계산."""
    # 초기 상태 (cand_w = 0 가정 시 모든 샘플은 other_best_cls로 예측됨)
    base_cm = np.bincount(
        y_true * n_classes + other_best_cls, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)

    valid_mask = p_k > 0.0
    if not np.any(valid_mask):
        tp = np.diag(base_cm).astype(np.float64)
        fp = base_cm.sum(axis=0).astype(np.float64) - tp
        fn = base_cm.sum(axis=1).astype(np.float64) - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
        return np.full(len(cand_weights), float(f1.mean()), dtype=np.float64)

    # cand_w * p_k > other_max_val  <=>  cand_w > other_max_val / p_k
    tau = other_max_val[valid_mask] / p_k[valid_mask]
    y_v = y_true[valid_mask]
    o_v = other_best_cls[valid_mask]

    order = np.argsort(tau, kind="mergesort")
    tau_sorted = tau[order]
    pair_idx = y_v[order] * n_classes + o_v[order]

    # 각 (true_cls, old_pred_cls) 쌍의 누적 전환 개수 행렬: (len(tau_sorted) + 1, 25)
    n_v = len(tau_sorted)
    one_hot = np.zeros((n_v + 1, n_classes * n_classes), dtype=np.int32)
    one_hot[np.arange(1, n_v + 1), pair_idx] = 1
    cum_transitions = np.cumsum(one_hot, axis=0)

    # cand_w > tau_sorted 인 샘플 개수: side='left' 이면 tau_sorted < cand_w 개수와 정확히 일치
    cut_indices = np.searchsorted(tau_sorted, cand_weights, side="left")
    trans = cum_transitions[cut_indices].reshape(-1, n_classes, n_classes)  # (M, K, K)

    # CM 업데이트: 기존 예측(old_pred)에서 빠지고 신규 클래스 k로 이동
    cm = base_cm[None, :, :] - trans
    switched_per_true = trans.sum(axis=2)  # (M, K)
    cm[:, :, k] += switched_per_true

    tp = np.diagonal(cm, axis1=1, axis2=2).astype(np.float64)  # (M, K)
    pred_sum = cm.sum(axis=1).astype(np.float64)  # (M, K)
    true_sum = cm.sum(axis=2).astype(np.float64)  # (M, K)
    denom = true_sum + pred_sum
    f1_per_cls = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0.0)
    return f1_per_cls.mean(axis=1)


def fit_class_multipliers(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_rounds: int = 4,
    reg_strength: float = 1e-4,
) -> np.ndarray:
    """Coordinate Ascent로 Macro F1을 최대화하는 클래스 승수 (w_0..w_4, w_4=1.0 고정) 탐색.

    누적 혼동행렬 전이(Cumulative Confusion Matrix Transition) 벡터화를 통해
    수십만 행에서도 수백 개 후보 그리드를 밀리초 단위로 전역 탐색합니다.
    """
    n_classes = probs.shape[1]
    y_int = np.asarray(y_true, dtype=np.int32)
    weights = np.ones(n_classes, dtype=np.float64)
    labels = list(range(n_classes))

    # 소수 클래스(1:B, 0:A)부터 중간 클래스(2:C, 3:D)까지 순차 최적화 (4:E는 1.0 기준점 고정)
    tune_order = [1, 0, 2, 3]
    coarse_grid = np.geomspace(0.05, 1500.0, 320)

    for _ in range(n_rounds):
        for k in tune_order:
            other_cols = [c for c in labels if c != k]
            other_scaled = probs[:, other_cols] * weights[other_cols].reshape(1, -1)
            other_max_idx = np.argmax(other_scaled, axis=1)
            other_max_val = other_scaled[np.arange(len(probs)), other_max_idx]
            other_best_cls = np.array(other_cols, dtype=np.int32)[other_max_idx]

            p_k = probs[:, k]
            other_pen = float(np.sum(np.log(np.clip(weights[other_cols], 1e-3, 1e4)) ** 2))

            # 1단계: 광역 로그 그리드 벡터 탐색
            scores_c = _fast_macro_f1_over_candidates(
                y_int, other_best_cls, other_max_val, p_k, k, coarse_grid, n_classes
            )
            penalties_c = reg_strength * (other_pen + np.log(coarse_grid) ** 2)
            objs_c = scores_c - penalties_c
            best_idx_c = int(np.argmax(objs_c))
            best_w = float(coarse_grid[best_idx_c])
            best_obj = float(objs_c[best_idx_c])

            # 2단계: 최적점 주변 초조밀 선형 그리드 벡터 탐색
            fine_grid = np.linspace(max(0.02, best_w * 0.65), min(2500.0, best_w * 1.45), 180)
            scores_f = _fast_macro_f1_over_candidates(
                y_int, other_best_cls, other_max_val, p_k, k, fine_grid, n_classes
            )
            penalties_f = reg_strength * (other_pen + np.log(fine_grid) ** 2)
            objs_f = scores_f - penalties_f
            best_idx_f = int(np.argmax(objs_f))
            if float(objs_f[best_idx_f]) > best_obj:
                best_w = float(fine_grid[best_idx_f])

            weights[k] = best_w

    return weights


def nested_oof_postprocess(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    n_rounds: int = 4,
    reg_strength: float = 2e-5,
) -> dict[str, Any]:
    """누수 없는 Nested Out-Of-Fold 방식으로 클래스 승수 최적화 및 일반화 성능 검증."""
    unique_folds = np.sort(np.unique(folds))
    nested_preds = np.zeros(len(y_true), dtype=np.int32)
    fold_weights: list[np.ndarray] = []

    for f in unique_folds:
        tr_mask = folds != f
        va_mask = folds == f
        w_f = fit_class_multipliers(
            oof_probs[tr_mask],
            y_true[tr_mask],
            n_rounds=n_rounds,
            reg_strength=reg_strength,
        )
        fold_weights.append(w_f)
        nested_preds[va_mask] = apply_multipliers(oof_probs[va_mask], w_f)

    # Test 추론용 기하평균 승수 (단일 Fold 이상치 완화)
    stacked_w = np.vstack(fold_weights)
    geo_mean_weights = np.exp(np.mean(np.log(np.clip(stacked_w, 1e-4, 1e4)), axis=0))
    geo_mean_weights = geo_mean_weights / geo_mean_weights[-1]

    raw_preds = np.argmax(oof_probs, axis=1)
    raw_metrics = evaluate_predictions(y_true, raw_preds)
    nested_metrics = evaluate_predictions(y_true, nested_preds)

    return {
        "raw_metrics": raw_metrics,
        "nested_metrics": nested_metrics,
        "fold_weights": stacked_w,
        "consensus_weights": geo_mean_weights,
        "nested_preds": nested_preds,
    }
