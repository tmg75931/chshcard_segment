"""고정 5-Fold 대칭 트리(Oblivious Trees) CatBoost 학습기.

불변 규칙 준수:
- 각 Fold 내부에서만 해당 Fold 학습 모델(lgbm_fold{f}.txt)의 Gain 중요도 기준 상위 피처 선택
- Oblivious Tree(depth=6)의 정규화 특성으로 초소수 등급(B=24명, A=162명) 확률 추정 분산을 억제하며,
  Leaf-wise(LightGBM) 및 Depth-wise(XGBoost)와 높은 앙상블 다양성을 제공합니다.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions, format_metrics_summary
from src.evaluation.postprocess import nested_oof_postprocess
from src.models.lgbm_trainer import fit_fold_smoothed_target_encoding, load_feature_matrix
from src.utils.config import get_process_memory_mb, load_config, resolve_path, set_seed


def train_catboost_cv(
    exp_id: str,
    train_paths: list[Path],
    test_paths: list[Path],
    ref_lgbm_exp_id: str,
    top_k_features: int = 550,
    cat_cols_for_te: list[str] | None = None,
    cb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    oof_dir = resolve_path(cfg["paths"]["oof_dir"])
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / exp_id
    ref_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / ref_lgbm_exp_id
    oof_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    folds_df = pd.read_parquet(processed_dir / "folds.parquet")
    y_all = folds_df["target"].to_numpy(dtype=np.int32)
    folds = folds_df["fold"].to_numpy(dtype=np.int32)

    print(f"[{exp_id}] CatBoost용 Float32 행렬 로드 중...")
    x_all, feat_names = load_feature_matrix(train_paths)
    x_test, feat_names_te = load_feature_matrix(test_paths)
    assert feat_names == feat_names_te

    te_col_indices: list[int] = []
    if cat_cols_for_te:
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        te_col_indices = [name_to_idx[c] for c in cat_cols_for_te if c in name_to_idx]

    default_params: dict[str, Any] = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1500,
        "learning_rate": 0.068,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "border_count": 128,
        "thread_count": 6,
        "random_seed": seed,
        "early_stopping_rounds": 80,
        "verbose": 150,
        "allow_writing_files": False,
    }
    if cb_params:
        default_params.update(cb_params)

    oof_probs = np.zeros((len(y_all), 5), dtype=np.float32)
    test_probs = np.zeros((len(x_test), 5), dtype=np.float32)
    best_iters: list[int] = []
    n_base_feats = x_all.shape[1]

    for fold_idx in range(5):
        f_t0 = time.time()
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]

        gains_npy = ref_dir / f"fold{fold_idx}_gains.npy"
        ref_gains = np.load(gains_npy)[:n_base_feats].copy()
        exp1_gains_npy = resolve_path(cfg["paths"]["artifacts_dir"]) / "exp_001" / f"fold{fold_idx}_gains.npy"
        if exp1_gains_npy.exists() and n_base_feats >= 2060:
            g1 = np.load(exp1_gains_npy)[:2060]
            g1_norm = g1 / (g1.sum() + 1e-9)
            g3_norm = ref_gains / (ref_gains.sum() + 1e-9)
            ref_gains[:2060] = np.maximum(g1_norm, g3_norm[:2060])
            ref_gains[2060:] = g3_norm[2060:]

        k_sel = min(top_k_features, n_base_feats)
        sel_idx = np.sort(np.argsort(ref_gains)[::-1][:k_sel])

        x_va_sel = x_all[va_idx][:, sel_idx]
        x_te_sel = x_test[:, sel_idx]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        if te_col_indices:
            tr_te, va_te, test_te = fit_fold_smoothed_target_encoding(
                x_all[tr_idx][:, te_col_indices],
                y_tr,
                x_all[va_idx][:, te_col_indices],
                x_test[:, te_col_indices],
            )
            x_va_sel = np.hstack([x_va_sel, va_te])
            x_te_sel = np.hstack([x_te_sel, test_te])
            del va_te, test_te
        else:
            tr_te = None

        cbm_path = artifacts_dir / f"catboost_fold{fold_idx}.cbm"
        if cbm_path.exists() and cbm_path.stat().st_size > 10_000:
            model = CatBoostClassifier()
            model.load_model(str(cbm_path))
            va_pool = Pool(x_va_sel, label=y_va)
            te_pool = Pool(x_te_sel)
            val_pred = model.predict_proba(va_pool).astype(np.float32)
            oof_probs[va_idx] = val_pred
            test_probs += model.predict_proba(te_pool).astype(np.float32) / 5.0
            best_iters.append(int(model.tree_count_))
            fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
            print(f"  [Fold {fold_idx} RESUMED] Raw Macro F1={fold_raw_m['macro_f1']:.5f}")
            del x_va_sel, x_te_sel, va_pool, te_pool, model, tr_te
            gc.collect()
            continue

        x_tr_sel = x_all[tr_idx][:, sel_idx]
        if tr_te is not None:
            x_tr_sel = np.hstack([x_tr_sel, tr_te])
            del tr_te

        tr_pool = Pool(x_tr_sel, label=y_tr)
        va_pool = Pool(x_va_sel, label=y_va)
        te_pool = Pool(x_te_sel)
        del x_tr_sel, x_va_sel, x_te_sel
        gc.collect()

        fold_params = dict(default_params)
        fold_params["random_seed"] = seed + fold_idx * 31
        model = CatBoostClassifier(**fold_params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        best_it = int(model.get_best_iteration())
        best_iters.append(best_it)
        val_pred = model.predict_proba(va_pool).astype(np.float32)
        oof_probs[va_idx] = val_pred
        test_probs += model.predict_proba(te_pool).astype(np.float32) / 5.0
        model.save_model(str(artifacts_dir / f"catboost_fold{fold_idx}.cbm"))

        fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
        print(
            f"  [Fold {fold_idx}] best_iter={best_it} | Raw Macro F1={fold_raw_m['macro_f1']:.5f} | "
            f"{time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
        )
        del tr_pool, va_pool, te_pool, model
        gc.collect()

    del x_all, x_test
    gc.collect()

    oof_path = oof_dir / f"{exp_id}_oof.npy"
    test_path = oof_dir / f"{exp_id}_test.npy"
    np.save(oof_path, oof_probs)
    np.save(test_path, test_probs)

    post_res = nested_oof_postprocess(oof_probs, y_all, folds, n_rounds=4, reg_strength=2e-5)
    print("\n" + "=" * 88)
    print(format_metrics_summary(post_res["raw_metrics"], prefix=f"{exp_id} RAW OOF"))
    print(format_metrics_summary(post_res["nested_metrics"], prefix=f"{exp_id} NESTED-POST OOF"))
    print("=" * 88)

    summary = {
        "exp_id": exp_id,
        "model_family": "CatBoost",
        "top_k_features": top_k_features,
        "best_iterations": best_iters,
        "raw_macro_f1": post_res["raw_metrics"]["macro_f1"],
        "raw_class_f1": post_res["raw_metrics"]["class_f1"],
        "nested_macro_f1": post_res["nested_metrics"]["macro_f1"],
        "nested_class_f1": post_res["nested_metrics"]["class_f1"],
        "nested_pred_counts": post_res["nested_metrics"]["pred_counts"],
        "consensus_weights": [float(x) for x in post_res["consensus_weights"]],
        "elapsed_sec": round(time.time() - t0, 1),
        "oof_path": str(oof_path.relative_to(resolve_path("."))),
        "test_path": str(test_path.relative_to(resolve_path("."))),
    }
    with (artifacts_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
