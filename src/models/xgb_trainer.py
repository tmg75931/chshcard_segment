"""고정 5-Fold 메모리 안전 XGBoost (hist) 학습기.

불변 규칙 준수:
- 각 Fold 내부에서만 해당 Fold 학습 모델(lgbm_fold{f}.txt)의 Gain 중요도 기준 상위 피처를 선택하여
  검증 폴드 정보 누수(Feature Selection Leakage)를 원천 차단합니다.
- 초소수 클래스(B=24명) 분할 억제를 막기 위해 min_child_weight=0.05, max_delta_step=2.0을 적용합니다.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import evaluate_predictions, format_metrics_summary
from src.evaluation.postprocess import nested_oof_postprocess
from src.models.lgbm_trainer import fit_fold_smoothed_target_encoding, load_feature_matrix
from src.utils.config import get_process_memory_mb, load_config, resolve_path, set_seed


def train_xgb_cv(
    exp_id: str,
    train_paths: list[Path],
    test_paths: list[Path],
    ref_lgbm_exp_id: str,
    top_k_features: int = 700,
    cat_cols_for_te: list[str] | None = None,
    xgb_params: dict[str, Any] | None = None,
    num_boost_round: int = 2200,
    early_stopping_rounds: int = 100,
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

    print(f"[{exp_id}] XGBoost용 Float32 행렬 로드 중...")
    x_all, feat_names = load_feature_matrix(train_paths)
    x_test, feat_names_te = load_feature_matrix(test_paths)
    assert feat_names == feat_names_te

    te_col_indices: list[int] = []
    if cat_cols_for_te:
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        te_col_indices = [name_to_idx[c] for c in cat_cols_for_te if c in name_to_idx]

    default_params: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_bin": 128,
        "learning_rate": 0.055,
        "max_depth": 7,
        "min_child_weight": 0.05,
        "max_delta_step": 2.0,
        "subsample": 0.80,
        "colsample_bytree": 0.45,
        "reg_alpha": 1.2,
        "reg_lambda": 5.0,
        "nthread": 6,
        "seed": seed,
        "verbosity": 0,
    }
    if xgb_params:
        default_params.update(xgb_params)

    oof_probs = np.zeros((len(y_all), 5), dtype=np.float32)
    test_probs = np.zeros((len(x_test), 5), dtype=np.float32)
    best_iters: list[int] = []
    n_base_feats = x_all.shape[1]

    for fold_idx in range(5):
        f_t0 = time.time()
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]

        # exp_001(전체 2,060 베이스) + ref_dir(파생 피처) Fold별 무누수 Gain 정규화 결합
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

        model_path = artifacts_dir / f"xgb_fold{fold_idx}.json"
        if model_path.exists() and model_path.stat().st_size > 10_000:
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            dval = xgb.DMatrix(x_va_sel)
            dtest = xgb.DMatrix(x_te_sel)
            val_pred = bst.predict(dval).astype(np.float32)
            oof_probs[va_idx] = val_pred
            test_probs += bst.predict(dtest).astype(np.float32) / 5.0
            best_iters.append(int(bst.num_boosted_rounds()))
            fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
            print(f"  [Fold {fold_idx} RESUMED] Raw Macro F1={fold_raw_m['macro_f1']:.5f}")
            del x_va_sel, x_te_sel, dval, dtest, bst, tr_te
            gc.collect()
            continue

        x_tr_sel = x_all[tr_idx][:, sel_idx]
        if tr_te is not None:
            x_tr_sel = np.hstack([x_tr_sel, tr_te])
            del tr_te

        dtrain = xgb.DMatrix(x_tr_sel, label=y_tr)
        dval = xgb.DMatrix(x_va_sel, label=y_va)
        dtest = xgb.DMatrix(x_te_sel)
        del x_tr_sel, x_va_sel, x_te_sel
        gc.collect()

        fold_params = dict(default_params)
        fold_params["seed"] = seed + fold_idx * 23

        bst = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=250,
        )

        best_it = int(bst.best_iteration) + 1
        best_iters.append(best_it)
        val_pred = bst.predict(dval, iteration_range=(0, best_it)).astype(np.float32)
        oof_probs[va_idx] = val_pred
        test_probs += bst.predict(dtest, iteration_range=(0, best_it)).astype(np.float32) / 5.0
        bst_sliced = bst[:best_it]
        bst_sliced.save_model(str(artifacts_dir / f"xgb_fold{fold_idx}.json"))

        fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
        print(
            f"  [Fold {fold_idx}] best_iter={best_it} | Raw Macro F1={fold_raw_m['macro_f1']:.5f} | "
            f"{time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
        )
        del dtrain, dval, dtest, bst
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
        "model_family": "XGBoost",
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
