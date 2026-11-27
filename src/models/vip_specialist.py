"""계층적 VIP 부분집합({A, B, C}) 스페셜리스트 다중 모델 학습 및 조건부 확률 결합기.

수리적 근거 (Law of Total Probability):
  S = {A(0), B(1), C(2)} 일 때, c in S 에 대하여
  P(Y = c | x) = P(Y in S | x)_global * P(Y = c | Y in S, x)_vip_specialist

40만 명 전체 모델에서 트리 분할의 95% 이상이 D·E 다수 클래스 분리에 소모되는 병목을 제거하고,
오직 각 Fold의 학습(Train) 데이터 내 {A, B, C} 고객(~17,160명)만 추출하여
LightGBM + CatBoost + XGBoost 3종 스페셜리스트가 A(162명)·B(24명)·C(21,265명) 간의
미세 도메인 경계(한도 규모 vs 할부·현금서비스·리볼빙 여신 집중도 vs 연간 복원 청구원금)만 전담 학습합니다.
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
import xgboost as xgb

from src.evaluation.metrics import format_metrics_summary
from src.evaluation.postprocess import nested_oof_postprocess
from src.models.lgbm_trainer import fit_fold_smoothed_target_encoding, load_feature_matrix
from src.utils.config import get_process_memory_mb, load_config, resolve_path, set_seed


def train_vip_specialist_cv(
    exp_id: str,
    train_paths: list[Path],
    test_paths: list[Path],
    base_global_exp_id: str,
    ref_lgbm_exp_id: str,
    top_k_features: int = 650,
    cat_cols_for_te: list[str] | None = None,
    vip_blend_alpha: float = 0.65,
) -> dict[str, Any]:
    """각 Fold 내부 {A, B, C} 부분집합에서 LGBM+XGB+CatBoost 3종 조건부 확률을 학습하고 전역 OOF와 결합."""
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

    global_oof = np.load(oof_dir / f"{base_global_exp_id}_oof.npy").astype(np.float64)
    global_test = np.load(oof_dir / f"{base_global_exp_id}_test.npy").astype(np.float64)

    x_all, feat_names = load_feature_matrix(train_paths)
    x_test, feat_names_te = load_feature_matrix(test_paths)
    assert feat_names == feat_names_te
    n_base_feats = x_all.shape[1]

    te_col_indices: list[int] = []
    if cat_cols_for_te:
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        te_col_indices = [name_to_idx[c] for c in cat_cols_for_te if c in name_to_idx]

    # 조건부 확률 P(Y = c | Y in {A,B,C}, x) 저장 배열 (N, 3)
    cond_oof = np.zeros((len(y_all), 3), dtype=np.float64)
    cond_test = np.zeros((len(x_test), 3), dtype=np.float64)

    for fold_idx in range(5):
        f_t0 = time.time()
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]

        # 오직 해당 Fold의 Train 내부 {A(0), B(1), C(2)} 고객만 필터링 (완전 무누수!)
        vip_tr_mask = y_all[tr_idx] <= 2
        vip_va_mask = y_all[va_idx] <= 2
        tr_vip_idx = tr_idx[vip_tr_mask]
        va_vip_idx = va_idx[vip_va_mask]

        y_tr_vip = y_all[tr_vip_idx]  # 값 범위: {0, 1, 2}
        y_va_vip = y_all[va_vip_idx]

        gains_npy = ref_dir / f"fold{fold_idx}_gains.npy"
        if gains_npy.exists():
            ref_gains = np.load(gains_npy)[:n_base_feats]
        else:
            ref_b = lgb.Booster(model_file=str(ref_dir / f"lgbm_fold{fold_idx}.txt"))
            ref_gains = ref_b.feature_importance(importance_type="gain")[:n_base_feats]
            del ref_b

        k_sel = min(top_k_features, n_base_feats)
        sel_idx = np.sort(np.argsort(ref_gains)[::-1][:k_sel])

        x_tr_vip = x_all[tr_vip_idx][:, sel_idx]
        x_va_vip = x_all[va_vip_idx][:, sel_idx]
        x_va_full = x_all[va_idx][:, sel_idx]
        x_te_full = x_test[:, sel_idx]

        if te_col_indices:
            tr_te, va_full_te, te_full_te = fit_fold_smoothed_target_encoding(
                x_all[tr_vip_idx][:, te_col_indices],
                y_tr_vip,
                x_all[va_idx][:, te_col_indices],
                x_test[:, te_col_indices],
                smoothing=15.0,
            )
            x_tr_vip = np.hstack([x_tr_vip, tr_te])
            x_va_full = np.hstack([x_va_full, va_full_te])
            x_va_vip = x_va_full[vip_va_mask]
            x_te_full = np.hstack([x_te_full, te_full_te])
            del tr_te, va_full_te, te_full_te

        # 1) VIP LightGBM (소수 B=19명 세밀 분할을 위해 min_child_samples=5 + 강한 정규화)
        lgb_p = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "force_col_wise": True,
            "max_bin": 127,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 5,
            "min_child_weight": 1e-3,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.40,
            "reg_alpha": 1.5,
            "reg_lambda": 6.0,
            "n_jobs": 6,
            "random_state": seed + fold_idx * 11,
            "verbose": -1,
        }
        dl_tr = lgb.Dataset(x_tr_vip, label=y_tr_vip, params=lgb_p, free_raw_data=True)
        dl_va = lgb.Dataset(x_va_vip, label=y_va_vip, reference=dl_tr, params=lgb_p, free_raw_data=True)
        m_lgb = lgb.train(
            lgb_p,
            dl_tr,
            num_boost_round=1200,
            valid_sets=[dl_va],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        p_va_lgb = m_lgb.predict(x_va_full, num_iteration=m_lgb.best_iteration)
        p_te_lgb = m_lgb.predict(x_te_full, num_iteration=m_lgb.best_iteration)

        # 2) VIP CatBoost (대칭 트리 depth=5로 극소수 B=19명 과적합 억제)
        m_cb = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1200,
            learning_rate=0.045,
            depth=5,
            l2_leaf_reg=6.0,
            random_strength=1.2,
            border_count=128,
            thread_count=6,
            random_seed=seed + fold_idx * 19,
            early_stopping_rounds=80,
            verbose=False,
            allow_writing_files=False,
        )
        m_cb.fit(Pool(x_tr_vip, label=y_tr_vip), eval_set=Pool(x_va_vip, label=y_va_vip), use_best_model=True)
        p_va_cb = m_cb.predict_proba(x_va_full)
        p_te_cb = m_cb.predict_proba(x_te_full)

        # 3) VIP XGBoost (hist, min_child_weight=0.02, max_delta_step=2.0)
        xgb_p = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "max_bin": 128,
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 0.02,
            "max_delta_step": 2.0,
            "subsample": 0.80,
            "colsample_bytree": 0.40,
            "reg_alpha": 1.5,
            "reg_lambda": 6.0,
            "nthread": 6,
            "seed": seed + fold_idx * 29,
            "verbosity": 0,
        }
        dx_tr = xgb.DMatrix(x_tr_vip, label=y_tr_vip)
        dx_va_vip = xgb.DMatrix(x_va_vip, label=y_va_vip)
        m_xgb = xgb.train(
            xgb_p,
            dx_tr,
            num_boost_round=1200,
            evals=[(dx_va_vip, "val")],
            early_stopping_rounds=80,
            verbose_eval=False,
        )
        b_it_x = int(m_xgb.best_iteration) + 1
        p_va_xgb = m_xgb.predict(xgb.DMatrix(x_va_full), iteration_range=(0, b_it_x))
        p_te_xgb = m_xgb.predict(xgb.DMatrix(x_te_full), iteration_range=(0, b_it_x))

        # 3종 VIP 스페셜리스트 조건부 확률 블렌딩
        fold_cond_va = 0.40 * p_va_lgb + 0.32 * p_va_cb + 0.28 * p_va_xgb
        fold_cond_te = 0.40 * p_te_lgb + 0.32 * p_te_cb + 0.28 * p_te_xgb
        cond_oof[va_idx] = fold_cond_va
        cond_test += fold_cond_te / 5.0

        print(
            f"  [VIP Fold {fold_idx}] n_vip_tr={len(y_tr_vip):,} | "
            f"iters=(LGB:{m_lgb.best_iteration}, CB:{m_cb.get_best_iteration()}, XGB:{b_it_x}) | "
            f"{time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
        )
        del x_tr_vip, x_va_vip, x_va_full, x_te_full, dl_tr, dl_va, m_lgb, m_cb, dx_tr, dx_va_vip, m_xgb
        gc.collect()

    del x_all, x_test
    gc.collect()

    # 조건부 확률 법칙 결합: P_hier(c|x) = P_global(S|x) * P_cond(c|S, x) for c in {0, 1, 2}
    p_s_oof = np.clip(global_oof[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)
    p_s_test = np.clip(global_test[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)

    hier_oof = global_oof.copy()
    hier_test = global_test.copy()

    # 전역 모델의 조건부 비율과 VIP 스페셜리스트 조건부 비율을 로그-선형(기하) 보간하여 안정성 극대화
    eps = 1e-9
    glob_cond_oof = global_oof[:, :3] / p_s_oof
    glob_cond_test = global_test[:, :3] / p_s_test

    fused_cond_oof = (np.clip(cond_oof, eps, 1.0) ** vip_blend_alpha) * (
        np.clip(glob_cond_oof, eps, 1.0) ** (1.0 - vip_blend_alpha)
    )
    fused_cond_oof /= fused_cond_oof.sum(axis=1, keepdims=True)

    fused_cond_test = (np.clip(cond_test, eps, 1.0) ** vip_blend_alpha) * (
        np.clip(glob_cond_test, eps, 1.0) ** (1.0 - vip_blend_alpha)
    )
    fused_cond_test /= fused_cond_test.sum(axis=1, keepdims=True)

    hier_oof[:, :3] = p_s_oof * fused_cond_oof
    hier_test[:, :3] = p_s_test * fused_cond_test

    oof_f32 = hier_oof.astype(np.float32)
    test_f32 = hier_test.astype(np.float32)

    oof_path = oof_dir / f"{exp_id}_oof.npy"
    test_path = oof_dir / f"{exp_id}_test.npy"
    np.save(oof_path, oof_f32)
    np.save(test_path, test_f32)
    np.save(artifacts_dir / "pure_cond_oof.npy", cond_oof.astype(np.float32))
    np.save(artifacts_dir / "pure_cond_test.npy", cond_test.astype(np.float32))

    post_res = nested_oof_postprocess(oof_f32, y_all, folds, n_rounds=4, reg_strength=2e-5)
    print("\n" + "=" * 88)
    print(format_metrics_summary(post_res["raw_metrics"], prefix=f"{exp_id} RAW OOF"))
    print(format_metrics_summary(post_res["nested_metrics"], prefix=f"{exp_id} NESTED-POST OOF"))
    print(
        f"  -> Consensus Multipliers (A..E): {[round(float(w), 4) for w in post_res['consensus_weights']]}"
    )
    print("=" * 88)

    summary = {
        "exp_id": exp_id,
        "model_family": "Hierarchical VIP Specialist (LGBM+CatBoost+XGB on {A,B,C})",
        "base_global_exp_id": base_global_exp_id,
        "top_k_features": top_k_features,
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
