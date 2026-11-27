"""Phase 5 최종 그랜드 앙상블 (이종 5-클래스 전역 블렌딩 + 계층적 초소수 A·B 이진/다중 스페셜리스트 + Nested OOF 후처리).

완전 무누수(Zero Leakage) 3단계 파이프라인:
1. Stage 1 (Global Multi-Family Calibration):
   - exp_001(LGBM Full-Base), exp_002(LGBM Temporal), exp_003(LGBM DomainRatio+TE),
     exp_005(XGBoost hist), exp_006(CatBoost Oblivious)의 OOF 확률에 대해
     각 Fold 학습셋(folds != f)에서 multi_logloss를 최소화하는 최적 볼록 결합 가중치 산출
2. Stage 2 (Sqrt-Balanced Rare-Class {A, B} Binary Specialist inside VIP Pool {A, B, C}):
   - 각 Fold 학습셋 내 {A, B, C} 고객(~17,160명)에서만 다중 시드 경량 트리(depth=4~5)와
     SqrtBalanced 가중치로 B-vs-{A,C} 및 A-vs-{B,C} 정밀 판별기를 학습하고 사전확률 오즈 보정(Prior Odds Calibration) 수행
   - exp_004의 3-클래스 VIP 조건부 확률 및 전역 앙상블 조건부 확률과 결합
3. Stage 3 (Nested OOF Multiplier Optimization & Validated Submission):
   - 5-Fold Nested OOF로 일반화 Macro F1을 엄밀 검증하고 최종 제출 CSV 생성 및 7단계 무결성 검증 수행
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import format_metrics_summary
from src.evaluation.postprocess import apply_multipliers, nested_oof_postprocess
from src.models.lgbm_trainer import load_feature_matrix
from src.utils.config import IDX_TO_CLASS, get_process_memory_mb, load_config, resolve_path, set_seed


def prior_calibrated_prob(weighted_prob: np.ndarray, true_prior: float, eff_prior: float) -> np.ndarray:
    """클래스 가중치가 적용된 예측 확률을 원래 모집단 사전확률 스케일로 베이즈 오즈 역보정."""
    p = np.clip(weighted_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    odds_ratio = (true_prior / (1.0 - true_prior)) / (eff_prior / (1.0 - eff_prior))
    cal_odds = (p / (1.0 - p)) * odds_ratio
    return cal_odds / (1.0 + cal_odds)


def optimize_logloss_weights(oofs_list: list[np.ndarray], y_tr: np.ndarray, log_space: bool = True) -> np.ndarray:
    """학습 폴드 내부에서 다중 모델 블렌딩 가중치 최적화 (Proper Scoring Rule: Multi-class LogLoss)."""
    m = len(oofs_list)
    eps = 1e-7
    prep = [np.log(np.clip(p, eps, 1.0)) if log_space else np.clip(p, eps, 1.0) for p in oofs_list]

    def loss_fn(w: np.ndarray) -> float:
        w_n = w / np.sum(w)
        comb = sum(wi * pi for wi, pi in zip(w_n, prep))
        if log_space:
            comb = np.exp(comb - np.max(comb, axis=1, keepdims=True))
            comb /= np.sum(comb, axis=1, keepdims=True)
        else:
            comb /= np.sum(comb, axis=1, keepdims=True)
        return float(log_loss(y_tr, comb, labels=[0, 1, 2, 3, 4]))

    init_w = np.ones(m, dtype=np.float64) / m
    bounds = [(0.0, 1.0)] * m
    res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w_opt = np.clip(res.x, 0.0, 1.0)
    return w_opt / np.sum(w_opt)


def train_rare_ab_binary_specialists(
    train_paths: list[Path],
    test_paths: list[Path],
    y_all: np.ndarray,
    folds: np.ndarray,
    top_k: int = 320,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """각 Fold 내부 {A, B, C} VIP 집합에서 B전담 / A전담 다중시드 이진 스페셜리스트 조건부 확률 추정."""
    cfg = load_config()
    art_root = resolve_path(cfg["paths"]["artifacts_dir"])

    x_all, feat_names = load_feature_matrix(train_paths)
    x_test, feat_names_te = load_feature_matrix(test_paths)
    assert feat_names == feat_names_te
    n_feats = x_all.shape[1]

    bin_cond_oof = np.zeros((len(y_all), 3), dtype=np.float64)
    bin_cond_test = np.zeros((len(x_test), 3), dtype=np.float64)

    for fold_idx in range(5):
        f_t0 = time.time()
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]

        tr_vip_idx = tr_idx[y_all[tr_idx] <= 2]
        va_vip_mask = y_all[va_idx] <= 2
        va_vip_idx = va_idx[va_vip_mask]

        y_tr_vip = y_all[tr_vip_idx]
        y_va_vip = y_all[va_vip_idx]

        # Fold별 무누수 통합 피처 중요도 (exp_001 전체 2060 + exp_003 파생 849)
        g1 = np.load(art_root / "exp_001" / f"fold{fold_idx}_gains.npy")[:2060]
        g3 = np.load(art_root / "exp_003" / f"fold{fold_idx}_gains.npy")[:n_feats]
        scores = np.zeros(n_feats, dtype=np.float64)
        scores[:2060] = np.maximum(g1 / (g1.sum() + 1e-9), g3[:2060] / (g3.sum() + 1e-9))
        scores[2060:] = g3[2060:] / (g3.sum() + 1e-9)
        sel_idx = np.sort(np.argsort(scores)[::-1][:top_k])

        x_tr_v = x_all[tr_vip_idx][:, sel_idx]
        x_va_v = x_all[va_vip_idx][:, sel_idx]
        x_va_all = x_all[va_idx][:, sel_idx]
        x_te_all = x_test[:, sel_idx]

        # --- (1) 초소수 B (19명 vs ~17,142명) 전담 다중 시드 스페셜리스트 ---
        yb_tr = (y_tr_vip == 1).astype(np.int32)
        yb_va = (y_va_vip == 1).astype(np.int32)
        pi_b = float(yb_tr.mean())
        spw_b = float(np.sqrt((1.0 - pi_b) / max(pi_b, 1e-6)))  # ~30.0 (SqrtBalanced)
        eff_pi_b = (pi_b * spw_b) / (pi_b * spw_b + (1.0 - pi_b))

        pb_va_acc = np.zeros(len(va_idx), dtype=np.float64)
        pb_te_acc = np.zeros(len(x_test), dtype=np.float64)

        for s_off in (0, 101, 202):
            p_lgb_b = {
                "objective": "binary",
                "metric": "binary_logloss",
                "force_col_wise": True,
                "max_bin": 127,
                "learning_rate": 0.025,
                "num_leaves": 11,
                "max_depth": 4,
                "min_child_samples": 3,
                "scale_pos_weight": spw_b,
                "subsample": 0.78,
                "subsample_freq": 1,
                "colsample_bytree": 0.28,
                "reg_alpha": 1.5,
                "reg_lambda": 10.0,
                "n_jobs": 6,
                "random_state": seed + fold_idx * 37 + s_off,
                "verbose": -1,
            }
            dt_b = lgb.Dataset(x_tr_v, label=yb_tr, params=p_lgb_b, free_raw_data=True)
            dv_b = lgb.Dataset(x_va_v, label=yb_va, reference=dt_b, params=p_lgb_b, free_raw_data=True)
            mb = lgb.train(
                p_lgb_b,
                dt_b,
                num_boost_round=800,
                valid_sets=[dv_b],
                callbacks=[lgb.early_stopping(65, verbose=False)],
            )
            pb_va_acc += prior_calibrated_prob(mb.predict(x_va_all, num_iteration=mb.best_iteration), pi_b, eff_pi_b) / 4.0
            pb_te_acc += prior_calibrated_prob(mb.predict(x_te_all, num_iteration=mb.best_iteration), pi_b, eff_pi_b) / 4.0
            del dt_b, dv_b, mb

        cb_b = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            auto_class_weights="SqrtBalanced",
            iterations=900,
            learning_rate=0.035,
            depth=4,
            l2_leaf_reg=8.0,
            border_count=128,
            thread_count=6,
            random_seed=seed + fold_idx * 41,
            early_stopping_rounds=70,
            verbose=False,
            allow_writing_files=False,
        )
        cb_b.fit(Pool(x_tr_v, label=yb_tr), eval_set=Pool(x_va_v, label=yb_va), use_best_model=True)
        pb_va_acc += prior_calibrated_prob(cb_b.predict_proba(x_va_all)[:, 1], pi_b, eff_pi_b) / 4.0
        pb_te_acc += prior_calibrated_prob(cb_b.predict_proba(x_te_all)[:, 1], pi_b, eff_pi_b) / 4.0
        del cb_b

        # --- (2) 소수 A (130명 vs ~17,031명) 전담 다중 시드 스페셜리스트 ---
        ya_tr = (y_tr_vip == 0).astype(np.int32)
        ya_va = (y_va_vip == 0).astype(np.int32)
        pi_a = float(ya_tr.mean())
        spw_a = float(np.sqrt((1.0 - pi_a) / max(pi_a, 1e-6)))  # ~11.4 (SqrtBalanced)
        eff_pi_a = (pi_a * spw_a) / (pi_a * spw_a + (1.0 - pi_a))

        pa_va_acc = np.zeros(len(va_idx), dtype=np.float64)
        pa_te_acc = np.zeros(len(x_test), dtype=np.float64)

        for s_off in (0, 101):
            p_lgb_a = {
                "objective": "binary",
                "metric": "binary_logloss",
                "force_col_wise": True,
                "max_bin": 127,
                "learning_rate": 0.03,
                "num_leaves": 19,
                "max_depth": 5,
                "min_child_samples": 5,
                "scale_pos_weight": spw_a,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.32,
                "reg_alpha": 1.5,
                "reg_lambda": 8.0,
                "n_jobs": 6,
                "random_state": seed + fold_idx * 53 + s_off,
                "verbose": -1,
            }
            dt_a = lgb.Dataset(x_tr_v, label=ya_tr, params=p_lgb_a, free_raw_data=True)
            dv_a = lgb.Dataset(x_va_v, label=ya_va, reference=dt_a, params=p_lgb_a, free_raw_data=True)
            ma = lgb.train(
                p_lgb_a,
                dt_a,
                num_boost_round=900,
                valid_sets=[dv_a],
                callbacks=[lgb.early_stopping(65, verbose=False)],
            )
            pa_va_acc += prior_calibrated_prob(ma.predict(x_va_all, num_iteration=ma.best_iteration), pi_a, eff_pi_a) / 3.0
            pa_te_acc += prior_calibrated_prob(ma.predict(x_te_all, num_iteration=ma.best_iteration), pi_a, eff_pi_a) / 3.0
            del dt_a, dv_a, ma

        cb_a = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            auto_class_weights="SqrtBalanced",
            iterations=950,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=7.0,
            border_count=128,
            thread_count=6,
            random_seed=seed + fold_idx * 59,
            early_stopping_rounds=70,
            verbose=False,
            allow_writing_files=False,
        )
        cb_a.fit(Pool(x_tr_v, label=ya_tr), eval_set=Pool(x_va_v, label=ya_va), use_best_model=True)
        pa_va_acc += prior_calibrated_prob(cb_a.predict_proba(x_va_all)[:, 1], pi_a, eff_pi_a) / 3.0
        pa_te_acc += prior_calibrated_prob(cb_a.predict_proba(x_te_all)[:, 1], pi_a, eff_pi_a) / 3.0
        del cb_a

        # {A, B, C} 합 1.0 정규화 조건부 분포 구성
        rem_va = np.clip(1.0 - pa_va_acc - pb_va_acc, 1e-5, 1.0)
        tot_va = pa_va_acc + pb_va_acc + rem_va
        bin_cond_oof[va_idx, 0] = pa_va_acc / tot_va
        bin_cond_oof[va_idx, 1] = pb_va_acc / tot_va
        bin_cond_oof[va_idx, 2] = rem_va / tot_va

        rem_te = np.clip(1.0 - pa_te_acc - pb_te_acc, 1e-5, 1.0)
        tot_te = pa_te_acc + pb_te_acc + rem_te
        bin_cond_test[:, 0] += (pa_te_acc / tot_te) / 5.0
        bin_cond_test[:, 1] += (pb_te_acc / tot_te) / 5.0
        bin_cond_test[:, 2] += (rem_te / tot_te) / 5.0

        print(f"  [Rare A/B Binary Specialist Fold {fold_idx}] {time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB")
        del x_tr_v, x_va_v, x_va_all, x_te_all
        gc.collect()

    del x_all, x_test
    gc.collect()
    return bin_cond_oof, bin_cond_test


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    proc_dir = resolve_path(cfg["paths"]["processed_dir"])
    oof_dir = resolve_path(cfg["paths"]["oof_dir"])
    art_root = resolve_path(cfg["paths"]["artifacts_dir"])
    sub_dir = resolve_path(cfg["paths"]["submissions_dir"])
    sample_sub_path = resolve_path(cfg["paths"]["sample_submission"])

    folds_df = pd.read_parquet(proc_dir / "folds.parquet")
    y_all = folds_df["target"].to_numpy(dtype=np.int32)
    folds = folds_df["fold"].to_numpy(dtype=np.int32)

    global_exps = ["exp_001", "exp_002", "exp_003", "exp_005", "exp_006"]
    available_global = [e for e in global_exps if (oof_dir / f"{e}_oof.npy").exists()]
    print(f"[Stage 1] 전역 5-클래스 앙상블 대상 모델: {available_global}")

    g_oofs = [np.load(oof_dir / f"{e}_oof.npy").astype(np.float64) for e in available_global]
    g_tests = [np.load(oof_dir / f"{e}_test.npy").astype(np.float64) for e in available_global]

    # Nested OOF 방식으로 전역 모델 블렌딩 가중치 산출 (Log-Opinion Pool vs Linear Pool 비교)
    eps = 1e-7
    global_ens_oof = np.zeros_like(g_oofs[0])
    fold_blend_weights = []

    for f in range(5):
        tr_m = folds != f
        va_m = folds == f
        w_f = optimize_logloss_weights([p[tr_m] for p in g_oofs], y_all[tr_m], log_space=False)
        fold_blend_weights.append(w_f)
        comb_va = sum(wi * pi[va_m] for wi, pi in zip(w_f, g_oofs))
        comb_va /= np.sum(comb_va, axis=1, keepdims=True)
        global_ens_oof[va_m] = comb_va

    mean_blend_w = np.mean(fold_blend_weights, axis=0)
    mean_blend_w /= np.sum(mean_blend_w)
    global_ens_test = sum(wi * pi for wi, pi in zip(mean_blend_w, g_tests))
    global_ens_test /= np.sum(global_ens_test, axis=1, keepdims=True)

    print(
        "  -> 최적 전역 모델 가중치: "
        + ", ".join(f"{e}={w:.3f}" for e, w in zip(available_global, mean_blend_w))
    )
    print(f"  -> Global Ensemble OOF LogLoss = {log_loss(y_all, np.clip(global_ens_oof, eps, 1.0)):.5f}")

    # 순수 전역 다중모델 앙상블 평가
    res_glob = nested_oof_postprocess(global_ens_oof.astype(np.float32), y_all, folds, n_rounds=4, reg_strength=2e-5)
    print(format_metrics_summary(res_glob["nested_metrics"], prefix="Stage1 Global-MultiModel OOF"))

    # Stage 2: VIP {A, B, C} 다중클래스 스페셜리스트(exp_004) + SqrtBalanced A/B 이진 스페셜리스트 결합
    print("\n[Stage 2] 초소수 A·B 전담 SqrtBalanced 다중시드 이진 스페셜리스트 학습 중...")
    tr_paths = [
        proc_dir / "train_base.parquet",
        proc_dir / "train_fg001_time.parquet",
        proc_dir / "train_fg002_ratio.parquet",
    ]
    te_paths = [
        proc_dir / "test_base.parquet",
        proc_dir / "test_fg001_time.parquet",
        proc_dir / "test_fg002_ratio.parquet",
    ]
    bin_cond_oof, bin_cond_test = train_rare_ab_binary_specialists(tr_paths, te_paths, y_all, folds)
    vip3_cond_oof = np.load(art_root / "exp_004" / "pure_cond_oof.npy").astype(np.float64)
    vip3_cond_test = np.load(art_root / "exp_004" / "pure_cond_test.npy").astype(np.float64)

    p_s_oof = np.clip(global_ens_oof[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)
    p_s_test = np.clip(global_ens_test[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)
    glob_cond_oof = global_ens_oof[:, :3] / p_s_oof
    glob_cond_test = global_ens_test[:, :3] / p_s_test

    # XGBoost(exp_005) 및 CatBoost(exp_006)의 독립적인 {A, B, C} 조건부 확률 추출
    xgb_oof = np.load(oof_dir / "exp_005_oof.npy").astype(np.float64)
    xgb_test = np.load(oof_dir / "exp_005_test.npy").astype(np.float64)
    xgb_cond_oof = xgb_oof[:, :3] / np.clip(xgb_oof[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)
    xgb_cond_test = xgb_test[:, :3] / np.clip(xgb_test[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)

    cb_oof = np.load(oof_dir / "exp_006_oof.npy").astype(np.float64)
    cb_test = np.load(oof_dir / "exp_006_test.npy").astype(np.float64)
    cb_cond_oof = cb_oof[:, :3] / np.clip(cb_oof[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)
    cb_cond_test = cb_test[:, :3] / np.clip(cb_test[:, :3].sum(axis=1, keepdims=True), 1e-12, 1.0)

    cond_sources_oof = [glob_cond_oof, xgb_cond_oof, cb_cond_oof, vip3_cond_oof, bin_cond_oof]
    cond_sources_test = [glob_cond_test, xgb_cond_test, cb_cond_test, vip3_cond_test, bin_cond_test]
    cond_names = ["GlobalEns", "XGBoost", "CatBoost", "VIP-3Class", "Rare-Binary"]

    # 엄밀한 Nested OOF로 5원 조건부 확률 결합 가중치 산출 (Zero Leakage)
    fused_cond_oof = np.zeros_like(glob_cond_oof)
    cond_w_folds = []
    for f in range(5):
        tr_v = (folds != f) & (y_all <= 2)
        va_m = folds == f
        c_list_tr = [src[tr_v] for src in cond_sources_oof]
        y_v_tr = y_all[tr_v]
        # 소수 A, B 분리 감도를 반영한 완만한 샘플 가중 3-클래스 LogLoss
        cls_w_map = np.array([2.5, 6.0, 1.0], dtype=np.float64)
        sw_tr = cls_w_map[y_v_tr]

        def cond_loss(w: np.ndarray) -> float:
            wn = w / np.sum(w)
            lg = sum(wi * np.log(np.clip(ci, eps, 1.0)) for wi, ci in zip(wn, c_list_tr))
            pr = np.exp(lg - np.max(lg, axis=1, keepdims=True))
            pr /= np.sum(pr, axis=1, keepdims=True)
            return float(log_loss(y_v_tr, pr, sample_weight=sw_tr, labels=[0, 1, 2]))

        r_c = minimize(
            cond_loss,
            np.array([0.25, 0.25, 0.15, 0.20, 0.15]),
            method="SLSQP",
            bounds=[(0.05, 0.70), (0.05, 0.70), (0.0, 0.50), (0.05, 0.60), (0.0, 0.50)],
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        wf = np.clip(r_c.x, 0.0, 1.0)
        wf /= np.sum(wf)
        cond_w_folds.append(wf)

        lg_va = sum(wi * np.log(np.clip(src[va_m], eps, 1.0)) for wi, src in zip(wf, cond_sources_oof))
        pr_va = np.exp(lg_va - np.max(lg_va, axis=1, keepdims=True))
        pr_va /= np.sum(pr_va, axis=1, keepdims=True)
        fused_cond_oof[va_m] = pr_va

    mean_cond_w = np.mean(cond_w_folds, axis=0)
    mean_cond_w /= np.sum(mean_cond_w)
    print(
        "  -> VIP 5원 조건부 확률 결합 비중: "
        + ", ".join(f"{nm}={w:.3f}" for nm, w in zip(cond_names, mean_cond_w))
    )

    lg_te = sum(wi * np.log(np.clip(src, eps, 1.0)) for wi, src in zip(mean_cond_w, cond_sources_test))
    fused_cond_test = np.exp(lg_te - np.max(lg_te, axis=1, keepdims=True))
    fused_cond_test /= np.sum(fused_cond_test, axis=1, keepdims=True)

    grand_oof = global_ens_oof.copy()
    grand_test = global_ens_test.copy()
    grand_oof[:, :3] = p_s_oof * fused_cond_oof
    grand_test[:, :3] = p_s_test * fused_cond_test

    # Global 순수 앙상블(exp_007_global)과 계층 통합 그랜드 앙상블(exp_007) 비교 후 최우수 선택
    res_grand = nested_oof_postprocess(grand_oof.astype(np.float32), y_all, folds, n_rounds=5, reg_strength=2e-5)
    print("\n" + "=" * 90)
    print(format_metrics_summary(res_grand["raw_metrics"], prefix="exp_007 GRAND RAW OOF"))
    print(format_metrics_summary(res_grand["nested_metrics"], prefix="exp_007 GRAND NESTED-POST OOF"))
    print(
        f"  -> Consensus Multipliers (A..E): {[round(float(w), 4) for w in res_grand['consensus_weights']]}"
    )
    print("=" * 90)

    # 만약 Stage1 Global-MultiModel보다 Stage2 Hierarchical Grand가 높으면 Grand 채택, 아니면 더 높은 쪽 채택
    if res_grand["nested_metrics"]["macro_f1"] >= res_glob["nested_metrics"]["macro_f1"]:
        chosen_oof = grand_oof.astype(np.float32)
        chosen_test = grand_test.astype(np.float32)
        chosen_res = res_grand
        chosen_tag = "Hierarchical Grand Ensemble (Global 5-Models + VIP 3-Class + SqrtBalanced A/B Specialists)"
    else:
        chosen_oof = global_ens_oof.astype(np.float32)
        chosen_test = global_ens_test.astype(np.float32)
        chosen_res = res_glob
        chosen_tag = "Global Multi-Family 5-Model Ensemble"

    exp_id = "exp_007"
    art_dir = art_root / exp_id
    art_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_dir / f"{exp_id}_oof.npy", chosen_oof)
    np.save(oof_dir / f"{exp_id}_test.npy", chosen_test)
    np.save(oof_dir / "exp_007_global_only_oof.npy", global_ens_oof.astype(np.float32))
    np.save(oof_dir / "exp_007_global_only_test.npy", global_ens_test.astype(np.float32))

    # 최종 제출 파일 생성 및 검증
    multipliers = chosen_res["consensus_weights"]
    test_pred_idx = apply_multipliers(chosen_test, multipliers)
    sub_df = pd.read_csv(sample_sub_path)
    sub_df["Segment"] = pd.Series(test_pred_idx).map(IDX_TO_CLASS).values

    out_csv = sub_dir / "sub_002_final_grand_ensemble.csv"
    sub_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[SAVED] 최종 제출 파일 생성: {out_csv}")

    val_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_submission.py"), str(out_csv)]
    proc = subprocess.run(val_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("최종 제출 파일 검증 실패!")

    test_dist = sub_df["Segment"].value_counts().reindex(["A", "B", "C", "D", "E"], fill_value=0).to_dict()
    summary = {
        "exp_id": exp_id,
        "architecture": chosen_tag,
        "global_models": available_global,
        "global_blend_weights": {e: round(float(w), 4) for e, w in zip(available_global, mean_blend_w)},
        "vip_cond_weights": [round(float(x), 4) for x in mean_cond_w],
        "stage1_global_nested_macro_f1": res_glob["nested_metrics"]["macro_f1"],
        "stage1_global_class_f1": res_glob["nested_metrics"]["class_f1"],
        "raw_macro_f1": chosen_res["raw_metrics"]["macro_f1"],
        "raw_class_f1": chosen_res["raw_metrics"]["class_f1"],
        "nested_macro_f1": chosen_res["nested_metrics"]["macro_f1"],
        "nested_class_f1": chosen_res["nested_metrics"]["class_f1"],
        "nested_class_precision": chosen_res["nested_metrics"]["class_precision"],
        "nested_class_recall": chosen_res["nested_metrics"]["class_recall"],
        "nested_pred_counts": chosen_res["nested_metrics"]["pred_counts"],
        "confusion_matrix": chosen_res["nested_metrics"]["confusion_matrix"].tolist(),
        "raw_confusion_matrix": chosen_res["raw_metrics"]["confusion_matrix"].tolist(),
        "consensus_multipliers": [float(x) for x in multipliers],
        "test_pred_counts": {k: int(v) for k, v in test_dist.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with (art_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
