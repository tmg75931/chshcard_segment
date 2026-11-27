"""고정 5-Fold 메모리 안전 LightGBM 학습기 (Fold 단위 체크포인트 Resume 및 무누수 피처 선택 지원).

불변 규칙 준수:
- 모든 분할은 data/processed/folds.parquet (고객 ID 단위 고정 5-Fold) 사용
- 빈도/타겟 인코딩, 베이스 피처 가지치기, 후처리 승수 탐색은 오직 각 Fold의 Train 내부에서만 fit
- 중단 후 재실행 시 이미 완료된 Fold 모델을 로드하여 즉시 이어서 학습(Checkpoint Resume)
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metrics import evaluate_predictions, format_metrics_summary
from src.evaluation.postprocess import nested_oof_postprocess
from src.utils.config import get_process_memory_mb, load_config, resolve_path, set_seed


def fit_fold_smoothed_target_encoding(
    x_tr_cat: np.ndarray,
    y_tr: np.ndarray,
    x_va_cat: np.ndarray,
    x_te_cat: np.ndarray,
    smoothing: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 Fold 내부에서만 다중 클래스 스무딩 타겟 인코딩 통계량을 계산하여 변환 (Zero Leakage)."""
    n_tr, n_cols = x_tr_cat.shape
    target_classes = [0, 1, 2, 3]
    n_out = n_cols * len(target_classes)

    priors = np.array([(y_tr == c).mean() for c in target_classes], dtype=np.float32)
    tr_encoded = np.zeros((n_tr, n_out), dtype=np.float32)
    va_encoded = np.zeros((len(x_va_cat), n_out), dtype=np.float32)
    te_encoded = np.zeros((len(x_te_cat), n_out), dtype=np.float32)

    rng = np.random.default_rng(42)
    inner_folds = rng.integers(0, 4, size=n_tr)

    for j in range(n_cols):
        col_tr = x_tr_cat[:, j].astype(np.int32)
        col_va = x_va_cat[:, j].astype(np.int32)
        col_te = x_te_cat[:, j].astype(np.int32)

        shift = 1
        max_cat = int(max(col_tr.max(initial=-1), col_va.max(initial=-1), col_te.max(initial=-1))) + shift + 1
        c_tr_s = np.clip(col_tr + shift, 0, max_cat - 1)
        c_va_s = np.clip(col_va + shift, 0, max_cat - 1)
        c_te_s = np.clip(col_te + shift, 0, max_cat - 1)

        for inf in range(4):
            in_tr = inner_folds != inf
            in_va = inner_folds == inf
            cnts = np.bincount(c_tr_s[in_tr], minlength=max_cat).astype(np.float32)
            for k_idx, cls_id in enumerate(target_classes):
                pos = np.bincount(
                    c_tr_s[in_tr],
                    weights=(y_tr[in_tr] == cls_id).astype(np.float32),
                    minlength=max_cat,
                )
                enc_map = (pos + smoothing * priors[k_idx]) / (cnts + smoothing)
                tr_encoded[in_va, j * len(target_classes) + k_idx] = enc_map[c_tr_s[in_va]]

        full_cnts = np.bincount(c_tr_s, minlength=max_cat).astype(np.float32)
        for k_idx, cls_id in enumerate(target_classes):
            pos = np.bincount(
                c_tr_s,
                weights=(y_tr == cls_id).astype(np.float32),
                minlength=max_cat,
            )
            enc_map = (pos + smoothing * priors[k_idx]) / (full_cnts + smoothing)
            va_encoded[:, j * len(target_classes) + k_idx] = enc_map[c_va_s]
            te_encoded[:, j * len(target_classes) + k_idx] = enc_map[c_te_s]

    return tr_encoded, va_encoded, te_encoded


def load_feature_matrix(
    parquet_paths: list[Path],
    selected_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """하나 이상의 고객 단위 Parquet 파일을 디스크 메모리맵(np.memmap) Float32 행렬 (N, F)로 로드."""
    cfg = load_config()
    interim_dir = resolve_path(cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    col_plan: list[tuple[Path, list[str]]] = []
    feature_names: list[str] = []
    n_rows = 0

    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        if n_rows == 0:
            n_rows = pf.metadata.num_rows
        schema_cols = [c for c in pf.schema_arrow.names if c != "ID"]
        cols_to_read = [c for c in schema_cols if c in selected_cols] if selected_cols is not None else schema_cols
        if cols_to_read:
            col_plan.append((p, cols_to_read))
            feature_names.extend(cols_to_read)

    sig_str = "|".join(f"{p.name}:{p.stat().st_mtime_ns}:{len(cs)}" for p, cs in col_plan) + f"|{len(feature_names)}"
    sig = hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:12]
    mmap_path = interim_dir / f"feat_mmap_{n_rows}x{len(feature_names)}_{sig}.dat"

    if mmap_path.exists() and mmap_path.stat().st_size == n_rows * len(feature_names) * 4:
        mm = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(n_rows, len(feature_names)))
        return mm, feature_names

    mm_w = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(n_rows, len(feature_names)))
    col_cursor = 0
    step = 160
    for p, cols_to_read in col_plan:
        for start in range(0, len(cols_to_read), step):
            sub_cols = cols_to_read[start : start + step]
            tbl = pq.read_table(p, columns=sub_cols)
            for offset, cname in enumerate(sub_cols):
                col_np = tbl.column(cname).to_numpy(zero_copy_only=False)
                mm_w[:, col_cursor + start + offset] = col_np.astype(np.float32, copy=False)
            del tbl
            gc.collect()
        col_cursor += len(cols_to_read)

    mm_w.flush()
    del mm_w
    gc.collect()
    mm_r = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(n_rows, len(feature_names)))
    return mm_r, feature_names


def train_lgbm_cv(
    exp_id: str,
    train_paths: list[Path],
    test_paths: list[Path],
    lgb_params: dict[str, Any] | None = None,
    selected_cols: list[str] | None = None,
    cat_cols_for_te: list[str] | None = None,
    ref_base_exp_id: str | None = None,
    n_base_cols_count: int = 2060,
    top_k_base_features: int | None = None,
    sample_weight_power: float = 0.0,
    num_boost_round: int = 1400,
    early_stopping_rounds: int = 75,
) -> dict[str, Any]:
    """고정 5-Fold LightGBM 학습 (체크포인트 Resume + Fold 내부 무누수 피처 선택 포함)."""
    t0 = time.time()
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    oof_dir = resolve_path(cfg["paths"]["oof_dir"])
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / exp_id
    oof_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    folds_df = pd.read_parquet(processed_dir / "folds.parquet")
    y_all = folds_df["target"].to_numpy(dtype=np.int32)
    folds = folds_df["fold"].to_numpy(dtype=np.int32)

    print(f"[{exp_id}] 학습/테스트 Float32 행렬 로드 중...")
    x_all, feat_names = load_feature_matrix(train_paths, selected_cols=selected_cols)
    x_test, feat_names_te = load_feature_matrix(test_paths, selected_cols=selected_cols)
    assert feat_names == feat_names_te, "Train/Test 피처 순서 불일치!"
    n_raw_feats = len(feat_names)
    print(
        f"  -> 로드 완료: Train {x_all.shape}, Test {x_test.shape} | RAM {get_process_memory_mb():.0f} MB"
    )

    te_col_indices: list[int] = []
    if cat_cols_for_te:
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        te_col_indices = [name_to_idx[c] for c in cat_cols_for_te if c in name_to_idx]

    default_params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "force_col_wise": True,
        "max_bin": 127,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 14,
        "min_child_weight": 1e-3,
        "max_delta_step": 0.0,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.35,
        "reg_alpha": 1.2,
        "reg_lambda": 5.0,
        "n_jobs": 6,
        "random_state": seed,
        "verbose": -1,
    }
    if lgb_params:
        default_params.update(lgb_params)

    oof_probs = np.zeros((len(y_all), 5), dtype=np.float32)
    test_probs = np.zeros((len(x_test), 5), dtype=np.float32)

    total_feat_names = list(feat_names)
    if te_col_indices:
        for idx_c in te_col_indices:
            cname = feat_names[idx_c]
            for cls_lbl in ("A", "B", "C", "D"):
                total_feat_names.append(f"{cname}__oof_te_{cls_lbl}")

    gain_importances = np.zeros(len(total_feat_names), dtype=np.float64)
    split_importances = np.zeros(len(total_feat_names), dtype=np.float64)
    best_iters: list[int] = []

    ref_base_dir = (
        resolve_path(cfg["paths"]["artifacts_dir"]) / ref_base_exp_id if ref_base_exp_id else None
    )

    for fold_idx in range(5):
        f_t0 = time.time()
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        # Fold 내부 무누수 베이스 피처 선택 (해당 Fold 학습 모델의 Gain만 참조)
        if ref_base_dir is not None and top_k_base_features is not None and n_raw_feats >= n_base_cols_count:
            ref_b = lgb.Booster(model_file=str(ref_base_dir / f"lgbm_fold{fold_idx}.txt"))
            base_gains = ref_b.feature_importance(importance_type="gain")[:n_base_cols_count]
            del ref_b
            sel_base = np.sort(np.argsort(base_gains)[::-1][:top_k_base_features])
            eng_idx = np.arange(n_base_cols_count, n_raw_feats, dtype=np.int64)
            raw_col_idx = np.concatenate([sel_base, eng_idx])
        else:
            raw_col_idx = np.arange(n_raw_feats, dtype=np.int64)

        n_te_feats = len(te_col_indices) * 4
        te_global_idx = np.arange(n_raw_feats, n_raw_feats + n_te_feats, dtype=np.int64)
        global_feat_idx = np.concatenate([raw_col_idx, te_global_idx]) if n_te_feats > 0 else raw_col_idx

        model_file = artifacts_dir / f"lgbm_fold{fold_idx}.txt"
        gains_file = artifacts_dir / f"fold{fold_idx}_gains.npy"

        if model_file.exists() and model_file.stat().st_size > 10_000:
            # 이미 완료된 Fold 체크포인트 즉시 복원!
            booster = lgb.Booster(model_file=str(model_file))
            x_va = x_all[va_idx][:, raw_col_idx]
            if te_col_indices:
                _, va_te, test_te = fit_fold_smoothed_target_encoding(
                    x_all[tr_idx][:, te_col_indices],
                    y_tr,
                    x_all[va_idx][:, te_col_indices],
                    x_test[:, te_col_indices],
                )
                x_va = np.hstack([x_va, va_te])
                x_te_fold = np.hstack([x_test[:, raw_col_idx], test_te])
                del va_te, test_te
            else:
                x_te_fold = x_test[:, raw_col_idx]

            best_it = booster.current_iteration()
            best_iters.append(int(best_it))
            val_pred = booster.predict(x_va).astype(np.float32)
            oof_probs[va_idx] = val_pred
            test_probs += booster.predict(x_te_fold).astype(np.float32) / 5.0

            f_gain = booster.feature_importance(importance_type="gain")
            f_split = booster.feature_importance(importance_type="split")
            gain_importances[global_feat_idx] += f_gain / 5.0
            split_importances[global_feat_idx] += f_split / 5.0

            full_raw_gains = np.zeros(n_raw_feats, dtype=np.float64)
            full_raw_gains[raw_col_idx] = f_gain[: len(raw_col_idx)]
            np.save(gains_file, full_raw_gains)

            fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
            print(
                f"  [Fold {fold_idx} RESUMED] iter={best_it} | Raw Macro F1={fold_raw_m['macro_f1']:.5f} | "
                f"{time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
            )
            del x_va, x_te_fold, booster
            gc.collect()
            continue

        # 신규 Fold 학습
        x_tr = x_all[tr_idx][:, raw_col_idx]
        x_va = x_all[va_idx][:, raw_col_idx]

        if te_col_indices:
            tr_te, va_te, test_te = fit_fold_smoothed_target_encoding(
                x_all[tr_idx][:, te_col_indices],
                y_tr,
                x_all[va_idx][:, te_col_indices],
                x_test[:, te_col_indices],
            )
            x_tr = np.hstack([x_tr, tr_te])
            x_va = np.hstack([x_va, va_te])
            x_te_fold = np.hstack([x_test[:, raw_col_idx], test_te])
            del tr_te, va_te, test_te
        else:
            x_te_fold = None

        tr_weights = None
        if sample_weight_power > 0.0:
            cls_cnts = np.bincount(y_tr, minlength=5).astype(np.float64)
            inv_w = (len(y_tr) / (5.0 * cls_cnts)) ** sample_weight_power
            tr_weights = inv_w[y_tr].astype(np.float32)

        fold_params = dict(default_params)
        fold_params["random_state"] = seed + fold_idx * 17

        dtrain = lgb.Dataset(x_tr, label=y_tr, weight=tr_weights, params=fold_params, free_raw_data=True)
        dval = lgb.Dataset(x_va, label=y_va, reference=dtrain, params=fold_params, free_raw_data=True)
        dtrain.construct()
        dval.construct()
        del x_tr
        gc.collect()

        booster = lgb.train(
            fold_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        best_it = booster.best_iteration
        best_iters.append(int(best_it))
        val_pred = booster.predict(x_va, num_iteration=best_it).astype(np.float32)
        oof_probs[va_idx] = val_pred

        if x_te_fold is None:
            x_te_fold = x_test[:, raw_col_idx]
        test_probs += booster.predict(x_te_fold, num_iteration=best_it).astype(np.float32) / 5.0

        f_gain = booster.feature_importance(importance_type="gain")
        f_split = booster.feature_importance(importance_type="split")
        gain_importances[global_feat_idx] += f_gain / 5.0
        split_importances[global_feat_idx] += f_split / 5.0

        full_raw_gains = np.zeros(n_raw_feats, dtype=np.float64)
        full_raw_gains[raw_col_idx] = f_gain[: len(raw_col_idx)]
        np.save(gains_file, full_raw_gains)

        booster.save_model(str(model_file), num_iteration=best_it)

        fold_raw_m = evaluate_predictions(y_va, np.argmax(val_pred, axis=1))
        print(
            f"  [Fold {fold_idx}] best_iter={best_it} | Raw Macro F1={fold_raw_m['macro_f1']:.5f} | "
            f"{time.time()-f_t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
        )

        del x_va, x_te_fold, dtrain, dval, booster
        gc.collect()

    del x_all, x_test
    gc.collect()

    oof_path = oof_dir / f"{exp_id}_oof.npy"
    test_path = oof_dir / f"{exp_id}_test.npy"
    np.save(oof_path, oof_probs)
    np.save(test_path, test_probs)

    imp_df = pd.DataFrame(
        {
            "feature": total_feat_names,
            "gain": gain_importances,
            "split": split_importances,
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    imp_df.to_csv(artifacts_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    post_res = nested_oof_postprocess(oof_probs, y_all, folds, n_rounds=4, reg_strength=2e-5)
    print("\n" + "=" * 88)
    print(format_metrics_summary(post_res["raw_metrics"], prefix=f"{exp_id} RAW OOF"))
    print(format_metrics_summary(post_res["nested_metrics"], prefix=f"{exp_id} NESTED-POST OOF"))
    print(
        f"  -> Consensus Multipliers (A..E): {[round(float(w), 4) for w in post_res['consensus_weights']]}"
    )
    print("=" * 88)

    summary = {
        "exp_id": exp_id,
        "model_family": "LightGBM",
        "n_features_active_per_fold": int(len(global_feat_idx)),
        "best_iterations": best_iters,
        "mean_best_iteration": float(np.mean(best_iters)),
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
