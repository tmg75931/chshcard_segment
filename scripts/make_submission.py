"""단일 실험 또는 다중 실험 OOF 가중 블렌딩 + Nested OOF 승수 보정 기반 제출 파일 생성 및 자동 검증기."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import format_metrics_summary
from src.evaluation.postprocess import apply_multipliers, nested_oof_postprocess
from src.utils.config import IDX_TO_CLASS, load_config, resolve_path


def generate_submission(
    sub_name: str,
    exp_ids: list[str],
    blend_weights: list[float] | None = None,
    use_log_blend: bool = False,
    save_ensemble_exp_id: str | None = None,
) -> dict:
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    oof_dir = resolve_path(cfg["paths"]["oof_dir"])
    submissions_dir = resolve_path(cfg["paths"]["submissions_dir"])
    sample_sub_path = resolve_path(cfg["paths"]["sample_submission"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    folds_df = pd.read_parquet(processed_dir / "folds.parquet")
    y_all = folds_df["target"].to_numpy(dtype=np.int32)
    folds = folds_df["fold"].to_numpy(dtype=np.int32)

    if blend_weights is None:
        blend_weights = [1.0 / len(exp_ids)] * len(exp_ids)
    w_arr = np.asarray(blend_weights, dtype=np.float64)
    w_arr = w_arr / w_arr.sum()

    oofs = [np.load(oof_dir / f"{eid}_oof.npy").astype(np.float64) for eid in exp_ids]
    tests = [np.load(oof_dir / f"{eid}_test.npy").astype(np.float64) for eid in exp_ids]

    if use_log_blend:
        eps = 1e-7
        log_oof = sum(w * np.log(np.clip(p, eps, 1.0)) for w, p in zip(w_arr, oofs))
        oof_blend = np.exp(log_oof)
        oof_blend /= oof_blend.sum(axis=1, keepdims=True)

        log_test = sum(w * np.log(np.clip(p, eps, 1.0)) for w, p in zip(w_arr, tests))
        test_blend = np.exp(log_test)
        test_blend /= test_blend.sum(axis=1, keepdims=True)
    else:
        oof_blend = sum(w * p for w, p in zip(w_arr, oofs))
        test_blend = sum(w * p for w, p in zip(w_arr, tests))

    oof_blend_f32 = oof_blend.astype(np.float32)
    test_blend_f32 = test_blend.astype(np.float32)

    if save_ensemble_exp_id:
        np.save(oof_dir / f"{save_ensemble_exp_id}_oof.npy", oof_blend_f32)
        np.save(oof_dir / f"{save_ensemble_exp_id}_test.npy", test_blend_f32)

    post_res = nested_oof_postprocess(oof_blend_f32, y_all, folds, n_rounds=4, reg_strength=2e-5)
    print(format_metrics_summary(post_res["raw_metrics"], prefix=f"{sub_name} RAW"))
    print(format_metrics_summary(post_res["nested_metrics"], prefix=f"{sub_name} NESTED-POST"))

    # Consensus Multipliers 적용하여 최종 Test 예측 도출
    multipliers = post_res["consensus_weights"]
    test_pred_idx = apply_multipliers(test_blend_f32, multipliers)
    test_pred_labels = pd.Series(test_pred_idx).map(IDX_TO_CLASS)

    sub_df = pd.read_csv(sample_sub_path)
    sub_df["Segment"] = test_pred_labels.values

    if not sub_name.endswith(".csv"):
        sub_name = f"{sub_name}.csv"
    out_csv = submissions_dir / sub_name
    sub_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[SAVED] 제출 파일 생성 완료: {out_csv}")

    # 필수 7단계 검증 게이트 실행
    val_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_submission.py"), str(out_csv)]
    proc = subprocess.run(val_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"제출 파일 검증 실패: {out_csv}")

    test_dist = sub_df["Segment"].value_counts().reindex(["A", "B", "C", "D", "E"], fill_value=0).to_dict()
    meta_out = {
        "submission_file": str(out_csv.relative_to(PROJECT_ROOT)),
        "exp_ids": exp_ids,
        "blend_weights": [float(x) for x in w_arr],
        "use_log_blend": use_log_blend,
        "raw_macro_f1": post_res["raw_metrics"]["macro_f1"],
        "nested_macro_f1": post_res["nested_metrics"]["macro_f1"],
        "nested_class_f1": post_res["nested_metrics"]["class_f1"],
        "consensus_multipliers": [float(x) for x in multipliers],
        "test_pred_counts": {k: int(v) for k, v in test_dist.items()},
    }
    if save_ensemble_exp_id:
        art_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / save_ensemble_exp_id
        art_dir.mkdir(parents=True, exist_ok=True)
        with (art_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)
    return meta_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-name", required=True)
    parser.add_argument("--exps", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--log-blend", action="store_true")
    parser.add_argument("--save-exp-id", default=None)
    args = parser.parse_args()
    generate_submission(args.sub_name, args.exps, args.weights, args.log_blend, args.save_exp_id)
