"""표준 실험 오케스트레이터 CLI (exp_001 ~ exp_006 단일 실행 및 체크포인트 Resume 지원)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.catboost_trainer import train_catboost_cv
from src.models.lgbm_trainer import train_lgbm_cv
from src.models.vip_specialist import train_vip_specialist_cv
from src.models.xgb_trainer import train_xgb_cv
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        choices=["exp_001", "exp_002", "exp_003", "exp_004", "exp_005", "exp_006"],
    )
    args = parser.parse_args()

    cfg = load_config()
    proc_dir = resolve_path(cfg["paths"]["processed_dir"])

    with (proc_dir / "base_feature_meta.json").open("r", encoding="utf-8") as f:
        base_meta = json.load(f)
    cat_cols = base_meta.get("categorical_cols", [])
    n_base = int(base_meta.get("n_features", 2060))

    tr_base = proc_dir / "train_base.parquet"
    te_base = proc_dir / "test_base.parquet"
    tr_fg1 = proc_dir / "train_fg001_time.parquet"
    te_fg1 = proc_dir / "test_fg001_time.parquet"
    tr_fg2 = proc_dir / "train_fg002_ratio.parquet"
    te_fg2 = proc_dir / "test_fg002_ratio.parquet"

    if args.exp == "exp_001":
        train_lgbm_cv(
            exp_id="exp_001",
            train_paths=[tr_base],
            test_paths=[te_base],
            cat_cols_for_te=None,
            lgb_params={"learning_rate": 0.055, "max_delta_step": 2.0, "colsample_bytree": 0.35},
            num_boost_round=1400,
            early_stopping_rounds=75,
        )

    elif args.exp == "exp_002":
        # Phase 3A: Fold별 상위 480 베이스 피처 + fg_001 (6개월 심화 변동성·최댓값·모멘텀 전체)
        train_lgbm_cv(
            exp_id="exp_002",
            train_paths=[tr_base, tr_fg1],
            test_paths=[te_base, te_fg1],
            cat_cols_for_te=None,
            ref_base_exp_id="exp_001",
            n_base_cols_count=n_base,
            top_k_base_features=480,
            lgb_params={
                "learning_rate": 0.08,
                "max_delta_step": 0.0,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 12,
                "colsample_bytree": 0.42,
            },
            num_boost_round=950,
            early_stopping_rounds=60,
        )

    elif args.exp == "exp_003":
        # Phase 3B: Fold별 상위 480 베이스 + fg_001 + fg_002(연간복원/도메인비율) + Fold내부 타겟인코딩
        train_lgbm_cv(
            exp_id="exp_003",
            train_paths=[tr_base, tr_fg1, tr_fg2],
            test_paths=[te_base, te_fg1, te_fg2],
            cat_cols_for_te=cat_cols,
            ref_base_exp_id="exp_001",
            n_base_cols_count=n_base,
            top_k_base_features=480,
            lgb_params={
                "learning_rate": 0.075,
                "max_delta_step": 0.0,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 10,
                "colsample_bytree": 0.38,
                "subsample": 0.82,
                "reg_alpha": 1.2,
                "reg_lambda": 5.0,
            },
            num_boost_round=1050,
            early_stopping_rounds=65,
        )

    elif args.exp == "exp_004":
        # Phase 3C: 계층적 VIP({A,B,C}) 3종 스페셜리스트(LGBM+CatBoost+XGB) 조건부 확률 결합
        train_vip_specialist_cv(
            exp_id="exp_004",
            train_paths=[tr_base, tr_fg1, tr_fg2],
            test_paths=[te_base, te_fg1, te_fg2],
            base_global_exp_id="exp_003",
            ref_lgbm_exp_id="exp_003",
            top_k_features=650,
            cat_cols_for_te=cat_cols,
            vip_blend_alpha=0.65,
        )

    elif args.exp == "exp_005":
        # Phase 4A: XGBoost (hist, Fold 내부 상위 420 피처 선택)
        train_xgb_cv(
            exp_id="exp_005",
            train_paths=[tr_base, tr_fg1, tr_fg2],
            test_paths=[te_base, te_fg1, te_fg2],
            ref_lgbm_exp_id="exp_003",
            top_k_features=420,
            cat_cols_for_te=cat_cols,
            xgb_params={"learning_rate": 0.08, "max_delta_step": 3.0},
            num_boost_round=950,
            early_stopping_rounds=60,
        )

    elif args.exp == "exp_006":
        # Phase 4B: CatBoost (Oblivious Trees depth=6, Fold 내부 상위 340 피처 선택)
        train_catboost_cv(
            exp_id="exp_006",
            train_paths=[tr_base, tr_fg1, tr_fg2],
            test_paths=[te_base, te_fg1, te_fg2],
            ref_lgbm_exp_id="exp_003",
            top_k_features=340,
            cat_cols_for_te=cat_cols,
            cb_params={"iterations": 1150, "learning_rate": 0.085, "early_stopping_rounds": 65},
        )


if __name__ == "__main__":
    main()
