"""하네스 파이프라인 무결성 및 누수 방지 단위 테스트 스위트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.data.cleaners import clean_domain_lazy_or_df, parse_ordinal_string_value
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.postprocess import (
    _fast_macro_f1_over_candidates,
    apply_multipliers,
    nested_oof_postprocess,
)
from src.utils.config import PROJECT_ROOT


def test_ordinal_and_sentinel_cleaning() -> None:
    assert parse_ordinal_string_value("연령", "40대") == 40.0
    assert parse_ordinal_string_value("연령", "70대이상") == 70.0
    assert parse_ordinal_string_value("이용금액대", "01.100만원+") == 100.0
    assert parse_ordinal_string_value("이용금액대", "09.미사용") == 0.0
    assert parse_ordinal_string_value("할인건수_R3M", "20회 이상") == 20.0
    assert parse_ordinal_string_value("할인건수_R3M", None) is None

    df = pl.DataFrame(
        {
            "ID": ["C1", "C2", "C3"],
            "기준년월": [201807, 201807, 201807],
            "연체일수_최근": [-999999, 0, 15],
            "rv최초시작후경과일": [99999999, 120, 365],
            "상수컬럼": [1, 1, 1],
            "거주시도명": ["서울", "경기", None],
        }
    )
    cleaned, num_cols, nom_cols = clean_domain_lazy_or_df(
        df,
        constant_cols=["상수컬럼"],
        nominal_vocab={"거주시도명": ["경기", "서울"]},
    )
    assert "상수컬럼" not in cleaned.columns
    assert cleaned["연체일수_최근"].to_list() == [None, 0.0, 15.0]
    assert cleaned["rv최초시작후경과일"].to_list() == [None, 120.0, 365.0]
    assert cleaned["거주시도명"].to_list() == [1, 0, -1]
    assert set(num_cols) == {"연체일수_최근", "rv최초시작후경과일"}
    assert nom_cols == ["거주시도명"]


def test_nested_postprocess_improves_or_preserves_imbalanced() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    # 불균형 시뮬레이션 (클래스 1이 초소수)
    y = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.02, 0.01, 0.07, 0.15, 0.75])
    folds = np.arange(n) % 5

    logits = rng.normal(0, 0.8, size=(n, 5))
    for i in range(n):
        logits[i, y[i]] += 1.8
    # 다수 클래스(4) 사전확률 편향 모사
    logits[:, 4] += 1.2
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)

    res = nested_oof_postprocess(probs, y, folds, n_rounds=3)
    assert res["nested_metrics"]["macro_f1"] >= res["raw_metrics"]["macro_f1"]
    preds = apply_multipliers(probs, res["consensus_weights"])
    assert preds.shape == (n,)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4})


def test_fast_macro_f1_matches_sklearn_exact() -> None:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(123)
    n = 600
    y_true = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.02, 0.01, 0.08, 0.19, 0.70]).astype(np.int32)
    probs = rng.dirichlet(np.ones(5), size=n)
    k = 1
    other_cols = [0, 2, 3, 4]
    other_max_idx = np.argmax(probs[:, other_cols], axis=1)
    other_max_val = probs[np.arange(n), np.array(other_cols)[other_max_idx]]
    other_best_cls = np.array(other_cols, dtype=np.int32)[other_max_idx]
    cands = np.geomspace(0.1, 20.0, 25)

    fast_scores = _fast_macro_f1_over_candidates(
        y_true, other_best_cls, other_max_val, probs[:, k], k, cands, n_classes=5
    )
    ref_scores = np.array(
        [
            f1_score(
                y_true,
                np.where(w * probs[:, k] > other_max_val, k, other_best_cls),
                labels=[0, 1, 2, 3, 4],
                average="macro",
                zero_division=0,
            )
            for w in cands
        ]
    )
    assert np.allclose(fast_scores, ref_scores, atol=1e-12)


def test_folds_file_if_exists() -> None:
    folds_path = PROJECT_ROOT / "data" / "processed" / "folds.parquet"
    if folds_path.exists():
        df = pd.read_parquet(folds_path)
        assert len(df) == 400_000
        assert df["ID"].is_unique
        assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}
        # 초소수 B 클래스(24명)가 모든 Fold에 최소 4명 이상 배분되었는지 검증
        b_counts = df[df["Segment"] == "B"]["fold"].value_counts()
        assert len(b_counts) == 5
        assert b_counts.min() >= 4
