"""원천 데이터 정제 모듈 (상수 컬럼 식별, 순서형 문자열 수치화, Sentinel 마스킹).

EDA 및 전수 스캔으로 확인된 데이터 특성을 Polars 벡터 연산으로 고속 처리합니다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq

# 명확한 서열/구간 의미를 갖는 문자열 컬럼 매핑
EXPLICIT_ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "연령": {
        "20대": 20.0,
        "30대": 30.0,
        "40대": 40.0,
        "50대": 50.0,
        "60대": 60.0,
        "70대이상": 70.0,
    },
    "이용금액대": {
        "09.미사용": 0.0,
        "05.10만원-": 5.0,
        "04.10만원+": 10.0,
        "03.30만원+": 30.0,
        "02.50만원+": 50.0,
        "01.100만원+": 100.0,
    },
    "카드론동의여부": {"N": 0.0, "Y": 1.0},
    "RV전환가능여부": {"N": 0.0, "Z": 1.0},
    "OS구분코드": {"Android": 0.0, "IOS": 1.0},
    "_1순위신용체크구분": {"체크": 0.0, "신용": 1.0},
    "_2순위신용체크구분": {"체크": 0.0, "신용": 1.0},
}

# 정규표현식으로 선행 숫자를 추출할 순서형 횟수/일수/개수 문자열 컬럼
REGEX_NUMERIC_STRING_COLS: set[str] = {
    "연회비발생카드수_B0M",
    "자발한도감액횟수_R12M",
    "한도증액횟수_R12M",
    "한도심사요청건수",
    "할인건수_R3M",
    "할인건수_B0M",
    "인입횟수_ARS_R6M",
    "이용메뉴건수_ARS_R6M",
    "방문횟수_PC_R6M",
    "방문일수_PC_R6M",
    "방문횟수_앱_R6M",
    "캠페인접촉건수_R12M",
    "캠페인접촉일수_R12M",
}

# 금융 합성데이터에서 미해당/미발생을 나타내는 대표적 극단 Sentinel 값
NEGATIVE_SENTINELS: list[int] = [-99999999, -9999999, -999999, -99999, -9999, -999]
POSITIVE_DATE_SENTINELS: list[int] = [99999999, 9999999]


def parse_ordinal_string_value(col_name: str, val: str | None) -> float | None:
    """단일 문자열 값을 수치형 순서값으로 변환 (단위 테스트 및 검증용 헬퍼)."""
    if val is None:
        return None
    if col_name in EXPLICIT_ORDINAL_MAPS:
        return EXPLICIT_ORDINAL_MAPS[col_name].get(val)
    if col_name in REGEX_NUMERIC_STRING_COLS:
        m = re.search(r"(\d+)", val)
        return float(m.group(1)) if m else None
    return None


def scan_domain_metadata(raw_dir: Path, cache_path: Path) -> dict[str, Any]:
    """Train 전체(240만 행)를 스캔하여 상수 컬럼(108개)과 명목형 문자열 카테고리 사전을 캐싱."""
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    domains = sorted([p.name for p in (raw_dir / "train").iterdir() if p.is_dir()])
    constant_cols: dict[str, list[str]] = {}
    nominal_vocab: dict[str, list[str]] = {}

    for dom in domains:
        tr_files = sorted((raw_dir / "train" / dom).glob("*.parquet"))
        schema = pq.read_schema(tr_files[0])
        feat_fields = [f for f in schema if f.name not in ("ID", "기준년월", "Segment")]

        dom_consts: list[str] = []
        col_names = [f.name for f in feat_fields]
        col_types = {f.name: str(f.type) for f in feat_fields}

        for i in range(0, len(col_names), 40):
            batch = col_names[i : i + 40]
            tbl = pq.read_table(tr_files, columns=batch)
            for c in batch:
                arr = tbl.column(c)
                n_uniq = pc.count_distinct(arr, mode="only_valid").as_py()
                total_states = n_uniq + (1 if arr.null_count > 0 else 0)
                if total_states <= 1:
                    dom_consts.append(c)
                    continue
                if (
                    col_types[c] == "string"
                    and c not in EXPLICIT_ORDINAL_MAPS
                    and c not in REGEX_NUMERIC_STRING_COLS
                ):
                    uniq_vals = sorted([v for v in pc.unique(arr).to_pylist() if v is not None])
                    nominal_vocab[c] = uniq_vals

        constant_cols[dom] = dom_consts

    meta = {
        "constant_cols": constant_cols,
        "nominal_vocab": nominal_vocab,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def clean_domain_lazy_or_df(
    df: pl.DataFrame,
    constant_cols: list[str],
    nominal_vocab: dict[str, list[str]],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """단일 도메인 Polars DataFrame에서 상수 제거, 문자열 변환, Sentinel 마스킹 수행.

    Returns:
        (정제된 DataFrame, 수치형 피처명 리스트, 명목형 피처명 리스트)
    """
    drop_targets = [c for c in constant_cols if c in df.columns]
    if drop_targets:
        df = df.drop(drop_targets)

    exprs: list[pl.Expr] = []
    numeric_cols: list[str] = []
    nominal_cols: list[str] = []

    for col_name, dtype in zip(df.columns, df.dtypes):
        if col_name in ("ID", "기준년월", "Segment"):
            continue

        if dtype == pl.String or dtype == pl.Utf8:
            if col_name in EXPLICIT_ORDINAL_MAPS:
                mapping = EXPLICIT_ORDINAL_MAPS[col_name]
                exprs.append(
                    pl.col(col_name).replace_strict(mapping, default=None).cast(pl.Float32).alias(col_name)
                )
                numeric_cols.append(col_name)
            elif col_name in REGEX_NUMERIC_STRING_COLS:
                exprs.append(
                    pl.col(col_name)
                    .str.extract(r"(\d+)", 1)
                    .cast(pl.Float32)
                    .alias(col_name)
                )
                numeric_cols.append(col_name)
            else:
                vocab = nominal_vocab.get(col_name, [])
                cat_map = {v: idx for idx, v in enumerate(vocab)}
                # 결측 또는 미등록 값은 -1 (Int16)
                exprs.append(
                    pl.col(col_name)
                    .replace_strict(cat_map, default=-1)
                    .fill_null(-1)
                    .cast(pl.Int16)
                    .alias(col_name)
                )
                nominal_cols.append(col_name)
        else:
            # 수치형 컬럼 Sentinel 마스킹
            col_expr = pl.col(col_name)
            # 극단 음수 코드 마스킹 (-99 포함 회차/연체 코드 처리)
            sentinels = list(NEGATIVE_SENTINELS)
            if "회차" in col_name or "연체" in col_name or "일수" in col_name or "경과" in col_name:
                sentinels.extend([-99, 99999999, 9999999])
            else:
                sentinels.extend(POSITIVE_DATE_SENTINELS)

            sentinels_f64 = [float(x) for x in sentinels]
            cleaned_expr = (
                pl.when(col_expr.cast(pl.Float64).is_in(sentinels_f64))
                .then(None)
                .otherwise(col_expr)
                .cast(pl.Float32)
                .alias(col_name)
            )
            exprs.append(cleaned_expr)
            numeric_cols.append(col_name)

    if exprs:
        df = df.with_columns(exprs)

    return df, numeric_cols, nominal_cols
