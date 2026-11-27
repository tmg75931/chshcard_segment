"""도메인별 스트리밍 정제 및 고객 단위(40만/10만 행) 기본 피처셋 구축 파이프라인.

가용 RAM(~6.2 GB) 제약을 안전하게 준수하기 위해 8개 도메인을 한 번에 1개씩만 로드하여:
1. 상수 컬럼(108개) 제거, 순서형 문자열 수치화, Sentinel(-999999, 99999999 등) null 마스킹
2. 고객(ID) 기준 정적/동적 피처 구분 후 효율적 시계열 요약 수행:
   - 최신월(201812) 스냅샷 (전체 유효 피처)
   - 동적 수치형 핵심 요약: 6개월 평균(mean), 6개월 변화량(diff = 201812 - 201807)
   - 고결측/희소 변수는 불필요한 파생 중복을 억제하고 Float32 / Int16으로 다운캐스팅
3. 도메인별 중간 캐시를 병합하여 data/processed/{train_base,test_base}.parquet 저장
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaners import clean_domain_lazy_or_df, scan_domain_metadata
from src.utils.config import get_process_memory_mb, load_config, resolve_path


def detect_dynamic_numeric_cols(df_tr: pl.DataFrame, numeric_cols: list[str]) -> list[str]:
    """상위 10,000명(6개월) 샘플로 월별 변동이 실제로 존재하는 동적 수치형 컬럼 판별."""
    m0 = df_tr.filter(pl.col("기준년월") == 201807).head(10_000)
    m5 = df_tr.filter(pl.col("기준년월") == 201812).head(10_000)
    m2 = df_tr.filter(pl.col("기준년월") == 201809).head(10_000)

    dyn_cols: list[str] = []
    for c in numeric_cols:
        s0, s2, s5 = m0[c], m2[c], m5[c]
        if not (s0.equals(s5) and s0.equals(s2)):
            dyn_cols.append(c)
    return dyn_cols


def aggregate_domain_for_split(
    files: list[Path],
    const_cols: list[str],
    nominal_vocab: dict[str, list[str]],
    dyn_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """단일 도메인 6개 월별 파일을 읽어 정제 후 고객(ID) 당 1행으로 요약."""
    # 월 순서(201807 -> 201812) 보장 로드
    dfs = [pl.read_parquet(f) for f in sorted(files)]
    df = pl.concat(dfs, how="vertical")
    del dfs

    df, numeric_cols, nominal_cols = clean_domain_lazy_or_df(df, const_cols, nominal_vocab)

    if dyn_cols is None:
        dyn_cols = detect_dynamic_numeric_cols(df, numeric_cols)

    # 원본 파일의 고객 정렬 순서(201807 기준) 보존
    base_order = df.filter(pl.col("기준년월") == 201807).select("ID")

    # 1) 최신월(201812) 스냅샷 값 (모든 유효 수치형 + 명목형)
    df_last = (
        df.filter(pl.col("기준년월") == 201812)
        .drop([c for c in ("기준년월", "Segment") if c in df.columns])
    )
    rename_last = {c: f"{c}__last" for c in (numeric_cols + nominal_cols)}
    df_last = df_last.rename(rename_last)

    if not dyn_cols:
        out = base_order.join(df_last, on="ID", how="left", maintain_order="left")
        return out, dyn_cols, [f"{c}__last" for c in nominal_cols]

    # 2) 초기월(201807) 대비 최신월(201812) 변화량(diff) 계산용 초기값 추출
    df_first = df.filter(pl.col("기준년월") == 201807).select(["ID"] + dyn_cols)

    # 3) 6개월 평균(mean) 집계 (Polars 병렬 group_by)
    mean_exprs = [pl.col(c).mean().cast(pl.Float32).alias(f"{c}__mean6") for c in dyn_cols]
    df_agg = df.group_by("ID", maintain_order=False).agg(mean_exprs)

    del df
    gc.collect()

    # 결합 및 diff(201812 - 201807) 생성
    out = (
        base_order.join(df_last, on="ID", how="left", maintain_order="left")
        .join(df_agg, on="ID", how="left", maintain_order="left")
        .join(
            df_first.rename({c: f"{c}__first_tmp" for c in dyn_cols}),
            on="ID",
            how="left",
            maintain_order="left",
        )
    )
    del df_last, df_agg, df_first

    diff_exprs = [
        (pl.col(f"{c}__last") - pl.col(f"{c}__first_tmp")).cast(pl.Float32).alias(f"{c}__diff6")
        for c in dyn_cols
    ]
    out = out.with_columns(diff_exprs).drop([f"{c}__first_tmp" for c in dyn_cols])
    return out, dyn_cols, [f"{c}__last" for c in nominal_cols]


def prune_exact_duplicate_or_constant_cols(
    tr_df: pl.DataFrame, te_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """집계 후 분산이 0이 되었거나 상위 샘플 해시가 완전 동일한 중복 파생 컬럼 제거."""
    sample = tr_df.head(20_000)
    drop_cols: list[str] = []
    seen_hashes: dict[int, str] = {}

    for c in tr_df.columns:
        if c == "ID":
            continue
        s = sample[c]
        if s.n_unique() <= 1 and tr_df[c].n_unique() <= 1:
            drop_cols.append(c)
            continue
        h = int((s.hash(seed=42) % 1_000_000_007).sum())
        if h in seen_hashes:
            prev_c = seen_hashes[h]
            if tr_df[c].equals(tr_df[prev_c]):
                drop_cols.append(c)
                continue
        else:
            seen_hashes[h] = c

    if drop_cols:
        tr_df = tr_df.drop(drop_cols)
        te_df = te_df.drop(drop_cols)
    return tr_df, te_df, drop_cols


def build_base_datasets() -> None:
    t0 = time.time()
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    interim_dir = resolve_path(cfg["paths"]["interim_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    meta_cache = interim_dir / "domain_metadata.json"
    print("[1/3] 도메인 메타데이터(상수 컬럼 및 명목형 사전) 점검 중...")
    meta = scan_domain_metadata(raw_dir, meta_cache)

    domains = sorted([p.name for p in (raw_dir / "train").iterdir() if p.is_dir()])
    tr_domain_paths: list[Path] = []
    te_domain_paths: list[Path] = []
    all_cat_cols: list[str] = []

    print("[2/3] 8대 도메인 순차 스트리밍 정제 및 고객 집계 시작...")
    for idx, dom in enumerate(domains, start=1):
        dt0 = time.time()
        tr_files = sorted((raw_dir / "train" / dom).glob("*.parquet"))
        te_files = sorted((raw_dir / "test" / dom).glob("*.parquet"))
        consts = meta["constant_cols"].get(dom, [])

        tr_dom, dyn_cols, cat_cols = aggregate_domain_for_split(
            tr_files, consts, meta["nominal_vocab"], dyn_cols=None
        )
        te_dom, _, _ = aggregate_domain_for_split(
            te_files, consts, meta["nominal_vocab"], dyn_cols=dyn_cols
        )
        tr_dom, te_dom, dropped = prune_exact_duplicate_or_constant_cols(tr_dom, te_dom)
        kept_cats = [c for c in cat_cols if c in tr_dom.columns]
        all_cat_cols.extend(kept_cats)

        tr_out = interim_dir / f"base_tr_{idx}.parquet"
        te_out = interim_dir / f"base_te_{idx}.parquet"
        tr_dom.write_parquet(tr_out)
        te_dom.write_parquet(te_out)
        tr_domain_paths.append(tr_out)
        te_domain_paths.append(te_out)

        mem_mb = get_process_memory_mb()
        print(
            f"  - [{idx}/8] {dom}: 컬럼 {len(tr_dom.columns)-1:,}개 유지 (중복/상수 {len(dropped)}개 제거) | "
            f"{time.time()-dt0:.1f}s | RAM {mem_mb:.0f} MB"
        )
        del tr_dom, te_dom
        gc.collect()

    print("[3/3] 도메인별 고객 피처 수평 결합 및 최종 저장 중...")
    # 모든 파일이 동일한 고객 ID 순서임을 assert 검증 후 수평 결합
    tr_parts = [pl.read_parquet(p) for p in tr_domain_paths]
    base_tr_id = tr_parts[0]["ID"]
    for p_df in tr_parts[1:]:
        assert base_tr_id.equals(p_df["ID"]), "Train 도메인 간 ID 순서 불일치!"
    train_full = pl.concat(
        [tr_parts[0]] + [p_df.drop("ID") for p_df in tr_parts[1:]], how="horizontal"
    )
    del tr_parts
    gc.collect()

    train_out_path = processed_dir / "train_base.parquet"
    train_full.write_parquet(train_out_path)
    n_features = len(train_full.columns) - 1
    del train_full
    gc.collect()

    te_parts = [pl.read_parquet(p) for p in te_domain_paths]
    base_te_id = te_parts[0]["ID"]
    for p_df in te_parts[1:]:
        assert base_te_id.equals(p_df["ID"]), "Test 도메인 간 ID 순서 불일치!"
    test_full = pl.concat(
        [te_parts[0]] + [p_df.drop("ID") for p_df in te_parts[1:]], how="horizontal"
    )
    del te_parts
    gc.collect()

    test_out_path = processed_dir / "test_base.parquet"
    test_full.write_parquet(test_out_path)
    del test_full
    gc.collect()

    feat_meta = {
        "n_features": n_features,
        "categorical_cols": all_cat_cols,
    }
    with (processed_dir / "base_feature_meta.json").open("w", encoding="utf-8") as f:
        json.dump(feat_meta, f, ensure_ascii=False, indent=2)

    print(
        f"\n[SUCCESS] 기본 고객 피처셋 생성 완료! 총 피처 {n_features:,}개 (명목형 {len(all_cat_cols)}개) | "
        f"총 소요시간 {time.time()-t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
    )


if __name__ == "__main__":
    build_base_datasets()
