"""Phase 3 파생변수 엔지니어링 파이프라인 (6개월 심화 시계열 통계 + 카드 금융 도메인 비율).

생성 피처 그룹 (독립 Parquet 아티팩트로 분리 저장하여 Ablation 실험 가능):
1. fg_001_time_agg (data/processed/{train,test}_fg001_time.parquet):
   - 핵심 금융 동적 변수(~78개)에 대한 6개월 표준편차(std6), 최댓값(max6), 최솟값(min6),
     하반기 모멘텀(최근 3개월 평균 - 초기 3개월 평균 및 비율), 최신월 대비 평균 괴리율, 유실적 개월수
2. fg_002_credit_ratio (data/processed/{train,test}_fg002_ratio.parquet):
   - 8대 도메인 교차 금융 비율(한도소진율, 한도성장률, 상향여력비율, 일시불/할부/유이자할부/CA/카드론 비중,
     이자성 여신잔액 비중, 건당 평균 승인티켓 금액, 입금충당률, 연체비율, 온라인/해외/프리미엄 라이프스타일 소비 합계 등)의
     최신월(last), 6개월 평균(mean6), 6개월 최댓값(max6), 6개월 변화량(diff6)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaners import NEGATIVE_SENTINELS, POSITIVE_DATE_SENTINELS
from src.utils.config import get_process_memory_mb, load_config, resolve_path

# 도메인별 시계열 심화 집계(std, max, min, 3m momentum) 핵심 타겟 컬럼
KEY_TEMPORAL_COLS: dict[str, list[str]] = {
    "1.회원정보": [
        "유효카드수_신용체크",
        "유효카드수_신용",
        "유효카드수_체크",
        "이용카드수_신용체크",
        "이용카드수_신용",
        "이용금액_R3M_신용체크",
        "이용금액_R3M_신용",
        "이용금액_R3M_체크",
        "_1순위카드이용금액",
        "_1순위카드이용건수",
        "_2순위카드이용금액",
        "_2순위카드이용건수",
    ],
    "2.신용정보": [
        "카드이용한도금액",
        "CA한도금액",
        "일시상환론한도금액",
        "월상환론한도금액",
        "상향가능한도금액",
        "상향가능CA한도금액",
        "CA이자율_할인전",
        "CL이자율_할인전",
        "RV일시불이자율_할인전",
    ],
    "3.승인매출정보": [
        "이용건수_신용_B0M",
        "이용건수_일시불_B0M",
        "이용건수_할부_B0M",
        "이용건수_CA_B0M",
        "이용건수_체크_B0M",
        "이용금액_일시불_B0M",
        "이용금액_할부_B0M",
        "이용금액_할부_유이자_B0M",
        "이용금액_할부_무이자_B0M",
        "이용금액_CA_B0M",
        "이용금액_체크_B0M",
        "이용금액_카드론_B0M",
        "이용금액_일시불_R12M",
        "이용금액_할부_R12M",
        "이용금액_CA_R12M",
        "이용금액_체크_R12M",
        "최대이용금액_일시불_R12M",
        "최대이용금액_할부_R12M",
        "최대이용금액_CA_R12M",
        "이용금액_일시불_R6M",
        "이용금액_할부_R6M",
        "이용금액_일시불_R3M",
        "이용금액_할부_R3M",
        "이용가맹점수",
        "이용금액_해외",
        "쇼핑_백화점_이용금액",
        "쇼핑_마트_이용금액",
        "쇼핑_온라인_이용금액",
        "_1순위업종_이용금액",
        "_2순위업종_이용금액",
        "RP금액_B0M",
        "이용금액_온라인_B0M",
        "이용금액_오프라인_B0M",
        "이용금액_온라인_R3M",
        "이용금액_오프라인_R3M",
        "이용금액_온라인_R6M",
        "이용금액_오프라인_R6M",
        "이용금액_간편결제_B0M",
        "최대이용금액_체크_R12M",
        "가맹점매출금액_B1M",
        "가맹점매출금액_B2M",
        "정상청구원금_B0M",
        "선입금원금_B0M",
        "정상입금원금_B0M",
        "연체입금원금_B0M",
        "정상청구원금_B2M",
        "정상입금원금_B2M",
        "연체입금원금_B2M",
        "정상청구원금_B5M",
        "선입금원금_B5M",
        "정상입금원금_B5M",
        "연체입금원금_B5M",
        "연속유실적개월수_기본_24M_카드",
        "연속무실적개월수_기본_24M_카드",
    ],
    "4.청구입금정보": [
        "청구금액_B0",
        "청구금액_R3M",
        "청구금액_R6M",
        "포인트_적립포인트_R12M",
        "포인트_적립포인트_R3M",
        "포인트_잔여포인트_B0M",
        "마일_적립포인트_R12M",
        "할인금액_R3M",
        "할인금액_B0M",
        "혜택수혜금액_R3M",
        "혜택수혜금액",
    ],
    "5.잔액정보": [
        "잔액_일시불_B0M",
        "잔액_할부_B0M",
        "잔액_현금서비스_B0M",
        "잔액_리볼빙일시불이월_B0M",
        "잔액_카드론_B0M",
        "월중평잔_일시불_B0M",
        "월중평잔_할부_B0M",
        "월중평잔_CA_B0M",
        "연체잔액_B0M",
        "RV_평균잔액_R12M",
        "RV_최대잔액_R12M",
        "월중평잔",
        "월중평잔_일시불",
        "월중평잔_RV일시불",
        "월중평잔_할부",
        "월중평잔_CA",
        "월중평잔_카드론",
        "평잔_3M",
        "평잔_일시불_3M",
        "평잔_6M",
        "평잔_일시불_6M",
        "평잔_할부_6M",
        "평잔_CA_6M",
        "평잔_카드론_6M",
    ],
    "8.성과정보": [
        "잔액_신판평균한도소진율_r6m",
        "잔액_신판최대한도소진율_r6m",
        "잔액_신판평균한도소진율_r3m",
        "잔액_신판ca평균한도소진율_r6m",
        "잔액_신판ca최대한도소진율_r6m",
        "증감율_이용금액_신용_분기",
        "증감율_이용금액_일시불_분기",
        "혜택수혜율_R3M",
        "혜택수혜율_B0M",
    ],
}


def _clean_numeric_exprs(cols: list[str]) -> list[pl.Expr]:
    sentinels_f64 = [float(x) for x in (NEGATIVE_SENTINELS + POSITIVE_DATE_SENTINELS + [-99])]
    return [
        pl.when(pl.col(c).cast(pl.Float64).is_in(sentinels_f64))
        .then(None)
        .otherwise(pl.col(c))
        .cast(pl.Float32)
        .alias(c)
        for c in cols
    ]


def build_fg001_temporal_for_split(raw_split_dir: Path, months: list[int]) -> pl.DataFrame:
    """각 고객은 매월 동일 행 순서로 정렬되어 있으므로 (N_customers, 6) 3D/수평 배열 벡터 연산으로 초고속 집계!"""
    f_ref = raw_split_dir / "1.회원정보" / f"{months[0]}_{raw_split_dir.name}_회원정보.parquet"
    base_ids = pl.read_parquet(f_ref, columns=["ID"])
    n_cust = len(base_ids)

    out_cols: dict[str, np.ndarray] = {}

    suffix_map = {
        "1.회원정보": "회원정보",
        "2.신용정보": "신용정보",
        "3.승인매출정보": "승인매출정보",
        "4.청구입금정보": "청구정보",
        "5.잔액정보": "잔액정보",
        "8.성과정보": "성과정보",
    }

    for dom, cols in KEY_TEMPORAL_COLS.items():
        sfx = suffix_map[dom]
        # (6, N_customers, len(cols)) 텐서 구축 -> group_by 없이 순수 NumPy 벡터화!
        month_mats = []
        for m in months:
            fpath = raw_split_dir / dom / f"{m}_{raw_split_dir.name}_{sfx}.parquet"
            df_m = pl.read_parquet(fpath, columns=["ID"] + cols)
            assert df_m["ID"].equals(base_ids["ID"]), f"{fpath} ID 정렬 불일치!"
            df_m = df_m.drop("ID").with_columns(_clean_numeric_exprs(cols))
            month_mats.append(df_m.to_numpy().astype(np.float32, copy=False))
            del df_m

        tensor = np.stack(month_mats, axis=1)  # (N_cust, 6, C)
        del month_mats

        with np.errstate(invalid="ignore", divide="ignore"):
            std6 = np.nanstd(tensor, axis=1).astype(np.float32)
            max6 = np.nanmax(tensor, axis=1).astype(np.float32)
            min6 = np.nanmin(tensor, axis=1).astype(np.float32)
            mean6 = np.nanmean(tensor, axis=1).astype(np.float32)
            f3_mean = np.nanmean(tensor[:, :3, :], axis=1).astype(np.float32)
            r3_mean = np.nanmean(tensor[:, 3:, :], axis=1).astype(np.float32)
            mom_diff = (r3_mean - f3_mean).astype(np.float32)
            last_val = tensor[:, -1, :]
            # 최신월 대비 6개월 평균 상대 배율 (부호 보존 안전 나눗셈)
            surge_ratio = np.where(
                np.abs(mean6) > 1.0,
                (last_val - mean6) / (np.abs(mean6) + 10.0),
                0.0,
            ).astype(np.float32)
            pos_months = (np.nan_to_num(tensor, nan=0.0) > 0.0).sum(axis=1).astype(np.float32)

        for j, cname in enumerate(cols):
            out_cols[f"{cname}__std6"] = std6[:, j]
            out_cols[f"{cname}__max6"] = max6[:, j]
            out_cols[f"{cname}__min6"] = min6[:, j]
            out_cols[f"{cname}__mom3m"] = mom_diff[:, j]
            out_cols[f"{cname}__surge"] = surge_ratio[:, j]
            if any(k in cname for k in ("이용금액", "청구금액", "잔액", "평잔", "입금원금")):
                out_cols[f"{cname}__pos_cnt6"] = pos_months[:, j]

        del tensor
        gc.collect()

    df_out = pl.DataFrame({"ID": base_ids["ID"], **{k: pl.Series(k, v) for k, v in out_cols.items()}})
    return df_out


def build_fg002_domain_ratios_for_split(raw_split_dir: Path, months: list[int]) -> pl.DataFrame:
    """매월 8대 도메인 교차 금융 비율을 계산한 뒤 고객별 last, mean6, max6, diff6 추출."""
    split_name = raw_split_dir.name
    f_ref = raw_split_dir / "1.회원정보" / f"{months[0]}_{split_name}_회원정보.parquet"
    base_ids = pl.read_parquet(f_ref, columns=["ID"])

    ratio_names: list[str] = []
    monthly_ratio_mats: list[np.ndarray] = []

    for m in months:
        mem = pl.read_parquet(
            raw_split_dir / "1.회원정보" / f"{m}_{split_name}_회원정보.parquet",
            columns=[
                "ID",
                "입회경과개월수_신용",
                "유효카드수_신용",
                "유효카드수_체크",
                "이용카드수_신용",
                "이용금액_R3M_신용체크",
                "이용금액_R3M_신용",
                "_1순위카드이용금액",
                "_2순위카드이용금액",
            ],
        )
        crd = pl.read_parquet(
            raw_split_dir / "2.신용정보" / f"{m}_{split_name}_신용정보.parquet",
            columns=["최초한도금액", "카드이용한도금액", "CA한도금액", "월상환론한도금액", "상향가능한도금액"],
        )
        apr = pl.read_parquet(
            raw_split_dir / "3.승인매출정보" / f"{m}_{split_name}_승인매출정보.parquet",
            columns=[
                "이용건수_일시불_B0M",
                "이용건수_할부_B0M",
                "이용금액_일시불_B0M",
                "이용금액_할부_B0M",
                "이용금액_할부_유이자_B0M",
                "이용금액_CA_B0M",
                "이용금액_체크_B0M",
                "이용금액_카드론_B0M",
                "이용금액_해외",
                "쇼핑_백화점_이용금액",
                "쇼핑_온라인_이용금액",
                "여유_항공이용금액",
                "여유_여행이용금액",
                "여유_숙박이용금액",
                "이용금액_온라인_R6M",
                "이용금액_오프라인_R6M",
                "이용금액_오프라인_B0M",
                "이용금액_오프라인_R3M",
                "이용금액_체크_R12M",
                "이용금액_일시불_R12M",
                "이용금액_할부_R12M",
                "정상청구원금_B0M",
                "선입금원금_B0M",
                "정상입금원금_B0M",
                "연체입금원금_B0M",
                "정상청구원금_B2M",
                "정상입금원금_B2M",
                "연체입금원금_B2M",
                "정상청구원금_B5M",
                "선입금원금_B5M",
                "정상입금원금_B5M",
                "연체입금원금_B5M",
            ],
        )
        bil = pl.read_parquet(
            raw_split_dir / "4.청구입금정보" / f"{m}_{split_name}_청구정보.parquet",
            columns=["청구금액_B0", "청구금액_R6M", "혜택수혜금액_R3M", "포인트_잔여포인트_B0M"],
        )
        bal = pl.read_parquet(
            raw_split_dir / "5.잔액정보" / f"{m}_{split_name}_잔액정보.parquet",
            columns=[
                "잔액_일시불_B0M",
                "잔액_할부_B0M",
                "잔액_현금서비스_B0M",
                "잔액_리볼빙일시불이월_B0M",
                "잔액_리볼빙CA이월_B0M",
                "잔액_카드론_B0M",
                "연체잔액_B0M",
                "월중평잔",
                "평잔_6M",
            ],
        )
        assert mem["ID"].equals(base_ids["ID"])

        d = pl.concat([mem.drop("ID"), crd, apr, bil, bal], how="horizontal")
        d = d.with_columns(_clean_numeric_exprs(d.columns))

        # 음수 승인금액(취소 등) 안전 클리핑 보조식
        lump = pl.col("이용금액_일시불_B0M").clip(0.0).fill_null(0.0)
        inst = pl.col("이용금액_할부_B0M").clip(0.0).fill_null(0.0)
        inst_int = pl.col("이용금액_할부_유이자_B0M").clip(0.0).fill_null(0.0)
        ca_amt = pl.col("이용금액_CA_B0M").clip(0.0).fill_null(0.0)
        cl_amt = pl.col("이용금액_카드론_B0M").clip(0.0).fill_null(0.0)
        chk_amt = pl.col("이용금액_체크_B0M").clip(0.0).fill_null(0.0)

        tot_credit_spend = lump + inst + ca_amt + cl_amt
        tot_all_spend = tot_credit_spend + chk_amt

        bal_lump = pl.col("잔액_일시불_B0M").clip(0.0).fill_null(0.0)
        bal_inst = pl.col("잔액_할부_B0M").clip(0.0).fill_null(0.0)
        bal_ca = pl.col("잔액_현금서비스_B0M").clip(0.0).fill_null(0.0)
        bal_rv1 = pl.col("잔액_리볼빙일시불이월_B0M").clip(0.0).fill_null(0.0)
        bal_rv2 = pl.col("잔액_리볼빙CA이월_B0M").clip(0.0).fill_null(0.0)
        bal_cl = pl.col("잔액_카드론_B0M").clip(0.0).fill_null(0.0)

        interest_bal = bal_inst + bal_ca + bal_rv1 + bal_rv2 + bal_cl
        tot_bal = bal_lump + interest_bal
        lim = pl.col("카드이용한도금액").clip(0.0).fill_null(0.0)
        init_lim = pl.col("최초한도금액").clip(0.0).fill_null(0.0)
        ca_lim = pl.col("CA한도금액").clip(0.0).fill_null(0.0)

        premium_spend = (
            pl.col("쇼핑_백화점_이용금액").clip(0.0).fill_null(0.0)
            + pl.col("이용금액_해외").clip(0.0).fill_null(0.0)
            + pl.col("여유_항공이용금액").clip(0.0).fill_null(0.0)
            + pl.col("여유_여행이용금액").clip(0.0).fill_null(0.0)
            + pl.col("여유_숙박이용금액").clip(0.0).fill_null(0.0)
        )

        ratios_df = d.select(
            [
                tot_credit_spend.cast(pl.Float32).alias("rat_총신용승인액"),
                tot_all_spend.cast(pl.Float32).alias("rat_총신용체크승인액"),
                tot_bal.cast(pl.Float32).alias("rat_총잔액합계"),
                interest_bal.cast(pl.Float32).alias("rat_이자성여신잔액합계"),
                premium_spend.cast(pl.Float32).alias("rat_프리미엄라이프소비액"),
                (tot_credit_spend / (lim + 10.0)).cast(pl.Float32).alias("rat_한도대비총승인율"),
                ((lump + inst) / (lim + 10.0)).cast(pl.Float32).alias("rat_한도대비신판소진율"),
                (tot_bal / (lim + 10.0)).cast(pl.Float32).alias("rat_한도대비총잔액비율"),
                (bal_ca / (ca_lim + 10.0)).cast(pl.Float32).alias("rat_CA한도소진율"),
                (lim / (init_lim + 10.0)).cast(pl.Float32).alias("rat_초기대비한도성장배율"),
                (
                    pl.col("상향가능한도금액").clip(0.0).fill_null(0.0) / (lim + 10.0)
                ).cast(pl.Float32).alias("rat_한도상향여력비율"),
                (inst / (tot_credit_spend + 10.0)).cast(pl.Float32).alias("rat_승인중할부비중"),
                (inst_int / (inst + 10.0)).cast(pl.Float32).alias("rat_할부중유이자비중"),
                ((ca_amt + cl_amt) / (tot_credit_spend + 10.0)).cast(pl.Float32).alias("rat_승인중현금대출비중"),
                (chk_amt / (tot_all_spend + 10.0)).cast(pl.Float32).alias("rat_전체승인중체크비중"),
                (interest_bal / (tot_bal + 10.0)).cast(pl.Float32).alias("rat_잔액중이자성여신비중"),
                ((bal_rv1 + bal_rv2) / (tot_bal + 10.0)).cast(pl.Float32).alias("rat_잔액중리볼빙비중"),
                (
                    lump / (pl.col("이용건수_일시불_B0M").clip(0.0).fill_null(0.0) + 1.0)
                ).cast(pl.Float32).alias("rat_일시불건당평균티켓금액"),
                (
                    pl.col("이용금액_R3M_신용").clip(0.0).fill_null(0.0)
                    / (pl.col("유효카드수_신용").clip(0.0).fill_null(0.0) + 1.0)
                ).cast(pl.Float32).alias("rat_유효신용카드당R3M이용액"),
                (
                    pl.col("이용카드수_신용").clip(0.0).fill_null(0.0)
                    / (pl.col("유효카드수_신용").clip(0.0).fill_null(0.0) + 0.5)
                ).cast(pl.Float32).alias("rat_보유신용카드활성률"),
                (
                    pl.col("_1순위카드이용금액").clip(0.0).fill_null(0.0)
                    / (pl.col("이용금액_R3M_신용체크").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_1순위카드집중도"),
                (
                    (
                        pl.col("정상입금원금_B0M").clip(0.0).fill_null(0.0)
                        + pl.col("선입금원금_B0M").clip(0.0).fill_null(0.0)
                    )
                    / (pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_청구원금대비정상선입금충당률"),
                (
                    pl.col("연체입금원금_B0M").clip(0.0).fill_null(0.0)
                    / (pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_청구원금대비연체원금비율"),
                (
                    pl.col("청구금액_B0").clip(0.0).fill_null(0.0) * 6.0
                    / (pl.col("청구금액_R6M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_최근6M평균대비당월청구배율"),
                (
                    interest_bal / (pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_정상청구원금대비이자성잔액배율"),
                (
                    pl.col("이용금액_온라인_R6M").clip(0.0).fill_null(0.0)
                    / (
                        pl.col("이용금액_온라인_R6M").clip(0.0).fill_null(0.0)
                        + pl.col("이용금액_오프라인_R6M").clip(0.0).fill_null(0.0)
                        + 10.0
                    )
                ).cast(pl.Float32).alias("rat_온라인결제비중R6M"),
                (premium_spend / (tot_credit_spend + 10.0)).cast(pl.Float32).alias("rat_승인중프리미엄소비비중"),
                (
                    pl.col("혜택수혜금액_R3M").clip(0.0).fill_null(0.0)
                    / (pl.col("이용금액_R3M_신용체크").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_실질혜택수혜율R3M"),
                # 11개월 연간 청구원금 복원 시그널 (B5M + B2M + B0M 결합 및 상·하반기 성장률)
                (
                    pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0)
                    + pl.col("정상청구원금_B2M").clip(0.0).fill_null(0.0)
                    + pl.col("정상청구원금_B5M").clip(0.0).fill_null(0.0)
                ).cast(pl.Float32).alias("rat_연간복원정상청구원금합"),
                (
                    pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0)
                    / (pl.col("정상청구원금_B5M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_5개월전대비정상청구원금성장배율"),
                (
                    pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0)
                    - pl.col("정상청구원금_B5M").clip(0.0).fill_null(0.0)
                ).cast(pl.Float32).alias("rat_5개월전대비정상청구원금증감"),
                (
                    (
                        pl.col("연체입금원금_B0M").clip(0.0).fill_null(0.0)
                        + pl.col("연체입금원금_B2M").clip(0.0).fill_null(0.0)
                        + pl.col("연체입금원금_B5M").clip(0.0).fill_null(0.0)
                    )
                    / (
                        pl.col("정상청구원금_B0M").clip(0.0).fill_null(0.0)
                        + pl.col("정상청구원금_B2M").clip(0.0).fill_null(0.0)
                        + pl.col("정상청구원금_B5M").clip(0.0).fill_null(0.0)
                        + 10.0
                    )
                ).cast(pl.Float32).alias("rat_연간복원연체원금비율"),
                (
                    pl.col("이용금액_일시불_R12M").clip(0.0).fill_null(0.0)
                    + pl.col("이용금액_할부_R12M").clip(0.0).fill_null(0.0)
                    + pl.col("이용금액_체크_R12M").clip(0.0).fill_null(0.0)
                ).cast(pl.Float32).alias("rat_연간R12M신판체크총이용액"),
                (
                    pl.col("이용금액_할부_R12M").clip(0.0).fill_null(0.0)
                    / (
                        pl.col("이용금액_일시불_R12M").clip(0.0).fill_null(0.0)
                        + pl.col("이용금액_할부_R12M").clip(0.0).fill_null(0.0)
                        + 10.0
                    )
                ).cast(pl.Float32).alias("rat_연간R12M신판중할부비중"),
                (
                    pl.col("이용금액_오프라인_B0M").clip(0.0).fill_null(0.0) * 3.0
                    / (pl.col("이용금액_오프라인_R3M").clip(0.0).fill_null(0.0) + 10.0)
                ).cast(pl.Float32).alias("rat_오프라인R3M대비당월가속도"),
            ]
        )
        if not ratio_names:
            ratio_names = list(ratios_df.columns)
        monthly_ratio_mats.append(ratios_df.to_numpy().astype(np.float32, copy=False))
        del mem, crd, apr, bil, bal, d, ratios_df
        gc.collect()

    tensor = np.stack(monthly_ratio_mats, axis=1)  # (N_cust, 6, R)
    del monthly_ratio_mats

    with np.errstate(invalid="ignore"):
        last_v = tensor[:, -1, :]
        mean_v = np.nanmean(tensor, axis=1).astype(np.float32)
        max_v = np.nanmax(tensor, axis=1).astype(np.float32)
        std_v = np.nanstd(tensor, axis=1).astype(np.float32)
        diff_v = (tensor[:, -1, :] - tensor[:, 0, :]).astype(np.float32)

    out_cols: dict[str, np.ndarray] = {}
    for j, rname in enumerate(ratio_names):
        out_cols[f"{rname}__last"] = last_v[:, j]
        out_cols[f"{rname}__mean6"] = mean_v[:, j]
        out_cols[f"{rname}__max6"] = max_v[:, j]
        out_cols[f"{rname}__std6"] = std_v[:, j]
        out_cols[f"{rname}__diff6"] = diff_v[:, j]

    del tensor
    gc.collect()
    return pl.DataFrame({"ID": base_ids["ID"], **{k: pl.Series(k, v) for k, v in out_cols.items()}})


def build_all_engineered_features() -> None:
    t0 = time.time()
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    months = cfg["project"]["months"]

    print("[Phase 3-A] fg_001_time_agg (6개월 심화 시계열 변동성·최댓값·모멘텀) 생성 중...")
    tr_t = build_fg001_temporal_for_split(raw_dir / "train", months)
    te_t = build_fg001_temporal_for_split(raw_dir / "test", months)
    tr_t.write_parquet(processed_dir / "train_fg001_time.parquet")
    te_t.write_parquet(processed_dir / "test_fg001_time.parquet")
    n_fg1 = len(tr_t.columns) - 1
    print(f"  -> fg_001 완료: {n_fg1}개 피처 | RAM {get_process_memory_mb():.0f} MB")
    del tr_t, te_t
    gc.collect()

    print("[Phase 3-B] fg_002_credit_ratio (카드 금융 도메인 교차 비율 28종 × 5요약) 생성 중...")
    tr_r = build_fg002_domain_ratios_for_split(raw_dir / "train", months)
    te_r = build_fg002_domain_ratios_for_split(raw_dir / "test", months)
    tr_r.write_parquet(processed_dir / "train_fg002_ratio.parquet")
    te_r.write_parquet(processed_dir / "test_fg002_ratio.parquet")
    n_fg2 = len(tr_r.columns) - 1
    print(
        f"  -> fg_002 완료: {n_fg2}개 피처 | 총 소요 {time.time()-t0:.1f}s | RAM {get_process_memory_mb():.0f} MB"
    )
    del tr_r, te_r
    gc.collect()


if __name__ == "__main__":
    build_all_engineered_features()
