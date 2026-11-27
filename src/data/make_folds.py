"""고객(ID) 단위 고정 5-Fold StratifiedKFold 생성기.

모든 400,000명 고객의 Segment는 6개월 내내 불변이므로, 고유 고객 테이블에서
StratifiedKFold를 수행하여 그룹 누수(Group Leakage)를 원천 차단하고
초소수 클래스 B(24명 -> Fold당 5,5,5,5,4명)와 A(162명 -> Fold당 33,33,32,32,32명)를 균등 배분합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CLASS_TO_IDX, load_config, resolve_path, set_seed


def create_fixed_folds() -> Path:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    n_folds = int(cfg["project"]["n_folds"])
    set_seed(seed)

    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    member_201807 = raw_dir / "train" / "1.회원정보" / "201807_train_회원정보.parquet"
    df = pq.read_table(member_201807, columns=["ID", "Segment"]).to_pandas()

    assert len(df) == 400_000, f"기대 고객 수 400,000명 불일치: {len(df)}"
    assert df["ID"].is_unique, "중복 고객 ID가 발견되었습니다."

    df["target"] = df["Segment"].map(CLASS_TO_IDX).astype("int8")
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df["ID"], df["target"])):
        df.loc[val_idx, "fold"] = fold_idx

    df["fold"] = df["fold"].astype("int8")
    assert (df["fold"] >= 0).all(), "미할당 Fold가 존재합니다."

    out_path = processed_dir / "folds.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[SUCCESS] 고정 {n_folds}-Fold 저장 완료: {out_path}")
    ct = pd.crosstab(df["fold"], df["Segment"])
    print("\n[Fold별 클래스 배분 현황]")
    print(ct.to_string())
    return out_path


if __name__ == "__main__":
    create_fixed_folds()
