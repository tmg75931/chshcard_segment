"""프로젝트 전역 설정, 시드 고정, 메모리 최적화 유틸리티."""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"

CLASS_NAMES: list[str] = ["A", "B", "C", "D", "E"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASS_NAMES)}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_process_memory_mb() -> float:
    """현재 프로세스 물리 메모리(RSS, MB) 반환."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def downcast_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 수치형 컬럼을 정밀도 손실 범위 내에서 최소 메모리 타입으로 다운캐스팅."""
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_float_dtype(dt):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dt):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    gc.collect()
    return df
