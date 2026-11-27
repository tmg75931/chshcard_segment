"""원본 데이터(data/raw/) 입고 상태 및 기본 스키마 검증 스크립트.

외부 패키지 미설치 환경에서도 기본 파일 트리·용량·sample_submission.csv 점검이 동작하며,
pyarrow가 설치되어 있으면 Parquet 메타데이터(행 수, 컬럼 수)를 메모리 로드 없이 빠르게 스캔합니다.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def inspect_sample_submission(csv_path: Path) -> None:
    print(f"\n[2] 제출 양식 점검: {csv_path.relative_to(PROJECT_ROOT)}")
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        row_count = 0
        sample_ids: list[str] = []
        values: Counter[str] = Counter()
        for row in reader:
            row_count += 1
            if len(sample_ids) < 3 and row:
                sample_ids.append(row[0])
            if len(row) > 1:
                values[row[1]] += 1

    print(f"  - 헤더 컬럼 : {header}")
    print(f"  - 총 행 수  : {row_count:,}행")
    print(f"  - 샘플 ID   : {sample_ids}")
    print(f"  - 초기 값 분포: {dict(values)}")


def inspect_parquet_metadata(parquet_files: list[Path]) -> None:
    print(f"\n[3] Parquet 파일 요약 (총 {len(parquet_files)}개 파일)")
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        print("  - (참고) pyarrow 미설치 상태이므로 파일 개수·용량만 표시합니다.")
        return

    by_parent: dict[str, list[Path]] = {}
    for p in sorted(parquet_files):
        rel_parent = str(p.parent.relative_to(RAW_DIR))
        by_parent.setdefault(rel_parent, []).append(p)

    for folder, files in sorted(by_parent.items()):
        first_meta = pq.read_metadata(files[0])
        total_rows = sum(pq.read_metadata(f).num_rows for f in files)
        print(
            f"  - [{folder}] 파일 {len(files)}개 | 총 {total_rows:,}행 | "
            f"첫 파일 컬럼 {first_meta.num_columns}개 ({files[0].name})"
        )


def main() -> int:
    print("=" * 70)
    print(f"원본 데이터 디렉토리 점검: {RAW_DIR}")
    print("=" * 70)

    if not RAW_DIR.exists():
        print("[ERROR] data/raw 디렉토리가 존재하지 않습니다.")
        return 1

    all_files = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not all_files:
        print("[WAIT] 현재 data/raw/ 폴더가 비어 있습니다.")
        print("       데이콘에서 다운로드한 데이터를 data/raw/ 경로에 압축 해제한 뒤 다시 실행해 주세요.")
        return 0

    total_size = sum(p.stat().st_size for p in all_files)
    ext_counts = Counter(p.suffix.lower() or "(no_ext)" for p in all_files)

    print("[1] 전체 파일 현황")
    print(f"  - 총 파일 수 : {len(all_files):,}개")
    print(f"  - 총 용량    : {format_bytes(total_size)}")
    print(f"  - 확장자 분포: {dict(ext_counts)}")

    sub_candidates = list(RAW_DIR.rglob("*sample_submission*.csv"))
    if sub_candidates:
        inspect_sample_submission(sub_candidates[0])
    else:
        print("\n[WARN] sample_submission.csv 파일을 찾지 못했습니다.")

    parquet_files = [p for p in all_files if p.suffix.lower() == ".parquet"]
    if parquet_files:
        inspect_parquet_metadata(parquet_files)

    print("\n[DONE] 원본 데이터 기본 점검 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
