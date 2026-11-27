"""데이콘 제출 파일(CSV) 무결성 7단계 자동 검증 게이트.

사용법:
    python scripts/validate_submission.py submissions/sub_001.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_SUB = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
ALLOWED_CLASSES = {"A", "B", "C", "D", "E"}
EXPECTED_HEADER = ["ID", "Segment"]


def find_sample_submission(explicit_path: Path | None) -> Path | None:
    if explicit_path and explicit_path.exists():
        return explicit_path
    if DEFAULT_SAMPLE_SUB.exists():
        return DEFAULT_SAMPLE_SUB
    matches = list((PROJECT_ROOT / "data" / "raw").rglob("*sample_submission*.csv"))
    return matches[0] if matches else None


def validate(sub_path: Path, sample_path: Path | None) -> int:
    print("=" * 70)
    print(f"제출 파일 무결성 검증: {sub_path}")
    print("=" * 70)

    # 1. 파일 존재 및 크기 확인
    if not sub_path.exists() or not sub_path.is_file():
        print(f"[FAIL 1/7] 제출 파일이 존재하지 않습니다: {sub_path}")
        return 1
    if sub_path.stat().st_size == 0:
        print("[FAIL 1/7] 제출 파일이 0 바이트입니다.")
        return 1
    print("[PASS 1/7] 파일 존재 및 비어있지 않음 확인")

    # 2. UTF-8 인코딩 및 BOM 점검
    raw_bytes = sub_path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        print("[WARN 2/7] UTF-8 BOM이 포함되어 있습니다. 순수 UTF-8(encoding='utf-8') 저장을 권장합니다.")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        print(f"[FAIL 2/7] UTF-8 디코딩 실패 (규칙 위반): {exc}")
        return 1
    print("[PASS 2/7] UTF-8 인코딩 검증 통과")

    # 3. CSV 파싱 및 헤더 점검
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header != EXPECTED_HEADER:
        print(f"[FAIL 3/7] 헤더 불일치: 기대값={EXPECTED_HEADER}, 실제값={header}")
        return 1
    print(f"[PASS 3/7] 헤더 일치: {header}")

    # 4~6. 행 단위 결측치, ID 중복, 허용 클래스 검증
    ids: list[str] = []
    class_counts: Counter[str] = Counter()
    for line_no, row in enumerate(reader, start=2):
        if len(row) != 2:
            print(f"[FAIL 4/7] {line_no}행 컬럼 개수 오류 (2개여야 함): {row}")
            return 1
        row_id, pred = row[0].strip(), row[1].strip()
        if not row_id or not pred or pred.lower() in {"nan", "none", "null"}:
            print(f"[FAIL 4/7] {line_no}행 결측치/빈 값 발견: {row}")
            return 1
        if pred not in ALLOWED_CLASSES:
            print(f"[FAIL 5/7] {line_no}행 허용되지 않은 Segment 값 '{pred}' (허용: {sorted(ALLOWED_CLASSES)})")
            return 1
        ids.append(row_id)
        class_counts[pred] += 1

    if not ids:
        print("[FAIL 4/7] 데이터 행이 0건입니다.")
        return 1
    print(f"[PASS 4/7] 결측치 및 컬럼 정합성 통과 (총 {len(ids):,}행)")
    print(f"[PASS 5/7] 모든 예측값이 허용 클래스 {sorted(ALLOWED_CLASSES)} 내에 존재함")

    if len(set(ids)) != len(ids):
        print(f"[FAIL 6/7] 중복된 ID가 존재합니다 (고유 {len(set(ids)):,}개 / 전체 {len(ids):,}개)")
        return 1
    print("[PASS 6/7] ID 고유성(중복 없음) 확인")

    # 7. sample_submission.csv 대비 행 수 및 ID 순서 완전 일치 검증
    if sample_path and sample_path.exists():
        with sample_path.open("r", encoding="utf-8-sig") as sf:
            s_reader = csv.reader(sf)
            s_header = next(s_reader, None)
            expected_ids = [r[0].strip() for r in s_reader if r]
        if s_header != EXPECTED_HEADER:
            print(f"[WARN 7/7] sample_submission 헤더가 예상과 다릅니다: {s_header}")
        if len(ids) != len(expected_ids):
            print(
                f"[FAIL 7/7] 행 수 불일치: 제출={len(ids):,}행 vs sample_submission={len(expected_ids):,}행"
            )
            return 1
        if ids != expected_ids:
            for idx, (got, exp) in enumerate(zip(ids, expected_ids), start=2):
                if got != exp:
                    print(f"[FAIL 7/7] {idx}행 ID 순서 불일치: 제출='{got}' vs 기준='{exp}'")
                    break
            return 1
        print(f"[PASS 7/7] sample_submission.csv ({sample_path.name})와 행 수·ID 순서 100% 일치!")
    else:
        print("[WARN 7/7] data/raw/ 내 sample_submission.csv가 아직 없어 기준 ID 대조는 건너뛰었습니다.")

    # 예측 클래스 분포 출력
    print("\n[예측 클래스 분포 리포트]")
    total = len(ids)
    for cls in sorted(ALLOWED_CLASSES):
        cnt = class_counts.get(cls, 0)
        pct = (cnt / total) * 100
        warn = "  <-- [주의: 0건 예측!]" if cnt == 0 else ""
        print(f"  - Segment {cls}: {cnt:>8,}건 ({pct:6.2f}%){warn}")

    print("\n[SUCCESS] 모든 제출 무결성 검증을 통과했습니다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="데이콘 236460 제출 CSV 검증기")
    parser.add_argument("submission_csv", type=Path, help="검증할 제출 CSV 파일 경로")
    parser.add_argument("--sample", type=Path, default=None, help="기준 sample_submission.csv 경로 (선택)")
    args = parser.parse_args()

    sample_path = find_sample_submission(args.sample)
    return validate(args.submission_csv, sample_path)


if __name__ == "__main__":
    sys.exit(main())
