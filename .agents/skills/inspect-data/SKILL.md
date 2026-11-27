---
name: inspect-data
description: >-
  원본 데이터가 data/raw/에 다운로드된 직후 폴더 구조, Parquet 스키마, 고객 ID 및 기준년월 정합성,
  타겟(Segment) 분포, sample_submission.csv 그레인을 검증하고 docs/data_schema.md를 업데이트할 때 사용한다.
---

# 원본 데이터 입고 점검 런북 (Inspect Data)

사용자가 `data/raw/`에 데이터를 배치하면 모델링 코드를 작성하기 전에 반드시 아래 순서로 데이터를 검증합니다.

## 실행 절차

1. **기본 파일 및 제출 양식 스캔**:
   ```powershell
   python scripts/verify_raw_data.py
   ```
2. **핵심 구조 검증 항목 (Python 1회 스크립트로 일괄 확인)**:
   - `sample_submission.csv`의 총 행 수와 고유 `ID` 수 비교 → **고객 단위 예측(예: 10만 행)**인지 **고객×월 행 단위 예측(예: 60만 행)**인지 확정
   - Train 내 동일 `ID`의 6개월(`201807`~`201812`) 동안 `Segment` 값이 항상 불변(100% 동일)인지 확인
   - 8개 도메인(`1.회원정보` ~ `8.성과정보`) 간 `['ID', '기준년월']` 키가 1:1로 정확히 매칭되는지 확인
   - 타겟 `Segment` (`A`, `B`, `C`, `D`, `E`) 정확한 건수 및 비율 확인
   - 분산이 0인 상수 컬럼 및 결측률 99% 이상 컬럼 목록 도출
3. **문서 업데이트**:
   - 확인된 실제 수치를 [data_schema.md](../../../docs/data_schema.md)에 즉시 반영합니다.
