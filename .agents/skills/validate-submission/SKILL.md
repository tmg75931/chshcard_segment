---
name: validate-submission
description: >-
  데이콘 리더보드에 제출할 CSV 파일을 생성한 후 포맷, UTF-8 인코딩, 행 수, ID 정렬, 결측치,
  클래스 분포 이상 여부를 검증하고 제출 로그를 남길 때 사용한다.
---

# 제출 파일 무결성 검증 런북 (Validate Submission)

데이콘 일일 제출 횟수는 **최대 5회**로 제한되어 있으므로 검증 게이트를 통과하지 않은 파일은 절대 제출하지 않습니다.

## 실행 절차

1. **자동 검증 스크립트 실행**:
   ```powershell
   python scripts/validate_submission.py submissions/<파일명>.csv
   ```
2. **분포 건전성(Sanity Check) 확인**:
   - 출력된 예측 클래스(`A`~`E`) 비율이 Train 타겟 분포 및 OOF 최적 예측 분포와 극단적으로 괴리되지 않았는지 확인합니다 (예: 소수 클래스 `A`, `B`가 0건으로 붕괴되었거나 반대로 과도하게 폭증했는지 점검).
3. **제출 원장 기록**:
   - 검증 통과 후 [submission_log.md](../../../submissions/submission_log.md)에 당일 제출 회차, 기반 `Exp ID`, Local CV 점수를 기록합니다.
