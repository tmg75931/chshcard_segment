# 신용카드 고객 세그먼트 분류 (DACON 236460)

데이콘 **[신용카드 고객 세그먼트 분류 AI 경진대회](https://dacon.io/competitions/official/236460/overview/description)** 해결을 위한 하네스 엔지니어링(Harness Engineering) 기반 머신러닝 프로젝트입니다.

## 빠른 시작 (Quick Start)

### 1. 데이터 배치
데이콘에서 다운로드한 대회 원본 데이터(`train`, `test`, `sample_submission.csv` 등)를 아래 폴더에 압축 해제해 주세요:
- 경로: `data/raw/`

### 2. 원본 데이터 검증
데이터 배치가 끝나면 아래 스크립트로 파일 트리, 행/컬럼 수, 제출 양식 정합성을 즉시 점검합니다:
```powershell
python scripts/verify_raw_data.py
```

### 3. 제출 파일 검증
제출용 CSV를 생성한 뒤에는 데이콘 업로드 전 반드시 검증 게이트를 실행합니다 (일일 제출 최대 5회):
```powershell
python scripts/validate_submission.py submissions/<파일명>.csv
```

## 핵심 문서 바로가기
- [AGENTS.md](./AGENTS.md): 프로젝트 불변 규칙 및 에이전트 행동 지침
- [docs/competition_overview.md](./docs/competition_overview.md): 대회 개요, Macro F1 평가 방식, 8대 도메인 데이터, 세부 규칙 총정리
- [docs/harness_architecture.md](./docs/harness_architecture.md): 하네스 폴더 계층 및 실험 운영 가이드
- [docs/data_schema.md](./docs/data_schema.md): 원본 데이터 스키마 및 EDA 요약
- [experiments/experiment_log.md](./experiments/experiment_log.md): 실험 가설 및 Local CV 기록장
- [submissions/submission_log.md](./submissions/submission_log.md): 제출 이력 및 CV-LB 비교 원장
