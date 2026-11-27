# AGENTS.md — 신용카드 고객 세그먼트 분류 하네스 규칙

이 저장소는 **데이콘 신용카드 고객 세그먼트 분류 AI 경진대회 (ID: 236460)** 해결을 위해 **하네스 엔지니어링(Harness Engineering)** 원칙으로 설계되었습니다. 이 공간에서 작업하는 모든 에이전트는 아래 불변 규칙(Invariants)과 디렉토리 계약을 반드시 준수해야 합니다.

---

## 1. 대회 핵심 요약
- **목표**: 6개월(`201807`~`201812`)간의 8대 금융 도메인 데이터를 바탕으로 고객 세그먼트 `Segment` (`A`, `B`, `C`, `D`, `E` 5개 클래스) 분류
- **평가지표**: **Macro F1-Score** (Public 50% / Private 100%)
- **상세 대회 규칙**: 작업 전 반드시 [competition_overview.md](./docs/competition_overview.md)를 참조하세요.

---

## 2. 절대 불변 규칙 (Hard Invariants)

1. **원본 데이터 불변성 (`data/raw/` Read-Only)**:
   - `data/raw/` 내부의 원본 Parquet 및 CSV 파일은 **절대 수정·덮어쓰기·삭제하지 않습니다.**
   - 병합/다운캐스팅 데이터는 `data/interim/`, 피처 엔지니어링 산출물과 고정 Fold 인덱스는 `data/processed/`에만 저장합니다.
2. **외부 데이터 절대 금지**:
   - 대회 공식 제공 데이터 외 외부 데이터(공휴일, 금리, 물가지수 등 외부 소스 일체) 결합을 엄격히 금지합니다.
3. **고객 `ID` 기준 그룹 교차검증 (Zero Group Leakage)**:
   - 동일 고객(`ID`)이 최대 6개 기준년월에 반복 등장하므로, 단순 Row 단위 K-Fold 분할을 **절대 금지**합니다.
   - 반드시 전 실험이 동일한 **고객 `ID` 단위 고정 Fold** (`StratifiedGroupKFold` 또는 고객 단위 집계 후 `StratifiedKFold`)를 공유해야 OOF 비교와 앙상블이 유효합니다.
   - 스케일러, 인코더, 결측치 대체 통계량, 피처 선택기, 후처리 Threshold는 반드시 **각 Fold의 Train 내부에서만 fit**합니다.
4. **메모리 안전성 (240만 행 × 870+ 변수)**:
   - 전체 데이터를 무작정 `pandas` 기본 타입(`float64`/`object`)으로 한 번에 로드하지 마세요.
   - 필요한 컬럼 선택 로딩, `float32`/`int32`/`int16`/`category` 다운캐스팅, 가비지 컬렉션(`gc.collect()`)을 기본 적용합니다.
5. **재현성 및 경로 추상화**:
   - 하드코딩된 절대 경로 대신 [paths.yaml](./configs/paths.yaml)을 통해 데이터/출력 경로를 참조합니다.
   - 모든 실험 스크립트는 고정 시드(`SEED = 42` 기본)를 설정하고, 패키지 변경 시 가상환경(`.venv`) 기준으로 `requirements.txt`를 최신 유지합니다 (글로벌 `pip install` 금지).
6. **실험 기록 및 제출 게이트 의무화**:
   - 모든 모델 학습 후에는 OOF 예측 확률(`.npy`/`.parquet`)을 `experiments/oof/`에 저장하고, 가설·피처·Macro F1 CV(전체 및 A~E 클래스별 F1)를 [experiment_log.md](./experiments/experiment_log.md)에 기록합니다.
   - 데이콘 일일 제출 제한은 **최대 5회**입니다. 제출 파일 생성 시 반드시 `python scripts/validate_submission.py <파일경로>` 검증을 통과한 뒤 [submission_log.md](./submissions/submission_log.md)에 기록합니다.

---

## 3. 저장소 맵 (Repository Map)

- [docs/competition_overview.md](./docs/competition_overview.md): 대회 개요, 평가 산식, 데이터 도메인, 세부 규칙 원문 요약
- [docs/harness_architecture.md](./docs/harness_architecture.md): 하네스 엔지니어링 아키텍처 및 실험 파이프라인 운영 가이드
- [docs/data_schema.md](./docs/data_schema.md): 실제 원본 데이터 스키마, 행/고객 수, 결측/상수 컬럼 분석 문서
- [docs/feature_catalog.md](./docs/feature_catalog.md): 파생변수 설계 명세 및 CV 성능 기여도 카탈로그
- [configs/paths.yaml](./configs/paths.yaml): 데이터·실험·제출 디렉토리 및 시드 중앙 설정
- `data/{raw,interim,processed}/`: 원본(읽기 전용) / 중간 병합 캐시 / 최종 피처셋 저장소
- `src/{utils,data,features,models,evaluation}/`: 모듈화된 핵심 소스코드
- `scripts/`: 데이터 검증(`verify_raw_data.py`), 제출 검증(`validate_submission.py`), 학습/추론 실행 CLI
- [experiments/experiment_log.md](./experiments/experiment_log.md): 실험 이력 및 CV 점수 원장 (`oof/`, `artifacts/` 포함)
- [submissions/submission_log.md](./submissions/submission_log.md): 제출 파일 버전 및 CV ↔ Public LB 추적 원장
- `.agents/skills/`: 데이터 점검(`inspect-data`), 실험 수행(`run-experiment`), 제출 검증(`validate-submission`) 표준 런북
