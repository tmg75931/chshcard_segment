---
name: run-experiment
description: >-
  새로운 피처 엔지니어링, 모델 아키텍처, 하이퍼파라미터, 또는 Macro F1 후처리(Threshold/Prior 최적화) 실험을
  설계·실행하고 OOF 예측값과 CV 점수를 기록할 때 사용한다.
---

# 표준 실험 실행 런북 (Run Experiment)

모든 모델링 실험은 재현 가능하고 앙상블 가능하도록 아래 절차를 따릅니다.

## 실행 절차

1. **가설 정의**:
   - 변경 요소를 1가지 단위로 명확히 정의합니다 (예: "고객별 6개월 잔액 증감 추세 피처 추가 시 C/D 등급 구분력 향상 기대").
2. **누수 방지 체크 (Pre-flight Check)**:
   - 고정된 고객 `ID` 그룹 기반 Fold 인덱스를 로드하는지 확인합니다.
   - 타겟 인코딩, 결측 대체, 스케일링, Threshold 튜닝이 모두 Fold 내부(Out-Of-Fold)에서만 수행되는지 확인합니다.
3. **학습 및 평가 실행**:
   - 전체 OOF **Macro F1-Score**와 클래스별(`A`, `B`, `C`, `D`, `E`) Precision / Recall / F1을 모두 산출합니다.
4. **아티팩트 및 로그 기록**:
   - OOF 확률 배열 `(N, 5)`를 `experiments/oof/{exp_id}_oof.npy`에, Test 예측 확률 배열을 `experiments/oof/{exp_id}_test.npy`에 저장합니다.
   - 결과를 [experiment_log.md](../../../experiments/experiment_log.md) 및 (파생변수 실험인 경우) [feature_catalog.md](../../../docs/feature_catalog.md)에 즉시 기록합니다.
