# 미리 준비한 LLM 경쟁 가설과 소수 관측 레짐 확정 — 겹침 검토

검토일: 2026-11-03  
상태: **아이디어 단계. 구현·실험은 미실행이다.**

[행동 모듈 검토](2026-11-03-behavior-modules.md) · [통합 검색 기록](../literature-map.md) · [결정 기록](../decision-log.md)

## 1. 검토한 아이디어

LLM이 물리 행동 모듈로 범위 밖 레짐의 경쟁 가설을 **미리** 만들어 학습 데이터로 피팅해 둔다.

- 범위 밖 데이터 0점: 점 예측과 신뢰도(후보 간 갈라짐)를 정직하게 낸다.
- 범위 밖 데이터 1~2점: 준비된 후보 중 하나를 빠르게 확정해, 데이터만 쓰는 방법보다 적은 점으로 오차를 줄인다.
- 이 구조에서 LLM에게 중요한 것은 1순위 정확도가 아니라 **재현율**(정답 가설을 후보에 포함하는 것)이다.

핵심 그림은 “범위 밖 새 데이터 점 수 k에 따른 외삽 오차” 곡선이다.

## 2. 확인 수준

웹 검색과 원문 확인. 전수 문헌조사가 아니다. 신경망 연산자 분할 논문은 본문 일부, LLM-AutoSciLab·HypoSpace·Prior or Feedback는 초록·검색 요약이다.

## 3. 겹침

| 요소 | 가까운 선행연구 | 확인 수준 | 판정 |
|---|---|---|---|
| 준비된 경쟁 모델을 적은 새 측정으로 판별 | Box & Hill(1967) 경쟁 모델 판별 실험설계, 모델 선택용 실험설계([생화학 네트워크](https://pmc.ncbi.nlm.nih.gov/articles/PMC3946009/)), 순차 Bayesian 모델 선택 | 검색 요약 | **고전적 메커니즘.** 신규성 아님 |
| LLM 가설 + 판별 실험 | Kabra et al., *LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs*, 2026-05 preprint. [arXiv](https://arxiv.org/abs/2605.24043) | 초록 | **메커니즘 겹침 큼.** LLM이 가설을 내고 구별 실험을 골라 2–5배 표본 효율을 보고. 평가는 식 복원 정확도. 외삽·새 레짐 설정 여부는 초록에 없음 |
| 대상 시스템의 소수 데이터로 적응 | Expert Augmentation(W3), MetaPhysiCa(W4), meta-learning 계열(iMODE, MAML 시스템 식별) | 기존 전달 노트 / 검색 요약 | 겹침. 후보·적응 능력이 **학습 분포의 여러 시스템**에서 나옴 |
| 한 궤적으로 연산자 사전에서 조합 탐색 | Serrano et al., *Test-time Generalization for Physics through Neural Operator Splitting*, 2026 preprint. [HTML](https://arxiv.org/html/2602.00884v2) | 본문 일부 | 대상 궤적 16프레임으로 사전 연산자 조합 선택. 사전은 학습 분포의 단일 물리에서 옴. LLM·외부 지식 없음. 사전이 표현하지 못하는 물리는 한계 |
| LLM 가설 집합의 재현율 측정 | Chen et al., *HypoSpace*, 2025–2026. [arXiv](https://arxiv.org/abs/2510.15614) | 초록 | 재현율(Recovery) 지표 자체는 있음. 인과 그래프·복셀·불리언 유전자 영역. 물리 외삽·데이터 기반 선택과 무관 |
| LLM의 prior와 피드백으로 적응 | Chan & Mora Jimenez, *Prior or Feedback? What an LLM Uses When Adapting Neural Operators*, NeurIPS 2026 목록. [NeurIPS](https://neurips.cc/virtual/2026/156586) | 검색 요약 | LLM이 미세조정 설정을 고름. 가설 준비와는 다름 |

## 4. 남는 차이

확인 범위에서 찾지 못한 것:

1. **모든 학습 시스템에 없는 레짐**을 대상으로 한 설정. meta-learning·연산자 사전처럼 학습 분포에서 후보를 얻는 방법은 원리상 그 레짐을 표현하지 못한다. LLM 지식은 표현할 수 있다.
2. **외삽 예측 오차**를 k(범위 밖 새 점 수)에 따라 평가하고, k=0에서 정직한 신뢰도를 함께 보고하는 것.
3. LLM 물리 레짐 가설을 **재현율 중심**으로 평가하고, 그것이 k-점 확정 성능과 어떻게 연결되는지 보이는 것.

## 5. 판정

**메커니즘은 신규성이 아니다. 설정과 평가 수준의 신규성만 남는다.** 독립 논문의 주 기여로는 약하다.

- 메커니즘(준비된 가설 + 판별 측정)은 고전 실험설계, LLM 버전은 LLM-AutoSciLab과 겹친다.
- 따라서 이 아이디어는 **행동 모듈 설계의 평가 축 하나**로 넣는 것을 권한다. 주 기여는 행동 모듈의 누락 처리와 재현율 확보 쪽에 둔다.
- LLM-AutoSciLab 본문(외삽 설정 여부, 실험 수)은 반드시 확인해야 한다.
