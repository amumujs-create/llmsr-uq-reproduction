# 학습 구간에서 구별되지 않는 식과 물리 지식 — 선행연구 비교

검토일: 2026-11-03  
상태: **선행연구 확인 단계. 복사 냉각 파일럿은 미실행이다.**

[통합 검색 기록](../literature-map.md) · [탐색 반복 간 불일치 비교](2026-11-03-search-disagreement.md) · [결정 기록](../decision-log.md)

## 1. 검토한 설계

학습 구간에서는 틀린 식(예: 선형 냉각)도 맞고 외삽 구간에서만 정답(선형 + 복사 T⁴)과 갈라지는 경우를 다룬다. 학습 오차로는 식을 구별할 수 없으므로 데이터 밖의 정보로 다음 중 하나를 한다.

1. 물리 조건(메커니즘·단위·극한 거동)으로 식을 **고른다.**
2. 상수 범위(예: 방사율 0–1)로 외삽 예측을 **범위로 묶는다.**
3. 지식도 없으면 여러 식을 유지하고 갈라지는 구간을 **경보**한다.
4. 식들이 가장 크게 갈라지는 **측정 위치**를 제안한다.
5. 틀린 지식의 **피해**를 측정한다.

## 2. 확인 수준과 한계

웹 검색과 원문 HTML 확인으로 수행했다. 전수 문헌조사가 아니다. PG-SR·KeplerAgent는 본문 앞부분(약 75%)을, 나머지는 초록·검색 요약만 확인했다. 부분 식별·prediction profile likelihood 계열 통계 문헌은 이번에 새로 검색하지 않았다.

## 3. 구성요소별 겹침

| 구성요소 | 가까운 선행연구 | 확인 수준 | 판정 |
|---|---|---|---|
| 1. 물리 조건으로 식 선택 | Kronberger et al., *Shape-constrained Symbolic Regression – Improving Extrapolation with Prior Knowledge*, Evolutionary Computation 30(1), 2022. [arXiv](https://arxiv.org/abs/2103.15624) | 초록 | **겹침 큼.** 단조성 등 형상 제약으로 외삽 개선. 학습·시험 정확도는 오히려 낮아질 수 있다고 보고 |
| | Tenachi et al., 단위 제약 deep SR(Φ-SO), 2023. [arXiv](https://arxiv.org/abs/2303.03192) | 검색 요약 | 단위 제약만으로 물리법칙 복원이 보장되지 않음 |
| | Xiao et al., *Prior-Guided Symbolic Regression: Towards Scientific Consistency in Equation Discovery* (PG-SR), 2026 preprint. [HTML](https://arxiv.org/html/2602.13021) | 본문 §3–4 | **겹침 큼.** LLM이 문제 설명에서 지식을 뽑고 전문가가 검토한 뒤 실행 가능한 제약 프로그램으로 변환. LLM-SRBench에서 OOD NMSE 평가. 틀린 prior 조건(E. coli 1개 문제)도 있음. 학습 구간에서 같은 식들이 갈라지는 경우는 개념 설명만 있고 실험으로 분리하지 않음. 구간·복수 가설 보고 없음 |
| | Yang et al., *Think like a Scientist: Physics-guided LLM Agent for Equation Discovery* (KeplerAgent), 2026 preprint. [HTML](https://arxiv.org/html/2602.12259v1) | 본문 일부 | 대칭성 등 물리 성질을 추론해 SR 제약으로 사용. 별도 외삽 분할 없음. 비슷한 오차면 단순한 식을 고르도록 지시함 |
| | PiSR, *Knowledge integration for physics-informed symbolic regression using pre-trained large language models*, Scientific Reports, 2026. [원문](https://www.nature.com/articles/s41598-026-35327-6) | 검색 요약 | LLM 점수(단위 일관성·물리 타당성)를 탐색 목적함수에 보조 penalty로 사용. 휴리스틱이며 타당성을 보장하지 않는다고 서술 |
| 2. 상수 범위로 외삽 묶기 | grey-box 모델의 물리 파라미터 범위 정규화(검색 요약), [prediction profile likelihood, Kreutz et al. 2012](https://link.springer.com/article/10.1186/1752-0509-6-120) | 검색 요약 / 기존 전달 노트 | 개념은 기존이다. LLM이 제공한 상수 범위로 기호식 외삽을 묶고 포함률을 평가한 연구는 확인 범위에서 찾지 못함 |
| 3. 갈라지는 구간 경보 | LLM-SR-UQ, Kotanchek et al. 2007 | [별도 기록](2026-11-03-search-disagreement.md) | 겹침 있음. 한 번 탐색 안의 불일치는 이미 있음 |
| 4. 측정 위치 제안 | Hunter–Reiner, Box & Hill(1967)의 경쟁 모델 판별 실험설계. [배경 설명](https://link.springer.com/article/10.1007/s00180-023-01370-4) · GP SR의 active learning, [Haut et al. 2023](https://arxiv.org/abs/2308.00672) | 검색 요약 | **고전적 방법.** 예측이 가장 크게 갈라지는 조건에서 측정하는 규칙 자체는 신규성이 아님 |
| 5. 틀린 지식의 피해 | PG-SR §4.6, PDNN(P2) | 본문 / 기존 검토 | 일부 겹침. PG-SR은 한 문제, 점 예측 NMSE만 보고 |

## 4. 남는 차이

구성요소 하나하나는 신규성이 아니다. 확인 범위에서 남은 것은 다음의 **통제 실험 질문**이다.

- **학습 구간 식별 불가능성을 독립 실험 축으로 조작:** 약한 항의 크기·노이즈·학습 범위를 바꿔 “데이터로 구별 가능 → 불가능”의 경계를 측정. PG-SR·KeplerAgent·LLM-SR-UQ는 이를 실험으로 분리하지 않았다.
- **필요한 지식의 종류를 분리:** 형태만 / 형태 + 상수 범위 / 틀린 지식. 형태를 알아도 계수가 식별되지 않으면 외삽이 살아나지 않는지.
- **프롬프트 지식과 적합도 기반 선택의 충돌:** 공정 설명에 복사를 언급해도 학습 오차·복잡도 기준이 단순한 틀린 식을 고르는지.
- **출력 형태의 비교:** 하나로 고르기 / 범위로 묶기 / 경보를 같은 조건에서 비교하고, 범위의 OOD 포함률을 평가.

기여 형태는 새 알고리즘보다 **통제된 분석 연구 + 단순한 절차(고르기·묶기·경보)**다.

## 5. 판정

**조건부로 논문 가능성 있음. 방법론 신규성은 약하다.**

가장 큰 위험은 “정답 형태와 상수를 주면 외삽이 잘 되는 것은 당연하다”는 반론이다. 이를 넘으려면 당연하지 않은 결과가 하나 이상 필요하다. 후보:

1. 물리 설명을 받은 LLM-SR도 학습 구간에서는 틀린 단순 식을 고른다.
2. 형태 지식만으로는 부족하고 상수 범위가 결정적이며, 그 경계가 약한 항의 크기·노이즈로 예측된다.
3. 상수 범위로 묶은 구간이 conformal 구간과 달리 OOD 포함률을 유지한다.
4. 틀린 지식은 데이터로 탐지되지 않지만, 대안 식 유지가 피해를 줄인다.

복사 냉각 하나는 파일럿이다. 논문에는 같은 구조의 물리 계열 2–3개(예: 항력 선형→제곱, 진자 작은 각→큰 각)와 PG-SR·shape-constrained SR류 비교군이 필요하다.

## 6. 빠른 결정 실험

복사 냉각 시뮬레이터에서 결과 1을 먼저 확인한다: 공정 설명(“고온 물체의 냉각”)을 받은 LLM-SR이 저온 학습 데이터에서 T⁴ 항을 유지하는지, 선형 식을 고르는지. 이 결과가 연구 이야기의 출발점을 정한다. 미실행.
