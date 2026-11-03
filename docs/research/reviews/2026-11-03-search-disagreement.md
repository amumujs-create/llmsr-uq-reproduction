# 탐색 반복 간 불일치 후보의 선행연구 비교

검토일: 2026-11-03  
상태: **선행연구 확인 단계. 후보는 조건부 유지이며, 파일럿 실험은 미실행이다.**

[통합 검색 기록](../literature-map.md) · [현재 방향](../README.md) · [결정 기록](../decision-log.md)

## 1. 후보

> 같은 데이터·분할·seed로 LLM 식 탐색을 독립적으로 여러 번 반복할 때, 탐색 간 외삽 예측의 불일치가 **한 번의 탐색 안의 후보식 불일치로는 포착되지 않는 OOD 오차**를 정답 없이 미리 알려 주는가?

이 후보는 외삽 예측을 더 정확하게 만드는 연구가 아니다. 외삽 예측을 언제 믿으면 안 되는지 아는 것이 목표다. 기존 조건부 물리 지식 후보를 대체하거나 같은 방법으로 합치지 않는다.

## 2. 확인 수준과 한계

이 검토는 웹 검색과 원문 HTML 확인으로 수행했다. 전수 문헌조사가 아니다. 딥러닝 앙상블의 불일치·OOD 탐지 문헌은 이번에 검색하지 않았다. 아래 “확인하지 못함”은 확인한 범위에서의 판단이며 해당 연구가 없다는 증명이 아니다.

## 3. 가장 가까운 방법

| ID | 자료 | 확인 수준 | 이미 해결한 부분 | 확인 범위에서 다루지 않은 부분 |
|---|---|---|---|---|
| S1 | *Know When the Equation Is Wrong: Structural Uncertainty for LLM-Guided Symbolic Regression under Extrapolation* (LLM-SR-UQ), 2026 preprint. [arXiv](https://arxiv.org/abs/2610.42024), [HTML](https://arxiv.org/html/2610.42024) | 본문 확인. 부록 일부는 페이지 길이 문제로 요약만 확인 | 후보식 간 불일치(σ_S)로 외삽 구간을 넓힘. split conformal이 ID에서는 맞고 OOD에서 자주 실패함을 LSR-Synth 109문제로 보고. 비교군은 LLM-SR, direct prompting, E-SINDy | 구간은 **한 번의 탐색** buffer(최대 24개)에서 계산. held-out 평가는 문제당 search seed 1개. 반복 간 식·OOD 예측의 변동을 보고하지 않음. σ_S와 OOD 오차의 상관·AUROC·risk–coverage를 보고하지 않음. 과적합된 단일 구조가 좁고 틀린 구간을 만들 수 있다고 부록에서 언급하지만 별도 평가는 없음 |
| S2 | *Programmatic Context Augmentation for LLM-based Symbolic Regression* (ProAug), 2026. [arXiv](https://arxiv.org/abs/2605.03101), [HTML](https://arxiv.org/html/2605.03101) | §4.1, §4.4.2 확인 | LLM-SR 결과가 seed에 민감함을 보고. 3회 반복에서 NMSE가 수 자릿수 차이(Chemistry·Qwen3-8B에서 약 10⁻⁴–10⁻⁹). 다회 반복 보고가 드물다고 지적 | 변동을 성능 보고의 문제로 다룸. 불확실성 신호나 구간으로 쓰지 않음. 변동 분석이 ID/OOD 중 어느 분할인지 명시하지 않음 |
| S3 | Kotanchek, Smits, Vladislavleva, *Trustable symbolic regression models*, GPTP V, 2007. [Springer](https://link.springer.com/chapter/10.1007/978-0-387-76308-8_12) · 후속 [Kotanchek & Haut, GPTP 2021](https://gpbib.cs.ucl.ac.uk/gp-html/Kotanchek_2021_GPTP.html) | 서지·요약 확인. 본문 미확인 | **다양한 GP 기호식 앙상블의 합의를 신뢰 지표로 사용**하는 개념의 원형. 2021 후속은 새 파라미터 영역으로의 외삽 탐지를 목표로 언급 | 정량적 외삽 결과와 평가 방법은 미확인. LLM 탐색이 아님 |
| S4 | Haut, Punch, Banzhaf, *Active Learning in Genetic Programming: Guiding Efficient Data Collection for Symbolic Regression*, 2023. [arXiv](https://arxiv.org/abs/2308.00672) | 검색 요약 확인 | GP 개체군 앙상블의 불확실성(차분 엔트로피 등)으로 다음 측정점 선택 | 데이터 수집용. 예측 거부나 오차 경보로 평가하지 않음 |
| S5 | Fasel, *Sparse Identification of Nonlinear Dynamics with Conformal Prediction*, 2025. [arXiv](https://arxiv.org/abs/2507.11739) | 초록 확인 | E-SINDy 앙상블과 conformal 결합으로 목표 포함률 달성 | 확인 범위에서는 학습 영역 내 예측. 외삽 포함률은 미확인 |

참고: [기호회귀 UQ 서베이(Reuter & Olivetti de França, 2026)](https://arxiv.org/abs/2606.06567)는 SR의 UQ가 아직 덜 연구됐다고 서술한다. 앞부분에서 BMA 앙상블(Agapitos et al.)과 bootstrap 기반 구조·계수 UQ 절을 확인했다. 확인한 부분에는 LLM 기반 SR의 UQ가 없었다. Discussion은 읽지 못했다. [Rashomon prediction intervals, PMLR 2026](https://proceedings.mlr.press/v337/nguyen26c.html)은 거의 최적인 모델 집합의 구간을 ridge 회귀에서 분석한다(초록 확인).

## 4. 남는 차이

| 구분 | 내용 |
|---|---|
| 이미 있는 것 | 후보식 불일치로 구간 넓히기(S1). 기호식 앙상블 합의를 신뢰 지표로 쓰는 개념(S3). 앙상블 불일치로 측정점 고르기(S4). LLM-SR 반복 간 변동의 존재(S2) |
| 확인 범위에서 남은 것 | ① 독립 LLM 탐색 간 **OOD 예측** 불일치의 크기를 한 번의 탐색 내부 σ_S와 비교. ② 불일치가 점별 OOD 오차를 예측하는지(상관·AUROC·risk–coverage). ③ 여러 탐색을 결합한 구간과 단일 탐색 구간의 비교. ④ 모든 탐색이 일치하지만 틀린 경우의 비율 |
| 추가로 쓰는 정보 | 같은 데이터에서 추가 LLM 호출(K배 계산). 목표 정답은 사용하지 않음 |
| 기여 형태 | 새로운 알고리즘보다 **평가·진단 연구 + 단순한 결합 방법**에 가깝다. “불일치를 신뢰 지표로 쓴다”는 개념 자체는 S3에 있으므로 신규성으로 주장하지 않는다 |

## 5. 예상 심사 질문

1. **“그냥 앙상블 아닌가?”** 개념은 기존이다. 주장할 것은 LLM 탐색에서 탐색 간 불일치가 탐색 내부 불일치에 **추가 정보**를 주는지, 그리고 그 크기와 실패 조건에 관한 실험적 지식이다.
2. **“K배 계산을 쓰면 당연히 좋아지지 않나?”** 같은 LLM 호출 예산에서 비교한다. 예: 탐색 1회 × 125라운드와 탐색 5회 × 25라운드.
3. **“학습 범위로부터의 거리만으로도 알 수 있지 않나?”** 거리 기준을 필수 비교군으로 둔다. 이를 넘지 못하면 기여가 약하다.
4. **“모두 같은 틀린 식을 내면?”** 데이터로 구별할 수 없는 경우는 불일치로 잡을 수 없다. 이 비율을 숨기지 않고 보고한다.

## 6. 판정과 파일럿

**조건부 유지.** 확인한 범위에서 S1–S5 중 위 ①–② 질문을 직접 다룬 연구는 찾지 못했다. 그러나 S3의 개념적 선행과 S1 저자의 후속 가능성 때문에 기여 형태를 진단 연구로 한정한다.

파일럿(미실행):

- 고정 개발 문제 `po10`, `po12`, `po25`에서 `repeat_search.py --runs 5 --iterations 125`.
- 점별 OOD 오차와 (a) 탐색 간 불일치, (b) 탐색 내부 σ_S, (c) 학습 범위로부터의 거리 사이의 순위 상관·AUROC를 계산. 이 분석 스크립트는 아직 없다.

진행 근거: (a)가 (b)·(c)보다 OOD 오차를 반복적으로 더 잘 예측하거나, (b)에 더했을 때 예측력이 의미 있게 늘어난다.  
중단 근거: 탐색 간 불일치가 σ_S와 거의 같거나, 거리 기준을 넘지 못한다. 이 경우 후보를 보류한다.

파일럿 3문제의 결과만으로 109문제 일반화나 논문 기여를 주장하지 않는다.

## 7. 다음 확인 항목

- S3 본문의 정량 평가 방법과 외삽 결과.
- 딥러닝 앙상블의 불일치 기반 OOD 오차 탐지 문헌(회귀·외삽).
- S1의 5-seed 개발 실험(부록 C) 상세 수치.
- 서베이 Discussion의 공백 서술.
