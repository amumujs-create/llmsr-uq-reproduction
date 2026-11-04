# 결빙 방지 세계 추가 — 겹침·논문성 검토

작성: 2026-11-04  
대상 설계: [CCPP 쌍둥이 결빙 방지 세계](../ccpp-anti-icing-world.md)

## 검색한 것

1. 결빙 방지의 작동 조건과 손실 (도메인 근거)
2. 시뮬레이터에 메커니즘을 심어 LLM의 도메인 진술을 채점하는 연구
3. LLM 사전 지식 추출(prior elicitation)과 레짐 변화·외삽

## 가장 가까운 연구

| 연구 | 하는 것 | 우리와 다른 점 |
|---|---|---|
| AutoElicit (Capstick 외, ICML 2025, arXiv 2411.17284) | LLM으로 선형 모델의 계수 prior를 만든다. 데이터가 적을 때 오차를 줄인다 | 계수 prior만 다룬다. 레짐 경계, 메커니즘 종류, 외삽 위험 제한은 없다 |
| LLM-BI (arXiv 2508.08300) | LLM이 prior 추출과 모델 설계를 맡는다. 선형 합성 데이터로 평가한다 | 물리 시스템이 아니고, 정답 구조를 바꿔 가며 채점하지 않는다 |
| 기상 prior 추출 (arXiv 2402.07770) | LLM이 전문가처럼 prior를 낸다. 사람 전문가와 비교한다 | 레짐과 외삽이 아니다 |
| TwinVoice, BehaviorChain (ACL Findings 2025–26) | "디지털 트윈"을 사람 페르소나 흉내로 쓴다 | 이름만 같다. 물리 시뮬레이터가 아니다 |
| LLM 보조 원자로 디지털 트윈 (ScienceDirect 2025) | 시뮬레이터와 도메인 LLM을 결합해 감시·제어한다 | LLM 진술의 신뢰도를 정답 구조로 채점하지 않는다 |
| Cooke 고전 모델, Zheng & Zhang (arXiv 2607.20529) | 정답 아는 시드 질문으로 전문가(LLM) 가중치를 정한다 | 시드가 같은 시스템의 하위 메커니즘이 아니다 ([이전 검토](2026-11-04-domain-grounded-delegation.md)) |

**메커니즘을 제어할 수 있는 물리 쌍둥이에서 LLM 레짐 진술을 "쌍둥이 정답"과 "문헌 현실성"으로 나눠 채점한 연구는 찾지 못했다.** 표준 검색 4회 기준이며, 검색 범위가 좁으므로 원고 단계에서 다시 확인한다.

## 판정

- **모듈 추가 자체는 기여가 아니다.** 실험 설계의 일부다. 리뷰어는 "자기 결과에 맞게 세계를 고쳤다"고 볼 수 있다. 설계 문서 §2의 순환 방지 조치(문헌 모수, 기존 세계 유지, 재수집, 사전 커밋)가 필수다.
- 이 설계가 논문에 주는 것은 세 가지다.
  1. **채점의 타당성:** 큰 모델의 현실 지식이 오답 처리되는 문제를 보이고 고친다. 벤치마크 설계 기여(B)의 근거가 된다.
  2. **어려운 세계:** 원인이 겹치는 저온 쪽에서 메커니즘 구분 능력을 잰다.
  3. **메커니즘별 λ의 재료:** 검사 가능한 메커니즘(창 안에 보이는 결빙 방지)과 불가능한 메커니즘(상한)이 한 세계에 함께 있다. 주 기여 후보(A: 검사 불가능 진술의 위험 제한)의 λ 규칙을 시험할 수 있다.
- **판정: 조건부로 진행.** 주 기여는 여전히 A다. 이 세계는 A의 검증 범위를 넓히는 도구로만 쓴다.

## 남은 위험

- 결빙 방지 효과(저온 평균 약 −4.5 MW)가 잡음 4.3 MW와 비슷하다. 결과 채점에서 차이가 안 날 수 있다. 사전 예측 Q5로 걸러 둔다.
- 실제 CCPP 데이터는 결빙 방지 신호를 보이지 않는다(8°C 아래 기압 계수 음수). 현실 근거는 문헌뿐이다.

## 출처

- [Camfil, Anti-icing Systems 제품 문서](https://www.camfil.com/product/documents/dam/47937/Product-Documentation-Anti-icing-Systems.pdf) — 5°C 이하·RH 70% 이상, 추출 공기 손실 최대 약 2%
- [Turbomachinery Magazine, types of anti-icing systems](https://www.turbomachinerymag.com/view/types-of-anti-icing-systems-in-gas-turbines) — 약 4°C·고습, 추출 공기 약 2%
- [SoftInWay, anti-icing systems for land-based gas turbines](https://www.softinway.com/anti-icing-systems-for-land-based-gas-turbines/) — 7–8°C 이상 가열 필요
- [US8899008 특허](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8899008) — 약 −15~8°C, RH 65% 미만이면 결빙 가능성 낮음
- [US4852343A 특허](https://patents.google.com/patent/US4852343) — 추출 공기 1%당 출력 3–4% 손실(항공 터보샤프트)
- [AutoElicit](https://arxiv.org/abs/2411.17284), [LLM-BI](https://arxiv.org/pdf/2508.08300), [기상 prior](https://arxiv.org/pdf/2402.07770)
- [TwinVoice](https://aclanthology.org/2026.findings-acl.981/), [BehaviorChain](https://aclanthology.org/2025.findings-acl.813/), [LLM 보조 원자로 트윈](https://www.sciencedirect.com/science/article/pii/S0149197025005700)
