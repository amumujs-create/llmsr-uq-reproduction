# 외삽 실패의 모듈별 원인 분해 — 후보와 선행연구 비교

검토일: 2026-11-03  
상태: **연구 후보와 선행연구 확인 단계. 같은 날 재검토로 신규성 범위를 좁혔다(§4.1). 분석·실험은 미실행이다.**

[통합 검색 기록](../literature-map.md) · [구별되지 않는 식 검토](2026-11-03-indistinguishable-candidates.md) · [탐색 반복 간 불일치 검토](2026-11-03-search-disagreement.md) · [결정 기록](../decision-log.md)

## 1. 후보

LLM 식 탐색을 모듈(식 생성 · 계수 피팅 · 선택 · 지식 제공)로 나누고, 외삽이 실패했을 때 **어느 모듈 때문인지** 분해한다. 이어서 정답 없이 계산되는 증상으로 실패 유형을 추정하고 유형별 처방으로 외삽 성능을 높인다.

정답 형태는 LLM에 주지 않는다. 정답식은 벤치마크·시뮬레이션에서 **사후 분석에만** 쓴다.

## 2. 실패 유형

| 유형 | 의미 | 정답을 아는 환경에서의 판정 | 처방 후보 |
|---|---|---|---|
| 1. 생성 실패 | 정답 형태가 한 번도 제안되지 않음 | 모든 LLM 응답에 정답 계열이 있었는지 | LLM 지식·RAG·경쟁 가설 생성 |
| 2. 피팅 실패 | 정답 형태는 나왔으나 계수를 잘못 맞춤 | 해당 후보를 더 강한 최적화로 재피팅하면 개선되는지 | 재시작·최적화 개선 |
| 3. 선택 실패 | 잘 맞춘 정답 형태가 최종 선택에서 탈락 | buffer 포함 여부와 가중치 | 복수 식 유지·선택 기준 변경 |
| 4. 식별 불가 | 정답 구조로 피팅해도 데이터만으로 외삽 불가 | 정답 구조를 학습 데이터로만 피팅한 OOD 오차 | 상수 범위·측정 제안·위험 경보 |
| 5. 지식 오류 | 제공한 지식 자체가 틀림 | 지식 조건 교란 비교 | 대안 유지 |

이 틀에 이전 후보가 들어간다: 1은 LLM 지식·RAG, 2는 SAGE-Fit류 피팅 개선, 3은 LLM-SR-UQ·탐색 반복 간 불일치, 4는 경쟁 가설에 의한 위험 감지.

## 3. 논문 구성안

- **1층 — 분석:** 정답을 아는 LLM-SRBench LSR-Synth 109문제에서 외삽 실패를 유형별로 분류한다. 어떤 비율이 나와도 보고 가치가 있다. 예를 들어 식별 불가가 많으면 더 좋은 LLM만으로는 해결되지 않는다는 결론이 된다.
- **2층 — 방법:** 정답 없이 계산되는 증상(재시작 간 계수 편차, 후보 간 적합도 차이, 경쟁 가설의 OOD 갈라짐 등)으로 유형을 추정하고 처방한다. 주장할 개선은 **같은 LLM 호출 예산에서** 맞춤 처방이 일괄 처방(라운드 추가)보다 OOD 오차를 줄인다는 것이다.

## 4. 가장 가까운 선행연구

확인 수준: 웹 검색과 원문 확인. 전수 문헌조사가 아니다.

| 자료 | 확인 수준 | 이미 한 것 | 확인 범위에서 하지 않은 것 |
|---|---|---|---|
| Wang et al., *When Good Equations Get Bad Scores: Improving Symbolic Regression Through Better Parameter Optimization* (SAGE-Fit), 2026 preprint. [HTML](https://arxiv.org/html/2605.23272v2) | 본문 앞부분(약 75%) | 구조 탐색과 피팅을 분리. 불안정한 피팅이 좋은 구조를 탈락시킴을 replay로 분석. LSR-Synth 사용 | 별도 OOD 분할 평가 없음. 정답 구조 oracle 없음. 사용 시점의 정답 없는 진단 없음 |
| Xie et al., *Language models guide symbolic equation discovery by controlling search* (LLM-PySR), 2026 preprint. [arXiv](https://arxiv.org/abs/2607.04156) | 초록 | LLM 역할(식 작성·선택·탐색 제어) 비교. 탐색 제어 역할이 가장 균형적 | 초록 기준 외삽 실패 원인 분해 없음. PDF 본문 미확인 |
| Saveliev et al., *Influence-Guided Symbolic Regression* (IGSR), ICML 2026 표기. [arXiv](https://arxiv.org/abs/2605.29184) | 초록 | 항별 기여도를 LLM에 피드백 | 초록 기준 외삽별 원인 진단 없음. 본문 미확인 |
| *In-Context Symbolic Regression* (ICSR), ACL SRW 2024. [원문](https://aclanthology.org/2024.acl-srw.49.pdf) | 검색 요약 | 기호 탐색과 수치 피팅 분리 | — |
| LLM-SR-UQ의 A1 진단 피드백 (`upstream/diagnostics.py`) | 코드 확인 | 약한 계수, 잔차 구조, 정체를 진단해 LLM에 전달 | 학습 구간 적합도를 위한 진단 |

판정(1차): **모듈화와 진단 피드백 구성은 기존이다.** 확인 범위에서 외삽 실패를 모듈별 원인으로 분해한 연구는 찾지 못했다. 신규성 주장은 분해 분석과 유형별 처방에 한정한다.

### 4.1 재검토 (같은 날, 검색 범위 확대) — 1차 판정 수정

일반 SR 실패 분석, 자기진단형 SR 에이전트, LLM 에이전트 실패 원인 추적까지 넓혀 다시 검색했다. **“정답 없이 실패 유형을 추정해 맞춤 처방한다”는 부분은 비어 있지 않다.**

| 자료 | 확인 수준 | 겹치는 부분 | 남는 차이 |
|---|---|---|---|
| Li et al., *From Score to Diagnosis: Search-Independent Post-Diagnosis for Symbolic Regression* (PostDiag), 2026-10-17 preprint. [HTML](https://arxiv.org/html/2610.24535) | 본문 §3–4, 부록 일부 | **직접 겹침.** 정답 없이 관측 데이터의 bootstrap 재피팅 편향·분산으로 후보를 Settled/Underfit/Overfit/Mismatched로 진단하고, 상태별 처방을 LLM 편집기에 전달. LSR-Synth 129문제, LLM-SR·SR-Scientist·PySR 등에 적용. OOD 정확도 개선도 보고 | 진단이 **관측(학습) 데이터만** 사용한다. 학습 구간에서 잘 맞아 Settled로 보이지만 외삽에서 틀리는 경우를 다루지 않는다. 진단 상태와 OOD 오차의 관계, 식별 불가능성은 확인한 본문에서 논의하지 않는다 |
| Kabra, Kriplani, Shojaee, Reddy, *SurfaceBench*, TMLR 표기. [arXiv](https://arxiv.org/abs/2511.10833) | 초록·검색 요약 | 실패를 탐색 실패(틀린 함수 계열)와 식 피팅 실패로 분류. LLM 방법은 구조 prior는 강하나 계수 보정이 약하다고 서술 | 초록 기준 OOD 평가 없음 |
| Zhao et al., *A-SR: Self-Evolving Agentic LLMs for Symbolic Regression via Hierarchical Coordination*, 2026-08 preprint. [arXiv](https://arxiv.org/abs/2608.04872) | 초록 | 이질적 탐색 실패가 스칼라 점수로 압축된다고 지적. 실패 흔적·유효성 진단을 역할별 에이전트에 분배. ID/OOD NMSE 보고 | 초록 기준 외삽 실패 원인 분해 없음 |
| STRIDE, Deliberate Evolution, DrSR, SR-Scientist 등 자기반성·진단형 SR 에이전트. [STRIDE](https://arxiv.org/abs/2605.17790), [DE](https://arxiv.org/abs/2606.04360), [DrSR](https://arxiv.org/abs/2506.04282) | 검색 요약 | 반성·비평·구조 진단 도구로 탐색을 수리 | 학습 구간 피드백 중심. 본문 미확인 |
| LLM-AutoSciLab, 2026 preprint. [arXiv](https://arxiv.org/abs/2605.24043) | 검색 요약 | LLM 방법이 교과서식 단순 형태로 수렴해 비선형 항을 놓치고, 판별 실험으로 복원이 개선된다고 보고 | 측정(실험) 접근. 본문 미확인 |
| Zhang et al., *Which Agent Causes Task Failures and When?*, ICML 2025. [PMLR](https://proceedings.mlr.press/v267/zhang25cq.html) | 초록 | LLM 다중 에이전트의 실패 원인 자동 추적이라는 일반 틀 | SR·외삽과 무관 |
| 근사·추정·최적화 오차 분해(일반 학습 이론) | 배경 지식, 이번에 검색하지 않음 | 생성 실패≈가설 공간, 피팅 실패≈최적화, 식별 불가≈데이터 정보 부족과 대응하는 고전적 개념 | 개념 자체는 신규성이 아님 |

**수정 판정:**

- 정답 없는 진단 + 유형별 처방 → **PostDiag와 겹친다.** 이것을 주 기여로 주장하지 않는다.
- 탐색 실패 vs 피팅 실패 분류 → SurfaceBench·SAGE-Fit과 겹친다.
- 확인 범위에서 남은 것은 더 좁다:
  1. **외삽 실패에 한정한** 원인 분해, 특히 정답 구조 oracle로 식별 불가능성을 분리하는 분석.
  2. **학습 구간 진단의 사각지대:** PostDiag류 진단이 Settled로 판정하지만 외삽에서 실패하는 경우가 얼마나 되는가. 이 경우는 관측 데이터만으로는 원리상 잡을 수 없다.
  3. 그 사각지대를 **외부 정보**(LLM 경쟁 가설, 상수 범위, 측정 제안)로 잡는 진단.

PostDiag는 이 후보의 핵심 비교군이 된다. 다만 이 재검토도 전수 조사가 아니고, PostDiag 부록 E(OOD 정의)와 SurfaceBench·A-SR 본문은 확인하지 못했다.

## 5. 이 저장소에서 이미 가능한 것

- 벤치마크 loader의 정답식 `gt_expression` (`upstream/benchmark/loader.py`).
- 실행기 출력의 모든 LLM 요청·응답 `llm_calls.jsonl`, 남은 후보 전체 `buffer.json`·`buffer_numeric.npz`, 예측 `predictions.npz`.
- 식 계열 비교 `upstream/canonical.py`의 `family_signature`. 동등성 판정 성능은 미확인.
- 1층 분석은 109문제 × 1회 실행으로 시작할 수 있다. 탐색 반복 간 불일치 후보의 545회보다 적다.

## 6. 위험과 사전에 정할 기준

- **수식 동등성 판정:** 같은 함수의 다른 표기를 정답 계열로 인정해야 한다. `canonical.py`를 수동 표본으로 점검한다.
- **2와 4의 구분:** 피팅 실패가 아니라고 판정할 최적화 강도(재시작 수, 전역 최적화 여부)를 실행 전에 고정한다.
- **합성 문제의 한계:** LSR-Synth는 합성 문제다. 복사 냉각 같은 시뮬레이터로 결과가 유지되는지 따로 확인한다.
- **유형의 중복:** 한 문제에 여러 원인이 겹칠 수 있다. 판정 순서와 중복 처리 규칙을 미리 정한다.
- 시험 정답으로 처방 규칙을 고치지 않는다. 처방은 개발 문제에서 정한다.

## 7. 다음 확인 항목

1. PostDiag 부록 E의 OOD 정의와 진단 상태별 OOD 오차. SAGE-Fit·IGSR·SurfaceBench·A-SR 본문.
2. LLM-PySR 본문의 오차 분해 여부.
3. `canonical.py`의 동등성 판정 점검.
4. 109문제 BF16 1회 실행의 소요 시간(`scripts/timing_summary.py`).
