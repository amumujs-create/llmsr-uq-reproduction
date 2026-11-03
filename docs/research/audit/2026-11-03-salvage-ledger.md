# 전체 외삽 실험의 활용 가능성 감사 — 2026-11-03

목적: 여러 저장소에 흩어진 외삽 실험 중 **논문 근거로 살릴 수 있는 것**을 가리고, 그 위에서 할 수 있는 연구를 정한다.

## 감사 범위와 한계

- 대상: `pp-extrapolation`(PP-X), `llm_prior`, `prior_module`, `pae`, `ca-css-ncmapss`, `pp_pae_total`.
- 방법: README·결과 문서·커밋 순서·일부 코드 읽기. **실험을 다시 실행하지 않았다.** 개별 수치·모든 코드 버그를 검증하지 않았다.
- 사전 검증 여부는 커밋 메시지와 결과 문서의 기록에 근거했다. 각 프로토콜이 데이터 개봉 전에 커밋됐는지는 건별로 다시 확인해야 한다.

## 등급 기준

| 등급 | 의미 | 조건 |
|---|---|---|
| A | 근거로 사용 가능 | 데이터 개봉 전 프로토콜·방법 고정, 한 번 평가 |
| B | 개발 근거 | 사후 분석, 또는 결과를 본 뒤 방법을 바꿈. 가설·동기로만 사용 |
| C | 자산 | 결과와 무관하게 재사용 가능한 코드·데이터 경로·프로토콜·기록 |
| D | 폐기·의심 | 누설, 분할 오류, 구현 버그, 라벨 불일치 |

## 작업 속도 — 공통 위험

| 저장소 | 커밋 | 기간 |
|---|---:|---|
| pp-extrapolation | 190 | 2026-09-05 ~ 09-17 |
| llm_prior | 225 | 2026-09-17 ~ 09-24 |
| prior_module | 1,216 | 2026-09-26 ~ 10-31 |

결과를 보며 빠르게 설계를 바꾸면, 개별 실험이 정직하게 기록돼도 연구 과정 전체가 시험셋에 적응한다. 아래 B 등급이 많은 주된 이유다.

## 1. pp-extrapolation (PP-X)

**A — 사전 1회 평가 기록 (대부분 실패):**

| 코호트 | 기록된 결과 |
|---|---|
| Axial fan untouched | 실패. PP macro R² −0.814, MLP −0.516 |
| XJTU-SY untouched | 실패. R² 음수 |
| MATWI tool life untouched | 실패. R² 음수 |
| Ferrara bearing external | 실패. 모든 모델 R² 음수 |
| NASA PCoE 미개봉 | 실패. persistence가 우세 |
| Na-ion prospective gate | 5개 중 1개만 올바른 승인 |
| MATR batch2 sealed | 사전 기준 하나 실패. 이후 개발에 재사용됨 |
| MATR 2019 confirmatory | pooled 기준 성공, unit-macro R² 음수 |
| Misata untouched | 절대 예측 성공, PP 우월성 실패 |
| N-CMAPSS DS03 prospective | 경로 선택 PASS(선택된 것은 PP를 쓰지 않는 fallback), Engression 0.901 > 0.882 |

**B — 개발 근거:** Main-9 사후 9/9(p=0.0039), 동일예산 8/9(p=0.039), 구성요소 ablation, CCMR v1.6–v2.3·BQ·RBPR·CTBF·ZNION 등 변형. 9개 결과는 데이터셋별로 다른 실험 산출물을 이어 붙였고(`experiments/ppx_final_result_reproduction_v1.py`), 선택 규칙은 9/11 동결 이후 9/15·9/17에 다시 바뀌었으며 주석에 “Order matches the frozen 9-setting portfolio”라고 적혀 있다. 약 35개 이상 데이터셋에서 고른 9개다. **p값은 추론 근거로 쓸 수 없다.**

**C — 자산:** 약 35개 열화 데이터셋 로더와 분할 어댑터, 1D hull 감사, 사전 프로토콜·SHA 기록 양식, 비교군 8종 × 후보 30 × seed 5의 동일예산 하네스, 물리 unit 단위 계층 bootstrap·BH 보정 코드, 재사용 금지 감사(`COHORT_REUSE_AUDIT_KO.md`).

**D — 의심:**
- N-CMAPSS DS02 validation 엔진이 train 엔진(2, 5, 10, 16, 18)과 겹친다(`MAIN_NINE_SPLIT_KO.md` §9). unit-disjoint 원칙과 모순.
- 데이터셋별 경로 하드코딩(`FINAL_BQ_ARM`, `ccmr_v23_frozen_12domain_audit.py`의 route map).
- RUL 라벨 정의 불일치(MATR: 기록 마지막 cycle 기준, HUST: 우측 검열 proxy EOL).
- 검증 표본 과소: Virkler test 20행, NASA 4셀 LOOCV.

## 2. llm_prior (Generic Prior Observability Lab)

**A — 합성 범위 안에서 사용 가능:** admission v2의 독립 4,500-task 재현(false admission 65.0% → 20.9%), E8 factorial confirmation, E10–E13 동결 합성 실험, E15-A confirmatory(증거가 약할 때 여러 가설 유지가 외삽 오차를 줄임), **E16-B CCPP 사전 평가**(주 가설 실패, 증거 가중 보조 결과는 회복). E15·E16은 프로토콜 → 결과 순서가 커밋으로 확인된다. E16-B의 CRPS 수정은 validation 개봉 전이다.

**제한:** E1–E8 결과는 첫 커밋(9/17)에 한꺼번에 들어와 순서를 확인할 수 없다. 생성기를 연구자가 설계한 1D 문법 안의 결과이며 외부 타당성은 별도다.

**C — 자산:** Prior Anatomy Framework·용어 체계, family/realization(ψ) 분해, 관측도 O 조작 설계, CCPP X-only AT 분할(목표값 미사용, 4.46°C 간격).

## 3. prior_module

**A:** 확인한 범위에서 없음. CCPP는 반복 노출됐고, Brayton·온도 합성 세계는 단일 task 반복이다.

**B — 개발 근거, 일관된 부정적 발견:**
- CCPP에서 물리 prior(외부 터빈 곡선, 습공기, 냉각 분기)는 같은 선형 backbone이나 Ridge를 확실히 넘지 못했다.
- 부분 정보 구조는 관계식을 보존해도 외삽 크기를 잘못 추정할 수 있다.
- 이상적 압력 세계에서는 정확한 관계 전달이 큰 이득(R² 0.99 대 RawNN 0.02)을 냈지만, 넓은 데이터에서는 Ridge가 근접했다.
- 기록: RC1–RC195 claim ledger(3,495줄).

**C — 자산:** freeze → replay → 독립 scorer 인프라, 자체 validity ledger, claim–evidence ledger, 2026-10-31 novelty audit(문헌 18편), Brayton·온도·전극 건조기 생성기, 문헌 지도.

**D — 자체 기록된 무효:** E01 CCPP 시트 불일치, E03 PM3-V0 provenance, E04 broadcasting 차원 오류, E06 생성기 명세 불일치, E08 비균일 grid index 오류.

## 4. pae

**B:** Virkler 사후 ablation. 전역 Paris 식이 직접 회귀보다 크게 우세(held-out pooled R² 0.967 대 HGB 0.930). 시편별 typed 계수 적응은 이득 없음. LLM front-end 사전등록 파일럿은 thinking 출력 예산 문제로 실패, 사후 no-think 실행은 통과.

**C:** 식 카드 schema, typed 정적 계약 검사, fail-closed 경로, 테스트.

## 5. ca-css-ncmapss

**B:** N-CMAPSS DS02 TRA 분할의 단일 데이터셋 결과. 자체 평가에서도 단일 데이터셋·SOTA 비교 부재·TRA 강제 채택(pseudo acc 0.49 < 0.6)을 약점으로 기록. Hard 분할에서 TabPFN RMSE가 더 낮음.

**C:** N-CMAPSS·C-MAPSS 로더, TRA quantile 분할(PP-X Main-9 #9가 사용), 물리 위반율 공동 지표.

## 6. 저장소를 가로지르는 결론

1. **실제 데이터의 긍정적 결과는 거의 모두 개발 근거다.** 사전 1회 평가는 약 10건 중 성공 1–2건이다.
2. **prior의 강도에 따른 일관된 패턴:** 정량 법칙(Virkler의 Paris 식)은 직접 모델보다 확실히 낫다. 약한 prior(단조, 경계, affine tail, 습공기)는 강한 단순 기준선(Ridge, persistence, Engression)을 안정적으로 넘지 못했다. 이는 llm_prior의 “맞는 prior도 실현 파라미터가 식별되지 않으면 쓸모없다”는 합성 결과와 일치한다.
3. **엄격한 꼬리 외삽에서는 대부분의 방법이 실패한다.** 사전 평가 다수에서 모든 모델의 R²가 음수였다.
4. 기록·사전등록·재사용 금지 문화는 강한 자산이다. 부족한 것은 **변경 속도의 통제**다.

## 7. 다음 확인 사항

- A 등급 사전 평가 각각의 프로토콜 커밋이 데이터 다운로드·라벨 열람보다 앞서는지 건별 확인.
- 사전 평가 코호트마다 동일 비교군 전체가 같은 조건으로 실행됐는지 확인.
- N-CMAPSS validation 중복이 다른 실험에도 있는지 확인.
- `model.py`, 분할 어댑터, 라벨 생성 코드의 독립 코드 리뷰.
