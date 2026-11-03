# 코드 검토 결과와 외삽 성능 중심 연구 방향 — 2026-11-03

배경 자료: [실험 활용 가능성 감사](2026-11-03-salvage-ledger.md)

## 검토 범위와 한계

- 읽은 코드: `pp-extrapolation/src/pp_extrapolation/model.py`, `paper_ppx.py`, `experiments/ppx_final_result_reproduction_v1.py`, `experiments/full_equal_candidate_budget.py`, `ca-css-ncmapss/run_affine_tail_external_three.py`(HUST·Virkler), `pae_boundary_realdata.py`(Sunwoda·RWTH·MICH), `ncmapss_tra_quantile_split.py`.
- llm_prior·prior_module은 커밋 순서와 자체 오류 기록 위주로 확인했고 코드 전체를 읽지 않았다.
- 실험을 다시 돌리지 않았다. 아래 Virkler 계산만 원자료 CSV로 직접 했다.

## Part 1. 코드에서 확인한 문제

### 치명 — 결론을 바꾸는 문제

| # | 문제 | 근거 | 영향 |
|---|---|---|---|
| C1 | **Virkler 시험의 pooled R²가 사소한 기준선보다 낮다** | test 행은 균열 39 mm와 49.8 mm 두 지점뿐이고, 49.8 mm의 RUL은 항상 0이다. 원자료로 계산하면 “39 mm면 평균 RUL, 49.8 mm면 0”이라는 시편 정보 없는 예측이 pooled R² **0.962**다. PP-X 보고값은 0.888 | Main-9의 Virkler 승리는 의미가 없다. pooled R²가 끝점 행 때문에 부풀려진 구조 |
| C2 | **9개 결과가 하나의 알고리즘이 아니다** | `ppx_final_result_reproduction_v1.py`가 데이터셋마다 다른 실험의 결과 파일을 읽는다. `TARGET` 딕셔너리에 목표 R²가 하드코딩돼 있다. `FINAL_BQ_ARM = {"Sunwoda": "bounded", "RWTH": "bounded", "MICH": "dual_scale"}` | “동결된 PP-X” 주장이 성립하지 않음 |
| C3 | **선택 규칙이 결과에 맞춰졌다** | `paper_ppx.py`의 `final_priority_executor` 주석: “Order matches the frozen 9-setting portfolio”. 9/11 동결 뒤 9/15·9/17에 규칙 변경 | 사후 적합. 9/9·8/9의 p값은 추론 근거가 아님 |
| C4 | **HUST 라벨이 선형 외삽으로 만들어졌다** | `causal_hust_frame`: RUL = proxy EOL − 현재 cycle. proxy EOL은 마지막 20% 구간의 선형 기울기로 0.88 Ah까지 연장한 값. 다수 셀이 0.88 Ah에 도달하지 않음 | 선형 tail을 가진 PP가 라벨 생성 방식과 같은 가정을 공유한다. 순환적 이점 |
| C5 | **가장 가까운 비교군이 빠졌다** | 동일예산 비교의 `MODELS`에 MLP, FT-Transformer, V-REx, GroupDRO, monotone NN, Engression, linear-tail RBF, SVGP만 있다. hull-out validation으로 alpha를 고른 **Ridge·선형 회귀·persistence가 없다** | 학습 범위 밖에서 PP는 사실상 “affine + 상수”다(아래 M1). 이와 같은 선형 외삽 기준선 없이는 PP의 기여를 분리할 수 없음 |

### 높음 — 평가 설계 문제

| # | 문제 | 근거 |
|---|---|---|
| H1 | N-CMAPSS validation이 train 엔진을 포함 | `make_tra_hard_split`: `val_units = list(cfg.train_unit_ids) + list(cfg.val_unit_ids)`. unit-disjoint 원칙과 모순 |
| H2 | validation과 test의 외삽 거리가 다름 | 배터리 3종은 validation이 train-q25 아래이면서 early 구간 안이고, test는 cutoff 아래 late 구간. 가까운 외삽으로 먼 외삽용 모델을 고른다 |
| H3 | validation 재사용 | 같은 validation으로 affine alpha, early stopping checkpoint, executor·경로를 모두 선택. Virkler validation 20행, NASA는 fold마다 셀 1개 |
| H4 | pooled R²가 unit 간 분산에 지배됨 | 셀마다 수명이 크게 달라 pooled R²는 “어느 셀이 오래 가나”를 맞히는 것만으로 높아질 수 있다. 셀 내부 외삽 정확도와 다름 |
| H5 | 라벨 정의 불일치 | MATR: 기록 마지막 cycle − 현재. HUST: 우측 검열 proxy. Sunwoda: 880 mAh 첫 교차. MICH: 공개 life label |

### 중간 — 모델·구현 문제

| # | 문제 | 근거 |
|---|---|---|
| M1 | 학습 범위 밖에서 PP ≈ 선형 외삽 | `PPNet`: tanh 잔차 경로는 범위 밖에서 포화해 상수가 된다. 예측 = affine(z) + 상수. `residual_decay`를 쓰면 상수도 0으로 감소 |
| M2 | soft-anchor 변형에서 weight decay가 affine을 0으로 당김 | AdamW `weight_decay=2.0`이 affine 파라미터에도 적용되어 anchor 항(초기값으로 당김)과 충돌 |
| M3 | gradient clipping이 일부 파라미터에만 적용 | `clip_grad_norm_(model.nonlinear.parameters(), …)`. affine·gate가 학습되는 변형에서는 클리핑 없음 |
| M4 | `target_scale = max(max(train_y), 1.0)` | 정규화된 작은 단위의 라벨이면 scale이 1로 고정되어 출력 상한 의미가 바뀜 |

### prior_module·llm_prior — 반복되는 위험

- prior_module 자체 기록의 무효 5건(E01 시트 불일치, E03 provenance, E04 broadcasting, E06 생성기 명세, E08 index). 그 뒤에도 입력 namespace 연결 오류가 RC144·RC149에서 다시 나왔다. **데이터 출처 연결 오류가 반복되는 패턴**이다.
- llm_prior E1–E8은 첫 커밋에 한꺼번에 들어와 사전등록 순서를 확인할 수 없다.

### 고칠 때 공통으로 적용할 규칙

1. 지표: pooled R² 단독 보고 금지. **unit 내부 RMSE, 고정 horizon 상대오차, 사소한 기준선 대비 skill**을 함께 보고.
2. 기준선 필수: persistence, 선형 tail(hull-out alpha), 사소한 구조 기준선(Virkler의 위치별 평균 같은 것).
3. 방법 하나 = 코드 경로 하나. 데이터셋 이름으로 분기하는 코드 금지.
4. validation은 test와 같은 외삽 거리 구조로 만든다.
5. 시험 결과를 본 데이터는 개발용으로 강등한다.

## Part 2. 외삽 성능을 올리는 연구 방향

### 지금까지의 실험이 알려 주는 것 — 방향 설정의 근거

| 관찰 | 출처 | 의미 |
|---|---|---|
| 정량 법칙은 확실히 이긴다 | pae Virkler: 전역 Paris 식 held-out R² 0.967 대 HGB 0.930 | 성능을 올리는 지렛대는 **정량적 관계** |
| 약한 prior는 단순 기준선을 못 넘는다 | PP-X 사전 평가 다수 실패, prior_module CCPP | 단조·경계·affine tail은 Ridge·persistence 수준 |
| 맞는 식 형태만으로는 부족하고 실현 파라미터가 결정적이다 | llm_prior 부분 realization 실험: onset+scale 제약이 regime 격차의 97.7%, scale+shape가 curvature 격차의 92.8%를 메움. onset만은 3.6% | 외부 지식은 **식 + 파라미터 범위**로 들어가야 한다 |
| 증거가 약하면 여러 가설 유지가 낫고, 균등 유지는 나쁘다 | llm_prior E15-A, E16-B | 점 예측과 불확실성 모두 **증거 가중 혼합**이 맞는 방식 |
| 이상적 관계 전달은 큰 이득 | prior_module 압력 세계: TransportNN R² 0.993 대 RawNN 0.019 | 관계가 정확하면 데이터 없는 영역도 살릴 수 있음 |

### 추천 방향: 정량 관계 + 파라미터 범위로 외삽 성능 올리기

> **실현 제약 외삽(realization-constrained extrapolation):** 문헌·물리에서 얻은 식 형태와 파라미터 범위로 허용 함수 집합을 만들고, 관측으로 가중한 혼합으로 점 예측과 구간을 낸다.

- **성능:** 학습 범위 밖의 모양을 데이터 대신 정량 관계가 정한다. 약한 prior가 실패한 이유를 정면으로 해결한다.
- **불확실성:** 파라미터 범위가 허용 집합을 줄이므로, 구간이 좁아지면서도 포함률을 유지할 수 있다. 범위가 틀리면 포함률이 깨지는 것을 검출해야 한다.
- **기존 자산:** pae의 식 카드·typed 컴파일러, llm_prior의 family/ψ 분해와 증거 가중, PP-X의 데이터 로더·사전 프로토콜 양식, prior_module의 freeze/replay 인프라를 그대로 쓴다.
- **겹침 주의:** 물리 모델 + 파라미터 prior 자체는 Bayesian calibration(Kennedy–O’Hagan 계열), 가속수명시험(ALT) 모델로 오래된 분야다. 기여는 “어떤 실현 제약이 외삽 오차를 얼마나 줄이는가”의 정량화, LLM/RAG가 범위를 가져올 때의 정확도와 피해, 엄격한 사전 검증이 되어야 한다. 착수 전 겹침 확인이 필요하다.

### 데이터셋 선택 — 정량 관계가 있고 엄격한 외삽이 의미 있는 것

| 후보 | 외삽 축 | 정량 관계와 범위 | 기존 자산 | 판단 |
|---|---|---|---|---|
| **배터리 온도 전이**(Sunwoda 25→35°C 외) | 온도 | Arrhenius 가속: 활성화 에너지 범위가 문헌에 있음. SEI 성장의 √t 법칙 | Sunwoda 로더, PP의 사후 성공 사례 | **1순위.** 온도 축 외삽이 명확하고 데이터가 있다. 다른 온도 조건 배터리 데이터셋을 사전 검증용으로 추가 확보 필요 |
| **피로 균열 성장**(Virkler, SJTU, PHM2019 crack) | 균열 길이, 하중 | Paris 법칙: 알루미늄의 지수 m 범위가 문헌에 있음 | pae Virkler, 기존 균열 코호트 프로토콜 | **2순위.** 정량 법칙의 이득이 이미 보임. 평가는 C1을 피해 초기 prefix(예: 17 mm까지)로 수명을 예측하는 방식으로 |
| 크리프 파단(Larson–Miller) | 시간·온도 | Larson–Miller 상수 범위 | 없음 | 고전적 외삽 문제라 이야기가 강하지만 공개 데이터 확보 여부 미확인 |
| CCPP | 외기 온도 | 터빈 보정 곡선 | E16-B, prior_module | **비추천.** 꼬리가 거의 선형이라 prior가 줄 정보가 적고, 이미 반복 노출됨. 대조 사례로만 |
| N-CMAPSS | TRA | 정량 관계 약함 | CA-CSS, PP-X | 비추천. 외삽 축과 RUL의 물리 관계가 약하고 H1 문제 |

### 실험 설계 원칙

1. **비교군:** 데이터만(MLP, Engression), 선형 tail, persistence, 식 + 학습 데이터 피팅(범위 없음), 식 + 범위, 식 + 틀린 범위, oracle 파라미터.
2. **측정:** 외삽 거리별 RMSE, 고정 horizon 상대오차, CRPS, 구간 포함률·폭. 사소한 기준선 대비 skill.
3. **개발/확증 분리:** Sunwoda·Virkler는 개발용(이미 노출됨). 확증은 한 번도 열지 않은 코호트로, 프로토콜 커밋 후 다운로드.
4. **한 방법, 한 코드 경로.** 데이터셋별 분기 금지. 식 카드만 데이터셋마다 다르다.

### 다음 단계

1. Part 1의 C1–C5, H1을 고치고 Sunwoda·RWTH·MICH·Virkler에서 선형 tail·persistence 기준선을 같은 지표로 다시 측정. PP가 선형 외삽보다 실제로 나은지 먼저 확인.
2. 배터리 온도 전이에 쓸 Arrhenius 식 카드와 활성화 에너지 범위의 문헌 근거 정리.
3. 사전 검증용 온도 조건 배터리 코호트 후보 조사(아직 열지 않은 것).
4. 실현 제약 외삽의 겹침 확인(Bayesian calibration, ALT, physics-informed degradation).
