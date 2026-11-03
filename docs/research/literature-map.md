# 검색·검토 자료 통합 기록 — 2026-11-03

범위: 이 대화에서 검색·확인한 자료와 사용자가 전달한 후속 조사 노트를 통합한다. 새로운 전수 문헌조사나 실험 결과가 아니다.

**현재 결론: 관련 연구 목록과 연구 후보는 확보했으나, 가장 가까운 선행연구와 비교해 구체적인 추가 기여를 충분히 뒷받침한 추천 주제는 아직 제시하지 못했다.** 목록이 길다는 사실은 신규성 검증을 뜻하지 않는다.

[현재 연구 방향](README.md) · [기존 상세 겹침 검토](reviews/2026-11-03-overlap.md) · [8편 비교와 약한 항 후보](reviews/2026-11-03-weak-term-candidate.md) · [결정 기록](decision-log.md)

## 확인 수준

- **본문 확인:** 앞선 대화에서 원문 또는 저자 원고의 관련 방법·실험 절을 직접 확인했다. 논문의 모든 주장이나 공개코드를 재현했다는 뜻은 아니다.
- **초록·서지 확인:** 해당 범위의 내용만 확인했다. 구체적 분할·실험 부재를 단정하지 않는다.
- **전달 노트:** 사용자가 붙여 넣은 후속 조사 내용이다. 이 저장소를 갱신하면서 원문·코드를 새로 확인한 것은 아니다.
- **검색 후보:** 검색 중 발견했거나 언급된 자료다. 핵심 주장이나 선행연구 부재의 근거로 사용하지 않는다.

이 문서는 논문의 문장을 재현하지 않고 검토 내용과 출처를 요약한다. 프리프린트의 게재 상태는 확인한 당시 상태로 기록하며, 저널 등재·등급을 임의로 추정하지 않는다.

## A. 기존 상세 검토에 포함된 12편

P 번호는 [기존 검토](reviews/2026-11-03-overlap.md)의 번호와 같다.

| ID | 논문·출처 | 확인 수준 | 연구 방향과의 관련성 / 미해결 확인 항목 |
|---|---|---|---|
| P1 | *Large Language Models for Extrapolative Modeling of Manufacturing Processes*, JIM online 2025. [출판사](https://doi.org/10.1007/s10845-025-02638-w), [원고](https://arxiv.org/pdf/2502.12185) | 본문 확인 | 문헌 추출과 분석식 정제를 통한 제조 외삽. LLM 외부 지식 활용 자체는 기존이다. RSM 생성 grid와 실제 미관측 공정 측정을 구분해야 한다. |
| P2 | *Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling*, 2026 preprint. [arXiv](https://arxiv.org/abs/2606.11605), [공개 원고](https://www.researchgate.net/publication/406909917_Physics-Distilled_Neural_Network_enabled_by_Large_Language_Models_for_Manufacturing_Process-Property_Predictive_Modeling) | 본문 §3.2–3.4 등 확인 | 물리 prior→교사→학생, 불완전한 prior도 평가. 합성 범위 외 평가와 실제 데이터 CV를 구별해야 한다. 명시적 조건 오류의 독립 조작은 별도 질문이다. |
| P3 | *Physics-informed kernel mixtures for structural dynamics*, MSSP 254,114330,2026. [기관 원고](https://eprints.whiterose.ac.uk/id/eprint/240468/), [DOI](https://doi.org/10.1016/j.ymssp.2026.114330) | 저자 본문·부록 확인 | 관측 switching variable에 따른 물리 의존도 조절이 이미 있다. 학습 자료로 전환 파라미터를 추정하는 조건과 완전히 미관측 전환을 구분한다. |
| P4 | *Physics-Informed Neural Networks With Weighted Losses by Uncertainty Evaluation for Accurate and Stable Prediction of Manufacturing Systems*, TNNLS 35(8),2024. [초록](https://pubmed.ncbi.nlm.nih.gov/37028329/), [DOI](https://doi.org/10.1109/TNNLS.2023.3247163) | 초록·서지 확인 | 부정확한 물리·잡음에서 prediction-error variance 기반 손실 가중치. 목표 범위 분할의 상세는 미확인이다. |
| P5 | *Correcting model misspecification in physics-informed neural networks (PINNs)*. [저자 원고](https://arxiv.org/abs/2310.10776), [JCP 505,112918,2024](https://doi.org/10.1016/j.jcp.2024.112918) | 저자·출판사 초록과 게재 정보 확인 | 신경망 discrepancy와 Bayesian/ensemble 보정. 관측이 주는 보정 근거와 목표 구간에만 존재하는 오류의 식별은 다르다. |
| P6 | *LLMs as designers of physics-constrained neural networks for constitutive modeling: a demonstration on hyperelastic solids*, 2026. [원문](https://doi.org/10.1007/s44379-026-00073-z) | 본문 확인 | GenCANN: LLM이 물리 제약 신경망을 구성하고 미관측 하중·범위를 평가한다. |
| P7 | *LLM-driven design of physics-constrained constitutive models: Two agents are better than one*, 2026. [원문](https://doi.org/10.1007/s44379-026-00099-3) | 본문 확인 | Creator–Inspector 및 수치 제약 검사. 제약을 만족하는가와 새 조건에서 가정이 타당한가는 구분해야 한다. |
| P8 | *LLM-SR: Scientific Equation Discovery via Programming with Large Language Models*, ICLR2025. [공식 페이지](https://proceedings.iclr.cc/paper_files/paper/2025/hash/28df8e730c054c5331855fd4d5403ba9-Abstract-Conference.html), [원문](https://arxiv.org/html/2404.18400v3) | 본문·공식 기록 확인 | 코드로 식 구조를 제안하고 계수를 최적화한다. OOD와 일부 잡음 실험이 있어, 잡음을 전혀 다루지 않았다는 해석은 부정확하다. |
| P9 | *LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models*, ICML2025. [공식 페이지](https://proceedings.mlr.press/v267/shojaee25a.html) | 공식 설명·평가 취지 확인 | 흔한 식의 암기로 인한 평가 왜곡을 다룬다. 일반 물리 사전 지식의 활용과 시험 정답 직접 누설을 구분하는 참고자료다. |
| P10 | *Extrapolation-Aware Nonparametric Statistical Inference*, JMLR27(150),2026. [공식 페이지](https://www.jmlr.org/papers/v27/24-0838.html) | 초록·서지 확인 | 명시적 방향별 도함수 가정 아래 외삽 경계를 구성한다. LLM 조건 추출과 동일한 방법은 아니다. |
| P11 | *LLM knowledge for improved Out-of-Distribution Tabular Prediction – A Cross-Validated Regularisation Approach*, 2026-10-27 preprint. [v1](https://arxiv.org/html/2610.36806v1) | 본문 확인 | LLM 계수 prior, source CV, 교란 prior, ID/OOD 효과 분리. 정규화와 외삽 지식의 이득을 구별하는 질문도 선행연구에 있다. |
| P12 | *What Language Models Know But Don’t Say: Non-Generative Prior Extraction for Generalization* (LoID), COLM2026. [공식 PDF](https://proceedings.colm.cc/paper_files/paper/2026/file/fb575ab4d882a4c734641155a5f30911-Paper-Conference.pdf) | 본문·§5.2 확인 | logits 기반 prior. 일반적 물리가 설비의 정량 관계에 맞지 않아 악화되는 발전소 사례가 있다. 시험셋 전체를 순수 외삽으로 해석하면 안 된다. |

## B. 후속 8편 비교 및 약한 물리항 후보

상세한 범위·잡음 수치와 실험안은 [별도 기록](reviews/2026-11-03-weak-term-candidate.md)에 보존했다. 다음 표에서 **전달 노트의 수치·코드 확인 주장과 이 대화에서 직접 확인한 내용을 합쳐 확정 사실로 만들지 않는다.**

| ID | 논문·출처 | 확인 수준 | 기록할 쟁점 |
|---|---|---|---|
| W1 | UQ-SINDy,2022 — 베이지안 식 발견. [저자 원문 링크](https://www.osti.gov/pages/servlets/purl/1861893) | 전달 노트 | 구조·계수 불확실성 및 prior 변화. 불확실한 항 유지 자체를 신규성으로 주장할 수 있는지 비교해야 한다. |
| W2 | E-SINDy / Ensemble-SINDy,2022. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006119/), [전달된 실험 코드](https://github.com/urban-fasel/EnsembleSINDy/blob/main/SINDY/main_runEnsembleSINDy_heatmap.m) | 방법의 개요는 앞서 논의; 아래 수치·코드 검증은 전달 노트 | 잡음×관측 길이 실험. 관측 시간·표본 수·상태 범위를 혼동하면 안 된다. |
| W3 | Expert Augmentation,2023. [원문](https://arxiv.org/html/2202.03881v3) | 전달 노트 | 일부 물리 누락과 시험 초기 궤적 제공. target 관측을 전혀 받지 않는 외삽과 비교 조건이 다르다. |
| W4 | MetaPhysiCa,2024. [ICLR 원문·부록](https://proceedings.iclr.cc/paper_files/paper/2024/file/f0e5cde3850e7dd0db125c0ebae16680-Paper-Conference.pdf) | 전달 노트 | 초기조건·계수·잡음·후보 라이브러리 변화. 시험 초기 궤적 재적합을 분리해야 한다. |
| W5 | 직교 투영 보정,2025 — 전달 노트의 한국어 설명명. [원문 링크](https://pure.tue.nl/ws/portalfiles/portal/366055041/gyorok25a.pdf) | 전달 노트; 정확한 서지 제목 추가 확인 필요 | 물리식과 신경망 역할 분리. 교대로 배정한 속도 시험과 최고 학습속도 밖의 외삽을 구분한다. |
| W6 | LLM 지식 통합,2026 — 전달 노트의 설명명. [원문 링크](https://www.nature.com/articles/s41598-026-35327-6) | 해당 실험 조건은 전달 노트 | 잡음·프롬프트 비교. 부정확한 지식의 목표 외삽 피해를 직접 평가했는지는 별개다. |
| W7 | OrthoReg,2026 preprint. [v1](https://arxiv.org/html/2606.19145v1) | 전달 노트 | 직교 정규화, 관측 수·잡음·누락 물리 강도 변화. 미분 추정 오염과 방법 실패를 분리할 필요가 있다는 메모다. |
| W8 | LLM-SR-UQ,2026 preprint. [v1](https://arxiv.org/html/2610.42024v1), [저자 코드](https://github.com/MostafaCham/LLM-SR-UQ) | 앞선 원문·이 저장소 재현 맥락; 후속 부재 주장은 전달 노트 | 후보식 간 불일치와 구간 평가. 현재 재현 실행 기록은 새 항 선택·조건 전달 방법의 성능 증거가 아니다. |
| W9 | Kreutz et al.,2012 — Prediction profile likelihood. [원문 링크](https://link.springer.com/article/10.1186/1752-0509-6-120) | 전달 노트 | 관측과 양립하는 예측 범위를 구하는 기존 접근. 작은 후보집합의 모델 평균과 함께 핵심 비교 대상으로 제시됐으나 비교를 실행하지 않았다. |

W5–W7의 설명명을 독립적으로 확인한 정식 제목으로 오인하지 않는다. W6와 아래의 PMC 지식 통합 자료가 같은 논문의 버전인지는 이 기록에서 확정하지 않는다.

## C. 검색 중 확인·언급했으나 주제 판정에 충분히 분석하지 않은 자료

이 절은 후속 검토에서 링크를 잃지 않기 위한 목록이다. 제목이나 초록의 유사성만으로 같은 문제를 해결했다고 판정하지 않는다.

| 자료 | 출처 | 현재 보존 범위 |
|---|---|---|
| SR-Scientist | [원문 v2](https://arxiv.org/html/2510.11661v2), [프로젝트 코드](https://github.com/GAIR-NLP/SR-Scientist) | 이전 검색에서 도구 기반 식 탐색·분석 및 잡음/OOD 평가를 논의. 조건부 지식 전달과의 정밀 비교는 미완료. |
| Knowledge integration for physics-informed symbolic regression using pre-trained large language models | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12800073/), [연구실 페이지](https://acml.teddylazebnik.com/publications/knowledge-integration-physics-informed-symbolic-regression-pre/) | 이전 검색에서 LLM 사전 지식과 기호회귀의 결합 자료로 발견. W6와 서지 대조 필요. |
| Prior or Feedback? What an LLM Uses When Adapting Neural Operators | [arXiv](https://arxiv.org/abs/2610.12325) | 이전 검색에서 과제 설명 교환·피드백 교란 대조군을 검토. target adaptation 설정을 label-free 외삽과 구별. |
| LLMPhy: Parameter-Identifiable Physical Reasoning Combining Large Language Models and Physics Engines | [AISTATS2026](https://proceedings.mlr.press/v300/cherian26a.html) | 공식 초록 확인. 시뮬레이터 피드백을 통한 잠재 파라미터 식별은 현재의 주 연구 범위와 다름. |
| Causal Discovery with Language Models as Imperfect Experts | [arXiv](https://arxiv.org/abs/2307.02390) | 불완전한 LLM 전문가 지식에 관한 검색 후보. 현재 물리 외삽과의 직접 비교 미완료. |
| Do causal predictors generalize better to new domains? | [NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/hash/3792ddbf94b68ff4369f510f7a3e1777-Abstract-Conference.html) | 인과적 변수 선택이면 항상 OOD가 좋아진다는 가정을 경계하기 위한 검색 후보. |
| Learning missing physics from legacy simulators with alternating neural integrators | [Nature Communications 링크](https://www.nature.com/articles/s41467-026-74002-2) | 누락 물리·시뮬레이터 보정의 검색 후보. 목표 정보 사용과 방법 비교 미완료. |
| Test-time Generalization for Physics through Neural Operator Splitting | [PMLR2026](https://proceedings.mlr.press/v306/serrano26a.html) | 미관측 물리 구성·파라미터 일반화 검색 후보. test-time에 받는 정보의 상세 확인 필요. |
| Bounds on Extrapolation Across Phase Transitions with Generalized Regression | [저자 프로젝트 페이지](https://jwei302.github.io/extrapolation/) | 상전이 외삽 경계 검색 후보. 이론 가정·실험 확인 미완료. |
| Rashomon Prediction Intervals | [전달된 PMLR 링크](https://proceedings.mlr.press/v337/nguyen26c.html) | 이전 자료에서 언급된 확인 대기 링크. 원문을 검증한 근거로 사용하지 않음. |
| Trust-Aware Domain Adaptation Using Physics-Guided Reliability Learning — milling 관련 자료 | [Sensors 링크](https://www.mdpi.com/1424-8220/26/14/4473) | 이전 검색에서 target 비라벨 자료와 분류 설정을 구분한 메모. 정확한 제목·평가 절 추가 확인 필요. |
| Interpretable models for extrapolation in scientific machine learning | [RSC2023 원문](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00082f) | 검색 결과에서 leave-one-cluster-out 평가 확인. 동일정보 단순 모델 비교의 참고. |
| Position: LLM for Physics Research Requires Domain-Specialized Training and Tooling | [ICML2026](https://proceedings.mlr.press/v306/lu26av.html) | 입장 논문. 현재 방법의 효과를 입증하는 실험 근거가 아님. |
| A self-correcting multi-agent LLM framework for language-based physics simulation and explanation | [npj AI 원문 링크](https://www.nature.com/articles/s44387-025-00057-z) | 시뮬레이션 생성·수정의 검색 결과. 목표 정답 없는 외삽과의 직접 비교 미완료. |
| PDEBench: An Extensive Benchmark for Scientific Machine Learning | [NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html), [데이터](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-2986) | 공식 설명 확인. 수치 시뮬레이션 평가의 선례이며 현재 연구에 쓸 데이터셋으로 확정하지 않음. |
| RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data | [arXiv](https://arxiv.org/abs/2601.01829) | 시뮬레이션 검색 중 발견한 실제 데이터 연계 후보. 원문 상세 검토 미완료. |
| Multiphysics Bench: Benchmarking and Investigating Scientific Machine Learning for Multiphysics PDEs | [arXiv](https://arxiv.org/abs/2505.17575) | 다중물리 평가 데이터 검색 후보. 원문 상세 검토 미완료. |
| Scientific Machine Learning Benchmarks | [arXiv](https://arxiv.org/abs/2110.12773) | 벤치마크 설계 검색 후보. 원문 상세 검토 미완료. |

SINDy는 배경 방법으로, SINDy와 conformal을 결합한 연구는 이전 논의의 키워드로 언급되었다. 이 기록에서 정확한 논문을 식별할 근거가 없는 항목에는 제목·DOI를 임의로 붙이지 않는다.

## C2. 탐색 반복 간 불일치 후보 검토에서 추가한 자료 — 2026-11-03

상세 비교는 [별도 기록](reviews/2026-11-03-search-disagreement.md)에 있다.

| ID | 자료 | 확인 수준 | 쟁점 |
|---|---|---|---|
| S2 | *Programmatic Context Augmentation for LLM-based Symbolic Regression*, 2026. [arXiv](https://arxiv.org/abs/2605.03101) | §4.1, §4.4.2 확인 | LLM-SR의 seed 민감도(3회 반복 NMSE 수 자릿수 차이). 불확실성 신호로는 사용하지 않음 |
| S3 | Kotanchek et al., *Trustable symbolic regression models*, GPTP V, 2007. [Springer](https://link.springer.com/chapter/10.1007/978-0-387-76308-8_12) · [2021 후속](https://gpbib.cs.ucl.ac.uk/gp-html/Kotanchek_2021_GPTP.html) | 서지·요약 | 기호식 앙상블 합의를 신뢰 지표로 쓰는 개념의 원형 |
| S4 | Haut et al., *Active Learning in Genetic Programming*, 2023. [arXiv](https://arxiv.org/abs/2308.00672) | 검색 요약 | 앙상블 불확실성으로 측정점 선택 |
| S5 | Fasel, *Sparse Identification of Nonlinear Dynamics with Conformal Prediction*, 2025. [arXiv](https://arxiv.org/abs/2507.11739) | 초록 | E-SINDy + conformal. 외삽 포함률 미확인 |
| — | Reuter & Olivetti de França, *Are you sure? … Survey of Uncertainty Quantification in Symbolic Regression*, 2026. [arXiv](https://arxiv.org/abs/2606.06567) | 앞부분 본문 | SR의 UQ가 덜 연구됐다는 서술. 확인 범위에 LLM 기반 SR UQ 없음 |

## C3. 구별되지 않는 식과 물리 지식 검토에서 추가한 자료 — 2026-11-03

상세 비교는 [별도 기록](reviews/2026-11-03-indistinguishable-candidates.md)에 있다.

| 자료 | 확인 수준 | 쟁점 |
|---|---|---|
| Kronberger et al., *Shape-constrained Symbolic Regression*, Evol. Comput. 2022. [arXiv](https://arxiv.org/abs/2103.15624) | 초록 | 형상 제약으로 외삽 개선 |
| Xiao et al., *Prior-Guided Symbolic Regression* (PG-SR), 2026 preprint. [HTML](https://arxiv.org/html/2602.13021) | 본문 §3–4 | LLM 추출·전문가 검토 제약 프로그램, OOD NMSE, 틀린 prior 1문제 |
| Yang et al., *Think like a Scientist* (KeplerAgent), 2026 preprint. [HTML](https://arxiv.org/html/2602.12259v1) | 본문 일부 | 대칭성 추론으로 SR 제약. 별도 외삽 분할 없음 |
| Tenachi et al., 단위 제약 deep SR, 2023. [arXiv](https://arxiv.org/abs/2303.03192) | 검색 요약 | 단위 제약만으로 복원 보장 안 됨 |
| Bartlett, Desmond, Ferreira, *Priors for symbolic regression*, 2023. [arXiv](https://arxiv.org/abs/2304.06333) | 초록 | 언어모델 기반 함수 prior와 Bayesian 모델 선택 |
| Box & Hill(1967) 등 경쟁 모델 판별 실험설계. [배경](https://link.springer.com/article/10.1007/s00180-023-01370-4) | 검색 요약 | 갈라지는 지점에서 측정하는 규칙은 고전적 |

## C4. 외삽 실패 원인 분해 검토에서 추가한 자료 — 2026-11-03

상세 비교는 [별도 기록](reviews/2026-11-03-failure-attribution.md)에 있다.

| 자료 | 확인 수준 | 쟁점 |
|---|---|---|
| Wang et al., *When Good Equations Get Bad Scores* (SAGE-Fit), 2026 preprint. [HTML](https://arxiv.org/html/2605.23272v2) | 본문 앞부분 | 피팅 불안정이 좋은 구조를 탈락시킴. OOD 분할·oracle 없음 |
| Xie et al., *Language models guide symbolic equation discovery by controlling search* (LLM-PySR), 2026. [arXiv](https://arxiv.org/abs/2607.04156) | 초록 | LLM 역할 비교 |
| Saveliev et al., *Influence-Guided Symbolic Regression* (IGSR), ICML 2026 표기. [arXiv](https://arxiv.org/abs/2605.29184) | 초록 | 항별 기여도 피드백 |
| *In-Context Symbolic Regression*, ACL SRW 2024. [원문](https://aclanthology.org/2024.acl-srw.49.pdf) | 검색 요약 | 기호 탐색과 피팅 분리 |
| *Towards Diverse Scientific Hypothesis Search with Large Language Models*, 2026. [HTML](https://arxiv.org/html/2606.10587) | 검색 결과 | 가설 다양성. 외삽에서 갈라지는 경쟁 가설 생성과는 다름 |

## C5. 외삽 실패 원인 분해 재검토에서 추가한 자료 — 2026-11-03

상세 비교는 [재검토 절](reviews/2026-11-03-failure-attribution.md)에 있다.

| 자료 | 확인 수준 | 쟁점 |
|---|---|---|
| Li et al., *From Score to Diagnosis* (PostDiag), 2026-10 preprint. [HTML](https://arxiv.org/html/2610.24535) | 본문 §3–4 | 정답 없는 bootstrap 편향·분산 진단과 상태별 LLM 수리. 관측 데이터만 사용 |
| Kabra et al., *SurfaceBench*, TMLR 표기. [arXiv](https://arxiv.org/abs/2511.10833) | 초록·검색 요약 | 탐색 실패 vs 피팅 실패 |
| Zhao et al., *A-SR*, 2026-08 preprint. [arXiv](https://arxiv.org/abs/2608.04872) | 초록 | 실패 흔적의 역할별 분배 |
| STRIDE, Deliberate Evolution, DrSR | 검색 요약 | 자기반성·진단형 SR 에이전트 |
| LLM-AutoSciLab, 2026 preprint. [arXiv](https://arxiv.org/abs/2605.24043) | 검색 요약 | 교과서식 단순 형태로의 수렴, 판별 실험 |
| Zhang et al., *Which Agent Causes Task Failures and When?*, ICML 2025. [PMLR](https://proceedings.mlr.press/v267/zhang25cq.html) | 초록 | LLM 에이전트 실패 원인 추적 |

## C6. 물리 행동 모듈 설계 검토에서 추가한 자료 — 2026-11-03

상세 비교는 [별도 기록](reviews/2026-11-03-behavior-modules.md)에 있다.

| 자료 | 확인 수준 | 쟁점 |
|---|---|---|
| von Rueden et al., *Informed Machine Learning*, IEEE TKDE 2023. [arXiv](https://arxiv.org/abs/1903.12394) | 검색 요약 | 지식 종류·통합 위치 분류 |
| von Kurnatowski et al., *Compensating data shortages in manufacturing with monotonicity knowledge*, 2021. [HTML](https://arxiv.org/html/2010.15955v2) | 본문 | 전문가 단조성 지식. 외삽 평가 없음 |
| Capstick et al., *AutoElicit*, ICML 2025. [PMLR](https://proceedings.mlr.press/v267/capstick25a.html) | 초록 | LLM prior 추출 |
| Zhu, *Incorporating LLM Priors into Tabular Learners*, 2023. [arXiv](https://arxiv.org/abs/2311.11628) | 검색 요약 | LLM 단조 방향 |
| Li, Fox, Goodman, *Automated Statistical Model Discovery with Language Models* (BoxLM), ICML 2024. [HTML](https://arxiv.org/html/2402.17879) | 본문 | LLM 모델 제안·비평, 커널 조립, ODE 외삽 구간 |
| *PiT-PO*, 2026. [arXiv](https://arxiv.org/abs/2602.10576) | 검색 요약 | 물리 타당성 제약을 LLM 강화학습에 사용 |

## D. 현재 후보의 상태

| 후보 | 제안 내용 | 가장 가까운 비교 축 | 현재 상태 |
|---|---|---|---|
| 조건부 물리 지식 전달 | LLM이 식·가정·적용 조건을 제공하고 신경망에 선택적으로 전달 | PDNN, KernelMix, adaptive PINN, GenCANN, LLM prior | 차별점 미확정. 넓은 방향은 유지하되 확정 추천으로 취급하지 않음 |
| 외삽에서 중요해지는 약한 항 보존 | 관측에서의 검출 가능성과 목표에서의 영향력을 분리하여 항 선택·신뢰 조절 | UQ-SINDy, E-SINDy, 모델 평균, prediction profile likelihood | 전달된 대화에서 추천 철회. 연구 가능성의 부정 판정이 아니라 추천 근거 미확보 상태로 보관 |
| 핵심 feature 누락 해결 | 관측되지 않는 필수 변수의 효과를 LLM으로 보완 | 잠재변수·식별가능성 접근, LLMPhy 등 | 사용자가 우선순위를 두지 않은 별도 문제 |
| 탐색 반복 간 불일치를 외삽 경보로 사용 | 같은 데이터로 독립 LLM 탐색을 반복하고 OOD 예측 불일치가 오차를 예측하는지 평가 | LLM-SR-UQ(단일 탐색 σ_S), ProAug, Kotanchek trust metric | 조건부 유지. 진단 연구로 기여 형태 한정. 파일럿 미실행 |
| 외삽 실패 원인 분해와 학습 구간 진단의 사각지대 | 외삽 실패를 원인별로 분해하고, 학습 구간 진단이 놓치는 실패를 외부 정보로 진단 | PostDiag, SurfaceBench, SAGE-Fit, A-SR | 정답 없는 진단·처방은 PostDiag와 겹쳐 범위 축소. 분석 미실행 |
| LLM-SR-UQ 재현 | 같은 예측에서 구간을 비교하고 반복 탐색 민감도 확인 | 원논문·저자 코드 | 별도 기존 작업. 새 연구 후보의 성능 검증으로 해석하지 않음 |

시뮬레이션 우선은 평가 경로다. 단순 함수 회귀는 선택적 진단이며, 후속 노트의 cubic 실험 제안을 사용자가 승인한 새 주제로 바꾸지 않는다.

## E. 정정과 남은 작업

- “잡음과 관측 부족을 함께 평가한 연구가 없다”, “불확실한 항을 여러 개 유지하면 새롭다”라는 주장은 사용하지 않는다.
- “LLM 물리 지식 + 신경망”, “조건별 gating”, “불확실성에 따른 가중치”, “residual 보정”만으로 독창성을 확보했다고 하지 않는다.
- 논문의 특정 실험을 확인하지 못한 것과 해당 문제가 연구 전체에서 미해결인 것은 다르다.
- 표본 수, 관측 시간, 상태 범위, 목표 범위, 시험 초기 궤적, target 정답 피드백을 따로 기록한다.
- 제공받은 논문 수치와 자체 실험 결과를 구분한다. 새 후보 실험은 아직 실행하지 않았다.
- 기존 문서의 “박사 연구로 진행할 가치”는 넓은 방향의 기획 의견이었다. 개별 주제의 차별점 검증 결과처럼 표현한 부분은 현재 문서에서 바로잡는다.
- 후보별로 가장 가까운 방법, 동일한 정보 제공 여부, 남는 차이, 차이를 가르는 실험을 확정해야 한다. 문헌 목록만으로 이 작업이 완료된 것으로 처리하지 않는다.
