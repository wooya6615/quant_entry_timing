# 사전등록: 지속성 필터(persistence filter) 진입 규칙 pilot

- 레포(제안): `quant_entry_timing`
- 브랜치(제안): `experiment/persistence-filter-pilot`
- 작성일: 2026-09-18
- 배경: `run_probability_timeline.py` 진단에서 064350(threshold=0.65) 신호일 997일 중
  89일(8.9%)만 실제 진입, 908일(91%)이 블랙아웃으로 소실됨을 확인 (seed=42). 확률
  곡선이 threshold 위에서 짧게 스파이크쳤다가 바로 꺾이는 경우가 많아 보여서, "하루라도
  threshold를 넘으면 즉시 진입" 대신 "N일 연속으로 threshold를 넘어야 진입"하는 지속성
  필터가 노이즈성 스파이크를 걸러내는지 확인한다.

⚠️ 이건 새 feature나 새 모델이 아니라 **진입 규칙(엔트리 타이밍)의 변경**이다. 학습된
모델(BASE 13개, XGBoost)은 그대로 두고, "언제 진입하는가"만 바꾼다.

⚠️ **메타 레이블링과 같은 계열**이다 -- 1차 모델의 확률을 보고 2차 필터로 재판단한다는
점에서 대한제강/현대로템 meta-labeling(둘 다 [실패])과 같은 위험을 안고 있음을 인지하고
시작한다.

## 1. 가설 / 질문

064350의 triple-barrier 방향 예측(threshold=0.65)에서, "확률이 threshold를 N_PERSIST일
연속 넘었을 때만 진입"하는 규칙이, "하루라도 넘으면 즉시 진입"하는 기존 규칙보다
거래비용 반영 순수익이 나은가?

## 2. 고정할 파라미터 (사전등록 -- 결과 보고 바꾸지 않음)

| 파라미터 | 값 |
|---|---|
| 대상 종목 | 064350 |
| feature set | FEATURE_COLS_BASE (13개) -- RELATED 등 추가 feature 없음, 어제 실험과 독립 |
| 라벨 | `label_tb_binary` (pt_sl=(2,1), num_days=20) |
| threshold | 0.65 (production 채택값 그대로 유지 -- 이번엔 진입 타이밍만 바꿈) |
| **N_PERSIST (지속성 일수)** | **3일** -- 사전 결과를 보지 않고 고른 값. 근거: 1일(기존)은
  스파이크에 취약했고, 5일 이상은 num_days=20 보유기간의 15% 이상을 확인 대기에만
  쓰게 돼 신호 포착 자체가 너무 늦어질 위험 -- 3일을 "짧은 확인 구간"으로 원칙적 절충 |
| walk-forward | train_size=300, test_size=60, step=60, embargo=20 |
| seeds | 42, 1, 7, 123, 2024 |
| 진입 규칙 | 최근 N_PERSIST일 연속 `proba >= threshold`일 때, N번째 날에 진입.
  진입 후에는 기존과 동일하게 `holding_rows_tb`만큼 블랙아웃 |

## 3. 베이스라인 (참고용, 어제 데이터에서 이미 확인됨)

같은 파이프라인(pt2sl1_nd20_hl, threshold=0.65, N_PERSIST=1과 동일)에서 BASE 단독
net_return이 Buy & Hold를 이긴 시드는 **2/5뿐**(seed=7, seed=2024)이었음
(`run_related_company_backtest.py`의 `base_net`/`buy_and_hold_net` 컬럼). 즉 이번
pilot의 베이스라인 자체가 원래도 강하지 않음 -- 지속성 필터가 이걸 넘어서면 그 자체로
의미 있는 개선.

## 4. 통과 기준

- 5-seed 중 4개 이상에서 `persistence_net > baseline_net(N=1)` **그리고**
  `persistence_net > buy_and_hold_net`
- 거래 집중도 체크: 상위 5개 거래의 기여도가 전체 수익의 50%를 넘으면 "거래 집중"
  경고로 별도 기록 (118990/threshold=0.01 실험에서 확인된 실패 유형 재발 방지)
- 거래 수가 30건 미만인 시드는 해석에 주의 (표본 과소 가능성 별도 기록, 자동 실패는 아님)

## 5. 이 pilot에서 하지 않는 것

- N_PERSIST를 여러 값으로 스윕해서 제일 좋은 걸 고르지 않음 (그 자체가 pt_sl 때
  PBO 95.6%를 만들었던 사후 탐색 패턴이므로) -- 3일 하나만 사전등록하고 그 결과를 받아들임
- threshold 자체를 재조정하지 않음

## 6. 판정 후 처리

- 통과 -> 지속성 필터를 production 후보 규칙에 반영 검토 (단, 이것도 새 규칙이므로
  PBO 재검증 필요)
- 실패 -> "이번엔 통과 못했다"로 기록. N_PERSIST를 바꿔서 재시도하고 싶으면 이 결과를
  본 뒤가 아니라 별도 addendum으로 새 이유(사전 원칙)를 남기고 진행할 것