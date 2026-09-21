# 사전등록: 확률 기울기 필터(slope filter) 진입 규칙 pilot

- 레포: `quant_entry_timing` (persistence filter pilot과 같은 레포, 같은 실험 라인)
- 브랜치(제안): `experiment/slope-filter-pilot`
- 작성일: 2026-09-18
- 배경: 지속성 필터(N_PERSIST=3) pilot이 0/5로 실패하고, 오히려 baseline보다 크게
  악화됨. "짧은 스파이크가 노이즈"라는 가설이 틀렸을 가능성이 제기됨 -- 그렇다면
  "얼마나 오래 버텼나(지속성)" 대신 "지금 오르는 중인가(기울기)"를 보는 게 다른
  결과를 낼 수 있는지 확인한다. 지속성 필터와 달리 진입을 늦추지 않는다는 점이 핵심
  차이 -- threshold를 넘은 바로 그날 진입하되, 그 확률이 상승 추세일 때만 진입한다.

⚠️ 지속성 필터와 마찬가지로 메타 레이블링 계열이며, 모델 자체는 바꾸지 않고 진입
규칙만 바꾼다.

## 1. 가설 / 질문

threshold를 넘은 날 중에서도, 최근 K_SLOPE일 전보다 확률이 더 높아진(상승 중인) 날만
진입하는 규칙이, 기존 규칙(threshold만 보고 즉시 진입)보다 거래비용 반영 순수익이
나은가?

## 2. 고정할 파라미터 (사전등록 -- 결과 보고 바꾸지 않음)

| 파라미터 | 값 |
|---|---|
| 대상 종목 / feature / 라벨 / threshold | persistence filter pilot과 동일 (BASE 13개,
  pt_sl=(2,1)/num_days=20, threshold=0.65) |
| **K_SLOPE** | **3일** -- persistence filter의 N_PERSIST=3과 같은 "짧은 확인 구간"
  원칙을 그대로 유지해서 두 실험이 비교 가능하게 함 (사후에 맞춘 값 아님) |
| 진입 조건 | `proba[i] >= threshold` **그리고** `proba[i] > proba[i - K_SLOPE]`
  (지속성 필터처럼 여러 날 연속 조건이 아니라, 오늘 하루만 판정 -- 진입 자체는
  지연시키지 않음) |
| walk-forward / seeds | persistence filter pilot과 동일 |

## 3. 베이스라인

persistence filter pilot과 동일 (baseline net, N=1) -- 같은 스크립트에서 나란히 재계산.

## 4. 통과 기준

- 5-seed 중 4개 이상에서 `slope_net > baseline_net` **그리고** `slope_net > buy_and_hold_net`
- 거래 집중도(상위 5건 기여도) 50% 초과 시 경고 기록
- 거래 수 30건 미만 시 경고 기록

## 5. 이 pilot에서 하지 않는 것

- K_SLOPE 스윕하지 않음 -- 3일 하나만 사전등록
- persistence filter와 slope filter를 섞은 조합(둘 다 만족해야 진입 등)은 이번엔
  시도하지 않음 -- 두 필터 각각의 개별 효과부터 독립적으로 확인

## 6. 판정 후 처리

- 통과 -> production 후보 규칙 반영 검토 (PBO 재검증 필요)
- 실패 -> 이번 세션에서 시도한 진입 타이밍 필터 3종(threshold만/지속성/기울기) 전부
  실패로 기록하고, 진입 규칙 개선 라인 종료. 원래 모델(BASE, threshold=0.65 즉시진입)을
  그대로 유지