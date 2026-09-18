"""
Pilot: 지속성 필터(persistence filter) 진입 규칙 -- N_PERSIST일 연속 threshold 초과 시 진입.

기존 규칙(하루라도 threshold 넘으면 즉시 진입)이 신호일의 91%를 블랙아웃으로 날리는
현상(run_probability_timeline.py 진단)을 확인한 뒤, "짧은 스파이크"와 "지속되는 신호"를
구분하기 위한 진입 규칙 변경. 모델 자체(BASE 13개, XGBoost)는 그대로 두고 진입 타이밍만
바꾼다.

사전등록: docs/prereg_persistence_filter_pilot.md 참고 (N_PERSIST=3, 결과 보고 바꾸지 않음)

사용법 (레포 루트에서):
    python -m src.experiments.run_persistence_filter_pilot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TICKER_KRX = "064350"
CONFIG_LABEL = "pt2sl1_nd20_hl"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20
SEEDS = [42, 1, 7, 123, 2024]
THRESHOLD = 0.65
N_PERSIST = 2  # 사전등록 값 -- 결과 보고 바꾸지 않음
ROUND_TRIP_COST = 0.002
TRADE_CONCENTRATION_LIMIT = 0.50
MIN_TRADES_WARN = 30

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
)


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_base.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음.")
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def load_close(start: str, end: str) -> pd.Series:
    px = yf.download(f"{TICKER_KRX}.KS", start=start, end=end, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    return px["Close"]


def buy_and_hold_return(close: pd.Series, start_date, end_date) -> float:
    window = close.loc[start_date:end_date]
    if len(window) < 2:
        return float("nan")
    return float(window.iloc[-1] / window.iloc[0] - 1)


def walk_forward_splits(n_rows, train_size, test_size, step, embargo):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = list(range(start, start + train_size))
        test_start = start + train_size + embargo
        test_idx = list(range(test_start, test_start + test_size))
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 진입 규칙: persist=1이면 기존과 동일, persist=N이면 N일 연속 넘어야 진입
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, seed: int, persist: int) -> pd.DataFrame:
    X, y = df[FEATURE_COLS_BASE], df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        i = 0
        while i < len(test_idx):
            window_ok = (
                i - persist + 1 >= 0
                and all(proba[i - persist + 1: i + 1] >= THRESHOLD)
            )
            if window_ok:
                row_idx = test_idx[i]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx]) if pd.notna(
                    df["holding_rows_tb"].iloc[row_idx]) else 1
                holding = max(holding, 1)
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": df.index[row_idx],
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += holding
            else:
                i += 1
    return pd.DataFrame(trades)


def compounded_return(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((1 + trades["net_return"]).prod() - 1)


def top5_concentration(trades: pd.DataFrame) -> float:
    if trades.empty or len(trades) < 5:
        return float("nan")
    total = trades["net_return"].abs().sum()
    if total == 0:
        return 0.0
    top5 = trades["net_return"].abs().nlargest(5).sum()
    return float(top5 / total)


def main():
    print(f"=== 지속성 필터 pilot: {TICKER_KRX}, N_PERSIST={N_PERSIST} ===\n")
    df = load_dataset()
    close = load_close(
        start=(df.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(df.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
    )

    rows = []
    for seed in SEEDS:
        baseline_trades = generate_trades(df, seed, persist=1)
        persist_trades = generate_trades(df, seed, persist=N_PERSIST)

        baseline_net = compounded_return(baseline_trades)
        persist_net = compounded_return(persist_trades)

        bh_start = min(baseline_trades["entry_date"].min(), persist_trades["entry_date"].min())
        bh_end = max(baseline_trades["exit_date"].max(), persist_trades["exit_date"].max())
        bh_net = buy_and_hold_return(close, bh_start, bh_end)

        rows.append({
            "seed": seed,
            "n_trades_baseline": len(baseline_trades),
            "n_trades_persist": len(persist_trades),
            "baseline_net": baseline_net,
            "persist_net": persist_net,
            "buy_and_hold_net": bh_net,
            "persist_beats_baseline": persist_net > baseline_net,
            "persist_beats_bh": persist_net > bh_net,
            "top5_concentration": top5_concentration(persist_trades),
            "low_trade_warning": len(persist_trades) < MIN_TRADES_WARN,
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    n_beats_baseline = int(result_df["persist_beats_baseline"].sum())
    n_beats_bh = int(result_df["persist_beats_bh"].sum())
    n_concentrated = int((result_df["top5_concentration"] > TRADE_CONCENTRATION_LIMIT).sum())
    n_low_trades = int(result_df["low_trade_warning"].sum())

    print(f"\n지속성 필터가 baseline(N=1)을 이긴 시드: {n_beats_baseline}/{len(SEEDS)}")
    print(f"지속성 필터가 Buy&Hold를 이긴 시드: {n_beats_bh}/{len(SEEDS)}")
    print(f"거래 집중도 50% 초과 시드: {n_concentrated}/{len(SEEDS)}")
    print(f"거래 수 30건 미만 경고 시드: {n_low_trades}/{len(SEEDS)}")

    if n_beats_baseline >= 4 and n_beats_bh >= 4:
        verdict = "[통과]"
        detail = "지속성 필터가 baseline과 Buy&Hold 둘 다 4/5 이상 이김."
    else:
        verdict = "[실패]"
        detail = "사전등록 기준(4/5) 미달."

    print(f"\n판정: {verdict} -- {detail}")
    if n_concentrated > 0:
        print("⚠️ 일부 시드에서 거래 집중도 50% 초과 -- 118990/threshold=0.01 때와 같은 "
              "'거래 집중' 실패 유형 가능성, 통과 판정이어도 참고할 것")
    if n_low_trades > 0:
        print("⚠️ 일부 시드에서 거래 수 30건 미만 -- 표본 과소로 결과가 불안정할 수 있음")


if __name__ == "__main__":
    main()