"""
국면 배제(regime exclusion) 재검증: 확률 기울기 필터가 특정 연도 쏠림으로 생긴
우연인지 확인.

slope 필터 거래를 연도별로 분해해서 가장 기여도가 큰 해를 찾아 빼고도, 여전히
baseline(threshold만 보는 원래 규칙)을 이기는지 재계산한다. related-company
backtest에서 썼던 것과 같은 방식 (집중 연도는 slope 거래 기준으로 찾고, 그 해를
뺀 slope_net을 baseline_net(전체 기간)과 비교).

사전등록: docs/prereg_slope_filter_pilot.md의 판정 기준을 그대로 적용
  (4/5 이상 유지 여부로 통과/실패)

사용법 (레포 루트에서, run_slope_filter_pilot.py 통과 확인 후):
    python -m src.experiments.run_slope_filter_regime_exclusion
"""

import pandas as pd

from src.experiments.run_slope_filter_pilot import (
    TICKER_KRX, SEEDS,
    load_dataset, load_close, buy_and_hold_return,
    generate_trades, compounded_return,
)


def dominant_year(trades: pd.DataFrame):
    if trades.empty:
        return None, None
    by_year = trades.groupby(trades["exit_date"].dt.year)["net_return"].apply(
        lambda s: float((1 + s).prod() - 1)
    )
    year = int(by_year.idxmax())
    return year, float(by_year.loc[year])


def exclude_year(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    return trades[trades["exit_date"].dt.year != year]


def main():
    print(f"=== 확률 기울기 필터 국면 배제 재검증: {TICKER_KRX} ===\n")
    df = load_dataset()
    close = load_close(
        start=(df.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(df.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
    )

    rows = []
    for seed in SEEDS:
        baseline_trades = generate_trades(df, seed, mode="baseline")
        slope_trades = generate_trades(df, seed, mode="slope")

        baseline_net = compounded_return(baseline_trades)
        slope_net = compounded_return(slope_trades)

        bh_start = min(baseline_trades["entry_date"].min(), slope_trades["entry_date"].min())
        bh_end = max(baseline_trades["exit_date"].max(), slope_trades["exit_date"].max())
        bh_net = buy_and_hold_return(close, bh_start, bh_end)

        dom_year, dom_contrib = dominant_year(slope_trades)
        slope_trades_excl = exclude_year(slope_trades, dom_year) if dom_year else slope_trades
        slope_net_excl = compounded_return(slope_trades_excl)
        bh_net_excl = buy_and_hold_return(
            close,
            slope_trades_excl["entry_date"].min() if not slope_trades_excl.empty else bh_start,
            slope_trades_excl["exit_date"].max() if not slope_trades_excl.empty else bh_end,
        )

        rows.append({
            "seed": seed,
            "baseline_net": baseline_net,
            "slope_net": slope_net,
            "buy_and_hold_net": bh_net,
            "dominant_year": dom_year,
            "dominant_year_contrib": dom_contrib,
            "n_trades_excl": len(slope_trades_excl),
            "slope_net_excl": slope_net_excl,
            "excl_beats_baseline": slope_net_excl > baseline_net,
            "excl_beats_bh": slope_net_excl > bh_net_excl,
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    n_excl_beats_baseline = int(result_df["excl_beats_baseline"].sum())
    n_excl_beats_bh = int(result_df["excl_beats_bh"].sum())

    print(f"\n집중 연도 제외 후에도 baseline을 이긴 시드: {n_excl_beats_baseline}/{len(SEEDS)}")
    print(f"집중 연도 제외 후에도 Buy&Hold를 이긴 시드: {n_excl_beats_bh}/{len(SEEDS)}")

    years = result_df["dominant_year"].value_counts()
    print(f"\n집중 연도 분포: {dict(years)}")

    if n_excl_beats_baseline >= 4 and n_excl_beats_bh >= 4:
        print(
            "\n판정: [통과] -- 특정 연도 쏠림이 아니라 여러 구간에 걸쳐 우위가 유지됨.\n"
            "다음 단계: PBO 검증 (K_SLOPE=3, threshold=0.65, num_days=20 조합 자체가 "
            "탐색의 산물이 아닌지 확인)."
        )
    else:
        print(
            "\n판정: [실패] -- 집중 연도를 빼면 우위가 사라짐. run_slope_filter_pilot.py의 "
            "5/5 통과는 국면 우연이었을 가능성. production 규칙 반영하지 말 것."
        )


if __name__ == "__main__":
    main()