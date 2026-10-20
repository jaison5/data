"""準確度門檻模擬往前延伸：QQQ（1999-03 起）與 Nasdaq-100 指數（1985-10 起，加 1%／年股利），分期間看
「每天方向要猜對多少才贏買進持有」。和 feasibility.accuracy_sim 相同的模擬（每個準確率 300 次、0.05% 成本、
錯在哪一天和漲跌幅大小無關）。輸出 results/threshold_long.csv 與 results/threshold_long_breakeven.csv。"""
import numpy as np
import pandas as pd

from common_d import RES, TDAYS, LAST
from feasibility import accuracy_sim
from realtime import ohlc, rf_series

P_GRID = [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65]


def frame(asset, div=0.0):
    c = ohlc(asset)["Close"]
    r = c.pct_change() + div / TDAYS
    return pd.DataFrame({asset: r, "RF": rf_series(c.index)}).dropna()


def main():
    cases = [
        ("NDX", 0.01, {"1986-01~1998-12": ("1986-01-02", "1998-12-31"), "1999-03~2002-12（泡沫與崩盤）": ("1999-03-10", "2002-12-31"),
                       "2003-01~2026-10": ("2003-01-02", LAST), "1986-01~2026-10（全部）": ("1986-01-02", LAST)}),
        ("QQQ", 0.0, {"1999-03~2002-12（泡沫與崩盤）": ("1999-03-11", "2002-12-31"), "2003-01~2026-10": ("2003-01-02", LAST),
                      "1999-03~2026-10（全部）": ("1999-03-11", LAST)}),
    ]
    rows = []
    for asset, div, periods in cases:
        ret = frame(asset, div)
        for per, (a, b) in periods.items():
            s = accuracy_sim(ret, asset, P_GRID, draws=300, seed=0, period=(a, b))
            s["period"] = per
            rows.append(s)
    res = pd.concat(rows)
    res.to_csv(RES / "threshold_long.csv", index=False)
    be = []
    for (asset, per, strat), g in res.groupby(["asset", "period", "strategy"], sort=False):
        g = g.sort_values("p")
        hit_c = g[g.CAGR_med > g.BH_CAGR]; hit_s = g[g.Sharpe_med > g.BH_Sharpe]
        be.append(dict(asset=asset, period=per, strategy=strat, BH_CAGR=g.BH_CAGR.iloc[0], BH_Sharpe=g.BH_Sharpe.iloc[0],
                       breakeven_CAGR=None if hit_c.empty else float(hit_c.p.min()),
                       breakeven_Sharpe=None if hit_s.empty else float(hit_s.p.min()),
                       CAGR_at_50=float(g[g.p == 0.50].CAGR_med.iloc[0]), CAGR_at_52=float(g[g.p == 0.52].CAGR_med.iloc[0])))
    be = pd.DataFrame(be)
    be.to_csv(RES / "threshold_long_breakeven.csv", index=False)
    pd.set_option("display.width", 220)
    print(be.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
