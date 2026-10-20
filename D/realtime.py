"""D3：老師的兩個問題。
  1. 即時（real time）進出也一樣有效嗎？—— 只用下單前已知的資料、改用不同成交時點、用真的可交易 ETF。
  2. 拉長到 30 年以上也一樣嗎？—— S&P 500 從 1928 年、Nasdaq-100 從 1985 年、SPY 從 1993 年，
     並看「每一個 30 年視窗」是不是都贏。
資料：data/rt_*.csv（yfinance，auto_adjust）。輸出 results/d3_*.csv。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common_d import RES, DATA, TDAYS, LAST

COST = 0.0005
SPREAD = 0.005


def ohlc(t):
    df = pd.read_csv(DATA / f"rt_{t}.csv", parse_dates=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated()].loc[:LAST]


def rf_series(index, pre1960=0.0):
    irx = ohlc("IRX")["Close"].reindex(index).ffill()
    rf = irx / 100 / TDAYS
    return rf.fillna(pre1960 / TDAYS)


def vt_weight(close_ret, target=0.20, look=20, cap=1.0):
    sig = close_ret.rolling(look).std() * math.sqrt(TDAYS)
    return (target / sig).clip(upper=cap)


def stats(r, rf):
    r = r.dropna()
    rf = rf.reindex(r.index)
    eq = (1 + r).cumprod()
    yrs = len(r) / TDAYS
    ex = r - rf
    dd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, Vol=r.std() * math.sqrt(TDAYS),
                Sharpe=ex.mean() / ex.std() * math.sqrt(TDAYS), MaxDD=dd, years=yrs)


# ------------------------------------------------------------ 1. 即時進出


def run_close(w_held, r, rf, cost=COST):
    """w_held[t] = 第 t 天（前一天收盤到第 t 天收盤）持有的比例。換倉成本在開始持有那天付。"""
    turn = w_held.diff().abs().fillna(w_held.abs())
    cash = 1 - w_held
    return w_held * r + cash * rf - np.maximum(-cash, 0) * SPREAD / TDAYS - cost * turn, turn


def run_open(w_new, df, rf, cost=COST):
    """第 t 天開盤換成 w_new[t]：隔夜（前收→今開）用舊比例，盤中（今開→今收）用新比例。"""
    on = df["Open"] / df["Close"].shift(1) - 1
    idr = df["Close"] / df["Open"] - 1
    w_old = w_new.shift(1).fillna(0.0)
    avg = (w_old + w_new) / 2
    gross = (1 + w_old * on) * (1 + w_new * idr) - 1
    turn = (w_new - w_old).abs()
    return gross + (1 - avg) * rf - np.maximum(avg - 1, 0) * SPREAD / TDAYS - cost * turn, turn


def band(target, width):
    """只有目標和目前持有差超過 width 才換倉。"""
    out, cur = [], np.nan
    for x in target.values:
        if np.isnan(x):
            out.append(np.nan); continue
        if np.isnan(cur) or abs(x - cur) > width:
            cur = x
        out.append(cur)
    return pd.Series(out, index=target.index)


def realtime_tests():
    rows = []
    for t in ("QQQ", "SPY"):
        df = ohlc(t)
        r = df["Close"].pct_change()
        rf = rf_series(df.index)
        bh = r
        for cap in (1.0, 1.5):
            w = vt_weight(r, cap=cap)                       # 用到第 t 天收盤（含當天）
            fri = pd.Series(df.index.dayofweek == 4, index=df.index)
            weekly = w.where(fri).ffill()
            V = {
                "原本：當天收盤算、當天收盤成交": (run_close, w.shift(1)),
                "即時①：只用到前一天的資料，今天收盤成交": (run_close, w.shift(2)),
                "即時②：昨天收盤算好，今天開盤成交": (run_open, w.shift(1)),
                "即時③：昨天收盤算好，晚一天（明天開盤）成交": (run_open, w.shift(2)),
                "即時④：同②，但差距超過 10% 才換倉": (run_open, band(w, 0.10).shift(1)),
                "即時⑤：每週五收盤算好，下週一開盤成交": (run_open, weekly.shift(1)),
            }
            for name, (fn, wh) in V.items():
                wh = wh.fillna(0.0)
                for cost in (0.0005, 0.001):
                    x, turn = (fn(wh, r, rf, cost) if fn is run_close else fn(wh, df, rf, cost))
                    for per, (p0, p1) in {"2000-04~2026-10": ("2000-04-03", LAST), "2008-01~2026-10": ("2008-01-02", LAST)}.items():
                        s, b = stats(x.loc[p0:p1], rf), stats(bh.loc[p0:p1], rf)
                        rows.append(dict(asset=t, cap=cap, variant=name, cost=cost, period=per, CAGR=s["CAGR"], Sharpe=s["Sharpe"],
                                         MaxDD=s["MaxDD"], BH_CAGR=b["CAGR"], BH_Sharpe=b["Sharpe"], BH_MaxDD=b["MaxDD"],
                                         trades_per_yr=float((turn.loc[p0:p1] > 1e-9).sum() / s["years"]),
                                         turnover_per_yr=float(turn.loc[p0:p1].sum() / s["years"])))
    return pd.DataFrame(rows)


def real_etf_test():
    """用真的 ETF 執行：不加槓桿＝QQQ＋BIL；1.5 倍＝QQQ＋QLD（2 倍 ETF，含其管理費與每日重設損耗）＋BIL。
    即時①：只用到前一天的資料，今天收盤成交。期間＝BIL 上市後 2007-06～2026-10。"""
    q, l, b = (ohlc(t)["Close"].pct_change() for t in ("QQQ", "QLD", "BIL"))
    idx = q.index.intersection(l.dropna().index).intersection(b.dropna().index)
    q, l, b = q.reindex(idx), l.reindex(idx), b.reindex(idx)
    rf = rf_series(idx)
    rows = []
    for cap in (1.0, 1.5):
        w = vt_weight(ohlc("QQQ")["Close"].pct_change(), cap=cap).reindex(idx).shift(2).fillna(0.0)
        wq = np.where(w <= 1, w, 2 - w)
        wl = np.where(w <= 1, 0.0, w - 1)
        wb = 1 - wq - wl
        W = pd.DataFrame({"q": wq, "l": wl, "b": wb}, index=idx)
        turn = W.diff().abs().sum(axis=1).fillna(0)
        x = W.q * q + W.l * l + W.b * b - COST * turn
        s, bb = stats(x.loc["2007-06-01":], rf), stats(q.loc["2007-06-01":], rf)
        rows.append(dict(implementation=f"QQQ＋{'QLD＋' if cap > 1 else ''}BIL（上限 {cap} 倍）", CAGR=s["CAGR"], Sharpe=s["Sharpe"],
                         MaxDD=s["MaxDD"], BH_CAGR=bb["CAGR"], BH_Sharpe=bb["Sharpe"], BH_MaxDD=bb["MaxDD"], years=s["years"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 2. 30 年以上


def long_history(t, div_yield=0.0, pre1960=0.0, cap=1.0, start=None):
    """即時①（只用前一天資料、收盤成交）。價格指數沒有股利：div_yield 給一個固定年股利率，依持有比例計入，兩邊公平。"""
    df = ohlc(t)
    if start:
        df = df.loc[start:]
    r = df["Close"].pct_change() + div_yield / TDAYS
    rf = rf_series(df.index, pre1960)
    w = vt_weight(df["Close"].pct_change(), cap=cap).shift(2).fillna(0.0)
    x, turn = run_close(w, r, rf)
    first = w[w > 0].index[0]
    return x.loc[first:], r.loc[first:], rf.loc[first:], turn.loc[first:]


def rolling_windows(x, b, rf, years=30):
    """每個月底起算的 30 年視窗：策略的年化報酬、Sharpe、最大回撤是否勝過買進持有。"""
    starts = x.index.to_series().groupby(x.index.to_period("M")).first()
    lx, lb = np.log1p(x).cumsum(), np.log1p(b).cumsum()
    out = []
    for s in starts:
        e = s + pd.DateOffset(years=years)
        if e > x.index[-1]:
            break
        xs, bs, fs = x.loc[s:e], b.loc[s:e], rf.loc[s:e]
        n = len(xs) / TDAYS
        cx, cb = math.exp(lx.loc[s:e].iloc[-1] - lx.loc[s:e].iloc[0]) ** (1 / n) - 1, math.exp(lb.loc[s:e].iloc[-1] - lb.loc[s:e].iloc[0]) ** (1 / n) - 1
        sh = lambda z: (z - fs).mean() / (z - fs).std() * math.sqrt(TDAYS)
        dd = lambda z: float(((1 + z).cumprod() / (1 + z).cumprod().cummax() - 1).min())
        out.append(dict(start=s.date(), CAGR=cx, BH_CAGR=cb, Sharpe=sh(xs), BH_Sharpe=sh(bs), MaxDD=dd(xs), BH_MaxDD=dd(bs)))
    return pd.DataFrame(out)


def long_tests():
    cases = [("GSPC", "S&P 500 指數（不含股利）", 0.0), ("GSPC", "S&P 500 指數＋股利 3%／年", 0.03),
             ("SP500TR", "S&P 500 總報酬指數", 0.0), ("SPY", "SPY（含股利）", 0.0),
             ("NDX", "Nasdaq-100 指數（不含股利）", 0.0), ("NDX", "Nasdaq-100 指數＋股利 1%／年", 0.01), ("QQQ", "QQQ（含股利）", 0.0)]
    full, roll = [], []
    for t, label, dy in cases:
        for cap in (1.0, 1.5):
            for pre in ((0.0, 0.03) if t == "GSPC" else (0.0,)):
                x, b, rf, turn = long_history(t, dy, pre, cap)
                s, sb = stats(x, rf), stats(b, rf)
                full.append(dict(series=label, cap=cap, rf_before_1960=pre, start=x.index[0].date(), end=x.index[-1].date(),
                                 years=s["years"], CAGR=s["CAGR"], BH_CAGR=sb["CAGR"], Sharpe=s["Sharpe"], BH_Sharpe=sb["Sharpe"],
                                 MaxDD=s["MaxDD"], BH_MaxDD=sb["MaxDD"], turnover_per_yr=float(turn.sum() / s["years"])))
                if pre == 0.0 and s["years"] >= 30.5:
                    w = rolling_windows(x, b, rf)
                    roll.append(dict(series=label, cap=cap, n_windows=len(w), first_start=w.start.iloc[0], last_start=w.start.iloc[-1],
                                     beat_CAGR=float((w.CAGR > w.BH_CAGR).mean()), beat_Sharpe=float((w.Sharpe > w.BH_Sharpe).mean()),
                                     beat_MaxDD=float((w.MaxDD > w.BH_MaxDD).mean()),
                                     dCAGR_min=float((w.CAGR - w.BH_CAGR).min()), dCAGR_med=float((w.CAGR - w.BH_CAGR).median()),
                                     dCAGR_max=float((w.CAGR - w.BH_CAGR).max()),
                                     dSharpe_min=float((w.Sharpe - w.BH_Sharpe).min()), dSharpe_med=float((w.Sharpe - w.BH_Sharpe).median())))
                    w.assign(series=label, cap=cap).to_csv(RES / f"d3_roll30_{t}_{int(dy*100)}_{cap}.csv", index=False)
    return pd.DataFrame(full), pd.DataFrame(roll)


def main():
    RES.mkdir(exist_ok=True)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
    rt = realtime_tests(); rt.to_csv(RES / "d3_realtime.csv", index=False)
    print(rt[rt.cost == 0.0005].round(3).to_string())
    etf = real_etf_test(); etf.to_csv(RES / "d3_real_etf.csv", index=False)
    print(etf.round(3).to_string())
    full, roll = long_tests()
    full.to_csv(RES / "d3_long_full.csv", index=False); roll.to_csv(RES / "d3_long_roll30.csv", index=False)
    print(full.round(3).to_string()); print(roll.round(3).to_string())


if __name__ == "__main__":
    main()
