"""D5：和美國資產規模前三大 ETF（VOO、IVV、SPY，2026 年 Lipper／FinanceCharts 排名）比。
老師的判定：夏普值＝年化報酬 ÷ 年化波動度（不扣無風險利率），比他們都高才算贏；報酬相同時，風險小的贏。
另附學界標準的 Sharpe（扣國庫券利率）與區塊自助法 p 值。策略用「即時①」時點（只用前一天資料、今天收盤成交）、單邊成本 0.05%。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common_d import RES, DATA, TDAYS, LAST
from realtime import ohlc, rf_series, run_close, vt_weight

ETFS = ["VOO", "IVV", "SPY", "VTI", "QQQ"]   # 前三大＋第 4、5 大當參考


def fetch():
    import yfinance as yf
    for t in ["VOO", "IVV", "VTI"]:
        df = yf.download(t, start="1993-01-01", end="2026-10-20", auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df[["Open", "High", "Low", "Close"]].rename_axis("date").to_csv(DATA / f"rt_{t}.csv")


def ratio(x):
    """老師的夏普值：年化報酬（幾何）÷ 年化波動度。"""
    yrs = len(x) / TDAYS
    cagr = (1 + x).prod() ** (1 / yrs) - 1
    vol = x.std() * math.sqrt(TDAYS)
    return cagr / vol, cagr, vol


def boot_ratio(a, b, reps=2000, block=21, seed=0):
    """策略與 ETF 的「年化報酬÷年化波動度」差，循環區塊自助法單尾 p 值（H1：策略較高）。"""
    a, b = a.values, b.values
    T = len(a)
    obs = ratio(pd.Series(a))[0] - ratio(pd.Series(b))[0]
    rng = np.random.default_rng(seed)
    nb = math.ceil(T / block)
    d = np.empty(reps)
    for k in range(reps):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:T] % T
        aa, bb = a[idx], b[idx]
        f = lambda z: ((np.prod(1 + z) ** (TDAYS / T) - 1) / (z.std(ddof=1) * math.sqrt(TDAYS)))
        d[k] = f(aa) - f(bb)
    return float(obs), float(np.mean(d - obs >= obs))


def main(extra=None):
    if not (DATA / "rt_VOO.csv").exists():
        fetch()
    rets = {t: ohlc(t)["Close"].pct_change() for t in ETFS}
    idx = rets["QQQ"].index
    rf = rf_series(idx)
    S = {}
    for base in ("QQQ", "SPY"):
        r = rets[base]
        for cap, lab in ((1.0, "不加槓桿"), (1.5, "上限 1.5 倍")):
            w = vt_weight(r, cap=cap).shift(2).fillna(0.0)
            S[f"{base} 波動度目標（{lab}）"] = run_close(w, r, rf)[0]
    if extra:
        S.update(extra(rets, rf))
    periods = {"VOO 上市後 2010-09～2026-10（五檔 ETF 共同期間）": ("2010-09-10", LAST),
               "2008-01～2026-10（D4 量子預測有資料的期間）": ("2008-01-02", LAST),
               "IVV 上市後 2000-06～2026-10": ("2000-06-01", LAST)}
    rows = []
    for per, (a0, a1) in periods.items():
        cand = {**{t: rets[t] for t in ETFS}, **S}
        for name, x in cand.items():
            x = x.loc[a0:a1].dropna()
            if x.index[0] > pd.Timestamp(a0) + pd.Timedelta(days=10):
                continue  # 這段期間還沒上市
            rr, cagr, vol = ratio(x)
            ex = x - rf.reindex(x.index)
            eq = (1 + x).cumprod()
            rows.append(dict(period=per, name=name, kind="ETF" if name in ETFS else "策略", CAGR=cagr, Vol=vol,
                             teacher_sharpe=rr, std_sharpe=float(ex.mean() / ex.std() * math.sqrt(TDAYS)),
                             MaxDD=float((eq / eq.cummax() - 1).min()), start=x.index[0].date()))
    res = pd.DataFrame(rows)
    # 每個策略對每檔前三大 ETF 的自助法 p 值
    tests = []
    for per, (a0, a1) in periods.items():
        for name, x in S.items():
            for t in ["VOO", "IVV", "SPY"]:
                xa, xb = x.loc[a0:a1].dropna(), rets[t].loc[a0:a1].dropna()
                common = xa.index.intersection(xb.index)
                late = pd.Timestamp(a0) + pd.Timedelta(days=10)
                if len(common) < 500 or xb.index[0] > late or xa.index[0] > late:
                    continue
                d, p = boot_ratio(xa.loc[common], xb.loc[common])
                tests.append(dict(period=per, strategy=name, vs=t, diff=d, p=p))
    tests = pd.DataFrame(tests)
    res.to_csv(RES / "d5_top_etf.csv", index=False); tests.to_csv(RES / "d5_top_etf_tests.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    for per in periods:
        print(per); print(res[res.period == per].sort_values("teacher_sharpe", ascending=False).drop(columns="period").round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    return res, tests


if __name__ == "__main__":
    main()
