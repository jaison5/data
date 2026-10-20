"""D2：候選策略是否「可以採用」的穩健性檢查（不用預測器的規則）。

對每個候選策略檢查：
  1. 兩段樣本外期間：2000-04～2007-12（含網路泡沫破裂）與 2008-01～2026-09
  2. 每個日曆年、每個滾動 3 年視窗，和買進持有同一標的比
  3. 參數敏感度（波動度目標的目標值、回看天數、上限；均線天數）
  4. 交易成本 0.01%～0.2%、晚一天成交
  5. Sharpe 差異檢定（Memmel–JK、區塊自助法）與 deflated Sharpe（把所有試過的變體都算進去）
輸出 results/d2_*.csv 與 results/summary_d2.md。
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd

from common_d import RES, TDAYS, LAST, load, backtest, stats
from report_d1 import sharpe_test, boot_sharpe, deflated_sharpe

PERIODS = {"2000-04~2007-12": ("2000-04-03", "2007-12-31"), "2008-01~2026-10": ("2008-01-02", LAST),
           "2000-04~2026-10": ("2000-04-03", LAST)}
MAIN = "2008-01~2026-10"


def vol_target(ret, a, target=0.20, look=20, cap=1.5):
    rv = ret[a].rolling(look).std() * math.sqrt(TDAYS)
    return pd.DataFrame({a: (target / rv).clip(upper=cap)})


def ma_rule(px, a, n=200, lev=1.0, alt=None):
    above = (px[a] > px[a].rolling(n).mean()).astype(float)
    w = {a: lev * above}
    if alt:
        w[alt] = 1 - above
    return pd.DataFrame(w)


def ma_vt(px, ret, a, n=200, target=0.20, look=20, cap=1.5):
    return vol_target(ret, a, target, look, cap).mul(ma_rule(px, a, n)[a], axis=0)


def dual_mom(px, ret, look=252):
    m_end = (px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()).values
    mom = px[["SPY", "QQQ", "IEF"]].pct_change(look)
    rf12 = ret["RF"].rolling(look).sum()
    dm = pd.DataFrame(np.nan, index=px.index, columns=["SPY", "QQQ", "IEF"])
    for t in px.index[m_end]:
        if pd.isna(mom.loc[t]).any() or pd.isna(rf12.loc[t]):
            continue
        best = "QQQ" if mom.loc[t, "QQQ"] >= mom.loc[t, "SPY"] else "SPY"
        dm.loc[t] = 0.0
        dm.loc[t, best if mom.loc[t, best] > rf12.loc[t] else "IEF"] = 1.0
    return dm.ffill().fillna(0.0)


def candidates(px, ret):
    """預設參數（文獻常用值，先固定）。key：(名稱, 比較的標的)"""
    C = {}
    for a in ("QQQ", "SPY"):
        C[(f"{a} 波動度目標 20%（上限 1.5 倍）", a)] = vol_target(ret, a)
        C[(f"{a} 波動度目標 20%（不加槓桿）", a)] = vol_target(ret, a, cap=1.0)
        C[(f"{a} 200 日均線→現金", a)] = ma_rule(px, a)
        C[(f"2 倍 {a}＋200 日均線", a)] = ma_rule(px, a, lev=2.0)
        C[(f"{a} 200 日均線＋波動度目標", a)] = ma_vt(px, ret, a)
    C[("雙動能 SPY／QQQ／IEF（每月）", "QQQ")] = dual_mom(px, ret)
    return C


def variants(px, ret):
    """參數敏感度：每個候選策略家族的變體（全部計入 deflated Sharpe 的試驗次數）。"""
    V = []
    for a in ("QQQ", "SPY"):
        for tg, lk, cp in itertools.product([0.15, 0.20, 0.25], [10, 20, 60], [1.0, 1.5, 2.0]):
            V.append((f"{a} 波動度目標", a, f"目標 {tg:.0%}／回看 {lk}／上限 {cp}", vol_target(ret, a, tg, lk, cp)))
        for n in (100, 150, 200, 250):
            V.append((f"{a} 均線→現金", a, f"{n} 日", ma_rule(px, a, n)))
            V.append((f"2 倍 {a}＋均線", a, f"{n} 日", ma_rule(px, a, n, lev=2.0)))
            for tg in (0.15, 0.20, 0.25):
                V.append((f"{a} 均線＋波動度目標", a, f"{n} 日／目標 {tg:.0%}", ma_vt(px, ret, a, n, tg)))
    for lk in (126, 189, 252):
        V.append(("雙動能", "QQQ", f"回看 {lk} 天", dual_mom(px, ret, lk)))
    return V


def rolling_wins(r, b, rf, years=3):
    """每月月底開始的滾動 3 年視窗：策略 Sharpe 高於買進持有的比例、報酬高於的比例。"""
    ends = r.index.to_series().groupby(r.index.to_period("M")).last()
    win_sh, win_ret = [], []
    for e in ends:
        s = e - pd.DateOffset(years=years)
        if s < r.index[0]:
            continue
        x, y, f = r.loc[s:e], b.loc[s:e], rf.loc[s:e]
        sh = lambda z: (z - f).mean() / (z - f).std()
        win_sh.append(sh(x) > sh(y))
        win_ret.append((1 + x).prod() > (1 + y).prod())
    return float(np.mean(win_sh)), float(np.mean(win_ret)), len(win_sh)


def main():
    px, ret, _ = load(start="1999-03-10")
    rf = ret["RF"]
    bh = {a: backtest(pd.DataFrame({a: pd.Series(1.0, index=px.index)}), ret)[0] for a in ("QQQ", "SPY")}
    rows, years, cost_rows = [], [], []
    for (name, a), w in candidates(px, ret).items():
        r, turn = backtest(w, ret)
        for per, (p0, p1) in PERIODS.items():
            if name.startswith("雙動能") and per != MAIN:
                continue  # IEF 2002-07 才上市，雙動能要到 2003-08 才有第一個訊號，前段無法公平比較
            x, b = r.loc[p0:p1], bh[a].loc[p0:p1]
            st, sb = stats(x, rf, turn.loc[p0:p1]), stats(b, rf)
            z, p_jk = sharpe_test(x - rf.loc[p0:p1], b - rf.loc[p0:p1])
            _, p_bt = boot_sharpe(x - rf.loc[p0:p1], b - rf.loc[p0:p1], reps=1000)
            yr = pd.DataFrame({"s": x, "b": b}).groupby(x.index.year).apply(lambda g: ((1 + g).prod() - 1))
            ws, wr, nw = rolling_wins(x, b, rf.loc[p0:p1])
            rows.append(dict(strategy=name, bench=a, period=per, **{k: st[k] for k in ("CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Turnover/yr")},
                             BH_CAGR=sb["CAGR"], BH_Sharpe=sb["Sharpe"], BH_MaxDD=sb["MaxDD"], JK_p=p_jk, boot_p=p_bt,
                             years_beat_return=float((yr.s > yr.b).mean()), n_years=len(yr),
                             roll3y_beat_sharpe=ws, roll3y_beat_return=wr, n_windows=nw,
                             worst_year_gap=float((yr.s - yr.b).min())))
            if per == MAIN:
                for y, g in yr.iterrows():
                    years.append(dict(strategy=name, bench=a, year=y, strat=g.s, bh=g.b))
        # 成本與成交時間
        p0, p1 = PERIODS[MAIN]
        for c in (0.0001, 0.0005, 0.001, 0.002):
            for delay in (0, 1):
                r2, t2 = backtest(w.shift(delay) if delay else w, ret, cost=c)
                st = stats(r2.loc[p0:p1], rf, t2.loc[p0:p1])
                cost_rows.append(dict(strategy=name, bench=a, cost=c, delay=delay, CAGR=st["CAGR"], Sharpe=st["Sharpe"],
                                      MaxDD=st["MaxDD"], BH_CAGR=stats(bh[a].loc[p0:p1], rf)["CAGR"],
                                      BH_Sharpe=stats(bh[a].loc[p0:p1], rf)["Sharpe"]))

    # 參數敏感度＋deflated Sharpe（所有變體都算試驗）
    p0, p1 = PERIODS[MAIN]
    var_rows, srs, series = [], [], {}
    for fam, a, par, w in variants(px, ret):
        r, turn = backtest(w, ret)
        x = r.loc[p0:p1]
        st = stats(x, rf, turn.loc[p0:p1])
        b = stats(bh[a].loc[p0:p1], rf)
        ex = x - rf.loc[p0:p1]
        srs.append(ex.mean() / ex.std())
        series[(fam, par)] = ex
        x0 = r.loc[PERIODS["2000-04~2007-12"][0]:PERIODS["2000-04~2007-12"][1]]
        st0 = stats(x0, rf)
        b0 = stats(bh[a].loc[PERIODS["2000-04~2007-12"][0]:PERIODS["2000-04~2007-12"][1]], rf)
        var_rows.append(dict(family=fam, bench=a, params=par, CAGR=st["CAGR"], Sharpe=st["Sharpe"], MaxDD=st["MaxDD"],
                             d_CAGR=st["CAGR"] - b["CAGR"], d_Sharpe=st["Sharpe"] - b["Sharpe"], d_MaxDD=st["MaxDD"] - b["MaxDD"],
                             pre_d_Sharpe=st0["Sharpe"] - b0["Sharpe"], pre_d_MaxDD=st0["MaxDD"] - b0["MaxDD"]))
    var = pd.DataFrame(var_rows)
    fam_sum = var.groupby(["family", "bench"]).agg(n=("params", "size"), share_beat_sharpe=("d_Sharpe", lambda s: (s > 0).mean()),
                                                   share_beat_mdd=("d_MaxDD", lambda s: (s > 0).mean()),
                                                   share_beat_cagr=("d_CAGR", lambda s: (s > 0).mean()),
                                                   med_d_Sharpe=("d_Sharpe", "median"), med_d_CAGR=("d_CAGR", "median"),
                                                   pre_share_beat_sharpe=("pre_d_Sharpe", lambda s: (s > 0).mean())).reset_index()
    dsr = {}
    for (name, a), w in candidates(px, ret).items():
        r, _ = backtest(w, ret)
        ex = (r - rf).loc[p0:p1]
        dsr[name] = deflated_sharpe(ex.values, srs, len(ex))[0]
    _, sr0 = deflated_sharpe(next(iter(series.values())).values, srs, len(next(iter(series.values()))))

    res = pd.DataFrame(rows)
    res["DSR"] = res.strategy.map(dsr).where(res.period == MAIN)
    RES.mkdir(exist_ok=True)
    res.to_csv(RES / "d2_candidates.csv", index=False)
    pd.DataFrame(years).to_csv(RES / "d2_years.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(RES / "d2_costs.csv", index=False)
    var.to_csv(RES / "d2_variants.csv", index=False)
    fam_sum.to_csv(RES / "d2_families.csv", index=False)
    (RES / "d2_meta.json").write_text(json.dumps({"n_trials": len(srs), "SR0_annual": sr0}, indent=1), encoding="utf-8")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    print(res.round(3).to_string())
    print(pd.DataFrame(cost_rows).round(4).to_string())
    print(fam_sum.round(3).to_string())
    print("trials", len(srs), "SR0 annual", round(sr0, 3))


if __name__ == "__main__":
    main()
