"""D0 feasibility: (1) how high must daily direction accuracy be to beat buy-and-hold QQQ / SPY after costs,
(2) rule-based strategies that need no forecast at all. Writes results/feasibility_*.csv and a markdown summary."""
import json

import numpy as np
import pandas as pd

from common_d import RES, COST, TDAYS, LAST, fetch, load, backtest, stats

PERIODS = {"full": ("2003-01-02", LAST), "design": ("2003-01-02", "2014-12-31"),
           "test": ("2015-01-02", LAST)}


def accuracy_sim(ret, asset, p_grid, draws=300, seed=0, cost=COST, period=PERIODS["full"]):
    """Synthetic forecaster: each day it calls the sign of the asset's next-day return correctly with prob p,
    independent of the size of the move. Three ways to trade the call."""
    r = ret.loc[period[0]:period[1]]
    ra, rf = r[asset].values, r["RF"].values
    up = ra > 0
    T = len(ra)
    yrs = T / TDAYS
    rng = np.random.default_rng(seed)
    bh = np.prod(1 + ra) ** (1 / yrs) - 1
    ex_bh = ra - rf
    sh_bh = ex_bh.mean() / ex_bh.std() * np.sqrt(TDAYS)
    rows = []
    for p in p_grid:
        correct = rng.random((draws, T)) < p
        pred_up = np.where(correct, up, ~up)
        for name, hi, lo in [("long/flat", 1.0, 0.0), ("long/short", 1.0, -1.0), ("overlay 1.5/0.5", 1.5, 0.5)]:
            w = np.where(pred_up, hi, lo)            # weight held on day t (decided at close t-1)
            turn = np.abs(np.diff(w, axis=1, prepend=w[:, :1]))
            cash = 1 - w
            port = w * ra + cash * rf - np.maximum(-cash, 0) * 0.005 / TDAYS - cost * turn
            eq_end = np.prod(1 + port, axis=1)
            cagr = eq_end ** (1 / yrs) - 1
            ex = port - rf
            sh = ex.mean(axis=1) / ex.std(axis=1) * np.sqrt(TDAYS)
            eq = np.cumprod(1 + port, axis=1)
            mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
            rows.append(dict(asset=asset, p=round(p, 3), strategy=name, CAGR_med=np.median(cagr),
                             CAGR_p05=np.quantile(cagr, .05), CAGR_p95=np.quantile(cagr, .95),
                             Sharpe_med=np.median(sh), MaxDD_med=np.median(mdd),
                             P_beat_CAGR=(cagr > bh).mean(), P_beat_Sharpe=(sh > sh_bh).mean(),
                             trades_per_yr=turn.sum(axis=1).mean() / yrs, BH_CAGR=bh, BH_Sharpe=sh_bh))
    return pd.DataFrame(rows)


def rule_strategies(px, ret):
    """Classic, parameter-light rules with the parameters fixed from the literature (no tuning here)."""
    q, s = px["QQQ"], px["SPY"]
    W = {}
    one = pd.Series(1.0, index=px.index)
    W["Buy & hold SPY"] = pd.DataFrame({"SPY": one})
    W["Buy & hold QQQ"] = pd.DataFrame({"QQQ": one})
    above = (q > q.rolling(200).mean()).astype(float)
    W["QQQ 200-day MA -> cash"] = pd.DataFrame({"QQQ": above})
    W["QQQ 200-day MA -> IEF"] = pd.DataFrame({"QQQ": above, "IEF": 1 - above})
    W["2x QQQ 200-day MA -> cash"] = pd.DataFrame({"QQQ": 2 * above})
    rv = ret["QQQ"].rolling(20).std() * np.sqrt(TDAYS)
    vt = (0.20 / rv).clip(upper=1.5).shift(0)
    W["QQQ vol target 20% (cap 1.5x)"] = pd.DataFrame({"QQQ": vt})
    W["QQQ 200-day MA + vol target 20%"] = pd.DataFrame({"QQQ": vt * above})
    # dual momentum, decided on the last trading day of each month
    m_end = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
    mom = px[["SPY", "QQQ", "IEF"]].pct_change(252)
    rf12 = ret["RF"].rolling(252).sum()
    dm = pd.DataFrame(np.nan, index=px.index, columns=["SPY", "QQQ", "IEF"])
    for t in px.index[m_end.values]:
        if pd.isna(mom.loc[t]).any() or pd.isna(rf12.loc[t]):
            continue
        best = "QQQ" if mom.loc[t, "QQQ"] >= mom.loc[t, "SPY"] else "SPY"
        dm.loc[t] = 0.0
        dm.loc[t, best if mom.loc[t, best] > rf12.loc[t] else "IEF"] = 1.0
    W["Dual momentum SPY/QQQ/IEF (monthly)"] = dm.ffill().fillna(0.0)
    return W


def main():
    fetch()
    px, ret, _ = load(start="1999-03-10")
    RES.mkdir(exist_ok=True)
    p_grid = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.65, 0.70]
    sim = pd.concat([accuracy_sim(ret, a, p_grid) for a in ["QQQ", "SPY"]])
    sim.to_csv(RES / "feasibility_accuracy.csv", index=False)

    W = rule_strategies(px, ret)
    rows = []
    for name, w in W.items():
        r, turn = backtest(w, ret)
        for per, (a, b) in PERIODS.items():
            st = stats(r.loc[a:b], ret["RF"], turn.loc[a:b])
            rows.append(dict(strategy=name, period=per, **st))
    rules = pd.DataFrame(rows)
    rules.to_csv(RES / "feasibility_rules.csv", index=False)

    # break-even accuracy per strategy (smallest p whose median CAGR beats buy-and-hold)
    be = {}
    for (a, s), g in sim.groupby(["asset", "strategy"]):
        hit = g[g.CAGR_med > g.BH_CAGR]
        hit_sh = g[g.Sharpe_med > g.BH_Sharpe]
        be[f"{a} {s}"] = dict(CAGR=None if hit.empty else float(hit.p.min()),
                              Sharpe=None if hit_sh.empty else float(hit_sh.p.min()))
    (RES / "feasibility_breakeven.json").write_text(json.dumps(be, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(be, indent=1))
    print(sim[sim.asset == "QQQ"].round(3).to_string())
    print(rules.round(3).to_string())


if __name__ == "__main__":
    main()
