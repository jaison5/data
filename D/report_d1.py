"""D1 報表：把每個預測器的 walk-forward 預測變成交易訊號，照 PROTOCOL.md 事先固定的 3 種策略回測，
和買進持有 SPY／QQQ 及不需預測的規則比較；附方向準確率、PT／DM 檢定、Sharpe 差異檢定與 deflated Sharpe。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))

import numpy as np
import pandas as pd
from scipy import stats as sst

import run_d1 as R
from common_d import RES, TDAYS, LAST, load as load_px, backtest, stats
from feasibility import rule_strategies

EVAL = ("2008-01-02", LAST)
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1", "LINEAR"]


def ensemble(asset, model, d):
    """每個測試年：在驗證年選學習率（前 5 個種子的平均驗證 RMSE），選中學習率的 10 個種子預測取平均。"""
    rows = []
    for y in R.TEST_YEARS:
        tr, va, te = R.fold_idx(d, y)
        if model == "LINEAR":
            p = np.array(json.loads((R.OUT / asset / f"y{y}" / "LINEAR.json").read_text())["pred_test"])
            lr = None
        else:
            val = {lr: np.mean([json.loads(R.path(asset, y, model, lr, s).read_text())["val_rmse"] for s in range(R.N_SEL)])
                   for lr in R.LRS}
            lr = min(val, key=val.get)
            p = np.mean([json.loads(R.path(asset, y, model, lr, s).read_text())["pred_test"]
                         for s in range(R.N_SEL + R.N_ADD)], axis=0)
        rows.append(pd.DataFrame({"date": d["ddate"][te], "pred": p, "last": d["last"][te], "tgt": d["tgt"][te],
                                  "year": y, "lr": lr}))
    return pd.concat(rows).set_index("date")


def direction_stats(e):
    import wf
    up_pred = np.sign(e["pred"] - e["last"]).values
    up_true = np.sign(e["tgt"] - e["last"]).values
    m = up_true != 0
    acc = float(np.mean(up_pred[m] == up_true[m]))
    pt, pt_p = wf.pt_test(up_pred, up_true)
    dm, dm_p = wf.dm_test(e["pred"].values - e["tgt"].values, e["last"].values - e["tgt"].values)
    frac_up = float(np.mean(up_pred > 0))
    n = int(m.sum())
    return {"acc": acc, "acc_ci95": 1.96 * math.sqrt(acc * (1 - acc) / n), "n": n, "pred_up_share": frac_up,
            "PT": pt, "PT_p": pt_p, "DM": dm, "DM_p": dm_p, "base_up": float(np.mean(up_true[m] > 0))}


def sharpe_test(a, b):
    """Memmel (2003) 修正的 Jobson–Korkie 檢定（日資料），單尾：H1 為 a 的 Sharpe 較高。"""
    a, b = a.dropna(), b.reindex(a.index)
    T = len(a)
    sa, sb = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    v = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2) - sa * sb * rho ** 2) / T
    z = (sa - sb) / math.sqrt(v) if v > 0 else 0.0
    return float(z), float(sst.norm.sf(z))


def boot_sharpe(a, b, reps=2000, block=21, seed=0):
    """循環區塊自助法：Sharpe 差（年化）的單尾 p 值。"""
    a, b = a.dropna().values, b.reindex(a.dropna().index).values
    T = len(a)
    f = lambda x: x.mean() / x.std() * math.sqrt(TDAYS)
    obs = f(a) - f(b)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    diffs = np.empty(reps)
    for k in range(reps):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        diffs[k] = f(a[idx]) - f(b[idx])
    return float(obs), float(np.mean(diffs - obs >= obs))


def deflated_sharpe(best, all_sr_daily, T):
    """Bailey & López de Prado (2014)：用全部試過的策略數 N 與其 Sharpe 的變異校正最佳策略。"""
    N = len(all_sr_daily)
    v = np.var(all_sr_daily, ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * sst.norm.ppf(1 - 1 / N) + g * sst.norm.ppf(1 - 1 / (N * math.e)))
    sr = best.mean() / best.std()
    sk, ku = sst.skew(best), sst.kurtosis(best, fisher=False)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(sst.norm.cdf(z)), float(sr0 * math.sqrt(TDAYS))


def main():
    px, ret, _ = load_px(start="1999-03-10")
    a0, a1 = EVAL
    rf = ret["RF"]
    excess = lambda r: (r - rf).loc[a0:a1]
    out, dirs, series = [], [], {}

    def add(name, asset, kind, model, w, extra_delay=False):
        if extra_delay:
            w = w.shift(1)
        r, turn = backtest(w.fillna(0.0), ret)
        r = r.loc[a0:a1]
        st = stats(r, rf, turn.loc[a0:a1])
        series[name] = r
        out.append(dict(name=name, asset=asset, kind=kind, model=model, delay=int(extra_delay), **st))

    # 不需要預測的基準（同 feasibility.py）
    for name, w in rule_strategies(px, ret).items():
        add(name, "QQQ" if "QQQ" in name else "SPY", "rule", "-", w)
    rv = {a: ret[a].rolling(20).std() * math.sqrt(TDAYS) for a in R.ASSETS}
    vt = {a: (0.20 / rv[a]).clip(upper=1.5) for a in R.ASSETS}
    add("SPY vol target 20% (cap 1.5x)", "SPY", "rule", "-", pd.DataFrame({"SPY": vt["SPY"]}))

    for asset in R.ASSETS:
        d = R.load(asset)
        for m in MODELS:
            e = ensemble(asset, m, d)
            ds = direction_stats(e)
            dirs.append(dict(asset=asset, model=m, **ds,
                             lr_counts=json.dumps(e.groupby("year")["lr"].first().value_counts().to_dict())))
            up = (e["pred"] > e["last"]).astype(float).reindex(ret.index)
            S = {"long/flat": up, "overlay 1.5/0.5": 0.5 + up, "vol target x 1.25/0.75": (vt[asset] * (0.75 + 0.5 * up)).clip(upper=2.0)}
            for sname, w in S.items():
                for delay in (False, True):
                    add(f"{asset} {m} {sname}" + (" [+1 day]" if delay else ""), asset, sname, m,
                        pd.DataFrame({asset: w}), extra_delay=delay)

    res = pd.DataFrame(out)
    bh_q, bh_s = series["Buy & hold QQQ"], series["Buy & hold SPY"]
    for i, r in res.iterrows():
        x = series[r["name"]]
        bench = bh_q if r["asset"] == "QQQ" else bh_s
        z, p = sharpe_test(x - rf.loc[x.index], bench - rf.loc[x.index])
        res.loc[i, "JK_z_vs_BH"], res.loc[i, "JK_p_vs_BH"] = z, p
        res.loc[i, "CAGR_minus_BH"] = r["CAGR"] - stats(bench, rf)["CAGR"]
    # 區塊自助法只給 QQQ 的主要結果（不延遲）
    for i, r in res[(res.asset == "QQQ") & (res.delay == 0)].iterrows():
        d_sr, p = boot_sharpe(series[r["name"]] - rf.loc[a0:a1], bh_q - rf.loc[a0:a1])
        res.loc[i, "boot_p_vs_QQQ"] = p
    trials = res[res.delay == 0]
    sr_daily = [(series[n] - rf.loc[a0:a1]).pipe(lambda x: x.mean() / x.std()) for n in trials["name"]]
    best = trials[trials.asset == "QQQ"].sort_values("Sharpe").iloc[-1]["name"]
    dsr, sr0 = deflated_sharpe((series[best] - rf.loc[a0:a1]).values, sr_daily, len(series[best]))

    RES.mkdir(exist_ok=True)
    res.to_csv(RES / "d1_strategies.csv", index=False)
    dirs = pd.DataFrame(dirs)
    dirs.to_csv(RES / "d1_direction.csv", index=False)
    meta = {"eval": EVAL, "n_trials": len(trials), "best_QQQ_by_Sharpe": best, "DSR": dsr,
            "SR0_annual (expected max Sharpe under no skill)": sr0}
    (RES / "d1_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(dirs.round(4).to_string())
    cols = ["name", "CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Turnover/yr", "CAGR_minus_BH", "JK_p_vs_BH", "boot_p_vs_QQQ"]
    print(res[res.delay == 0].sort_values(["asset", "Sharpe"], ascending=[True, False])[cols].round(4).to_string())
    print(res[res.delay == 1][cols].round(4).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
