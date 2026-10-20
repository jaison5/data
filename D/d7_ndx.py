"""D7：用 Nasdaq-100 指數（^NDX，1985-10 起）把量子版本往前延伸到 1995 年。規格見 PROTOCOL.md 的 D7 增補（先寫好才跑）。

直接呼叫 D4（波動度預測）與 D6（端到端）的訓練函式，只換標的（NDX）、測試年（1995～2026）與輸出資料夾，
所以訓練方法和 QQQ 版完全相同。
用法：python d7_ndx.py --procs 20                     （CPU 程序，認領全部設定）
      python d7_ndx.py --procs 3 --device cuda --only QLSTM   （另開少量 GPU 程序只跑 QLSTM）
      python d7_ndx.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

from common_d import RES, TDAYS, LAST

OUT, LOCKS = RES / "d7", RES / "_locks_d7"
ASSET = "NDX"
TEST_YEARS = list(range(1995, 2027))
VOL_MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]
E2E_MODELS = ["QLSTM", "FWPC", "LSTM3", "LSTM1"]
DIV = 0.01   # 指數不含股利：1%／年，依持有比例計入（同 D3）
WCOST = {"QLSTM": 10, "QFWP": 3, "FWPC": 3, "LSTM3": 1, "LSTM1": 1}
PERIODS = {"1995-01~2026-10": ("1995-01-03", LAST), "1995-01~2007-12": ("1995-01-03", "2007-12-31"),
           "2008-01~2026-10": ("2008-01-02", LAST)}
MAIN = "1995-01~2026-10"


def setup():
    """把 D4／D6 的全域設定換成 NDX 版（每個程序都要呼叫）。"""
    import d4_vol as D4
    import d6_e2e as D6
    D4.OUT, D4.LOCKS, D4.TEST_YEARS, D4.ASSETS = OUT / "vol", LOCKS, TEST_YEARS, [ASSET]
    D6.OUT, D6.LOCKS, D6.TEST_YEARS, D6.CAPS, D6.MODELS = OUT / "e2e", LOCKS, TEST_YEARS, [1.0], E2E_MODELS
    return D4, D6


def ndx_returns():
    from realtime import ohlc, rf_series
    df = ohlc(ASSET)
    r = df["Close"].pct_change() + DIV / TDAYS
    return df, r, rf_series(df.index)


def e2e_data(D4):
    """D6.data() 的 NDX 版：決策日 t 的特徵，以及第 t+2 天的報酬與無風險利率。"""
    R = D4.raw(ASSET)
    df, r, rf = ndx_returns()
    pos = df.index.get_indexer(R["dates"])
    assert (pos >= 0).all()
    p2 = pos + 2
    ok = p2 < len(df)
    R["r2"] = np.where(ok, r.values[np.minimum(p2, len(df) - 1)], np.nan)
    R["rf2"] = np.where(ok, rf.values[np.minimum(p2, len(df) - 1)], np.nan)
    return R


def claim(key):
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCKS / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(wid, device="cpu", only=None, skip=None):
    import torch
    torch.set_num_threads(1)
    D4, D6 = setup()
    cv, ce = {}, {}
    jobs = [("vol", y, m) for y in TEST_YEARS for m in VOL_MODELS] + [("e2e", y, m) for y in TEST_YEARS for m in E2E_MODELS]
    jobs.sort(key=lambda j: (-WCOST[j[2]] * (3 if j[0] == "e2e" else 1), -j[1]))
    for kind, y, m in jobs:
        if only and m not in only:
            continue
        if skip and f"{kind}:{m}" in skip:   # 例如 e2e:QLSTM 留給 GPU 程序
            continue
        key = f"{kind}_{y}_{m}"
        if not claim(key):
            continue
        dev = device if m == "QLSTM" else "cpu"
        try:
            if kind == "vol":
                D4.run_config((ASSET, y, m), cv, dev)
            else:
                if "R" not in ce:
                    ce["R"] = e2e_data(D4)
                D6.run_config((1.0, y, m), ce, dev)
        except Exception as e:
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} failed: {e!r}", flush=True)
            raise


# ------------------------------------------------------------ 報表


def stats(x, rf):
    x = x.dropna()
    yrs = len(x) / TDAYS
    cagr = (1 + x).prod() ** (1 / yrs) - 1
    vol = x.std() * math.sqrt(TDAYS)
    ex = x - rf.reindex(x.index)
    eq = (1 + x).cumprod()
    return dict(CAGR=cagr, Vol=vol, ratio=cagr / vol, Sharpe=float(ex.mean() / ex.std() * math.sqrt(TDAYS)),
                MaxDD=float((eq / eq.cummax() - 1).min()))


def report():
    from scipy import stats as sst
    from realtime import ohlc, run_close
    from d5_top_etf import boot_ratio
    from report_d1 import boot_sharpe
    D4, D6 = setup()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)

    # 1. 波動度預測準確度
    F = D4.forecasts(ASSET)
    F.to_csv(RES / "d7_forecasts_NDX.csv")
    ok = F["realized"].notna()
    rv2 = np.exp(2 * F.loc[ok, "realized"])
    base = rv2 / np.exp(2 * F.loc[ok, "hist20"]); qb = base - np.log(base) - 1
    frows = []
    for m in ["hist20", "ewma", "HAR", "LINEAR"] + VOL_MODELS:
        f2 = np.exp(2 * F.loc[ok, m]); ql = rv2 / f2 - np.log(rv2 / f2) - 1
        dlt = (ql - qb).values; T = len(dlt); u = dlt - dlt.mean(); lrv = u @ u / T
        for k in range(1, 20):
            lrv += 2 * (1 - k / 20) * (u[k:] @ u[:-k]) / T
        dm = dlt.mean() / math.sqrt(lrv / T) if lrv > 0 else np.nan
        frows.append(dict(method=m, QLIKE=float(ql.mean()), DM=float(dm), p_better_than_hist20=float(sst.norm.cdf(dm)) if np.isfinite(dm) else np.nan))
    fr = pd.DataFrame(frows); fr.to_csv(RES / "d7_forecast_eval.csv", index=False)
    print(fr.round(4).to_string(index=False))

    # 2. 策略
    df, r, rf = ndx_returns()
    idx = df.index
    S = {}
    hist_w = (0.20 / np.exp(F["hist20"])).reindex(idx).clip(upper=1.0).shift(2)
    for m in ["hist20", "ewma", "HAR"] + VOL_MODELS:
        w = (0.20 / np.exp(F[m])).reindex(idx).clip(upper=1.0).shift(2).fillna(hist_w).fillna(0.0)
        S[f"VT_{m}"] = run_close(w, r, rf)
    R = e2e_data(D4)
    avg, per_seed, lrt = D6.weights_table(R)
    lrt.to_csv(RES / "d7_e2e_lr_choice.csv", index=False)
    for m in E2E_MODELS:
        w = avg[(1.0, m)].reindex(idx).shift(2).fillna(hist_w).fillna(0.0)
        S[f"E2E_{m}"] = run_close(w, r, rf)
    pd.DataFrame({k: v[0] for k, v in S.items()}).to_csv(RES / "d7_daily_returns.csv")
    a0, a1 = PERIODS[MAIN]
    # 事後配對：把「過去 20 天」波動度目標調到和端到端 QLSTM 同樣的實現波動度
    target_vol = S["E2E_QLSTM"][0].loc[a0:a1].std() * math.sqrt(TDAYS)
    sig20 = np.exp(F["hist20"]).reindex(idx)
    lo, hi = 0.01, 0.40
    for _ in range(40):
        mid = (lo + hi) / 2
        x = run_close((mid / sig20).clip(upper=1.0).shift(2).fillna(0.0), r, rf)[0].loc[a0:a1]
        lo, hi = (mid, hi) if x.std() * math.sqrt(TDAYS) < target_vol else (lo, mid)
    S[f"VT_hist20_matched_{mid:.3f}"] = run_close((mid / sig20).clip(upper=1.0).shift(2).fillna(0.0), r, rf)
    matched = f"VT_hist20_matched_{mid:.3f}"

    bench = {t: ohlc(t)["Close"].pct_change() for t in ["SPY", "IVV", "VOO", "QQQ"]}
    bench["NDX_buy&hold"] = r
    rows = []
    for per, (p0, p1) in PERIODS.items():
        for name, (x, turn) in S.items():
            s = stats(x.loc[p0:p1], rf)
            rows.append(dict(period=per, name=name, **s, turnover=float(turn.loc[p0:p1].sum() / (len(x.loc[p0:p1]) / TDAYS))))
        for name, x in bench.items():
            x = x.loc[p0:p1].dropna()
            if len(x) == 0 or x.index[0] > pd.Timestamp(p0) + pd.Timedelta(days=10):
                continue
            rows.append(dict(period=per, name=name, **stats(x, rf), turnover=0.0))
    res = pd.DataFrame(rows); res.to_csv(RES / "d7_strategy.csv", index=False)
    print(res.round(4).to_string(index=False))

    # 3. 事先固定的檢定
    tests = []
    win = {"SPY": ("1995-01-03", LAST), "IVV": ("2000-06-01", LAST), "VOO": ("2010-09-10", LAST)}
    Q = ["VT_QLSTM", "VT_QFWP", "E2E_QLSTM"]
    for q in Q:
        x = S[q][0]
        for t, (p0, p1) in win.items():
            xa, xb = x.loc[p0:p1], bench[t].loc[p0:p1].dropna()
            c = xa.index.intersection(xb.index)
            d, p = boot_ratio(xa.loc[c], xb.loc[c])
            tests.append(dict(strategy=q, vs=t, window=f"{p0}~{p1}", d_ratio=d, p_ratio=p))
        rivals = ["VT_hist20"] + ([f"VT_{m}" for m in ["FWPC", "LSTM3", "LSTM1"]] if q.startswith("VT") else
                                  [f"E2E_{m}" for m in ["FWPC", "LSTM3", "LSTM1"]] + [matched])
        for v in rivals:
            xa, xb = x.loc[a0:a1], S[v][0].loc[a0:a1]
            d, p = boot_ratio(xa, xb)
            ds, ps = boot_sharpe(xa - rf.loc[a0:a1], xb - rf.loc[a0:a1], reps=2000)
            tests.append(dict(strategy=q, vs=v, window=f"{a0}~{a1}", d_ratio=d, p_ratio=p, d_sharpe=ds, p_sharpe=ps))
    tests = pd.DataFrame(tests); tests.to_csv(RES / "d7_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))
    verdict = []
    for q in Q:
        t = tests[tests.strategy == q]
        etf = t[t.vs.isin(["SPY", "IVV", "VOO"])]
        beats_spy = bool(etf[etf.vs == "SPY"].p_ratio.iloc[0] < 0.0167 and etf[etf.vs == "SPY"].d_ratio.iloc[0] > 0)
        beats_point = bool((etf.d_ratio > 0).all())
        riv = t[~t.vs.isin(["SPY", "IVV", "VOO"])]
        contrib = bool(((riv.p_ratio < 0.0125) & (riv.p_sharpe < 0.0125) & (riv.d_ratio > 0) & (riv.d_sharpe > 0)).all())
        verdict.append(dict(strategy=q, beats_top3_etf=beats_spy and beats_point, p_vs_SPY=float(etf[etf.vs == "SPY"].p_ratio.iloc[0]),
                            quantum_contribution=contrib, max_p_vs_rivals_ratio=float(riv.p_ratio.max()),
                            max_p_vs_rivals_sharpe=float(riv.p_sharpe.max())))
    verdict = pd.DataFrame(verdict); verdict.to_csv(RES / "d7_verdict.csv", index=False)
    print(verdict.to_string(index=False))
    meta = {"matched_target_vol": mid, "E2E_QLSTM_vol": target_vol, "mean_w": {m: float(avg[(1.0, m)].loc[a0:a1].mean()) for m in E2E_MODELS}}
    (RES / "d7_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None, help="逗號分隔的 kind:model，例如 e2e:QLSTM")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        D4, D6 = setup()
        R = e2e_data(D4)
        for y in (1995, 2008, 2026):
            d, tr, va, te = D4.fold_data(R, y)
            print(y, "train", len(tr), R["dates"][tr[0]].date(), R["dates"][tr[-1]].date(), "val", len(va), "test", len(te),
                  "X range", float(np.nanmin(d["X"][tr])), float(np.nanmax(d["X"][tr])), "r2 nan in test", int(np.isnan(R["r2"][te]).sum()))
        raise SystemExit
    if a.report:
        report(); raise SystemExit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    only = a.only.split(",") if a.only else None
    skip = a.skip.split(",") if a.skip else None
    ps = [mp.get_context("spawn").Process(target=worker, args=(w, a.device, only, skip)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("D7 done", flush=True)
