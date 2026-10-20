"""D4：用 QLSTM／QFWP（以及同規模古典模型、HAR、EWMA）預測未來 20 天波動度，取代波動度目標策略裡的「過去 20 天 σ」。
規格見 PROTOCOL.md 的 D4 增補（先寫好才跑）。
用法：python d4_vol.py --procs 24      （訓練，可中斷後續跑）
      python d4_vol.py --report        （預測誤差＋策略回測）
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

from common_d import LAST

DATA, OUT, LOCKS = HERE / "data", HERE / "results" / "d4", HERE / "results" / "_locks_d4"
ASSETS = ["QQQ", "SPY"]
TEST_YEARS = list(range(2008, 2027))
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]
LRS = [0.005, 0.02, 0.05]
N_SEL, N_ADD = 5, 5
H = 20          # 預測未來 20 個交易日
W, STEPS = 5, 5  # 5 週 × 每週 5 天
COST = {"QLSTM": 10, "QFWP": 3, "FWPC": 3, "LSTM3": 1, "LSTM1": 1}


def raw(asset):
    """回傳每個決策日 t 的原始特徵 (N, 5, 4)、目標 log 已實現波動度（未來 20 天）、日期與基準需要的序列。"""
    df = pd.read_csv(DATA / f"ohlc_{asset}.csv", parse_dates=["date"]).set_index("date").sort_index().loc[:LAST]
    r = np.log(df["Close"]).diff().values
    pk = (np.log(df["High"] / df["Low"]).values ** 2) / (4 * math.log(2))
    n = len(df)
    rr = np.nan_to_num(r ** 2)
    c = np.r_[0, np.cumsum(rr)]
    cp = np.r_[0, np.cumsum(np.nan_to_num(pk))]
    mean_r2 = lambda a, b: (c[b + 1] - c[a]) / (b - a + 1)     # 第 a..b 天（含）
    mean_pk = lambda a, b: (cp[b + 1] - cp[a]) / (b - a + 1)
    lv = lambda m: 0.5 * math.log(252 * max(m, 1e-10))          # log 年化波動度
    X, y, dates, hist20, har = [], [], [], [], []
    first = W * STEPS + 20
    for t in range(first, n):
        f = []
        for j in range(STEPS):
            e = t - W * (STEPS - 1 - j)
            s = e - W + 1
            f.append([lv(mean_r2(s, e)), lv(mean_pk(s, e)), float(np.nansum(r[s:e + 1])), lv(mean_r2(e - 19, e))])
        X.append(f)
        y.append(lv(mean_r2(t + 1, t + H)) if t + H < n else np.nan)
        dates.append(df.index[t])
        hist20.append(lv(mean_r2(t - 19, t)))
        har.append([lv(rr[t]) if rr[t] > 0 else lv(1e-8), lv(mean_r2(t - 4, t)), lv(mean_r2(t - 21, t))])
    # EWMA（RiskMetrics λ=0.94）
    ew, v = [], np.nanmean(rr[1:61])
    for t in range(1, n):
        v = 0.94 * v + 0.06 * rr[t]
        ew.append(v)
    ew = np.r_[np.nan, ew]
    ewma = np.array([lv(ew[t]) for t in range(first, n)])
    return {"X": np.array(X), "ylog": np.array(y), "dates": pd.DatetimeIndex(dates), "hist20": np.array(hist20),
            "har": np.array(har), "ewma": ewma, "year": np.array([d.year for d in dates])}


def fold_idx(R, year):
    yr = R["year"]
    tr = np.where(yr <= year - 2)[0][:-H]
    va = np.where(yr == year - 1)[0][:-H]
    te = np.where(yr == year)[0]
    return tr, va, te


def fold_data(R, year):
    """依訓練期的 1%／99% 分位數把特徵與目標縮到 [0,1]（特徵再截在 [0,1]）。"""
    tr, va, te = fold_idx(R, year)
    X = R["X"]
    lo = np.nanquantile(X[tr], 0.01, axis=(0, 1)); hi = np.nanquantile(X[tr], 0.99, axis=(0, 1))
    Xs = np.clip((X - lo) / (hi - lo), 0, 1)
    ylo, yhi = np.nanquantile(R["ylog"][tr], 0.01), np.nanquantile(R["ylog"][tr], 0.99)
    N = len(X)
    d = {"X": Xs, "y": (R["ylog"] - ylo) / (yhi - ylo), "lo": np.full(N, ylo), "hi": np.full(N, yhi), "tgt": R["ylog"]}
    return d, tr, va, te


def path(asset, year, model, lr, seed):
    return OUT / asset / f"y{year}" / f"{model}_lr{lr:g}_s{seed}.json"


def claim(key):
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCKS / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def run_config(cfg, cache, device="cpu"):
    from gpu_batch import train_group
    asset, year, model = cfg
    if asset not in cache:
        cache[asset] = raw(asset)
    d, tr, va, te = fold_data(cache[asset], year)

    def go(pairs):
        todo = [(lr, s) for lr, s in pairs if not path(asset, year, model, lr, s).exists()]
        if not todo:
            return
        t0 = time.time()
        res, _ = train_group(d, tr, va, te, model, [lr for lr, _ in todo], [s for _, s in todo], device)
        for (lr, s), r in zip(todo, res):
            p = path(asset, year, model, lr, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**r, "asset": asset, "test_year": year, "model": model, "lr": lr, "seed": s,
                                     "n_train": int(len(tr)), "device": device, "seconds": (time.time() - t0) / len(todo)}))
        print(f"{asset} {year} {model:5s} {len(todo):2d} models, {time.time() - t0:.0f} s [{device}]", flush=True)

    go([(lr, s) for lr in LRS for s in range(N_SEL)])
    val = {lr: np.mean([json.loads(path(asset, year, model, lr, s).read_text())["val_rmse"] for s in range(N_SEL)]) for lr in LRS}
    best = min(val, key=val.get)
    go([(best, s) for s in range(N_SEL, N_SEL + N_ADD)])


def worker(wid, device="cpu", only=None):
    import torch
    torch.set_num_threads(1)
    cache = {}
    cfgs = sorted([(a, y, m) for a in ASSETS for y in TEST_YEARS for m in MODELS], key=lambda c: (-COST[c[2]], -c[1]))
    for c in cfgs:
        key = "_".join(map(str, c))
        if only and c[2] not in only:
            continue
        if not claim(key):
            continue
        dev = device if c[2] == "QLSTM" else "cpu"   # 只有 QLSTM 值得放 GPU
        try:
            run_config(c, cache, dev)
        except Exception as e:
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} failed: {e!r}", flush=True)
            raise


# ------------------------------------------------------------ 報表


def forecasts(asset):
    """每個方法在每個測試年的 log σ̂（決策日 t 的預測），合併成一張表。"""
    R = raw(asset)
    out = pd.DataFrame(index=R["dates"])
    out["realized"] = R["ylog"]
    out["hist20"] = R["hist20"]
    out["ewma"] = R["ewma"]
    for c in ["HAR", "LINEAR"] + MODELS:
        out[c] = np.nan
    for y in TEST_YEARS:
        d, tr, va, te = fold_data(R, y)
        # HAR 與線性：最小平方法（訓練期，已 purge）
        A = np.c_[R["har"][tr], np.ones(len(tr))]
        w, *_ = np.linalg.lstsq(A, R["ylog"][tr], rcond=None)
        out.iloc[te, out.columns.get_loc("HAR")] = np.c_[R["har"][te], np.ones(len(te))] @ w
        A = np.c_[d["X"][tr].reshape(len(tr), -1), np.ones(len(tr))]
        w, *_ = np.linalg.lstsq(A, d["y"][tr], rcond=None)
        out.iloc[te, out.columns.get_loc("LINEAR")] = (np.c_[d["X"][te].reshape(len(te), -1), np.ones(len(te))] @ w) * (d["hi"][te] - d["lo"][te]) + d["lo"][te]
        for m in MODELS:
            val = {lr: np.mean([json.loads(path(asset, y, m, lr, s).read_text())["val_rmse"] for s in range(N_SEL)]) for lr in LRS}
            lr = min(val, key=val.get)
            p = np.mean([json.loads(path(asset, y, m, lr, s).read_text())["pred_test"] for s in range(N_SEL + N_ADD)], axis=0)
            out.iloc[te, out.columns.get_loc(m)] = p
    return out.loc[f"{TEST_YEARS[0]}-01-01":]


def report():
    from scipy import stats as sst
    from common_d import RES
    from report_d1 import sharpe_test, boot_sharpe
    from realtime import ohlc, rf_series, run_close, stats
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    METH = ["hist20", "ewma", "HAR", "LINEAR"] + MODELS
    frows, srows = [], []
    for asset in ASSETS:
        F = forecasts(asset)
        F.to_csv(RES / f"d4_forecasts_{asset}.csv")
        ok = F["realized"].notna()
        rv2 = np.exp(2 * F.loc[ok, "realized"])
        for m in METH:
            f2 = np.exp(2 * F.loc[ok, m])
            ql = rv2 / f2 - np.log(rv2 / f2) - 1
            base = rv2 / np.exp(2 * F.loc[ok, "hist20"]); qb = base - np.log(base) - 1
            dlt = (ql - qb).values
            # 目標重疊 20 天：DM 用 Newey–West（20 期）標準誤
            T = len(dlt); u = dlt - dlt.mean(); lrv = u @ u / T
            for k in range(1, H):
                lrv += 2 * (1 - k / H) * (u[k:] @ u[:-k]) / T
            dm = dlt.mean() / math.sqrt(lrv / T)
            frows.append(dict(asset=asset, method=m, QLIKE=float(ql.mean()), RMSE_log=float(np.sqrt(((F.loc[ok, m] - F.loc[ok, "realized"]) ** 2).mean())),
                              DM_vs_hist20=float(dm), p_better_than_hist20=float(sst.norm.cdf(dm))))
        # 策略：即時①（第 t 天收盤後算，第 t+1 天收盤成交 → 第 t+2 天起持有）
        df = ohlc(asset)
        r = df["Close"].pct_change()
        rf = rf_series(df.index)
        a0, a1 = "2008-01-02", LAST
        bh = r.loc[a0:a1]
        series = {}
        for cap in (1.0, 1.5):
            for m in METH:
                sig = np.exp(F[m]).reindex(df.index)
                w = (0.20 / sig).clip(upper=cap).shift(2)
                w = w.fillna((0.20 / np.exp(F["hist20"]).reindex(df.index)).clip(upper=cap).shift(2)).fillna(0.0)
                x, turn = run_close(w, r, rf)
                x = x.loc[a0:a1]
                series[(cap, m)] = x
                s = stats(x, rf)
                srows.append(dict(asset=asset, cap=cap, method=m, CAGR=s["CAGR"], Sharpe=s["Sharpe"], MaxDD=s["MaxDD"],
                                  turnover=float(turn.loc[a0:a1].sum() / s["years"])))
            b = stats(bh, rf)
            srows.append(dict(asset=asset, cap=cap, method="buy&hold", CAGR=b["CAGR"], Sharpe=b["Sharpe"], MaxDD=b["MaxDD"], turnover=0.0))
            for row in srows:
                if row["asset"] == asset and row["cap"] == cap and row["method"] in METH and "p_vs_hist20" not in row:
                    x = series[(cap, row["method"])] - rf.loc[a0:a1]
                    h = series[(cap, "hist20")] - rf.loc[a0:a1]
                    row["p_vs_hist20_JK"] = sharpe_test(x, h)[1]
                    row["p_vs_hist20_boot"] = boot_sharpe(x, h, reps=1000)[1]
                    row["p_vs_BH_boot"] = boot_sharpe(x, bh - rf.loc[a0:a1], reps=1000)[1]
                    row["p_vs_hist20"] = True
    fr, sr = pd.DataFrame(frows), pd.DataFrame(srows).drop(columns=["p_vs_hist20"], errors="ignore")
    fr.to_csv(RES / "d4_forecast_eval.csv", index=False); sr.to_csv(RES / "d4_strategy.csv", index=False)
    print(fr.round(4).to_string()); print(sr.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="逗號分隔的模型名稱：只認領這些模型（例如 QLSTM）")
    a = ap.parse_args()
    if a.check:
        R = raw("QQQ"); d, tr, va, te = fold_data(R, 2020)
        print(R["X"].shape, len(tr), len(va), len(te), np.nanmin(d["X"][tr]), np.nanmax(d["X"][tr]), np.nanmean(d["y"][tr]))
        print("corr(hist20, realized)", np.corrcoef(R["hist20"][~np.isnan(R["ylog"])], R["ylog"][~np.isnan(R["ylog"])])[0, 1])
        raise SystemExit
    if a.report:
        report(); raise SystemExit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    only = a.only.split(",") if a.only else None
    ps = [mp.get_context("spawn").Process(target=worker, args=(w, a.device, only)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("D4 done", flush=True)
