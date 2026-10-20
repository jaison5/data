"""D1：在 SPY／QQQ 上用 walk-forward 訓練方向 A 的同一組預測器（量子模型用 B/fastsim），產生每天的隔天收盤預測。

沿用 A/gpu_batch.py 的 train_group（多個種子與學習率疊成一個張量一起訓練，和逐一訓練的結果相同）。
用法：python run_d1.py --fetch      （用有 yfinance 的環境下載 OHLC）
      python run_d1.py --procs 12   （用 B/.venv-gpu 訓練）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
from common_d import END
import pandas as pd

DATA, OUT, LOCKS = HERE / "data", HERE / "results" / "d1", HERE / "results" / "_locks_d1"
ASSETS = ["QQQ", "SPY"]
TEST_YEARS = list(range(2008, 2027))
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]
LRS = [0.005, 0.02, 0.05]
COST = {"QLSTM": 10, "QFWP": 3, "FWPC": 3, "LSTM3": 1, "LSTM1": 1}
SEQ = 5
N_SEL, N_ADD = 5, 5   # 5 個種子選學習率，選中的再加 5 個


def fetch():
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    for a in ASSETS:
        df = yf.download(a, start="1999-03-10", end=END, auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df[["Open", "High", "Low", "Close"]].rename_axis("date").to_csv(DATA / f"ohlc_{a}.csv")
        print(a, len(df), df.index[0].date(), df.index[-1].date())


def load(asset):
    """同 A/wf.load：每個 5 天視窗各自 min–max，目標＝隔天收盤（同一尺度）。美股的 Close 是真正的收盤價，
    第 i 天的 OHLC 在第 i 天收盤時都已知，所以沒有前視（見 A 的探針：SPY 50.7%、QQQ 50.0%）。"""
    df = pd.read_csv(DATA / f"ohlc_{asset}.csv", parse_dates=["date"]).set_index("date")
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna().sort_index()
    o = df.values
    X, y, lo, hi, last, tgt, tdate, ddate = [], [], [], [], [], [], [], []
    for i in range(SEQ - 1, len(o) - 1):
        w = o[i - SEQ + 1:i + 1]
        mn, mx = w.min(), w.max()
        if mx - mn < 1e-12:
            continue
        X.append((w - mn) / (mx - mn)); y.append((o[i + 1, 3] - mn) / (mx - mn))
        lo.append(mn); hi.append(mx); last.append(o[i, 3]); tgt.append(o[i + 1, 3])
        tdate.append(df.index[i + 1]); ddate.append(df.index[i])
    return {"X": np.array(X), "y": np.array(y), "lo": np.array(lo), "hi": np.array(hi), "last": np.array(last),
            "tgt": np.array(tgt), "tyear": np.array([d.year for d in tdate]), "tdate": np.array(tdate),
            "ddate": np.array(ddate)}


def fold_idx(d, year):
    ty = d["tyear"]
    return np.where(ty <= year - 2)[0], np.where(ty == year - 1)[0], np.where(ty == year)[0]


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


def run_config(cfg, device, cache):
    from gpu_batch import train_group
    asset, year, model = cfg
    if asset not in cache:
        cache[asset] = load(asset)
    d = cache[asset]
    tr, va, te = fold_idx(d, year)
    meta = {"asset": asset, "test_year": year, "model": model, "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_test": int(len(te)), "runner": f"run_d1/{device}"}

    def go(pairs):
        todo = [(lr, s) for lr, s in pairs if not path(asset, year, model, lr, s).exists()]
        if not todo:
            return
        t0 = time.time()
        res, _ = train_group(d, tr, va, te, model, [lr for lr, _ in todo], [s for _, s in todo], device)
        sec = (time.time() - t0) / len(todo)
        for (lr, s), r in zip(todo, res):
            p = path(asset, year, model, lr, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**r, **meta, "lr": lr, "seed": s, "seconds": sec, "group_size": len(todo)}))
        print(f"{asset} {year} {model:5s} {len(todo):2d} models, {time.time() - t0:.0f} s", flush=True)

    go([(lr, s) for lr in LRS for s in range(N_SEL)])
    val = {lr: np.mean([json.loads(path(asset, year, model, lr, s).read_text())["val_rmse"] for s in range(N_SEL)])
           for lr in LRS}
    best = min(val, key=val.get)
    go([(best, s) for s in range(N_SEL, N_SEL + N_ADD)])


def linear_all():
    """線性迴歸基準（21 個參數），每個 fold 一次，不需要種子。"""
    for a in ASSETS:
        d = load(a)
        for y in TEST_YEARS:
            p = OUT / a / f"y{y}" / "LINEAR.json"
            if p.exists():
                continue
            tr, va, te = fold_idx(d, y)
            A = np.c_[d["X"][tr].reshape(len(tr), -1), np.ones(len(tr))]
            w, *_ = np.linalg.lstsq(A, d["y"][tr], rcond=None)
            f = lambda idx: (np.c_[d["X"][idx].reshape(len(idx), -1), np.ones(len(idx))] @ w) * (d["hi"][idx] - d["lo"][idx]) + d["lo"][idx]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"asset": a, "test_year": y, "model": "LINEAR", "pred_test": f(te).tolist(),
                                     "val_rmse": float(np.sqrt(np.mean((f(va) - d["tgt"][va]) ** 2)))}))


def worker(wid, device):
    import torch
    torch.set_num_threads(1)
    cache = {}
    cfgs = sorted([(a, y, m) for a in ASSETS for y in TEST_YEARS for m in MODELS], key=lambda c: (-COST[c[2]], -c[1]))
    for c in cfgs:
        key = "_".join(map(str, c))
        if not claim(key):
            continue
        dev = device if (c[2] == "QLSTM" and torch.cuda.is_available()) else "cpu"
        try:
            run_config(c, dev, cache)
        except Exception as e:
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} failed: {e!r}", flush=True)
            raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--one", default=None, help="asset,year,model：只跑一組（計時用）")
    a = ap.parse_args()
    if a.fetch:
        fetch(); raise SystemExit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.one:
        s, y, m = a.one.split(",")
        import torch
        torch.set_num_threads(1)
        run_config((s, int(y), m), a.device if m == "QLSTM" else "cpu", {}); raise SystemExit
    linear_all()
    ps = [mp.get_context("spawn").Process(target=worker, args=(w, a.device)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("D1 done", flush=True)
