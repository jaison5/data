"""方向 B 實驗 B2：用 fastsim 重跑實驗一的完整訓練（2 epochs、Adam lr 0.05、batch non-batch／4／8／16／32／64、10 個種子），
和 PennyLane 版（qfin-main/results_win/exp1a_pl_adam0.05）比較整體訓練時間與準確率。訓練流程和 pl_exp1.py 逐行對應。

用法：
  PennyLane 環境（先產生 results/grid_data.npz）：python train_grid.py --device cpu --jobs 16
  GPU 環境：python train_grid.py --device cuda --seeds 3        （單一行程依序執行）
輸出：results/grid_<device>/<模型>_b<batch>_s<seed>.json
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
sys.path.insert(0, str(HERE))

import numpy as np

DATA = HERE / "results" / "grid_data.npz"


def da_paper(yhat, y):
    dy, dp = np.diff(y), np.diff(yhat)
    m = dy != 0
    return float(100 * np.mean(np.sign(dp[m]) == np.sign(dy[m])))


def make_data():
    sys.path.insert(0, str(HERE.parent / "qfin-main"))
    from exp1a import prep
    df, X, y, lo, hi, last, tgt, end, n_tr = prep("paper")
    np.savez(DATA, X=X, y=y, lo=lo, hi=hi, last=last, tgt=tgt, n_tr=n_tr)


def run(job):
    import torch
    import fastsim as F
    torch.set_num_threads(1)
    name, batch, seed, device, outdir = job
    out = Path(outdir) / f"{name}_b{batch}_s{seed}.json"
    if out.exists():
        return
    d = np.load(DATA)
    X, y, lo, hi, last, tgt, n_tr = (d[k] for k in ("X", "y", "lo", "hi", "last", "tgt", "n_tr"))
    n_tr = int(n_tr)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    torch.manual_seed(seed)
    model = F.MODELS[name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xtr, ytr = torch.tensor(X[:n_tr], device=device), torch.tensor(y[:n_tr], device=device)
    rng = np.random.default_rng(seed + 1000)
    tf = tb = 0.0
    sync(); t0 = time.perf_counter()
    for _ in range(2):
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            j = torch.as_tensor(idx[s:s + batch], device=device)
            opt.zero_grad()
            sync(); a = time.perf_counter()
            loss = ((model(Xtr[j])[:, 0] - ytr[j]) ** 2).mean()
            sync(); b = time.perf_counter()
            loss.backward()
            sync(); c = time.perf_counter()
            opt.step()
            tf += b - a; tb += c - b
    sync(); full = time.perf_counter() - t0
    Xte = torch.tensor(X[n_tr:], device=device)
    with torch.no_grad():
        sync(); t1 = time.perf_counter()
        p = torch.cat([model(Xte[s:s + 256])[:, 0] for s in range(0, len(Xte), 256)]).cpu().numpy()
        sync(); t_inf = time.perf_counter() - t1
        sync(); t2 = time.perf_counter()
        for s in range(0, len(Xte), batch):
            model(Xte[s:s + batch])
        sync(); t_inf_same = time.perf_counter() - t2
    yhat = p * (hi[n_tr:] - lo[n_tr:]) + lo[n_tr:]
    yt = tgt[n_tr:]
    res = {"model": name, "batch": batch, "seed": seed, "backend": f"fastsim/{device}",
           "rmse": float(np.sqrt(np.mean((yhat - yt) ** 2))), "da_paper": da_paper(yhat, yt),
           "train_forward_s": tf, "backward_s": tb, "full_train_s": full, "inference_s": t_inf,
           "inference_same_batch_s": t_inf_same, "steps_per_epoch": int(np.ceil(n_tr / batch))}
    out.write_text(json.dumps(res, indent=1))
    print(f"{name:5s} b={batch:2d} s={seed} {device} rmse={res['rmse']:.5f} DA={res['da_paper']:.2f} "
          f"train={full:.1f}s (fwd {tf:.1f} bwd {tb:.1f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--models", default="QLSTM,QFWP")
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    a = ap.parse_args()
    if not DATA.exists():
        make_data()
    outdir = HERE / "results" / f"grid_{a.device}"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, int(b), s, a.device, str(outdir)) for m in a.models.split(",") for s in range(a.seeds)
            for b in a.batches.split(",")]
    if a.device == "cuda":
        for j in jobs:
            run(j)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            pool.map(run, jobs, chunksize=1)
