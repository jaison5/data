"""方向 A 實驗 A2：batch 大小的影響，是 batch 本身，還是「更新次數變少」？

論文一固定 2 個 epoch，batch 越大更新次數越少（batch 4：2,188 次；batch 64：138 次），兩個因素混在一起。
這裡在論文原本的 EUR/USD 單一 80/20 切分上比較兩種訓練預算：
  epochs：固定 2 個 epoch（論文的做法）
  steps ：固定 2,188 次更新（= batch 4 跑 2 個 epoch 的次數）
模型 QLSTM、QFWP；batch 4／16／64；Adam lr 0.05；leaky 與 aligned 兩種資料；10 個種子。

用法：python run_a2.py --jobs 8
輸出：results/a2/<資料>/<模型>_b<batch>_<預算>_s<種子>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results" / "a2"
STEPS = 2188
_CACHE = {}


def run(job):
    import torch
    import wf
    torch.set_num_threads(1)
    variant, model_name, batch, budget, seed = job
    out = OUT / variant / f"{model_name}_b{batch}_{budget}_s{seed}.json"
    if out.exists():
        return
    if variant not in _CACHE:
        _CACHE[variant] = wf.load("EURUSD", variant)
    d = _CACHE[variant]
    n = len(d["X"]); n_tr = int(round(0.8 * n))
    tr, te = np.arange(n_tr), np.arange(n_tr, n)
    torch.manual_seed(seed)
    model = wf.make_model(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xtr, ytr = torch.tensor(d["X"][tr]), torch.tensor(d["y"][tr])
    rng = np.random.default_rng(seed + 1000)
    total = 2 * int(np.ceil(n_tr / batch)) if budget == "epochs" else STEPS
    t0, k = time.time(), 0
    while k < total:
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            if k >= total:
                break
            j = torch.as_tensor(perm[s:s + batch])
            opt.zero_grad()
            ((model(Xtr[j])[:, 0] - ytr[j]) ** 2).mean().backward()
            opt.step()
            k += 1
    with torch.no_grad():
        p = model(torch.tensor(d["X"][te]))[:, 0].numpy()
    yhat, yt = wf.to_price(d, te, p), d["tgt"][te]
    res = {"variant": variant, "model": model_name, "batch": batch, "budget": budget, "seed": seed, "updates": k,
           "n_train": n_tr, "n_test": int(len(te)), "rmse": wf.rmse(yhat, yt), "rw_rmse": wf.rmse(d["last"][te], yt),
           "da_paper": wf.da_paper(yhat, yt), "da_known": wf.da_known(yhat, yt, d["last"][te]),
           "seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"{variant:7s} {model_name:5s} b={batch:2d} {budget:6s} s={seed} updates={k} rmse={res['rmse']:.5f} "
          f"DA={res['da_paper']:.2f} {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(v, m, b, bud, s) for m in ("QLSTM", "QFWP") for bud in ("steps", "epochs") for v in ("leaky", "aligned")
            for b in (64, 16, 4) for s in range(10)]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        pool.map(run, jobs, chunksize=1)
    print("A2 完成", flush=True)
