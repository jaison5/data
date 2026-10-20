"""實驗一的補充計時：把 adjoint 梯度改在「反向」才計算（device_vjp=True），重量各部分時間，
用來檢驗論文 Table 3、4 的「前向加速約 2.2～2.4 倍、反向只有約 1.0～1.2 倍」。
訓練流程和 pl_exp1.py 相同（2 epochs、Adam lr 0.05），差別：
  - device_vjp 可選（預設 True）；pl_exp1.py 用 PennyLane 預設，adjoint 時前向就把梯度算好
    （lightning.qubit 會忽略 grad_on_execution=False，只有 device_vjp=True 才會把 adjoint 移到反向）。
  - 推論用和訓練相同的 batch 大小（non-batch 為逐筆），對應論文「batched vs. non-batched」的 Infer-Forward。

用法：
  python pl_exp1_timing.py --seeds 3 --jobs 16
  python pl_exp1_timing.py --summary
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from common import RES, da_paper, dump
from exp1a import prep

CFG = {}


def _dir(vjp):
    return RES / f"exp1_timing_vjp{str(vjp).lower()}"


def _init(cfg):
    import torch
    torch.set_num_threads(1)
    CFG.update(cfg)


def run(job):
    import torch
    import pl_models as P
    P.configure(CFG["device"], CFG["diff"], CFG["vjp"])
    name, batch, seed = job
    out = _dir(CFG["vjp"]) / f"{name}_b{batch}_s{seed}.json"
    if out.exists():
        return
    df, X, y, lo, hi, last, tgt, end, n_tr = prep("paper")
    torch.manual_seed(seed)
    model = P.MODELS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    Xtr, ytr = torch.tensor(X[:n_tr]), torch.tensor(y[:n_tr])
    rng = np.random.default_rng(seed + 1000)
    tf = tb = 0.0
    t0 = time.perf_counter()
    for _ in range(2):
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            j = torch.as_tensor(idx[s:s + batch])
            opt.zero_grad()
            a = time.perf_counter()
            loss = ((model(Xtr[j])[:, 0] - ytr[j]) ** 2).mean()
            b = time.perf_counter()
            loss.backward()
            c = time.perf_counter()
            opt.step()
            tf += b - a
            tb += c - b
    full = time.perf_counter() - t0
    Xte = torch.tensor(X[n_tr:])
    with torch.no_grad():
        model(Xte[:batch])  # warm-up
        t1 = time.perf_counter()
        p = torch.cat([model(Xte[s:s + batch])[:, 0] for s in range(0, len(Xte), batch)]).numpy()
        t_inf = time.perf_counter() - t1
    yhat = p * (hi[n_tr:] - lo[n_tr:]) + lo[n_tr:]
    yt = tgt[n_tr:]
    res = {"model": name, "batch": batch, "seed": seed, "device_vjp": CFG["vjp"],
           "rmse": float(np.sqrt(np.mean((yhat - yt) ** 2))), "da_paper": da_paper(yhat, yt),
           "train_forward_s": tf, "backward_s": tb, "full_train_s": full, "inference_s": t_inf,
           "inference_batch": batch}
    dump(res, out)
    print(f"{name:5s} b={batch:2d} s={seed} fwd={tf:.1f}s bwd={tb:.1f}s full={full:.1f}s inf={t_inf:.2f}s "
          f"rmse={res['rmse']:.5f}", flush=True)


def summary(vjp):
    import pandas as pd
    d = _dir(vjp)
    df = pd.DataFrame([json.loads(p.read_text()) for p in d.glob("*_b*_s*.json")])
    cols = ["train_forward_s", "backward_s", "full_train_s", "inference_s"]
    rows = []
    for name in sorted(df.model.unique()):
        nb = df[(df.model == name) & (df.batch == 1)].set_index("seed")
        for b in sorted(df.batch.unique()):
            g = df[(df.model == name) & (df.batch == b)].set_index("seed")
            r = {"model": name, "batch": int(b), "n": len(g)}
            for c in cols:
                r[c] = float(g[c].median())
                if b != 1:
                    r[c + "_speedup"] = float((nb[c] / g[c]).dropna().median())
            rows.append(r)
    out = {"device_vjp": vjp, "rows": rows}
    (RES / f"exp1_timing_vjp{str(vjp).lower()}_summary.json").write_text(json.dumps(out, indent=1))
    print(f"device_vjp={vjp}（中位數秒數；括號內為 non-batch ÷ batch 的加速比中位數）")
    for r in rows:
        sp = lambda c: f" ({r[c + '_speedup']:.2f}x)" if r["batch"] != 1 else ""
        print(f"{r['model']:5s} b={r['batch']:2d} n={r['n']} " +
              " ".join(f"{c[:-2]}={r[c]:.1f}{sp(c)}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--models", default="QLSTM,QFWP")
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--vjp", default="true", choices=["false", "true"])
    ap.add_argument("--device", default="lightning.qubit")
    ap.add_argument("--diff", default="adjoint")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    vjp = a.vjp == "true"
    if a.summary:
        summary(vjp)
        raise SystemExit
    cfg = {"lr": a.lr, "device": a.device, "diff": a.diff, "vjp": vjp}
    _dir(vjp).mkdir(parents=True, exist_ok=True)
    jobs = [(m, int(b), s) for m in a.models.split(",") for s in range(a.seeds) for b in a.batches.split(",")]
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    with ctx.Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
        pool.map(run, jobs, chunksize=1)
    summary(vjp)
