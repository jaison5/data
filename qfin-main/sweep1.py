"""實驗一的訓練設定掃描：論文未說明最佳化器與學習率，找最接近論文 Table 2 的組合。
用法：python3 sweep1.py [--jobs 2]；結果在 results/sweep1/。"""
from __future__ import annotations
import argparse, itertools, json
from multiprocessing import Pool
import numpy as np
from common import *
from models import MODELS
from exp1a import prep

OPTS = [("adam", 0.03), ("adam", 0.05), ("adam", 0.1)]
BATCHES = (4, 16)
PAPER = {("QLSTM", 4): (0.0034, 69.04), ("QFWP", 4): (0.0030, 72.72),
         ("QLSTM", 16): (0.0039, 65.56), ("QFWP", 16): (0.0029, 72.34)}


def run(args):
    name, batch, seed, opt, lr = args
    out = RES / "sweep1" / f"{name}_{opt}_{lr}_b{batch}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    df, X, y, lo, hi, last, tgt, end, n_tr = prep("paper")
    model = MODELS[name](np.random.default_rng(seed))
    train_regressor(model, X[:n_tr], y[:n_tr], batch, epochs=2, lr=lr, seed=seed, opt=opt)
    p, _ = predict(model, X[n_tr:])
    yhat = p * (hi[n_tr:] - lo[n_tr:]) + lo[n_tr:]
    r = {"model": name, "batch": batch, "seed": seed, "opt": opt, "lr": lr,
         "rmse": float(np.sqrt(np.mean((yhat - tgt[n_tr:]) ** 2))), "da_paper": da_paper(yhat, tgt[n_tr:])}
    dump(r, out)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args()
    jobs = [(m, b, s, o, l) for (o, l) in OPTS for m in ("QFWP", "QLSTM") for b in BATCHES for s in range(3)]
    with Pool(a.jobs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    import pandas as pd
    d = pd.DataFrame(rows).groupby(["opt", "lr", "model", "batch"])[["rmse", "da_paper"]].mean().reset_index()
    d["gap_da"] = [abs(r.da_paper - PAPER[(r.model, r.batch)][1]) for r in d.itertuples()]
    d["gap_rmse_pct"] = [100 * abs(r.rmse - PAPER[(r.model, r.batch)][0]) / PAPER[(r.model, r.batch)][0] for r in d.itertuples()]
    print(d.round(5).to_string())
    score = d.groupby(["opt", "lr"])[["gap_da", "gap_rmse_pct"]].mean().sort_values("gap_da")
    print(score.round(2))
    (RES / "sweep1_summary.txt").write_text(d.round(5).to_string() + "\n\n" + score.round(2).to_string())
