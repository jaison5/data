"""實驗 1A：照論文原樣重現 QLSTM vs QFWP（EUR/USD 日線 2003-12～2024-12、序列長度 5、80/20、2 epochs）。
用法：python3 exp1a.py [--seeds 10] [--jobs 2]"""
from __future__ import annotations
import argparse, itertools, time
from multiprocessing import Pool
import numpy as np
from common import *
from models import MODELS

BATCHES = (1, 4, 8, 16, 32, 64)  # 1 = 論文的 non-batch


VARIANT = "paper"
LR, OPT, TAG = 0.01, "adam", ""  # 由 --lr/--opt 改寫；TAG 讓不同設定的結果分開存


def prep(variant=None):
    """variant：paper＝論文原樣（同一列的 O/H/L/C）；aligned＝去除前視偏誤：第 i 列改用前一根 K 棒的 High/Low
    （Yahoo 匯率日線的 Close ≈ 該 K 棒開始時的價格，但 High/Low 涵蓋整天，會偷看到隔天的收盤）。"""
    variant = variant or VARIANT
    df = load_ohlc("EURUSD_X").loc["2003-12-01":"2024-12-31"]
    if variant == "aligned":
        df = df.assign(High=df["High"].shift(1), Low=df["Low"].shift(1)).iloc[1:]
    X, y, lo, hi, last, tgt, end = windows_minmax(df.values, 5)
    n_tr = int(round(0.8 * len(X)))
    return df, X, y, lo, hi, last, tgt, end, n_tr


def run(args):
    name, batch, seed, variant = args
    out = RES / f"exp1a_{variant}{TAG}" / f"{name}_b{batch}_s{seed}.json"
    if out.exists():
        return
    df, X, y, lo, hi, last, tgt, end, n_tr = prep(variant)
    rng = np.random.default_rng(seed)
    model = MODELS[name](rng)
    tm = train_regressor(model, X[:n_tr], y[:n_tr], batch, epochs=2, lr=LR, seed=seed, opt=OPT)
    p, t_inf = predict(model, X[n_tr:])
    yhat = p * (hi[n_tr:] - lo[n_tr:]) + lo[n_tr:]
    yt = tgt[n_tr:]
    res = {"model": name, "batch": batch, "seed": seed, "variant": variant, "n_params": n_params(model.params()),
           "rmse": float(np.sqrt(np.mean((yhat - yt) ** 2))),
           "da_paper": da_paper(yhat, yt), "da_known": da_known(yhat, yt, last[n_tr:]),
           "inference_s": t_inf, **tm}
    dump(res, out)
    print(f"[{variant}] {name:5s} b={batch:2d} s={seed} rmse={res['rmse']:.5f} DA(paper)={res['da_paper']:.2f} "
          f"DA(known)={res['da_known']:.2f} train={tm['full_train_s']:.1f}s", flush=True)


def _init(lr, opt, tag):
    global LR, OPT, TAG
    LR, OPT, TAG = lr, opt, tag


def _base(variant):
    df, X, y, lo, hi, last, tgt, end, n_tr = prep(variant)
    yt = tgt[n_tr:]
    base = {"rows": len(df), "windows": len(X), "train": n_tr, "test": len(X) - n_tr,
            "train_end": str(df.index[end[n_tr - 1]].date()), "test_start": str(df.index[end[n_tr]].date()),
            "rw_rmse": float(np.sqrt(np.mean((last[n_tr:] - yt) ** 2))),
            "rw_da_paper": da_paper(last[n_tr:], yt)}
    base["rw_da_known"] = 0.0
    dump(base, RES / f"exp1a_{variant}{TAG}" / "_data.json")
    print(variant, base, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--models", default="QFWP,LSTM,QLSTM")
    ap.add_argument("--variants", default="paper")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--opt", default="adam")
    a = ap.parse_args()
    LR, OPT = a.lr, a.opt
    TAG = "" if (a.lr, a.opt) == (0.01, "adam") else f"_{a.opt}{a.lr:g}"
    for v in a.variants.split(","):
        _base(v)
    jobs = [(m, b, s, v) for v in a.variants.split(",") for s in range(a.seeds) for m in a.models.split(",")
            for b in BATCHES]
    with Pool(a.jobs, initializer=_init, initargs=(LR, OPT, TAG)) as pool:
        pool.map(run, jobs, chunksize=1)



