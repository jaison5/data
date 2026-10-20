"""方向 A 實驗 A1：無前視、walk-forward 的量子序列模型基準。

設定：3 個匯率 × 2 種資料（leaky 有前視／aligned 去前視）× 10 個 fold（測試年 2015～2024）× 5 個模型
（QLSTM、QFWP、FWPC＝QFWP 換成古典函數的消融、LSTM1＝30 參數、LSTM3＝112 參數），batch 16。
超參數只看驗證集：
  階段 1：學習率 {0.005, 0.02, 0.05} 各跑種子 0～4，取驗證 RMSE 平均最低者（每個 fold、每個模型各自選）。
  階段 2：選定的學習率再跑種子 5～19，合計 20 個種子。
每次訓練最多 10 個 epoch，驗證 RMSE 連續 3 次沒進步就停，取驗證最好的那次去預測測試年。

用法：python run_a1.py --jobs 8
輸出：results/a1/<匯率>_<資料>/y<測試年>/<模型>_lr<lr>_s<種子>.json、results/a1/lr_choice.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results" / "a1"
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]  # QLSTM 最慢，排最前面
LRS = [0.005, 0.02, 0.05]
VARIANTS = ["aligned", "leaky"]
_CACHE = {}


def path(asset, variant, year, model, lr, seed):
    return OUT / f"{asset}_{variant}" / f"y{year}" / f"{model}_lr{lr:g}_s{seed}.json"


def run(job):
    import wf
    asset, variant, year, model, lr, seed = job
    out = path(*job)
    if out.exists():
        return
    key = (asset, variant)
    if key not in _CACHE:
        _CACHE[key] = wf.load(asset, variant)
    d = _CACHE[key]
    tr, va, te = wf.fold_idx(d, year)
    t0 = time.time()
    r = wf.train_eval(d, tr, va, te, model, lr, seed)
    r.update({"asset": asset, "variant": variant, "test_year": year, "model": model, "lr": lr, "seed": seed,
              "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "seconds": time.time() - t0})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r))
    print(f"{asset} {variant:7s} {year} {model:5s} lr={lr:<5g} s={seed:2d} val={r['val_rmse']:.6f} "
          f"ep={r['best_epoch']}/{r['epochs_run']} {r['seconds']:.0f}s", flush=True)


def choose_lr(assets, years):
    import numpy as np
    choice = {}
    for asset in assets:
        for variant in VARIANTS:
            for year in years:
                for model in MODELS:
                    scores = {}
                    for lr in LRS:
                        v = [json.loads(path(asset, variant, year, model, lr, s).read_text())["val_rmse"] for s in range(5)]
                        scores[lr] = float(np.mean(v))
                    choice[f"{asset}|{variant}|{year}|{model}"] = {"lr": min(scores, key=scores.get), "val_rmse": scores}
    (OUT / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    return choice


if __name__ == "__main__":
    import wf
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    assets, years = a.assets.split(","), wf.TEST_YEARS
    ctx = mp.get_context("spawn")
    stage1 = [(asset, v, y, m, lr, s) for m in MODELS for asset in assets for v in VARIANTS for y in years
              for lr in LRS for s in range(5)]
    print(f"階段 1：{len(stage1)} 次訓練", flush=True)
    with ctx.Pool(a.jobs) as pool:
        pool.map(run, stage1, chunksize=1)
    choice = choose_lr(assets, years)
    stage2 = [(asset, v, y, m, choice[f"{asset}|{v}|{y}|{m}"]["lr"], s) for m in MODELS for asset in assets
              for v in VARIANTS for y in years for s in range(5, 20)]
    print(f"階段 2：{len(stage2)} 次訓練", flush=True)
    with ctx.Pool(a.jobs) as pool:
        pool.map(run, stage2, chunksize=1)
    print("A1 完成", flush=True)
