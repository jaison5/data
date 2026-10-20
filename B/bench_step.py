"""方向 B 實驗 B1：每一個訓練步（一個 mini-batch）的前向、反向時間，比較不同模擬器與微分方式。
單執行緒、機器上沒有其他重的工作時執行。

用法：
  PennyLane 環境：python bench_step.py --backends pl_adjoint,pl_vjp,pl_pshift,pl_backprop,fast_cpu,lstm
  GPU 環境     ：python bench_step.py --backends fast_gpu        （讀 PennyLane 環境存下的 results/bench_X.npy）
輸出：results/bench_step_<backends>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qfin-main"))

import numpy as np
import torch

import fastsim as F

BATCHES = [1, 4, 16, 64, 256]


def load_X():
    p = HERE / "results" / "bench_X.npy"
    if not p.exists():
        from exp1a import prep
        df, X, y, *_ , n_tr = prep("paper")
        (HERE / "results").mkdir(exist_ok=True)
        np.save(p, X[:256]); np.save(HERE / "results" / "bench_y.npy", y[:256])
    return np.load(p), np.load(HERE / "results" / "bench_y.npy")


def make(model, backend):
    """回傳 (模型, torch device)。PennyLane 後端只在需要時才 import。"""
    if backend == "fast_cpu":
        return F.MODELS[model](), "cpu"
    if backend == "fast_gpu":
        return F.MODELS[model]().cuda(), "cuda"
    if backend == "lstm":
        return F.LSTM(), "cpu"
    import pl_models as P
    cfg = {"pl_adjoint": ("lightning.qubit", "adjoint", False), "pl_vjp": ("lightning.qubit", "adjoint", True),
           "pl_pshift": ("lightning.qubit", "parameter-shift", False),
           "pl_backprop": ("default.qubit", "backprop", False)}[backend]
    P.configure(cfg[0], cfg[1], cfg[2])
    return P.MODELS[model](), "cpu"


def time_steps(m, Xb, yb, dev, budget=1.5, rmax=50, rmin=3):
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    fw, bw = [], []
    for k in range(rmax + 2):
        opt.zero_grad()
        sync(); a = time.perf_counter()
        loss = ((m(Xb)[:, 0] - yb) ** 2).mean()
        sync(); b = time.perf_counter()
        loss.backward()
        sync(); c = time.perf_counter()
        opt.step()
        if k >= 2:  # 前兩步是暖身
            fw.append(b - a); bw.append(c - b)
            if len(fw) >= rmin and sum(fw) + sum(bw) > budget:
                break
    return float(np.median(fw)), float(np.median(bw)), len(fw)


def count_executions():
    """lightning.qubit 與 default.qubit 對一個 16 筆的批次各執行幾次電路（用 PennyLane 的 Tracker）。"""
    import pennylane as qml
    import pl_models as P
    res = {}
    for dev_name, diff in (("lightning.qubit", "adjoint"), ("default.qubit", "backprop")):
        P.configure(dev_name, diff, False)
        circ = P.qlstm_gate_qnode(7, 3, 1)
        x = torch.rand(16, 7, requires_grad=True); w = torch.rand(1, 7, requires_grad=True)
        with qml.Tracker(circ.device) as tr:
            torch.stack(circ(x, w), -1).sum().backward()
        res[dev_name] = {k: v for k, v in tr.totals.items()}
    P.configure("lightning.qubit", "adjoint", False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="pl_adjoint,pl_vjp,pl_pshift,pl_backprop,fast_cpu,lstm")
    ap.add_argument("--models", default="QLSTM,QFWP")
    a = ap.parse_args()
    torch.set_num_threads(1)
    X, y = load_X()
    rows = []
    for backend in a.backends.split(","):
        for model in (["LSTM"] if backend == "lstm" else a.models.split(",")):
            for B in BATCHES:
                if backend == "pl_pshift" and B > 4:  # parameter-shift 只支援 batch 1；試到 4 記錄錯誤即可
                    continue
                torch.manual_seed(0)
                m, dev = make(model, backend)
                Xb = torch.tensor(X[:B], device=dev); yb = torch.tensor(y[:B], device=dev)
                try:
                    f, b, r = time_steps(m, Xb, yb, dev, rmax=10 if backend == "pl_pshift" else 50)
                except NotImplementedError as e:  # parameter-shift 不支援對批次輸入求梯度（PennyLane #4462）
                    rows.append({"model": model, "backend": backend, "batch": B, "error": str(e)[:200]})
                    print(f"{model:5s} {backend:11s} B={B:3d}  不支援：{str(e)[:90]}", flush=True)
                    continue
                rows.append({"model": model, "backend": backend, "batch": B, "fwd_ms": 1e3 * f, "bwd_ms": 1e3 * b,
                             "step_ms": 1e3 * (f + b), "per_sample_ms": 1e3 * (f + b) / B, "reps": r})
                print(f"{model:5s} {backend:11s} B={B:3d}  fwd {1e3*f:9.2f} ms  bwd {1e3*b:9.2f} ms  "
                      f"每筆 {1e3*(f+b)/B:8.3f} ms  ({r} 次)", flush=True)
    out = {"rows": rows, "torch": torch.__version__, "threads": torch.get_num_threads()}
    if any(b.startswith("pl_") for b in a.backends.split(",")):
        out["executions_per_call_batch16"] = count_executions()
        print("Tracker（16 筆的批次，一次前向＋反向）：", out["executions_per_call_batch16"], flush=True)
    tag = a.backends.replace(",", "_")
    (HERE / "results" / f"bench_step_{tag}.json").write_text(json.dumps(out, indent=1))
