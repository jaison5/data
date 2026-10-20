"""彙整方向 B 的結果 → results/summary_b.json、results/summary_b.md。
用法（PennyLane 環境）：python report_b.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
PL_DIR = HERE.parent / "qfin-main" / "results_win" / "exp1a_pl_adam0.05"
BACKENDS = ["pl_adjoint", "pl_vjp", "pl_backprop", "fast_cpu", "fast_gpu", "lstm"]
NAMES = {"pl_adjoint": "PennyLane adjoint（預設）", "pl_vjp": "PennyLane adjoint＋device_vjp",
         "pl_backprop": "PennyLane default.qubit backprop", "fast_cpu": "fastsim CPU", "fast_gpu": "fastsim GPU",
         "lstm": "古典 LSTM（參考）", "pl_pshift": "PennyLane parameter-shift"}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    md, out = [], {}
    # ---- B1：每一步的時間
    rows, execs = [], None
    for p in RES.glob("bench_step_*.json"):
        o = json.loads(p.read_text())
        rows += o["rows"]
        execs = o.get("executions_per_call_batch16", execs)
    b1 = pd.DataFrame(rows)
    out["b1"] = rows
    out["tracker_batch16"] = execs
    for model in ("QLSTM", "QFWP"):
        d = b1[(b1.model.isin([model, "LSTM"])) & b1.get("error", pd.Series(index=b1.index)).isna()]
        piv = d.pivot_table(index="backend", columns="batch", values="per_sample_ms").reindex(
            [b for b in BACKENDS if b in d.backend.unique()])
        step = d.pivot_table(index="backend", columns="batch", values="step_ms").reindex(piv.index)
        t = pd.DataFrame({"後端": [NAMES[b] for b in piv.index]})
        for B in piv.columns:
            t[f"B={B} 每步 ms"] = [f"{x:.2f}" for x in step[B]]
        for B in piv.columns:
            t[f"B={B} 每筆 ms"] = [f"{x:.3f}" for x in piv[B]]
        md += [f"\n### B1：{model} 每個訓練步（前向＋反向）的時間\n", md_table(t)]
        ref = piv.loc["pl_adjoint"]
        sp = pd.DataFrame({"後端": [NAMES[b] for b in piv.index if b != "pl_adjoint"]})
        for B in piv.columns:
            sp[f"B={B}"] = [f"{ref[B] / piv.loc[b, B]:.1f}×" for b in piv.index if b != "pl_adjoint"]
        md += [f"\n{model}：相對 PennyLane adjoint 的每筆加速倍數\n", md_table(sp)]
    # ---- B2：完整訓練
    def load_dir(d, tag):
        return pd.DataFrame([{**json.loads(p.read_text()), "src": tag} for p in Path(d).glob("*_b*_s*.json")])
    pl = load_dir(PL_DIR, "PennyLane")
    pl = pl[pl.model.isin(["QLSTM", "QFWP"])]
    fc = load_dir(RES / "grid_cpu", "fastsim CPU")
    fg = load_dir(RES / "grid_cuda", "fastsim GPU") if (RES / "grid_cuda").exists() else pd.DataFrame()
    allr = pd.concat([pl, fc, fg], ignore_index=True)
    b2 = []
    for model in ("QLSTM", "QFWP"):
        for b in sorted(allr.batch.unique()):
            r = {"model": model, "batch": "non-batch" if b == 1 else int(b)}
            for src in ("PennyLane", "fastsim CPU", "fastsim GPU"):
                d = allr[(allr.src == src) & (allr.model == model) & (allr.batch == b)]
                nb = allr[(allr.src == src) & (allr.model == model) & (allr.batch == 1)].set_index("seed")
                if len(d):
                    r[f"{src} 秒"] = float(d.full_train_s.median())
                    if b != 1:
                        s = (nb.full_train_s / d.set_index("seed").full_train_s).dropna()
                        r[f"{src} 加速"] = float(s.median()) if len(s) else None
            if "PennyLane 秒" in r and "fastsim CPU 秒" in r:
                r["fastsim CPU 比 PennyLane 快"] = r["PennyLane 秒"] / r["fastsim CPU 秒"]
            if "PennyLane 秒" in r and "fastsim GPU 秒" in r:
                r["fastsim GPU 比 PennyLane 快"] = r["PennyLane 秒"] / r["fastsim GPU 秒"]
            for src in ("PennyLane", "fastsim CPU"):
                d = allr[(allr.src == src) & (allr.model == model) & (allr.batch == b)]
                r[f"{src} RMSE"] = f"{d.rmse.mean():.5f} ± {d.rmse.std():.5f}"
                r[f"{src} DA"] = f"{d.da_paper.mean():.2f} ± {d.da_paper.std():.2f}"
            m = pd.merge(allr[(allr.src == "PennyLane") & (allr.model == model) & (allr.batch == b)][["seed", "rmse"]],
                         allr[(allr.src == "fastsim CPU") & (allr.model == model) & (allr.batch == b)][["seed", "rmse"]],
                         on="seed")
            r["同種子 RMSE 最大差"] = float((m.rmse_x - m.rmse_y).abs().max()) if len(m) else None
            r["同種子 RMSE 相同（差<1e-8）的比例"] = float(((m.rmse_x - m.rmse_y).abs() < 1e-8).mean()) if len(m) else None
            b2.append(r)
    out["b2"] = b2
    t = pd.DataFrame(b2)
    fmt = lambda x: f"{x:.1f}" if isinstance(x, float) and x == x else ("—" if x is None or x != x else x)
    show = ["model", "batch", "PennyLane 秒", "fastsim CPU 秒", "fastsim GPU 秒", "PennyLane 加速", "fastsim CPU 加速",
            "fastsim GPU 加速", "fastsim CPU 比 PennyLane 快", "fastsim GPU 比 PennyLane 快"]
    md += ["\n### B2：完整訓練（2 epochs）的時間（秒，中位數）與加速比（non-batch ÷ batch）\n",
           md_table(t[[c for c in show if c in t]].map(fmt))]
    show = ["model", "batch", "PennyLane RMSE", "fastsim CPU RMSE", "PennyLane DA", "fastsim CPU DA", "同種子 RMSE 最大差",
            "同種子 RMSE 相同（差<1e-8）的比例"]
    md += ["\n### B2：準確率（10 個種子）——換模擬器不改變結果\n",
           md_table(t[show].map(lambda x: f"{x:.2e}" if isinstance(x, float) and x < 0.01 else fmt(x)))]
    # ---- 等價性檢查與零梯度參數
    chk = json.loads((RES / "check_fastsim.json").read_text())
    out["check"] = chk
    t = pd.DataFrame([{"模型": k, "參數量": v["n_params_pl"], "初始參數最大差": f"{v['max_abs_diff_init']:.1e}",
                       "輸出最大差": f"{v['max_abs_diff_output']:.1e}", "梯度最大差": f"{v['max_abs_diff_grad']:.1e}"}
                      for k, v in chk.items() if k != "dead_parameters"])
    md += ["\n### fastsim 與 PennyLane 的等價性\n", md_table(t)]
    t = pd.DataFrame([{"模型": k, "參數量": v["n_params"], "梯度恆為 0 的參數": v["n_zero_grad"],
                       "實際有作用的參數": v["n_params"] - v["n_zero_grad"]} for k, v in chk["dead_parameters"].items()])
    md += ["\n### 結構上不影響輸出的參數\n", md_table(t)]
    (RES / "summary_b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    (RES / "summary_b.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
