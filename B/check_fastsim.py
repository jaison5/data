"""核對 fastsim.py 和 PennyLane 版（qfin-main/pl_models.py）是否完全等價，並找出結構上永遠是零梯度的參數。
用法（qfin-main 的 PennyLane 環境）：python check_fastsim.py
輸出：results/check_fastsim.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "qfin-main"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import fastsim as F
import pl_models as P
from exp1a import prep

torch.set_num_threads(1)
out = {}
ok = True


def flat(ts):
    return torch.cat([t.detach().reshape(-1) for t in ts])


def compare(name, make_pl, make_fs, run):
    """同一個種子建兩個模型 → 比初始參數、輸出、每個參數的梯度。"""
    global ok
    torch.manual_seed(123)
    mp = make_pl()
    torch.manual_seed(123)
    mf = make_fs()
    d_init = (flat(mp.parameters()) - flat(mf.parameters())).abs().max().item()
    yp, yf = run(mp), run(mf)
    d_out = (yp - yf).abs().max().item()
    (yp ** 2).sum().backward()
    (yf ** 2).sum().backward()
    d_grad = (flat(p.grad for p in mp.parameters()) - flat(p.grad for p in mf.parameters())).abs().max().item()
    good = max(d_init, d_out, d_grad) < 1e-10
    ok &= good
    out[name] = {"n_params_pl": P.n_params(mp), "n_params_fastsim": F.n_params(mf),
                 "max_abs_diff_init": d_init, "max_abs_diff_output": d_out, "max_abs_diff_grad": d_grad, "ok": good}
    print(f"{name:6s} 參數 {P.n_params(mp)}／{F.n_params(mf)}  初始參數差 {d_init:.1e}  輸出差 {d_out:.1e}  梯度差 {d_grad:.1e}  "
          f"{'OK' if good else '不一致'}", flush=True)


df, X, y, *_ , n_tr = prep("paper")
Xb = torch.tensor(X[:16])
compare("QLSTM", lambda: P.QLSTM(), lambda: F.QLSTM(), lambda m: m(Xb))
compare("QFWP", lambda: P.QFWP(), lambda: F.QFWP(), lambda m: m(Xb))
Xc = torch.rand(16, 4, 6)
compare("QLSTM2", lambda: P.QLSTM(n_in=6, H=2, n_out=2), lambda: F.QLSTM(n_in=6, H=2, n_out=2), lambda m: m(Xc))
S = torch.rand(16, 10)
compare("QA3C", lambda: P.QNet(), lambda: F.QNet(), lambda m: torch.cat([m.pi(S), m.v(S)], dim=1))

# ---- 結構上永遠是零梯度的參數（用 1,000 筆真實訓練資料、10 組隨機初始參數）
print("\n結構上不影響輸出的參数（梯度恆為 0）：", flush=True)
dead = {}
Xd = torch.tensor(X[:1000])
for name, make in (("QLSTM", F.QLSTM), ("QFWP", F.QFWP)):
    mx = None
    for s in range(10):
        torch.manual_seed(s)
        m = make()
        (m(Xd) ** 2).mean().backward()
        g = torch.cat([p.grad.abs().reshape(-1) for p in m.parameters()])
        mx = g if mx is None else torch.maximum(mx, g)
    names = [f"{n}[{i}]" for n, p in make().named_parameters() for i in range(p.numel())]
    zero = [nm for nm, v in zip(names, mx.tolist()) if v < 1e-14]
    dead[name] = {"n_params": len(names), "n_zero_grad": len(zero), "zero_grad_params": zero}
    print(f"  {name}: {len(zero)}／{len(names)} 個參數的梯度恆為 0：{zero}", flush=True)
out["dead_parameters"] = dead

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "check_fastsim.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("\n全部一致" if ok else "\n有不一致！", flush=True)
