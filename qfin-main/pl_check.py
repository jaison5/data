"""裝好 PennyLane 後先跑這支：檢查 PennyLane 電路和 numpy 模擬器（ad.py）的輸出與梯度是否一致，並印出參數量。
用法：python pl_check.py"""
import numpy as np
import torch
import pennylane as qml

import pl_models as P
from ad import Circuit, Param, T, sum_


def np_circuit(kind):
    ops, k = [], 0
    if kind == "qlstm":
        n, init, meas = 7, "plus", range(3)
        for q in range(n): ops.append(("ry", q, k)); k += 1
        for q in range(0, n - 1, 2): ops.append(("cnot", q, q + 1))
        for q in range(1, n - 1, 2): ops.append(("cnot", q, q + 1))
        for q in range(n): ops.append(("ry", q, k)); k += 1
    elif kind == "qfwp":
        n, init, meas = 3, "zero", [0]
        for q in range(n): ops.append(("ry", q, k)); k += 1
        for q in range(n - 1): ops.append(("cnot", q, q + 1))
        for q in range(n): ops.append(("ry", q, k)); k += 1
    else:
        n, init, meas = 8, "plus", range(8)
        for q in range(n): ops.append(("ry", q, k)); k += 1
        for rot in ("rz", "ry"):
            for q in range(n): ops.append(("cnot", q, (q + 1) % n))
            for q in range(n): ops.append((rot, q, k)); k += 1
    return Circuit(n, ops, k, measure=meas, init=init), n


rng = np.random.default_rng(0)
B = 5
ok = True
for kind in ("qlstm", "qfwp", "qa3c"):
    C, n = np_circuit(kind)
    enc = rng.normal(size=(B, n))
    w = rng.uniform(0, 2 * np.pi, size=(n,) if kind != "qa3c" else (2, n))
    G = rng.normal(size=(B, len(C.measure)))
    # numpy
    if kind == "qfwp":
        Pn = Param(np.concatenate([enc, np.tile(w, (B, 1))], axis=1))
    else:
        Pn = Param(np.concatenate([enc, np.tile(w.reshape(-1), (B, 1))], axis=1))
    out_np = C(Pn)
    sum_(out_np * T(G)).backward()
    g_np = Pn.grad
    # pennylane
    e = torch.tensor(enc, requires_grad=True)
    if kind == "qlstm":
        wt = torch.tensor(w.reshape(1, n), requires_grad=True)
        out = P._col(P.qlstm_gate_qnode(7, 3, 1)(e, wt), B)
    elif kind == "qfwp":
        wt = torch.tensor(np.tile(w, (B, 1)), requires_grad=True)  # Θ 每個樣本各自一份
        out = P._col(P.qfwp_qnode(3, 1, 1)(e, wt), B)
    else:
        wt = torch.tensor(w, requires_grad=True)
        out = P._col(P.qa3c_qnode(8)(e, wt), B)
    (out * torch.tensor(G)).sum().backward()
    d_out = np.abs(out.detach().numpy() - out_np.data).max()
    d_enc = np.abs(e.grad.numpy() - g_np[:, :n]).max()
    g_w_np = g_np[:, n:].sum(axis=0) if kind != "qfwp" else g_np[:, n:]
    d_w = np.abs(wt.grad.numpy().reshape(g_w_np.shape) - g_w_np).max()
    good = max(d_out, d_enc, d_w) < 1e-6
    ok &= good
    print(f"{kind:5s}: 輸出差 {d_out:.1e}、輸入梯度差 {d_enc:.1e}、權重梯度差 {d_w:.1e}  {'OK' if good else '不一致！'}")

for name, cls in P.MODELS.items():
    print(f"{name} 參數量：{P.n_params(cls())}（論文：QLSTM 32、QFWP 33）")
for name, cls in P.AGENTS.items():
    print(f"{name} 參數量：{P.n_params(cls())}（論文：QA3C 244、A3C 3332）")
print("PennyLane", qml.__version__, "| torch", torch.__version__, "| 全部一致" if ok else "| 有不一致，請回報")
