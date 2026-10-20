import numpy as np
from ad import *

rng = np.random.default_rng(0)

def fd_check(f, params, eps=1e-6):
    worst = 0
    for p in params:
        for p_ in params: p_.grad = None
        out = f(); out.backward()
        g = p.grad.copy()
        num = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p.data[i]
            p.data[i] = old + eps; fp = f().data
            p.data[i] = old - eps; fm = f().data
            p.data[i] = old
            num[i] = (fp - fm) / (2 * eps)
        worst = max(worst, np.abs(g - num).max() / (np.abs(num).max() + 1e-12))
    return worst

# 1) 電路：Ry + Rz + CNOT，|0> 與 |+> 起始
for init in ("zero", "plus"):
    n = 4
    ops = []
    k = 0
    for q in range(n): ops.append(("ry", q, k)); k += 1
    for q in range(n - 1): ops.append(("cnot", q, q + 1))
    for q in range(n): ops.append(("rz", q, k)); k += 1
    for q in range(n): ops.append(("ry", q, k)); k += 1
    ops.append(("cnot", 3, 0))
    C = Circuit(n, ops, k, measure=[0, 2, 3], init=init)
    P = Param(rng.normal(size=(5, k)))
    G = T(rng.normal(size=(5, 3)))
    print(init, "circuit grad rel err:", fd_check(lambda: sum_(C(P) * G), [P]))

# 2) 已知結果：單一 qubit Ry(θ) 後 <Z> = cos θ
C1 = Circuit(1, [("ry", 0, 0)], 1, [0])
th = np.linspace(-3, 3, 7)
print("Ry<Z>=cosθ max err:", np.abs(C1(T(th[:, None])).data[:, 0] - np.cos(th)).max())
# Bell：H 由 |+> 起始；CNOT 後 <Z0>=0
C2 = Circuit(2, [("ry", 0, 0), ("cnot", 0, 1)], 1, [0, 1])
print("Ry(pi/2)+CNOT <Z0>,<Z1>:", C2(T([[np.pi / 2]])).data)

# 3) 古典部分：小型 LSTM 式組合
W = Param(rng.normal(size=(3, 4))); b = Param(rng.normal(size=4)); x = T(rng.normal(size=(6, 3)))
w2 = Param(rng.normal(size=(4,)))
def f():
    h = tanh(x @ W + b) * sigmoid(x @ W)
    z = concat([h, h[:, :2]], axis=-1)
    ls = log_softmax(z[:, :3])
    return sum_(square(h - 0.3)) + mean(ls[:, 0]) + sum_(tile_rows(w2, 6) * h) + sum_(reshape(h, (24,)) * 0.1)
print("classical grad rel err:", fd_check(f, [W, b, w2]))

# 4) 合併版電路和原始版逐值相同（含 Rz、CNOT 環）
from ad import CircuitRef
for init in ("zero", "plus"):
    n = 8; ops = []; k = 0
    for q in range(n): ops.append(("ry", q, k)); k += 1
    for rot in ("rz", "ry", "rz"):
        for q in range(n): ops.append(("cnot", q, (q + 1) % n))
        for q in range(n): ops.append((rot, q, k)); k += 1
    A, Bc = Circuit(n, ops, k, range(n), init), CircuitRef(n, ops, k, range(n), init)
    P1 = Param(rng.normal(size=(7, k))); P2 = Param(P1.data.copy()); G = rng.normal(size=(7, n))
    o1, o2 = A(P1), Bc(P2)
    sum_(o1 * T(G)).backward(); sum_(o2 * T(G)).backward()
    print(init, "fused vs ref: out diff", np.abs(o1.data - o2.data).max(), "grad diff", np.abs(P1.grad - P2.grad).max(),
          "ops", len(A.prog), "vs", len(Bc.ops))
    print(init, "fused grad vs finite diff:", fd_check(lambda: sum_(A(P1) * T(G)), [P1]))
