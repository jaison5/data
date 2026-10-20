"""實驗一的三個序列模型：QLSTM、QFWP（依 ICASSP 2026 論文 Table 1 的參數配置），以及同樣大小的古典 LSTM 基準。"""
from __future__ import annotations

import numpy as np

from ad import (Circuit, Linear, Param, T, concat, getitem, mul, reshape, sigmoid, tanh, tile_rows, n_params)

TWO_PI = 2 * np.pi


class QLSTM:
    """Chen et al. (ICASSP 2022) 的 QLSTM：每個 LSTM 閘門換成一個 VQC。
    論文設定：輸入 4（OHLC）、hidden H=3、n_q = 4+H = 7、深度 1、|+> 起始、Ry 編碼、even-odd CNOT 階梯、
    量測前 H 條線的 <Z>；4 個 VQC × 7 = 28 個量子參數 + Linear(3→1) 4 個 = 32。
    論文未說明的細節（本實作的設定）：可訓練的那層是 CNOT 階梯之後每條線一個 Ry；權重初始化用
    PennyLane TorchLayer 的預設 U[0, 2π)。"""

    def __init__(self, rng, n_in=4, H=3, depth=1, n_out=1):
        self.n_in, self.H = n_in, H
        nq = n_in + H
        ops, k = [], 0
        for q in range(nq):
            ops.append(("ry", q, k)); k += 1
        for _ in range(depth):
            for q in range(0, nq - 1, 2):
                ops.append(("cnot", q, q + 1))
            for q in range(1, nq - 1, 2):
                ops.append(("cnot", q, q + 1))
            for q in range(nq):
                ops.append(("ry", q, k)); k += 1
        self.circ = Circuit(nq, ops, k, measure=range(H), init="plus")
        self.w = [Param(rng.uniform(0, TWO_PI, size=k - nq)) for _ in range(4)]  # f, i, c, o
        self.out = Linear(rng, H, n_out)

    def params(self):
        return self.w + self.out.params()

    def hidden(self, X):
        B, L, _ = X.shape
        h = T(np.zeros((B, self.H)))
        c = T(np.zeros((B, self.H)))
        for t in range(L):
            v = concat([T(X[:, t, :]), h], axis=-1)
            P = concat([concat([v, tile_rows(w, B)], axis=-1) for w in self.w], axis=0)
            E = self.circ(P)
            f = sigmoid(getitem(E, slice(0, B)))
            i = sigmoid(getitem(E, slice(B, 2 * B)))
            g = tanh(getitem(E, slice(2 * B, 3 * B)))
            o = sigmoid(getitem(E, slice(3 * B, 4 * B)))
            c = f * c + i * g
            h = o * tanh(c)
        return h

    def __call__(self, X):
        return self.out(self.hidden(X))


class QFWP:
    """Chen (IJCNN 2024) 的 Quantum Fast Weight Programmer：古典的「慢程式」每一步產生
    Θ_t = Θ_{t-1} + α_t β_tᵀ，用來設定淺層 VQC 的旋轉角度（Θ 在每個序列開頭歸零）。
    論文設定：s_dim=4、latent=3、3 qubits、深度 D=1、量測 A=1 條線；
    Linear(4→3) 15 + Linear(3→1) 4 + Linear(3→3) 12 + Linear(1→1) 2 = 33 個參數，全部是古典參數。
    論文未說明的細節（本實作的設定）：4 維 OHLC 放不進 3 個 qubit，所以把 encoder 輸出的 3 維 z_t
    用 Ry 編碼；糾纏層為 CNOT(0,1)、CNOT(1,2)。"""

    def __init__(self, rng, s_dim=4, latent=3, nq=3, D=1, A=1):
        self.nq, self.D = nq, D
        self.enc = Linear(rng, s_dim, latent)
        self.lay = Linear(rng, latent, D)
        self.qub = Linear(rng, latent, nq)
        self.post = Linear(rng, A, 1)
        assert latent == nq
        ops, k = [], 0
        for q in range(nq):
            ops.append(("ry", q, k)); k += 1
        for d in range(D):
            for q in range(nq - 1):
                ops.append(("cnot", q, q + 1))
            for q in range(nq):
                ops.append(("ry", q, k)); k += 1
        self.circ = Circuit(nq, ops, k, measure=range(A), init="zero")

    def params(self):
        return self.enc.params() + self.lay.params() + self.qub.params() + self.post.params()

    def __call__(self, X):
        B, L, _ = X.shape
        theta = T(np.zeros((B, self.D * self.nq)))
        y = None
        for t in range(L):
            z = self.enc(T(X[:, t, :]))
            a = reshape(self.lay(z), (B, self.D, 1))
            b = reshape(self.qub(z), (B, 1, self.nq))
            theta = theta + reshape(mul(a, b), (B, self.D * self.nq))
            y = self.post(self.circ(concat([z, theta], axis=-1)))
        return y


class LSTM:
    """同樣 hidden 大小的古典 LSTM（基準）。"""

    def __init__(self, rng, n_in=4, H=3, n_out=1):
        self.H = H
        self.gates = Linear(rng, n_in + H, 4 * H)
        self.out = Linear(rng, H, n_out)

    def params(self):
        return self.gates.params() + self.out.params()

    def hidden(self, X):
        B, L, _ = X.shape
        H = self.H
        h = T(np.zeros((B, H)))
        c = T(np.zeros((B, H)))
        for t in range(L):
            G = self.gates(concat([T(X[:, t, :]), h], axis=-1))
            f = sigmoid(getitem(G, (slice(None), slice(0, H))))
            i = sigmoid(getitem(G, (slice(None), slice(H, 2 * H))))
            g = tanh(getitem(G, (slice(None), slice(2 * H, 3 * H))))
            o = sigmoid(getitem(G, (slice(None), slice(3 * H, 4 * H))))
            c = f * c + i * g
            h = o * tanh(c)
        return h

    def __call__(self, X):
        return self.out(self.hidden(X))


MODELS = {"QLSTM": QLSTM, "QFWP": QFWP, "LSTM": LSTM}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, cls in MODELS.items():
        m = cls(rng)
        print(name, "params:", n_params(m.params()))
