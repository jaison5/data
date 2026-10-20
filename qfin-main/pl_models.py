"""PennyLane + PyTorch 版的模型（照論文使用的工具：lightning.qubit + adjoint 微分）。
電路結構和 numpy 版（models.py、exp2.QNet）完全相同，pl_check.py 會逐一核對輸出與梯度。"""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DEVICE = "lightning.qubit"   # 可改成 "default.qubit"
DIFF = "adjoint"             # 論文的設定；遇到版本問題可改成 "best"
VJP = False                  # PennyLane 預設：adjoint 在前向就把梯度算好；True（device_vjp）則改在反向才用 adjoint 算 VJP


def configure(device: str = None, diff: str = None, device_vjp: bool = None):
    global DEVICE, DIFF, VJP
    DEVICE = device or DEVICE
    DIFF = diff or DIFF
    VJP = VJP if device_vjp is None else device_vjp


# ---------------------------------------------------------------- 電路


def qlstm_gate_qnode(nq: int, H: int, depth: int = 1):
    """QLSTM 一個閘門的 VQC：|+> 起始 → Ry(輸入) 編碼 → [even-odd CNOT 階梯 → 每條線 Ry(權重)] × depth → 量測前 H 條線 <Z>。"""
    dev = qml.device(DEVICE, wires=nq)

    @qml.qnode(dev, interface="torch", diff_method=DIFF, device_vjp=VJP)
    def circuit(inputs, weights):
        for w in range(nq):
            qml.Hadamard(wires=w)
            qml.RY(inputs[..., w], wires=w)
        for d in range(depth):
            for w in range(0, nq - 1, 2):
                qml.CNOT(wires=[w, w + 1])
            for w in range(1, nq - 1, 2):
                qml.CNOT(wires=[w, w + 1])
            for w in range(nq):
                qml.RY(weights[d, w], wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(H)]

    return circuit


def qfwp_qnode(nq: int = 3, D: int = 1, A: int = 1):
    """QFWP 的 VQC：|0> 起始 → Ry(z_t) 編碼 → [CNOT 鏈 → Ry(Θ_t)] × D → 量測前 A 條線。Θ_t 由慢程式產生（不是可訓練權重）。"""
    dev = qml.device(DEVICE, wires=nq)

    @qml.qnode(dev, interface="torch", diff_method=DIFF, device_vjp=VJP)
    def circuit(z, theta):
        for q in range(nq):
            qml.RY(z[..., q], wires=q)
        for d in range(D):
            for q in range(nq - 1):
                qml.CNOT(wires=[q, q + 1])
            for q in range(nq):
                qml.RY(theta[..., d * nq + q], wires=q)
        return qml.expval(qml.PauliZ(0)) if A == 1 else [qml.expval(qml.PauliZ(q)) for q in range(A)]

    return circuit


def qa3c_qnode(nq: int = 8):
    """QA3C 的 VQC：|+> 起始 → Ry(x) 編碼 → [CNOT 環 → Rz(w0)] → [CNOT 環 → Ry(w1)] → 量測 8 條線 <Z>。"""
    dev = qml.device(DEVICE, wires=nq)

    @qml.qnode(dev, interface="torch", diff_method=DIFF, device_vjp=VJP)
    def circuit(inputs, weights):
        for q in range(nq):
            qml.Hadamard(wires=q)
            qml.RY(inputs[..., q], wires=q)
        for q in range(nq):
            qml.CNOT(wires=[q, (q + 1) % nq])
        for q in range(nq):
            qml.RZ(weights[0, q], wires=q)
        for q in range(nq):
            qml.CNOT(wires=[q, (q + 1) % nq])
        for q in range(nq):
            qml.RY(weights[1, q], wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(nq)]

    return circuit


def _col(out, B):
    if isinstance(out, (list, tuple)):
        out = torch.stack(list(out), dim=-1)
    return out.reshape(B, -1)


# ---------------------------------------------------------------- 實驗一的模型


class QLSTM(nn.Module):
    """論文：輸入 4、H=3、7 qubits、深度 1；4 個閘門 × 7 = 28 個量子參數 + Linear(3→1) 4 = 32。
    TorchLayer 的權重預設初始化為 U[0, 2π)。"""

    def __init__(self, n_in=4, H=3, depth=1, n_out=1):
        super().__init__()
        self.H = H
        nq = n_in + H
        self.gates = nn.ModuleList([qml.qnn.TorchLayer(qlstm_gate_qnode(nq, H, depth), {"weights": (depth, nq)})
                                    for _ in range(4)])
        self.out = nn.Linear(H, n_out)

    def forward(self, X):
        B, L, _ = X.shape
        h = X.new_zeros(B, self.H)
        c = X.new_zeros(B, self.H)
        for t in range(L):
            v = torch.cat([X[:, t, :], h], dim=-1)
            f = torch.sigmoid(self.gates[0](v))
            i = torch.sigmoid(self.gates[1](v))
            g = torch.tanh(self.gates[2](v))
            o = torch.sigmoid(self.gates[3](v))
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.out(h)


class QFWP(nn.Module):
    """論文：s_dim 4、latent 3、3 qubits、D=1、A=1；15 + 4 + 12 + 2 = 33 個古典參數。Θ 在每個序列開頭歸零。"""

    def __init__(self, s_dim=4, latent=3, nq=3, D=1, A=1):
        super().__init__()
        self.nq, self.D = nq, D
        self.enc = nn.Linear(s_dim, latent)
        self.lay = nn.Linear(latent, D)
        self.qub = nn.Linear(latent, nq)
        self.post = nn.Linear(A, 1)
        self.circ = qfwp_qnode(nq, D, A)

    def forward(self, X):
        B, L, _ = X.shape
        theta = X.new_zeros(B, self.D * self.nq)
        y = None
        for t in range(L):
            z = self.enc(X[:, t, :])
            a = self.lay(z).unsqueeze(2)       # (B, D, 1)
            b = self.qub(z).unsqueeze(1)       # (B, 1, nq)
            theta = theta + (a * b).reshape(B, self.D * self.nq)
            y = self.post(_col(self.circ(z, theta), B))
        return y


class LSTM(nn.Module):
    """同樣 hidden 大小的古典 LSTM（基準，論文沒有）。"""

    def __init__(self, n_in=4, H=3, n_out=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, H, batch_first=True)
        self.out = nn.Linear(H, n_out)

    def forward(self, X):
        o, _ = self.lstm(X)
        return self.out(o[:, -1, :])


MODELS = {"QLSTM": QLSTM, "QFWP": QFWP, "LSTM": LSTM}


# ---------------------------------------------------------------- 實驗二的代理人


def _init_normal(*layers):
    """MorvanZhou/pytorch-A3C 的初始化：權重 N(0, 0.1)、偏差 0。"""
    for l in layers:
        nn.init.normal_(l.weight, mean=0.0, std=0.1)
        nn.init.constant_(l.bias, 0.0)


class QNet(nn.Module):
    """QA3C：actor、critic 各自 tanh(Linear 10→8) → 8-qubit VQC（16 個量子參數）→ Linear 頭。
    參數：古典 2×88 + 27 + 9 = 212，量子 2×16 = 32，共 244（和論文相同）。"""

    def __init__(self, s_dim=10, a_dim=3, nq=8):
        super().__init__()
        self.pa, self.pc = nn.Linear(s_dim, nq), nn.Linear(s_dim, nq)
        self.va = qml.qnn.TorchLayer(qa3c_qnode(nq), {"weights": (2, nq)})
        self.vc = qml.qnn.TorchLayer(qa3c_qnode(nq), {"weights": (2, nq)})
        self.ha, self.hc = nn.Linear(nq, a_dim), nn.Linear(nq, 1)
        _init_normal(self.pa, self.pc, self.ha, self.hc)

    def pi(self, s):
        return self.ha(self.va(torch.tanh(self.pa(s))))

    def v(self, s):
        return self.hc(self.vc(torch.tanh(self.pc(s))))


class CNet(nn.Module):
    """古典 A3C（MorvanZhou/pytorch-A3C）：actor 10→128→3、critic 10→128→1，tanh；3,332 個參數（和論文相同）。"""

    def __init__(self, s_dim=10, a_dim=3, h=128):
        super().__init__()
        self.pi1, self.pi2 = nn.Linear(s_dim, h), nn.Linear(h, a_dim)
        self.v1, self.v2 = nn.Linear(s_dim, h), nn.Linear(h, 1)
        _init_normal(self.pi1, self.pi2, self.v1, self.v2)

    def pi(self, s):
        return self.pi2(torch.tanh(self.pi1(s)))

    def v(self, s):
        return self.v2(torch.tanh(self.v1(s)))


AGENTS = {"QA3C": QNet, "A3C": CNet}


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
