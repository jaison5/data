"""批次 state-vector 模擬器（純 PyTorch，可在 CPU 或 GPU 上跑），重現 qfin-main/pl_models.py 的三種電路與模型。

為什麼需要：PennyLane lightning.qubit 會把一個批次拆成「每個樣本一條電路」逐一模擬，批次只省下 Python 呼叫開銷，
所以論文一的批次加速卡在約 2 倍。這裡把整個批次當成一個張量 (batch, 2^n) 一次算完，梯度由 PyTorch autograd 精確計算
（等同 backprop，不是近似）。

和 pl_models.py 的對應（電路結構、量測、參數形狀、初始化順序都相同；同一個種子會得到完全相同的初始參數）：
  QLSTM  ← pl_models.QLSTM   （每個閘門：|+> → Ry(輸入) → even-odd CNOT 階梯 → Ry(權重) → 量測前 H 條線 <Z>）
  QFWP   ← pl_models.QFWP    （|0> → Ry(z) → CNOT 鏈 → Ry(Θ_t) → 量測第 0 條線 <Z>）
  QNet   ← pl_models.QNet    （QA3C：|+> → Ry(x) → CNOT 環 → Rz → CNOT 環 → Ry → 量測 8 條線）
  CNet、LSTM 為古典模型，和 pl_models 相同（另加參數量相近的 CNet(h=9)，給方向 C 的公平比較用）。
比對結果見 check_fastsim.py。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)  # 和 pl_models.py 相同，確保初始化的亂數消耗一致


# ---------------------------------------------------------------- 基本運算


def _bit(idx: torch.Tensor, w: int, n: int) -> torch.Tensor:
    """基底索引的第 w 條線（PennyLane 慣例：第 0 條線是最高位元）。"""
    return (idx >> (n - 1 - w)) & 1


def cnot_perm(pairs, n: int) -> torch.Tensor:
    """一串依序作用的 CNOT（控制, 目標）合成一個排列 perm，使得 新狀態 = 舊狀態[..., perm]。"""
    N = 1 << n
    idx = torch.arange(N)
    perm = idx.clone()
    for c, t in pairs:
        f = torch.where(_bit(idx, c, n) == 1, idx ^ (1 << (n - 1 - t)), idx)  # CNOT 是自己的反函數
        perm = perm[f]
    return perm


def z_signs(n: int) -> torch.Tensor:
    """(N, n) 矩陣：基底 j 在第 w 條線量測 Z 的本徵值 (+1 或 −1)。<Z_w> = |ψ|² @ signs[:, w]。"""
    idx = torch.arange(1 << n)
    return torch.stack([1.0 - 2.0 * _bit(idx, w, n).to(torch.get_default_dtype()) for w in range(n)], dim=1)


def product_state(theta: torch.Tensor, plus: bool) -> torch.Tensor:
    """每條線 Ry(θ_w) 作用在 |0>（plus=False）或 |+>（plus=True，即先 Hadamard）上的乘積態。
    theta: (..., n) → 回傳 (..., 2^n)，第 0 條線為最高位元。"""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    if plus:
        v0, v1 = (c - s) / math.sqrt(2), (s + c) / math.sqrt(2)
    else:
        v0, v1 = c, s
    v = torch.stack([v0, v1], dim=-1)  # (..., n, 2)
    psi = v[..., 0, :]
    for w in range(1, theta.shape[-1]):
        psi = (psi.unsqueeze(-1) * v[..., w, None, :]).reshape(*psi.shape[:-1], -1)
    return psi


def apply_ry(psi: torch.Tensor, theta: torch.Tensor, w: int, n: int) -> torch.Tensor:
    """在第 w 條線作用 Ry(θ)。psi: (..., 2^n)；theta 可廣播到 psi 的前導維度（例如 (G,1) 或 (B,)）。"""
    L, R = 1 << w, 1 << (n - 1 - w)
    s = psi.reshape(*psi.shape[:-1], L, 2, R)
    a0, a1 = s[..., 0, :], s[..., 1, :]
    c = torch.cos(theta / 2)[..., None, None]
    si = torch.sin(theta / 2)[..., None, None]
    out = torch.stack([c * a0 - si * a1, si * a0 + c * a1], dim=-2)
    return out.reshape(*out.shape[:-3], -1)


def rz_layer_phase(phi: torch.Tensor, n: int, signs: torch.Tensor) -> torch.Tensor:
    """每條線 Rz(φ_w) 合起來是對角相位：基底 j 乘上 exp(−i/2 · Σ_w φ_w · z_w(j))，z_w = ±1。"""
    ang = -0.5 * (signs @ phi)  # (N,)
    return torch.polar(torch.ones_like(ang), ang)


def expval_z(psi: torch.Tensor, signs: torch.Tensor, wires) -> torch.Tensor:
    p = psi.real ** 2 + psi.imag ** 2 if psi.is_complex() else psi ** 2
    return p @ signs[:, list(wires)]


# ---------------------------------------------------------------- 實驗一的模型


class QLSTM(nn.Module):
    """同 pl_models.QLSTM：輸入 n_in、hidden H、qubits n_in+H、深度 depth；4 個閘門各一組 (depth, nq) 權重 + Linear(H→n_out)。
    四個閘門吃同一個輸入向量，所以乘積態和 CNOT 階梯只算一次，最後一層 Ry 才分成 4 組（深度 1 時完全等價）。"""

    def __init__(self, n_in=4, H=3, depth=1, n_out=1):
        super().__init__()
        self.H, self.depth = H, depth
        nq = self.nq = n_in + H
        self.w = nn.ParameterList([nn.Parameter(torch.empty(depth, nq).uniform_(0, 2 * math.pi)) for _ in range(4)])
        self.out = nn.Linear(H, n_out)
        ladder = [(w, w + 1) for w in range(0, nq - 1, 2)] + [(w, w + 1) for w in range(1, nq - 1, 2)]
        self.register_buffer("perm", cnot_perm(ladder, nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def gates(self, v):
        """v: (B, nq) → (4, B, H)，四個閘門的 <Z>。"""
        W = torch.stack(list(self.w), dim=0)  # (4, depth, nq)
        psi = product_state(v, plus=True).unsqueeze(0)  # (1, B, N)
        for d in range(self.depth):
            psi = psi[..., self.perm]
            for q in range(self.nq):
                psi = apply_ry(psi, W[:, d, q][:, None], q, self.nq)  # (4, B, N)
        return expval_z(psi, self.signs, range(self.H))

    def forward(self, X):
        B, L, _ = X.shape
        h = X.new_zeros(B, self.H)
        c = X.new_zeros(B, self.H)
        for t in range(L):
            g = self.gates(torch.cat([X[:, t, :], h], dim=-1))
            f, i, gg, o = torch.sigmoid(g[0]), torch.sigmoid(g[1]), torch.tanh(g[2]), torch.sigmoid(g[3])
            c = f * c + i * gg
            h = o * torch.tanh(c)
        return self.out(h)


class QFWP(nn.Module):
    """同 pl_models.QFWP：s_dim 4、latent 3、3 qubits、D=1、A=1；33 個古典參數，Θ 由慢程式產生。"""

    def __init__(self, s_dim=4, latent=3, nq=3, D=1, A=1):
        super().__init__()
        self.nq, self.D, self.A = nq, D, A
        self.enc = nn.Linear(s_dim, latent)
        self.lay = nn.Linear(latent, D)
        self.qub = nn.Linear(latent, nq)
        self.post = nn.Linear(A, 1)
        self.register_buffer("perm", cnot_perm([(q, q + 1) for q in range(nq - 1)], nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def circ(self, z, theta):
        psi = product_state(z, plus=False)
        for d in range(self.D):
            psi = psi[..., self.perm]
            for q in range(self.nq):
                psi = apply_ry(psi, theta[:, d * self.nq + q], q, self.nq)
        return expval_z(psi, self.signs, range(self.A))

    def forward(self, X):
        B, L, _ = X.shape
        theta = X.new_zeros(B, self.D * self.nq)
        y = None
        for t in range(L):
            z = self.enc(X[:, t, :])
            a = self.lay(z).unsqueeze(2)
            b = self.qub(z).unsqueeze(1)
            theta = theta + (a * b).reshape(B, self.D * self.nq)
            y = self.post(self.circ(z, theta))
        return y


class FWPClassical(QFWP):
    """方向 A 的消融：QFWP 的慢程式與快權重累加完全相同，只把 3-qubit 電路換成無參數的古典函數 tanh(z·Θ)。參數同為 33。"""

    def circ(self, z, theta):
        return torch.tanh((z * theta[:, : self.nq]).sum(-1, keepdim=True))


class LSTM(nn.Module):
    """古典 LSTM 基準（同 pl_models.LSTM）。H=1 時 30 個參數，和 QLSTM 的 32 個接近。"""

    def __init__(self, n_in=4, H=3, n_out=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, H, batch_first=True)
        self.out = nn.Linear(H, n_out)

    def forward(self, X):
        o, _ = self.lstm(X)
        return self.out(o[:, -1, :])


# ---------------------------------------------------------------- 實驗二的代理人


def _init_normal(*layers):
    for l in layers:
        nn.init.normal_(l.weight, mean=0.0, std=0.1)
        nn.init.constant_(l.bias, 0.0)


class _VQCWeights(nn.Module):
    """只裝一組 VQC 權重，讓參數順序與名稱（va.weights、vc.weights）和 PennyLane 的 TorchLayer 相同。"""

    def __init__(self, shape):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(*shape).uniform_(0, 2 * math.pi))


class QNet(nn.Module):
    """同 pl_models.QNet（QA3C）：actor、critic 各自 tanh(Linear 10→8) → 8-qubit VQC → Linear 頭；244 個參數。"""

    def __init__(self, s_dim=10, a_dim=3, nq=8):
        super().__init__()
        self.nq = nq
        self.pa, self.pc = nn.Linear(s_dim, nq), nn.Linear(s_dim, nq)
        self.va, self.vc = _VQCWeights((2, nq)), _VQCWeights((2, nq))
        self.ha, self.hc = nn.Linear(nq, a_dim), nn.Linear(nq, 1)
        _init_normal(self.pa, self.pc, self.ha, self.hc)
        self.register_buffer("perm", cnot_perm([(q, (q + 1) % nq) for q in range(nq)], nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def vqc(self, x, w):
        psi = product_state(x, plus=True).to(torch.complex128)
        psi = psi[..., self.perm] * rz_layer_phase(w[0], self.nq, self.signs)
        psi = psi[..., self.perm]
        for q in range(self.nq):
            psi = apply_ry(psi, w[1, q], q, self.nq)
        return expval_z(psi, self.signs, range(self.nq))

    def refresh(self):
        """把 actor 電路中「編碼之後」的固定部分（CNOT 環、Rz、CNOT 環、Ry）合成一個 256×256 矩陣快取起來。
        權重更新之前，每一步選動作只要一次矩陣乘法（給強化學習的 worker 用；不需要梯度）。"""
        with torch.no_grad():
            w = self.va.weights
            M = torch.eye(1 << self.nq, dtype=torch.complex128, device=w.device)
            M = M[..., self.perm] * rz_layer_phase(w[0], self.nq, self.signs)
            M = M[..., self.perm]
            for q in range(self.nq):
                M = apply_ry(M, w[1, q], q, self.nq)
            self._M = M  # 第 i 列 = 基底 |i> 經過固定部分後的狀態

    def pi_fast(self, s):
        """和 pi(s) 數值相同（不含梯度）；使用 refresh() 快取的矩陣。"""
        with torch.no_grad():
            psi = product_state(torch.tanh(self.pa(s)), plus=True).to(torch.complex128) @ self._M
            return self.ha(expval_z(psi, self.signs, range(self.nq)))

    def pi(self, s):
        return self.ha(self.vqc(torch.tanh(self.pa(s)), self.va.weights))

    def v(self, s):
        return self.hc(self.vqc(torch.tanh(self.pc(s)), self.vc.weights))


class CNet(nn.Module):
    """古典 A3C（同 pl_models.CNet）。h=128 → 3,332 個參數（論文）；h=9 → 238 個參數，和 QA3C 的 244 個相近。"""

    def __init__(self, s_dim=10, a_dim=3, h=128):
        super().__init__()
        self.pi1, self.pi2 = nn.Linear(s_dim, h), nn.Linear(h, a_dim)
        self.v1, self.v2 = nn.Linear(s_dim, h), nn.Linear(h, 1)
        _init_normal(self.pi1, self.pi2, self.v1, self.v2)

    def pi(self, s):
        return self.pi2(torch.tanh(self.pi1(s)))

    def v(self, s):
        return self.v2(torch.tanh(self.v1(s)))


MODELS = {"QLSTM": QLSTM, "QFWP": QFWP, "FWPC": FWPClassical, "LSTM": LSTM}
AGENTS = {"QA3C": QNet, "A3C": CNet, "A3C_small": lambda: CNet(h=9)}


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
