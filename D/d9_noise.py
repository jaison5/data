"""D9：把訓練好的量子模型放到「接近現有量子硬體」的雜訊與有限測量次數下推論，看論文的主要結果是否仍成立。
規格見 PROTOCOL.md 的 D9 增補（先寫好才跑）。只測推論，不在雜訊下訓練。

流程：
  1. 重訓並存參數（原本的 JSON 只存預測）：D4 的 QLSTM／QFWP 波動度預測、D6 的端到端 QLSTM（上限 1.0），
     用原本選中的學習率、同樣的 10 個種子與種子分組（0～4、5～9），記錄和原結果的差距。
  2. 雜訊推論：密度矩陣精確計算去極化雜訊（單量子位元閘 p1、CNOT p2）與讀出翻轉（e），再從量測線的聯合分佈
     抽 S 次量測（S = ∞ 時用精確機率）。雜訊輸出會傳到下一個時間步。
  3. 報表：策略回測、對三大 ETF 的檢定、事先固定的判定。

用法：python d9_noise.py --train vol:QLSTM --device cuda     （也可 vol:QFWP、e2e:QLSTM；可中斷後續跑）
      python d9_noise.py --check                             （無雜訊的密度矩陣路徑 = 原本的前向計算）
      python d9_noise.py --infer --device cuda [--only vol:QLSTM]（全部雜訊設定）
      python d9_noise.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

import d4_vol as D4
import d6_e2e as D6
from common_d import LAST, RES, TDAYS

OUT = RES / "d9"
TEST_YEARS = D4.TEST_YEARS
LEVELS = {"ideal": (0.0, 0.0, 0.0), "hw": (0.001, 0.01, 0.02), "hw3": (0.003, 0.03, 0.06)}   # (p1, p2, e)
SHOTS = [0, 10000, 1000, 100]   # 0 = ∞（精確期望值）
N_REAL = 5
JOBS = [("vol", "QLSTM"), ("vol", "QFWP"), ("e2e", "QLSTM")]
SEEDS = list(range(D4.N_SEL + D4.N_ADD))
GROUPS = [list(range(D4.N_SEL)), list(range(D4.N_SEL, D4.N_SEL + D4.N_ADD))]   # 同原本的訓練分組
CHUNK = 64
PRIMARY = ("hw", 1000, False)
A0, A10 = "2008-01-02", "2010-09-10"

_made = []


def capture():
    """讓 gpu_batch.Stacked 建立時記下實例，訓練完就能取得（已還原成最佳 epoch 的）參數。不改變任何計算。"""
    import gpu_batch as GB
    if getattr(GB.Stacked, "_d9", False):
        return

    class Capture(GB.Stacked):
        _d9 = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _made.append(self)
    GB.Stacked = Capture


def data(kind, cache):
    if kind not in cache:
        cache[kind] = D4.raw("QQQ") if kind == "vol" else D6.data()
    return cache[kind]


def ppath(kind, model, year):
    return OUT / "params" / f"{kind}_{model}_y{year}.pt"


# ------------------------------------------------------------ 1. 重訓並存參數


def claim(key):
    (OUT / "_locks").mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(OUT / "_locks" / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def chosen(kind, model, year):
    """原本在驗證年選中的學習率，與 10 個種子的原始測試期輸出。"""
    if kind == "vol":
        val = {lr: np.mean([json.loads(D4.path("QQQ", year, model, lr, s).read_text())["val_rmse"] for s in range(D4.N_SEL)])
               for lr in D4.LRS}
        lr = min(val, key=val.get)
        ref = np.array([json.loads(D4.path("QQQ", year, model, lr, s).read_text())["pred_test"] for s in SEEDS])
    else:
        lr = D6.best_lr(1.0, year, model)[0]
        ref = np.array([D6.load(1.0, year, model, lr, s)["w_test"] for s in SEEDS])
    return lr, ref


def train(kind, model, device):
    import torch
    import fastsim  # noqa: F401  （float64）
    from gpu_batch import train_group
    torch.set_num_threads(1)
    capture()
    cache = {}
    R = data(kind, cache)
    for year in sorted(TEST_YEARS, reverse=True):
        p = ppath(kind, model, year)
        if p.exists() or not claim(f"{kind}_{model}_{year}"):
            continue
        d, tr, va, te = D4.fold_data(R, year)
        lr, ref = chosen(kind, model, year)
        P, out = {}, []
        t0 = time.time()
        # 完全照原本的分組重訓（分組改變會改變浮點運算順序，大學習率時會被放大成不同的 early stopping）：
        # D4 第一階段＝3 個學習率 × 種子 0～4 一起訓練（只留選中的學習率），第二階段＝選中的學習率 × 種子 5～9。
        # D6 原本也是這樣分組；實測只用選中的學習率重訓種子 0～4 已完全重現（差距 < 1e-14），所以只訓練需要的模型。
        if kind == "vol":
            plan = [([l for l in D4.LRS for _ in GROUPS[0]], [s_ for _ in D4.LRS for s_ in GROUPS[0]]), ([lr] * len(GROUPS[1]), GROUPS[1])]
        else:
            plan = [([lr] * len(g), g) for g in GROUPS]
        for lrs, g in plan:
            _made.clear()
            if kind == "vol":
                res, _ = train_group(d, tr, va, te, model, lrs, g, device)
                key = "pred_test"
            else:
                res = D6.e2e_group(R, d, tr, va, te, model, 1.0, lrs, g, device)
                key = "w_test"
            keep = [i for i, l in enumerate(lrs) if l == lr]
            out += [res[i][key] for i in keep]
            S = _made[-1]
            for k in S.keys:
                P.setdefault(k, []).append(S.P[k].detach().cpu()[keep])
        out = np.array(out)
        diff = np.abs(out - ref)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"P": {k: torch.cat(v) for k, v in P.items()}, "lr": lr, "seeds": SEEDS, "out": out, "device": device,
                    "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                    "seeds_differing": int((diff.max(1) > 1e-6).sum())}, p)
        print(f"{kind}:{model} {year} lr={lr:g} {time.time() - t0:.0f} s [{device}]  "
              f"max|Δ| vs original {diff.max():.1e}, seeds with |Δ|>1e-6: {(diff.max(1) > 1e-6).sum()}", flush=True)


# ------------------------------------------------------------ 2. 雜訊電路


class Circuit:
    """n 條線的實數密度矩陣／狀態向量運算（這些電路只有 H、R_y、CNOT，振幅都是實數）。第 0 條線是最高位元。"""

    def __init__(self, n, device):
        import torch
        import fastsim as F
        self.F, self.n, self.dev = F, n, device
        idx = torch.arange(1 << n)
        self.signs = F.z_signs(n).to(device)
        self.flip = [(idx ^ (1 << (n - 1 - w))).to(device) for w in range(n)]
        self._perm = {}

    def perm(self, c, t):
        if (c, t) not in self._perm:
            self._perm[(c, t)] = self.F.cnot_perm([(c, t)], self.n).to(self.dev)
        return self._perm[(c, t)]

    def twirl(self, rho, w):
        """(ρ + XρX + YρY + ZρZ)/4 on wire w = Tr_w(ρ) ⊗ I/2。YρY = X(ZρZ)X。"""
        s = self.signs[:, w]
        a = rho + rho * s[:, None] * s[None, :]
        f = self.flip[w]
        return (a + a[..., f, :][..., f]) / 4

    def depol1(self, rho, w, p):
        """單量子位元去極化：錯誤機率 p（X、Y、Z 各 p/3）。"""
        if p == 0:
            return rho
        lam = 4 * p / 3
        return (1 - lam) * rho + lam * self.twirl(rho, w)

    def depol2(self, rho, c, t, p):
        """雙量子位元去極化：錯誤機率 p（15 個非單位 Pauli 各 p/15）。"""
        if p == 0:
            return rho
        lam = 16 * p / 15
        return (1 - lam) * rho + lam * self.twirl(self.twirl(rho, c), t)

    def ry(self, rho, theta, w):
        """U ρ Uᵀ，U = R_y(θ) on wire w；theta 可廣播到 rho.shape[:-2]。"""
        th = theta[..., None]
        m = self.F.apply_ry(rho, th, w, self.n)                       # ρ Uᵀ
        return self.F.apply_ry(m.transpose(-1, -2), th, w, self.n)    # U ρ Uᵀ（對稱）

    def cnot(self, x, c, t, dm):
        p = self.perm(c, t)
        x = x[..., p]
        return x[..., p, :] if dm else x


def marginal(p, n_meas):
    """量測線（最高的 n_meas 個位元）的聯合分佈。"""
    p = p.clamp_min(0)
    pm = p.reshape(*p.shape[:-1], 1 << n_meas, -1).sum(-1)
    return pm / pm.sum(-1, keepdim=True)


def sample_freq(pm, shots, gen):
    """從 (…, 2^k) 的分佈抽 shots 次，回傳各結果的頻率。逐位元二分：先抽第 0 條線，再在每個分支抽下一條線（等同多項分佈抽樣）。"""
    import torch
    K = pm.shape[-1]
    k = K.bit_length() - 1
    lead = pm.shape[:-1]
    cnt = torch.full((*lead, 1), float(shots), dtype=pm.dtype, device=pm.device)
    for lev in range(k):
        q = pm.reshape(*lead, 2 ** (lev + 1), -1).sum(-1).reshape(*lead, 2 ** lev, 2)
        tot = q.sum(-1)
        pl = torch.where(tot > 0, q[..., 0] / tot.clamp_min(1e-300), torch.zeros_like(tot)).clamp(0, 1)
        left = torch.binomial(cnt, pl, generator=gen)
        cnt = torch.stack([left, cnt - left], -1).reshape(*lead, -1)
    return cnt / shots


def measure(pm, n_meas, e, shots, gen, mitigate):
    """讀出翻轉 → 有限次量測 → ⟨Z⟩；mitigate：除以 1 − 2e（對稱讀出混淆矩陣的反矩陣）。"""
    import torch
    import fastsim as F
    K = 1 << n_meas
    idx = torch.arange(K, device=pm.device)
    if e > 0:
        for w in range(n_meas):
            pm = (1 - e) * pm + e * pm[..., idx ^ (1 << (n_meas - 1 - w))]
    if shots:
        pm = sample_freq(pm, shots, gen)
    z = pm @ F.z_signs(n_meas).to(pm.device)
    if mitigate and e > 0:
        z = (z / (1 - 2 * e)).clamp(-1, 1)
    return z


def qlstm_cell_z(C, v, W, n_meas, p1, p2, e, shots, gen, mitigate, dm):
    """QLSTM 一個時間步的 4 個閘門電路：v (G,B,nq) 輸入角度、W (G,4,nq) 權重 → ⟨Z⟩ (G,4,B,n_meas)。
    只有量測線上的最後一層 R_y 會影響量測結果（其他線的 R_y 與雜訊在量測前被追蹤掉），所以只算量測線。"""
    import torch
    F, n = C.F, C.n
    ladder = [(w, w + 1) for w in range(0, n - 1, 2)] + [(w, w + 1) for w in range(1, n - 1, 2)]
    psi = F.product_state(v, plus=True)   # (G,B,N)；H·R_y(輸入) 在硬體上合成一個單量子位元閘
    if not dm:
        for c, t in ladder:
            psi = C.cnot(psi, c, t, False)
        psi = psi.unsqueeze(1)
        for q in range(n_meas):
            psi = F.apply_ry(psi, W[:, :, q][..., None], q, n)
        p = psi ** 2
    else:
        rho = psi[..., :, None] * psi[..., None, :]
        for w in range(n):
            rho = C.depol1(rho, w, p1)
        for c, t in ladder:
            rho = C.depol2(C.cnot(rho, c, t, True), c, t, p2)
        rho = rho.unsqueeze(1)
        for q in range(n_meas):
            rho = C.depol1(C.ry(rho, W[:, :, q][..., None], q), q, p1)
        p = torch.diagonal(rho, dim1=-2, dim2=-1)
    return measure(marginal(p, n_meas), n_meas, e, shots, gen, mitigate)


def qfwp_z(C, z, theta, p1, p2, e, shots, gen, mitigate, dm):
    """QFWP 最後一個時間步的電路（只有它的輸出會被用到）：|0⟩ → R_y(z) → CNOT 鏈 → R_y(Θ) → 量測第 0 條線。"""
    import torch
    F, n = C.F, C.n
    chain = [(q, q + 1) for q in range(n - 1)]
    psi = F.product_state(z, plus=False)
    if not dm:
        for c, t in chain:
            psi = C.cnot(psi, c, t, False)
        p = F.apply_ry(psi, theta[..., 0], 0, n) ** 2
    else:
        rho = psi[..., :, None] * psi[..., None, :]
        for w in range(n):
            rho = C.depol1(rho, w, p1)
        for c, t in chain:
            rho = C.depol2(C.cnot(rho, c, t, True), c, t, p2)
        rho = C.depol1(C.ry(rho, theta[..., 0], 0), 0, p1)
        p = torch.diagonal(rho, dim1=-2, dim2=-1)
    return measure(marginal(p, 1), 1, e, shots, gen, mitigate)


def noisy_forward(S, X, level, shots, gen, mitigate=False, force_dm=False):
    """和 gpu_batch.Stacked.forward 相同的模型（X: (G,B,L,4) → (G,B)），但電路輸出改成有雜訊、有限次量測的估計值。"""
    import torch
    p1, p2, e = LEVELS[level]
    dm = force_dm or p1 > 0 or p2 > 0
    P = S.P
    G, B, L, _ = X.shape
    if S.name == "QLSTM":
        m = S.template
        assert m.depth == 1
        C = Circuit(m.nq, X.device)
        W = torch.stack([P[f"w.{k}"] for k in range(4)], dim=1)[:, :, 0, :]   # (G,4,nq)
        h = X.new_zeros(G, B, m.H); c = X.new_zeros(G, B, m.H)
        for t in range(L):
            g = qlstm_cell_z(C, torch.cat([X[:, :, t, :], h], dim=-1), W, m.H, p1, p2, e, shots, gen, mitigate, dm)
            f, i, gg, o = torch.sigmoid(g[:, 0]), torch.sigmoid(g[:, 1]), torch.tanh(g[:, 2]), torch.sigmoid(g[:, 3])
            c = f * c + i * gg
            h = o * torch.tanh(c)
        return torch.einsum("gbh,goh->gbo", h, P["out.weight"])[..., 0] + P["out.bias"]
    assert S.name == "QFWP"
    lin = lambda x, k: torch.einsum("gbi,goi->gbo", x, P[f"{k}.weight"]) + P[f"{k}.bias"][:, None, :]
    C = Circuit(3, X.device)
    theta = X.new_zeros(G, B, 3)
    for t in range(L):
        z = lin(X[:, :, t, :], "enc")
        theta = theta + lin(z, "lay") * lin(z, "qub")
    out = qfwp_z(C, z, theta, p1, p2, e, shots, gen, mitigate, dm)
    return lin(out, "post")[..., 0]


def load_stacked(kind, model, year, device):
    import torch
    import fastsim  # noqa: F401
    from gpu_batch import Stacked
    z = torch.load(ppath(kind, model, year), weights_only=False)
    S = Stacked(model, SEEDS, device)
    S.P = {k: z["P"][k].to(device) for k in S.keys}
    return S, z


def fold_outputs(S, d, te, kind, level, shots, gen, mitigate, device, force_dm=False):
    """一個測試年的 10 個模型輸出：波動度預測（log σ̂）或端到端持有比例。"""
    import torch
    X = torch.tensor(d["X"][te], device=device)
    G = len(SEEDS)
    with torch.no_grad():
        o = torch.cat([noisy_forward(S, X[s:s + CHUNK].unsqueeze(0).expand(G, -1, -1, -1), level, shots, gen, mitigate, force_dm)
                       for s in range(0, len(te), CHUNK)], 1)
    o = o.cpu().numpy()
    if kind == "vol":
        return o * (d["hi"][te] - d["lo"][te]) + d["lo"][te]
    return 1.0 / (1.0 + np.exp(-o))   # 上限 1.0


def settings():
    for level in LEVELS:
        for shots in SHOTS:
            for mit in ([False] if level == "ideal" else [False, True]):
                for real in ([0] if shots == 0 else range(N_REAL)):
                    yield level, shots, mit, real


def tag(kind, model, level, shots, mit, real):
    return f"{kind}_{model}_{level}_S{shots}_{'mit' if mit else 'raw'}_r{real}"


def check(device):
    """(1) 無雜訊、S = ∞ 的狀態向量路徑與密度矩陣路徑都要等於 Stacked.forward；(2) 重訓輸出與存下的參數一致。"""
    import torch
    cache = {}
    for kind, model in JOBS:
        R = data(kind, cache)
        for year in (2008, 2020, 2026):
            if not ppath(kind, model, year).exists():
                continue
            S, z = load_stacked(kind, model, year, device)
            d, tr, va, te = D4.fold_data(R, year)
            X = torch.tensor(d["X"][te], device=device).unsqueeze(0).expand(len(SEEDS), -1, -1, -1)
            with torch.no_grad():
                ref = S.forward(X)
                a = noisy_forward(S, X, "ideal", 0, None)
                b = noisy_forward(S, X, "ideal", 0, None, force_dm=True)
            out = fold_outputs(S, d, te, kind, "ideal", 0, None, False, device)
            print(f"{kind}:{model} {year}: |pure−forward| {float((a - ref).abs().max()):.1e}, |dm−forward| {float((b - ref).abs().max()):.1e}, "
                  f"|saved−recomputed| {np.abs(out - z['out']).max():.1e}, original vs retrained max|Δ| {z['max_abs_diff']:.1e} "
                  f"({z['seeds_differing']} seeds differ)", flush=True)
    # 雜訊模型自我檢查：單一 qubit 讀出 → ⟨Z⟩ 乘 (1−2e)；去極化 → 完全混合時 ⟨Z⟩ = 0
    C = Circuit(2, device)
    psi = torch.tensor([1.0, 0, 0, 0], device=device)
    rho = psi[:, None] * psi[None, :]
    full = C.depol1(rho, 0, 0.75)
    print("depol1(p=0.75) on |00⟩, ⟨Z0⟩ =", float(torch.diagonal(full) @ C.signs[:, 0]), "(expect 0)")
    print("readout e=0.02 on ⟨Z⟩=1:", float(measure(torch.tensor([1.0, 0.0], device=device), 1, 0.02, 0, None, False)[0]), "(expect 0.96)")
    g = torch.Generator(device=device); g.manual_seed(0)
    f = sample_freq(torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 20000, device=device, dtype=torch.float64), 1000, g)
    print("shot sampler mean", f.mean(0).cpu().numpy().round(4), "sd", f.std(0).cpu().numpy().round(4),
          "expect sd", np.sqrt(np.array([0.1, 0.2, 0.3, 0.4]) * np.array([0.9, 0.8, 0.7, 0.6]) / 1000).round(4))


def infer(device, only=None):
    import torch
    torch.set_num_threads(1)
    cache = {}
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    for ji, (kind, model) in enumerate(JOBS):
        if only and f"{kind}:{model}" not in only:
            continue
        R = data(kind, cache)
        folds = {y: D4.fold_data(R, y) for y in TEST_YEARS}
        stacks = {y: load_stacked(kind, model, y, device)[0] for y in TEST_YEARS}
        dates = np.concatenate([R["dates"][folds[y][3]] for y in TEST_YEARS])
        if not (OUT / "pred" / "dates.npy").exists():   # 各模型的測試日期相同；先寫暫存檔再改名，避免並行時寫壞
            tmp = OUT / "pred" / f"dates_{os.getpid()}.npy"
            np.save(tmp, dates.astype("datetime64[ns]"))
            os.replace(tmp, OUT / "pred" / "dates.npy")
        for li, (level, shots, mit, real) in enumerate(settings()):
            f = OUT / "pred" / (tag(kind, model, level, shots, mit, real) + ".npy")
            if f.exists():
                continue
            t0 = time.time()
            out = []
            for y in TEST_YEARS:
                gen = torch.Generator(device=device)
                gen.manual_seed(1_000_003 * (real + 1) + 10_007 * ji + 101 * list(LEVELS).index(level) + 7 * SHOTS.index(shots)
                                + 3 * int(mit) + y)
                d, tr, va, te = folds[y]
                out.append(fold_outputs(stacks[y], d, te, kind, level, shots, gen, mit, device))
            np.save(f, np.concatenate(out, axis=1))
            print(f"{tag(kind, model, level, shots, mit, real)}  {time.time() - t0:.0f} s", flush=True)
    print("D9 inference done", flush=True)


# ------------------------------------------------------------ 3. 報表


def _boot(args):
    from d5_top_etf import boot_ratio
    from report_d1 import boot_sharpe
    key, a, b, rfa = args
    c = a.index.intersection(b.index)
    a, b = a.loc[c], b.loc[c]
    out = dict(zip(("d_ratio", "p_ratio"), boot_ratio(a, b, reps=2000)))
    if rfa is not None:
        out.update(zip(("d_sharpe", "p_sharpe"), boot_sharpe(a - rfa.loc[c], b - rfa.loc[c], reps=2000)))
    return key, out


def report(procs=12):
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sst
    import d5_top_etf as D5
    from realtime import ohlc, rf_series, run_close, stats, vt_weight
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    df = ohlc("QQQ")
    r = df["Close"].pct_change()
    rf = rf_series(df.index)
    etf = {t: ohlc(t)["Close"].pct_change().dropna() for t in ("VOO", "IVV", "SPY")}
    F = pd.read_csv(RES / "d4_forecasts_QQQ.csv", index_col=0, parse_dates=True)
    dates = pd.DatetimeIndex(np.load(OUT / "pred" / "dates.npy"))
    hist_w = vt_weight(r, cap=1.0)

    def strategy(kind, arr, cap=1.0):
        """arr: (10, T) 的模型輸出 → 10 個模型平均後的策略日報酬（同 D4／D6 的做法）。"""
        a = pd.Series(arr.mean(0), index=dates)
        if kind == "vol":
            w = (0.20 / np.exp(a)).reindex(df.index).clip(upper=cap).shift(2)
            w = w.fillna((0.20 / np.exp(F["hist20"]).reindex(df.index)).clip(upper=cap).shift(2)).fillna(0.0)
        else:
            w = a.reindex(df.index).shift(2).fillna(vt_weight(r, cap=cap).shift(2)).fillna(0.0)
        return run_close(w, r, rf, D6.COST)[0].loc[A0:LAST], w.loc[A0:LAST]

    def ratio_on(x, a0, ref=None):
        x = x.loc[a0:LAST].dropna()
        if ref is not None:
            x = x.loc[x.index.intersection(ref.index)]
        return D5.ratio(x)[0]

    def qlike(arr):
        a = pd.Series(arr.mean(0), index=dates)
        ok = F["realized"].notna()
        rv2 = np.exp(2 * F.loc[ok, "realized"])
        ql = rv2 / np.exp(2 * a.reindex(F.index)[ok]) - np.log(rv2 / np.exp(2 * a.reindex(F.index)[ok])) - 1
        base = rv2 / np.exp(2 * F.loc[ok, "hist20"]); qb = base - np.log(base) - 1
        dlt = (ql - qb).values
        T = len(dlt); u = dlt - dlt.mean(); lrv = u @ u / T
        for k in range(1, D4.H):
            lrv += 2 * (1 - k / D4.H) * (u[k:] @ u[:-k]) / T
        dm = dlt.mean() / math.sqrt(lrv / T)
        return float(ql.mean()), float(dm), float(sst.norm.cdf(dm))

    # 古典對照（端到端，次要比較）：D6 三個古典模型、D8 主要對照（驗證年最好的正則化古典模型）、兩個傅立葉特徵 LSTM
    import d8_controls as D8
    R6 = D6.data()
    avg6, _, d6lr = D6.weights_table(R6)
    CL = {}
    strat_w = lambda w: run_close(w.reindex(df.index).shift(2).fillna(hist_w.shift(2)).fillna(0.0), r, rf, D6.COST)[0].loc[A0:LAST]
    for m in D6.CLASSICAL:
        CL[f"D6_{m}"] = strat_w(avg6[(1.0, m)])
    regs = {}
    for t in D8.REG:
        for m in D6.CLASSICAL:
            w, vm, _ = D8.weights_series(R6, "reg", t, m)
            regs[f"reg_{t}_{m}"] = (w, vm)
    best_reg = max(regs, key=lambda k: regs[k][1])
    CL[f"D8 primary ({best_reg})"] = strat_w(regs[best_reg][0])
    for m in D8.FF_MODELS:
        CL[f"ff_std_{m}"] = strat_w(D8.weights_series(R6, "ff", "std", m)[0])

    rows, boots, seeds = [], [], []
    original = {("vol", "QLSTM"): "D4_QLSTM", ("vol", "QFWP"): "D4_QFWP", ("e2e", "QLSTM"): "D6_QLSTM"}
    for kind, model in JOBS:
        for level, shots, mit, real in settings():
            f = OUT / "pred" / (tag(kind, model, level, shots, mit, real) + ".npy")
            if not f.exists():
                continue
            arr = np.load(f)
            caps = [1.0, 1.5] if kind == "vol" else [1.0]
            for cap in caps:
                x, w = strategy(kind, arr, cap)
                st = stats(x, rf)
                row = dict(strategy=f"{kind}:{model}", cap=cap, level=level, shots=shots, mitigated=mit, real=real,
                           ratio=D5.ratio(x)[0], CAGR=st["CAGR"], Vol=st["Vol"], Sharpe=st["Sharpe"], MaxDD=st["MaxDD"],
                           mean_w=float(w.mean()), ratio_2010=ratio_on(x, A10))
                if kind == "vol":
                    row["QLIKE"], row["DM_vs_hist20"], row["p_DM"] = qlike(arr)
                rows.append(row)
                if cap == 1.0:
                    key = (f"{kind}:{model}", level, shots, mit, real)
                    for t in ("IVV", "SPY"):
                        boots.append((key + (t,), x, etf[t].loc[A0:LAST], None))
                    if kind == "e2e" and shots in (0, 1000):
                        for nm, y in CL.items():
                            boots.append((key + (nm,), x, y, rf.loc[A0:LAST]))
            if (level, shots, mit, real) in [("ideal", 0, False, 0), PRIMARY + (0,)]:
                for s in range(arr.shape[0]):
                    x = strategy(kind, arr[s:s + 1])[0]
                    seeds.append(dict(strategy=f"{kind}:{model}", level=level, shots=shots, seed=s, ratio=D5.ratio(x)[0],
                                      Sharpe=stats(x, rf)["Sharpe"]))
    res = pd.DataFrame(rows)
    etf_ratio = {(t, a0): ratio_on(etf[t], a0) for t in etf for a0 in (A0, A10)}
    with ProcessPoolExecutor(procs) as ex:
        bt = dict(ex.map(_boot, boots, chunksize=4))
    tests = pd.DataFrame([dict(strategy=k[0], level=k[1], shots=k[2], mitigated=k[3], real=k[4], vs=k[5], **v) for k, v in bt.items()])
    res.to_csv(RES / "d9_strategy.csv", index=False)
    tests.to_csv(RES / "d9_tests.csv", index=False)
    seeds = pd.DataFrame(seeds); seeds.to_csv(RES / "d9_seeds.csv", index=False)

    # 重訓與原結果的差距
    import torch
    rep = []
    for kind, model in JOBS:
        for y in TEST_YEARS:
            z = torch.load(ppath(kind, model, y), weights_only=False)
            rep.append(dict(strategy=f"{kind}:{model}", year=y, lr=z["lr"], max_abs_diff=z["max_abs_diff"],
                            mean_abs_diff=z["mean_abs_diff"], seeds_differing=z["seeds_differing"], device=z["device"]))
    rep = pd.DataFrame(rep); rep.to_csv(RES / "d9_retrain_check.csv", index=False)
    d6s = pd.read_csv(RES / "d6_strategy.csv").query("cap == 1.0 and period == '2008-01~2026-10'").set_index("name")

    # 事先固定的判定
    c1 = res[res.cap == 1.0]
    verdict = []
    for s in ["vol:QLSTM", "vol:QFWP", "e2e:QLSTM"]:
        base = float(c1[(c1.strategy == s) & (c1.level == "ideal") & (c1.shots == 0)].ratio.iloc[0])
        kind, model = s.split(":")
        orig = float(d6s.loc[original[(kind, model)], "teacher_sharpe"])
        for real in range(N_REAL):
            q = c1[(c1.strategy == s) & (c1.level == PRIMARY[0]) & (c1.shots == PRIMARY[1]) & (c1.mitigated == PRIMARY[2]) & (c1.real == real)]
            if q.empty:
                continue
            q = q.iloc[0]
            t = tests[(tests.strategy == s) & (tests.level == PRIMARY[0]) & (tests.shots == PRIMARY[1]) & (tests.mitigated == PRIMARY[2])
                      & (tests.real == real)].set_index("vs")
            a = bool(all(t.loc[e_, "p_ratio"] < 0.05 and t.loc[e_, "d_ratio"] > 0 for e_ in ("IVV", "SPY")))
            b = bool(q.ratio >= base - 0.05)
            c = bool(all(q.ratio_2010 > etf_ratio[(e_, A10)] for e_ in etf))
            verdict.append(dict(strategy=s, real=real, ratio_original=orig, ratio_noiseless_retrained=base, ratio_noisy=q.ratio,
                                drop=base - q.ratio, p_vs_IVV=t.loc["IVV", "p_ratio"], p_vs_SPY=t.loc["SPY", "p_ratio"],
                                ratio_2010=q.ratio_2010, a_beats_IVV_SPY=a, b_drop_le_0_05=b, c_above_top3_2010=c, passed=a and b and c))
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(RES / "d9_verdict.csv", index=False)
    meta = dict(etf_ratio={f"{t}_{a0}": v for (t, a0), v in etf_ratio.items()}, best_reg=best_reg, primary=list(PRIMARY),
                all_passed=bool(len(verdict) == 15 and verdict.passed.all()),
                circuits_per_day={"QLSTM (per model)": "5 time steps x 4 circuits = 20 (7 qubits, 6 CNOTs)",
                                  "QFWP (per model)": "1 (3 qubits, 2 CNOTs)", "models": len(SEEDS)})
    (RES / "d9_meta.json").write_text(json.dumps(meta, indent=1))

    # 摘要
    agg = res.groupby(["strategy", "cap", "level", "shots", "mitigated"]).agg(
        ratio=("ratio", "mean"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"), Sharpe=("Sharpe", "mean"),
        MaxDD=("MaxDD", "mean"), ratio_2010=("ratio_2010", "mean"), QLIKE=("QLIKE", "mean"), p_DM=("p_DM", "max"), n=("real", "count"))
    print(agg.round(4).to_string())
    pt = tests.groupby(["strategy", "level", "shots", "mitigated", "vs"]).agg(d_ratio=("d_ratio", "mean"), p_ratio_max=("p_ratio", "max"),
                                                                             d_sharpe=("d_sharpe", "mean"), p_sharpe_max=("p_sharpe", "max"))
    print(pt.round(4).to_string())
    print(seeds.groupby(["strategy", "level", "shots"])[["ratio", "Sharpe"]].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(rep.groupby("strategy")[["max_abs_diff", "mean_abs_diff", "seeds_differing"]].agg(["max", "mean"]).round(6).to_string())
    print({k: round(v, 3) for k, v in meta["etf_ratio"].items()}, "best_reg:", best_reg)
    print(verdict.round(4).to_string(index=False))
    print("ALL PASSED:", meta["all_passed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="vol:QLSTM、vol:QFWP 或 e2e:QLSTM")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--only", default=None, help="逗號分隔，例如 vol:QLSTM,e2e:QLSTM（只跑這些模型的推論）")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.train:
        train(*a.train.split(":"), a.device)
    elif a.check:
        check(a.device)
    elif a.infer:
        infer(a.device, a.only.split(",") if a.only else None)
    elif a.report:
        report(a.procs)
