"""實驗二：重現 Globecom Wkshps 2025《Quantum-Enhanced Forecasting for Deep Reinforcement Learning in
Algorithmic Trading》：QLSTM 預測 USD/TWD 5 日趨勢 → 機率加進狀態 → QA3C（量子 A3C）交易代理，只做多。

階段：
  python3 exp2.py qlstm  --seeds 3        # 訓練 QLSTM 分類器（論文：測試準確率約 71.5%）
  python3 exp2.py rl --agent QA3C --seeds 3 --episodes 1600
  python3 exp2.py rl --agent A3C  --seeds 3 --episodes 1600
  python3 exp2.py report
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ad import (Adam, Circuit, Linear, Param, RMSprop, T, concat, getitem, log_softmax, mean, mul, n_params,
                sum_, tanh, exp, square)
from common import DATA, RES, dump, load_ohlc
from models import LSTM, QLSTM

TRAIN_START, TRAIN_END = "2000-01-01", "2020-03-31"
TEST_START, TEST_END = "2020-04-01", "2025-04-30"
OUT = RES / "exp2"

# ---------------------------------------------------------------- 資料


def load_data():
    """論文：依時間排序、去重複、去掉收盤為空或 0 的列；MA5/10/20/60 與 20 日波動。
    資料清理：Yahoo 的 USD/TWD 有離譜的錯誤報價（例如 2011-10-26 單日 +1571%），和前後兩天的平均差 25% 以上的列
    視為錯誤報價拿掉（反覆檢查直到沒有）；論文的資料從 2000 年開始，應該不是 Yahoo（Yahoo 從 2004-03 才有）。"""
    df = load_ohlc("TWD_X").loc[TRAIN_START:TEST_END].copy()
    removed = []
    for _ in range(10):
        lc = np.log(df[["Open", "High", "Low", "Close"]])
        dev = (lc - (lc.shift(1) + lc.shift(-1)) / 2).abs()
        bad = (dev > 0.25).any(axis=1)
        if not bad.any():
            break
        removed += [str(d.date()) for d in df.index[bad]]
        df = df[~bad]
    c = df["Close"]
    for n in (5, 10, 20, 60):
        df[f"MA{n}"] = c.rolling(n).mean()
    df["VOL20"] = c.pct_change().rolling(20).std()
    # 「20 日相對波動」：20 日波動 ÷ 過去一年 20 日波動的平均（論文未定義，此處的定義）
    df["VOLREL"] = df["VOL20"] / df["VOL20"].rolling(252, min_periods=60).mean()
    df = df.dropna()
    return df, removed


SCALE = "train"  # train＝只用訓練期間估計 min–max；full＝用全期間（論文可能的做法，會偷看測試期間的範圍）
IN_COLS = ["Open", "High", "Low", "Close", "MA5", "MA10"]
SEQ = 4
THR = 0.012


def qlstm_dataset(df):
    """輸入：最近 4 天 × (O, H, L, C, MA5, MA10)，min–max 縮放到 [0,1]（只用訓練期間估計尺度）。
    標籤：5 日後上漲超過 1.2% → 1、下跌超過 1.2% → 0，中間的不用。"""
    is_tr = df.index <= TRAIN_END
    raw = df[IN_COLS].to_numpy()
    ref = raw[is_tr] if SCALE == "train" else raw
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    Z = (raw - lo) / (hi - lo)
    c = df["Close"].to_numpy()
    n = len(df)
    Xall = np.full((n, SEQ, len(IN_COLS)), np.nan)
    for t in range(SEQ - 1, n):
        Xall[t] = Z[t - SEQ + 1:t + 1]
    fwd = np.full(n, np.nan)
    fwd[:-5] = c[5:] / c[:-5] - 1
    lab = np.where(fwd > THR, 1, np.where(fwd < -THR, 0, -1))
    idx = np.arange(n)
    tr_last = np.where(is_tr)[0].max()
    tr = (idx >= SEQ - 1) & (idx + 5 <= tr_last) & (lab >= 0)  # 標籤要完全落在訓練期間內
    te = (~is_tr) & (idx + 5 < n) & (lab >= 0)
    return Xall, lab, tr, te


def SUFFIX():
    return "" if SCALE == "train" else "_fullscale"


def weighted_ce(logits, y, w):
    """torch.nn.CrossEntropyLoss(weight=w)：sum(w_y · −log p_y) / sum(w_y)。"""
    lp = log_softmax(logits)
    pick = getitem(lp, (np.arange(len(y)), y))
    wy = w[y]
    return mul(sum_(mul(pick, T(wy))), -1.0 / wy.sum())


def softmax_np(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train_qlstm(seed, kind="QLSTM", epochs=50, batch=32, lr=5e-3):
    """論文：hidden 2、dense-softmax 輸出、50 epochs、RMSprop lr 5e-3、weighted cross-entropy。
    batch size 論文未說明，此處用 32。"""
    df, removed = load_data()
    Xall, lab, tr, te = qlstm_dataset(df)
    rng = np.random.default_rng(seed)
    model = (QLSTM if kind == "QLSTM" else LSTM)(rng, n_in=len(IN_COLS), H=2, n_out=2)
    Xtr, ytr = Xall[tr], lab[tr]
    counts = np.bincount(ytr, minlength=2)
    w = len(ytr) / (2.0 * counts)
    opt = RMSprop(model.params(), lr)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), batch):
            j = perm[s:s + batch]
            opt.zero()
            loss = weighted_ce(model(Xtr[j]), ytr[j], w)
            loss.backward()
            opt.step()
            tot += float(loss.data) * len(j)
        pte = softmax_np(model(Xall[te]).data)
        ptr = softmax_np(model(Xtr).data)
        hist.append({"epoch": ep + 1, "loss": tot / len(Xtr),
                     "train_acc": float(np.mean(ptr.argmax(1) == ytr)),
                     "test_acc": float(np.mean(pte.argmax(1) == lab[te]))})
    # 每天都要有機率給 RL 用（凍結的模型、只用最近 4 天）
    valid = np.arange(len(df)) >= SEQ - 1
    probs = np.full((len(df), 2), 0.5)
    probs[valid] = softmax_np(model(Xall[valid]).data)
    pte = probs[te]
    yte = lab[te]
    res = {"kind": kind, "seed": seed, "n_params": n_params(model.params()), "epochs": epochs, "batch": batch,
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "train_counts": counts.tolist(),
           "test_counts": np.bincount(yte, minlength=2).tolist(), "removed_bad_ticks": removed,
           "test_acc": float(np.mean(pte.argmax(1) == yte)),
           "test_acc_majority": float(max(np.mean(yte), 1 - np.mean(yte))),
           "test_balanced_acc": float(np.mean([np.mean(pte[yte == k].argmax(1) == k) for k in (0, 1)])),
           "seconds": time.time() - t0, "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": df.index, "p_down": probs[:, 0], "p_up": probs[:, 1]}).to_csv(
        OUT / f"probs_{kind}{SUFFIX()}_s{seed}.csv", index=False)
    dump(res, OUT / f"forecaster_{kind}{SUFFIX()}_s{seed}.json")
    print(f"{kind}{SUFFIX()} seed {seed}: test acc {res['test_acc']:.4f} (majority {res['test_acc_majority']:.4f}, "
          f"balanced {res['test_balanced_acc']:.4f}), n_train {res['n_train']}, n_test {res['n_test']}, "
          f"{res['seconds']:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------- 交易環境（論文 Table I～III）


class Env:
    """向量化的交易環境：W 個 worker 在同一段價格上各自交易。
    動作：0 持有、1 現金足夠就買 1 單位、2 全部賣出；只做多、可加碼。
    獎勵：r = −0.02 + 進場 + 出場 + 持有，截斷在 [−15, 30]（pnl 以百分比為單位）。
    論文未說明的細節（本實作的設定）：初始資金 = 10 單位（10 × 期初價格）；無效動作罰 −0.1；
    虧損出場 −2 − 10×|pnl%|（論文表格寫 −2 − 10×pnl%，對虧損來說會變成獎勵，應是筆誤）。"""

    def __init__(self, df, probs, W):
        self.p = df["Close"].to_numpy()
        self.ma5, self.ma20, self.ma60 = (df[f"MA{n}"].to_numpy() for n in (5, 20, 60))
        self.volrel = df["VOLREL"].to_numpy()
        self.pb, self.pr = probs[:, 1], probs[:, 0]
        self.W = W
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = np.full(self.W, 10.0 * self.p[0])
        self.init = self.cash.copy()
        self.units = np.zeros(self.W)
        self.avg = np.zeros(self.W)
        self.trades = [[] for _ in range(self.W)]
        self.buys = np.zeros(self.W, dtype=int)
        self.values = [self.cash.copy()]
        return self.state()

    def state(self):
        t, p = self.t, self.p[self.t]
        hold = self.units > 0
        value = self.cash + self.units * p
        upnl = np.where(hold, (p / np.where(hold, self.avg, 1) - 1) * 100, 0.0)
        apr = np.where(hold, p / np.where(hold, self.avg, 1), 1.0)
        common = np.array([self.pb[t], self.pr[t]])
        tech = np.array([(p / self.ma20[t] - 1) * 100, (p / self.ma60[t] - 1) * 100,
                         (self.ma5[t] / self.ma20[t] - 1) * 100, self.volrel[t]])
        return np.column_stack([np.tile(common, (self.W, 1)), self.cash / value, self.units * p / value, upnl, apr,
                                np.tile(tech, (self.W, 1))])

    def step(self, a):
        t, p = self.t, self.p[self.t]
        r = np.full(self.W, -0.02)
        buy = a == 1
        ok = buy & (self.cash >= p)
        r[buy & ~ok] -= 0.1
        r[ok] += np.where(p > self.ma20[t], 0.5, -2.0)
        self.avg[ok] = (self.avg[ok] * self.units[ok] + p) / (self.units[ok] + 1)
        self.units[ok] += 1
        self.cash[ok] -= p
        self.buys += ok
        sell = a == 2
        oks = sell & (self.units > 0)
        r[sell & ~oks] -= 0.1
        for w in np.where(oks)[0]:
            pnl = (p / self.avg[w] - 1) * 100
            self.trades[w].append(pnl)
            r[w] += (10 + 50 * pnl) if pnl > 0 else (-2 - 10 * abs(pnl))
            self.cash[w] += self.units[w] * p
            self.units[w] = 0
            self.avg[w] = 0
        self.t += 1
        p2 = self.p[self.t]
        hold = self.units > 0
        upnl = np.where(hold, (p2 / np.where(hold, self.avg, 1) - 1) * 100, 0.0)
        r += np.where(upnl < 0, -5 * upnl ** 2, 0.0)
        r = np.clip(r, -15, 30)
        self.values.append(self.cash + self.units * p2)
        done = self.t >= len(self.p) - 1
        return r, done


# ---------------------------------------------------------------- 代理人


class QNet:
    """QA3C：狀態 → tanh(Linear 10→8) → 8-qubit VQC → Linear 頭。actor、critic 各一個 VQC。
    論文：244 個參數（量子 32、古典 212）。212 = 2×Linear(10→8) 176 + Linear(8→3) 27 + Linear(8→1) 9，
    所以是 actor、critic 各自一組投影 + VQC，每個 VQC 16 個量子參數。
    VQC（論文細節不完整，本實作採用符合「8 qubits、深度 2、angle encoding、可訓練 Ry 與 Rz、CNOT」）：
    |+> 起始 → Ry(x_i) 編碼 → [CNOT 環 → Rz(w1_i)] → [CNOT 環 → Ry(w2_i)] → 量測 8 條線的 <Z>。"""

    def __init__(self, rng, s_dim=10, a_dim=3, nq=8):
        ops, k = [], 0
        for q in range(nq):
            ops.append(("ry", q, k)); k += 1
        for layer, rot in ((0, "rz"), (1, "ry")):
            for q in range(nq):
                ops.append(("cnot", q, (q + 1) % nq))
            for q in range(nq):
                ops.append((rot, q, k)); k += 1
        self.circ = Circuit(nq, ops, k, measure=range(nq), init="plus")
        self.nq = nq
        self.pa, self.pc = Linear(rng, s_dim, nq), Linear(rng, s_dim, nq)
        self.wa = Param(rng.uniform(0, 2 * np.pi, size=k - nq))
        self.wc = Param(rng.uniform(0, 2 * np.pi, size=k - nq))
        self.ha, self.hc = Linear(rng, nq, a_dim), Linear(rng, nq, 1)

    def params(self):
        return self.pa.params() + self.pc.params() + [self.wa, self.wc] + self.ha.params() + self.hc.params()

    def _vqc(self, proj, w, s):
        from ad import tile_rows
        x = tanh(proj(T(s)))
        return self.circ(concat([x, tile_rows(w, s.shape[0])], axis=-1))

    def logits(self, s):
        return self.ha(self._vqc(self.pa, self.wa, s))

    def value(self, s):
        return self.hc(self._vqc(self.pc, self.wc, s))


class CNet:
    """古典 A3C（MorvanZhou/pytorch-A3C 的結構）：actor 10→128→3、critic 10→128→1，tanh。
    參數 = 3332，和論文的古典 A3C 一樣。"""

    def __init__(self, rng, s_dim=10, a_dim=3, h=128):
        self.pi1, self.pi2 = Linear(rng, s_dim, h), Linear(rng, h, a_dim)
        self.v1, self.v2 = Linear(rng, s_dim, h), Linear(rng, h, 1)

    def params(self):
        return self.pi1.params() + self.pi2.params() + self.v1.params() + self.v2.params()

    def logits(self, s):
        return self.pi2(tanh(self.pi1(T(s))))

    def value(self, s):
        return self.v2(tanh(self.v1(T(s))))


AGENTS = {"QA3C": QNet, "A3C": CNet}


def a3c_loss(net, S, A, Vt, beta):
    """MorvanZhou A3C 的損失＋熵正則：L = td² − log π(a|s)·td(停止梯度) − β·H(π)。"""
    v = net.value(S)
    td = T(Vt[:, None]) - v
    lp = log_softmax(net.logits(S))
    pick = getitem(lp, (np.arange(len(A)), A))
    td_d = td.data[:, 0]
    ent = mul(sum_(mul(exp(lp), lp), axis=-1), -1.0)
    c_loss = mean(getitem(square(td), (slice(None), 0)))
    a_loss = mean(mul(pick, T(-td_d)))
    return c_loss + a_loss - mul(mean(ent), beta)


def train_rl(agent, seed, episodes, probs_kind="QLSTM", probs_seed=0, W=8, n_step=30, gamma=0.995, beta=0.05,
             lr=1e-5, log_every=8):
    """論文：8 個 worker、rollout 長度 30、Shared Adam lr 1e-5、γ=0.995、β=0.05。
    實作方式：A3C 原本是 8 個行程非同步更新；這裡 8 個 worker 同步收集資料（快很多），
    但仍然「每個 worker 各自計算梯度、依序更新共享參數」——每收集 30 步就有 8 次更新，和 A3C 相同。
    一個 episode = 跑完整個訓練期間一次（論文沒定義）；episodes 計的是全部 worker 加總（同 MorvanZhou 的 MAX_EP）。"""
    df, _ = load_data()
    pr = pd.read_csv(OUT / f"probs_{probs_kind}_s{probs_seed}.csv", parse_dates=["date"]).set_index("date")
    probs = pr.loc[df.index, ["p_down", "p_up"]].to_numpy()
    tr_mask = df.index <= TRAIN_END
    dtr, ptr = df[tr_mask], probs[tr_mask]
    dte, pte = df[~tr_mask], probs[~tr_mask]
    rng = np.random.default_rng(seed)
    net = AGENTS[agent](rng)
    opt = Adam(net.params(), lr, b1=0.92, b2=0.999)  # MorvanZhou SharedAdam 的 betas
    env = Env(dtr, ptr, W)
    log = []
    t0 = time.time()
    rounds = int(np.ceil(episodes / W))
    for rd in range(rounds):
        s = env.reset()
        ep_r = np.zeros(W)
        bs, ba, br = [], [], []
        done = False
        while not done:
            pi = softmax_np(net.logits(s).data)
            a = (rng.random((W, 1)) < np.cumsum(pi, axis=1)).argmax(axis=1)
            r, done = env.step(a)
            ep_r += r
            bs.append(s); ba.append(a); br.append(r)
            s = env.state() if not done else s
            if len(bs) == n_step or done:
                S = np.stack(bs, 1)  # (W, n, 10)
                A = np.stack(ba, 1)
                R = np.stack(br, 1)
                for w in range(W):
                    v_next = 0.0 if done else float(net.value(s[w:w + 1]).data[0, 0])
                    vt = np.empty(len(bs))
                    for i in range(len(bs) - 1, -1, -1):
                        v_next = R[w, i] + gamma * v_next
                        vt[i] = v_next
                    opt.zero()
                    a3c_loss(net, S[w], A[w], vt, beta).backward()
                    opt.step()
                bs, ba, br = [], [], []
        log.append({"episode": (rd + 1) * W, "mean_reward": float(ep_r.mean()),
                    "mean_return_pct": float(100 * np.mean(env.values[-1] / env.init - 1)),
                    "minutes": (time.time() - t0) / 60})
        if rd % max(1, log_every // W) == 0 or rd == rounds - 1:
            print(f"{agent} s{seed} ep {log[-1]['episode']}: reward {log[-1]['mean_reward']:.1f}, "
                  f"train return {log[-1]['mean_return_pct']:.2f}%, {log[-1]['minutes']:.1f} min", flush=True)
    res = {"agent": agent, "seed": seed, "episodes": rounds * W, "n_params": n_params(net.params()),
           "probs": f"{probs_kind}_s{probs_seed}", "train_minutes": (time.time() - t0) / 60, "log": log,
           "test": evaluate(net, dte, pte), "train_eval": evaluate(net, dtr, ptr)}
    dump(res, OUT / f"rl_{agent}_s{seed}.json")
    tst = res["test"]
    print(f"{agent} s{seed} TEST: return {tst['total_return_pct']:.2f}%, MDD {tst['max_drawdown_pct']:.2f}%, "
          f"trades {tst['trades']}, win {tst['win_rate_pct']}%", flush=True)
    return res


def evaluate(net, d, p):
    """測試：每一步選機率最高的動作（貪婪）。期末以收盤價計算資產（不額外交易）。"""
    env = Env(d, p, 1)
    s = env.reset()
    done = False
    acts, held = [], []
    while not done:
        a = net.logits(s).data.argmax(axis=1)
        acts.append(int(a[0]))
        _, done = env.step(a)
        held.append(bool(env.units[0] > 0))
        if not done:
            s = env.state()
    v = np.array([x[0] for x in env.values])
    peak = np.maximum.accumulate(v)
    tr = np.array(env.trades[0])
    dr = np.diff(v) / v[:-1]
    return {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "days": int(len(d)),
            "total_return_pct": float(100 * (v[-1] / v[0] - 1)),
            "max_drawdown_pct": float(100 * np.max(1 - v / peak)),
            "trades": int(len(tr)), "buys": int(env.buys[0]),
            "win_rate_pct": float(100 * np.mean(tr > 0)) if len(tr) else None,
            "best_trade_pct": float(tr.max()) if len(tr) else None,
            "worst_trade_pct": float(tr.min()) if len(tr) else None,
            "sum_trade_pct": float(tr.sum()) if len(tr) else 0.0,
            "time_in_market_pct": float(100 * np.mean(held)),
            "sharpe_ann": float(np.sqrt(252) * dr.mean() / dr.std()) if dr.std() > 0 else None,
            "action_counts": np.bincount(acts, minlength=3).tolist(),
            "buy_hold_pct": float(100 * (d["Close"].iloc[-1] / d["Close"].iloc[0] - 1))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["qlstm", "rl", "report"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--agent", default="QA3C")
    ap.add_argument("--episodes", type=int, default=1600)
    ap.add_argument("--kind", default="QLSTM")
    ap.add_argument("--scale", default="train", choices=["train", "full"])
    a = ap.parse_args()
    SCALE = a.scale
    if a.stage == "qlstm":
        for s in range(a.seed0, a.seed0 + a.seeds):
            train_qlstm(s, a.kind)
    elif a.stage == "rl":
        for s in range(a.seed0, a.seed0 + a.seeds):
            train_rl(a.agent, s, a.episodes)
