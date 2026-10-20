"""實驗二（PennyLane 版）：QLSTM 預測 USD/TWD 5 日趨勢 → QA3C／A3C 交易代理。
資料、交易環境、測試方式和 numpy 版共用（exp2.load_data、exp2.Env、exp2.evaluate）。
A3C 照論文用真正的非同步多行程（MorvanZhou/pytorch-A3C 的寫法）：8 個 worker 各自一個行程，共用 Shared Adam。

用法：
  python pl_exp2.py qlstm --seeds 3                       # QLSTM 預測器（每個種子各自存機率）
  python pl_exp2.py rl --agent A3C  --seeds 3             # 每個種子用 8 個 CPU 核心
  python pl_exp2.py rl --agent QA3C --seeds 1 --seed0 0   # 16 核心可以同時開兩個（--seed0 0 和 --seed0 1）
  python report2.py exp2_pl
注意：A3C 在 Linux／macOS 用 fork 建立 worker；Windows 沒有 fork，改用 spawn，在 worker 內重建網路並接上共享記憶體。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import types

import numpy as np
import pandas as pd

import exp2 as E
from common import RES, dump

OUT = RES / "exp2_pl"


# ---------------------------------------------------------------- QLSTM 預測器


def train_qlstm(seed, kind="QLSTM", epochs=50, batch=32, lr=5e-3, device="lightning.qubit", diff="adjoint"):
    """論文：hidden 2、dense-softmax、50 epochs、RMSprop lr 5e-3、weighted cross-entropy（batch 32 為本實作的設定）。"""
    import torch
    import pl_models as P
    P.configure(device, diff)
    torch.manual_seed(seed)
    df, removed = E.load_data()
    Xall, lab, tr, te = E.qlstm_dataset(df)
    model = P.QLSTM(n_in=len(E.IN_COLS), H=2, n_out=2) if kind == "QLSTM" else P.LSTM(len(E.IN_COLS), 2, 2)
    Xtr, ytr = torch.tensor(Xall[tr]), torch.tensor(lab[tr])
    counts = np.bincount(lab[tr], minlength=2)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (2.0 * counts)))
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Xte, yte = torch.tensor(Xall[te]), lab[te]
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), batch):
            j = torch.as_tensor(perm[s:s + batch])
            opt.zero_grad()
            loss = lossf(model(Xtr[j]), ytr[j])
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        with torch.no_grad():
            acc = float((model(Xte).argmax(1).numpy() == yte).mean())
        hist.append({"epoch": ep + 1, "loss": tot / len(Xtr), "test_acc": acc})
        print(f"  {kind} s{seed} epoch {ep + 1}/{epochs}: loss {hist[-1]['loss']:.4f}, test acc {acc:.4f}", flush=True)
    valid = np.arange(len(df)) >= E.SEQ - 1
    probs = np.full((len(df), 2), 0.5)
    with torch.no_grad():
        probs[valid] = torch.softmax(model(torch.tensor(Xall[valid])), dim=1).numpy()
    pte = probs[te]
    res = {"kind": kind, "seed": seed, "backend": f"{device}/{diff}", "n_params": P.n_params(model),
           "epochs": epochs, "batch": batch, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "train_counts": counts.tolist(), "test_counts": np.bincount(yte, minlength=2).tolist(),
           "removed_bad_ticks": removed, "test_acc": float(np.mean(pte.argmax(1) == yte)),
           "test_acc_majority": float(max(np.mean(yte), 1 - np.mean(yte))),
           "test_balanced_acc": float(np.mean([np.mean(pte[yte == k].argmax(1) == k) for k in (0, 1)])),
           "seconds": time.time() - t0, "history": hist}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": df.index, "p_down": probs[:, 0], "p_up": probs[:, 1]}).to_csv(
        OUT / f"probs_{kind}_s{seed}.csv", index=False)
    dump(res, OUT / f"forecaster_{kind}_s{seed}.json")
    print(f"{kind} seed {seed}: test acc {res['test_acc']:.4f}（論文約 0.715；多數類別 {res['test_acc_majority']:.4f}）",
          flush=True)


# ---------------------------------------------------------------- A3C（非同步、多行程）


def shared_adam(params, lr, betas=(0.92, 0.999), eps=1e-8):
    """MorvanZhou 的 SharedAdam：Adam 的狀態放在共享記憶體，所有 worker 共用。"""
    import torch
    opt = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            st["step"] = torch.zeros((), dtype=torch.float32).share_memory_()
            st["exp_avg"] = torch.zeros_like(p.data).share_memory_()
            st["exp_avg_sq"] = torch.zeros_like(p.data).share_memory_()
    return opt


def a3c_loss(net, S, A, Vt, beta):
    """L = td² − log π(a|s)·td（停止梯度）− β·H(π)，取平均。"""
    import torch
    v = net.v(S)[:, 0]
    td = Vt - v
    dist = torch.distributions.Categorical(logits=net.pi(S))
    return (td.pow(2) - dist.log_prob(A) * td.detach() - beta * dist.entropy()).mean()


def _shared_tensors(gnet, opt):
    """spawn（Windows）用：QNode 無法 pickle，只把共享記憶體裡的參數和 Adam 狀態傳給 worker。"""
    ps = list(gnet.parameters())
    return [p.data for p in ps], [(opt.state[p]["step"], opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]) for p in ps]


def _attach(cfg, shared, lr, betas=(0.92, 0.999), eps=1e-8):
    """在 worker 內重建全域網路與 Shared Adam，參數和 Adam 狀態都指向主行程的共享記憶體。"""
    import torch
    import pl_models as P
    params, states = shared
    gnet = P.AGENTS[cfg["agent"]]()
    for p, d in zip(gnet.parameters(), params):
        p.data = d
    opt = torch.optim.Adam(gnet.parameters(), lr=lr, betas=betas, eps=eps)
    for p, (step, m, v) in zip(gnet.parameters(), states):
        opt.state[p] = {"step": step, "exp_avg": m, "exp_avg_sq": v}
    return gnet, opt


def worker(wid, gnet, opt, gep, rq, cfg, dtr, ptr):
    import torch
    import pl_models as P
    torch.set_num_threads(1)
    P.configure(cfg["device"], cfg["diff"])
    if gnet is None:  # spawn：opt 傳進來的是 _shared_tensors() 的結果
        gnet, opt = _attach(cfg, opt, cfg["lr"])
    torch.manual_seed(cfg["seed"] * 1000 + wid)
    lnet = P.AGENTS[cfg["agent"]]()
    lnet.load_state_dict(gnet.state_dict())
    env = E.Env(dtr, ptr, 1)
    while True:
        with gep.get_lock():
            if gep.value >= cfg["episodes"]:
                break
            gep.value += 1
            ep_id = gep.value
        s = env.reset()
        bs, ba, br = [], [], []
        ep_r, t, done = 0.0, 0, False
        while not done:
            with torch.no_grad():
                a = torch.distributions.Categorical(logits=lnet.pi(torch.from_numpy(s))).sample().numpy()
            r, done = env.step(a)
            ep_r += float(r[0])
            bs.append(s[0]); ba.append(int(a[0])); br.append(float(r[0]))
            if not done:
                s = env.state()
            t += 1
            if t % cfg["n_step"] == 0 or done:
                with torch.no_grad():
                    v_next = 0.0 if done else float(lnet.v(torch.from_numpy(s))[0, 0])
                vt = np.empty(len(br))
                for i in range(len(br) - 1, -1, -1):
                    v_next = br[i] + cfg["gamma"] * v_next
                    vt[i] = v_next
                lnet.zero_grad()
                loss = a3c_loss(lnet, torch.tensor(np.array(bs)), torch.tensor(ba), torch.tensor(vt), cfg["beta"])
                loss.backward()
                opt.zero_grad()
                for lp, gp in zip(lnet.parameters(), gnet.parameters()):
                    gp._grad = lp.grad.clone()
                opt.step()
                lnet.load_state_dict(gnet.state_dict())
                bs, ba, br = [], [], []
        rq.put((ep_id, ep_r, float(100 * (env.values[-1][0] / env.init[0] - 1))))
    rq.put(None)


class _Adapter:
    """讓 exp2.evaluate()（numpy 版的測試函式）可以直接用 torch 模型。"""

    def __init__(self, net):
        self.net = net

    def logits(self, s):
        import torch
        with torch.no_grad():
            return types.SimpleNamespace(data=self.net.pi(torch.from_numpy(s)).numpy())


def train_rl(agent, seed, episodes, workers=8, n_step=30, gamma=0.995, beta=0.05, lr=1e-5,
             device="lightning.qubit", diff="adjoint", probs_seed=0):
    """論文：8 workers、rollout 30、Shared Adam lr 1e-5、γ 0.995、β 0.05。episodes 為所有 worker 加總。"""
    import torch
    import pl_models as P
    P.configure(device, diff)
    torch.manual_seed(seed)
    df, _ = E.load_data()
    pr = pd.read_csv(OUT / f"probs_QLSTM_s{probs_seed}.csv", parse_dates=["date"]).set_index("date")
    probs = pr.loc[df.index, ["p_down", "p_up"]].to_numpy()
    m = df.index <= E.TRAIN_END
    dtr, ptr, dte, pte = df[m], probs[m], df[~m], probs[~m]
    gnet = P.AGENTS[agent]()
    gnet.share_memory()
    opt = shared_adam(gnet.parameters(), lr)
    fork = "fork" in mp.get_all_start_methods()
    ctx = mp.get_context("fork" if fork else "spawn")
    gep, rq = ctx.Value("i", 0), ctx.Queue()
    cfg = {"agent": agent, "seed": seed, "episodes": episodes, "n_step": n_step, "gamma": gamma, "beta": beta,
           "device": device, "diff": diff, "lr": lr}
    wargs = (gnet, opt) if fork else (None, _shared_tensors(gnet, opt))
    procs = [ctx.Process(target=worker, args=(w, *wargs, gep, rq, cfg, dtr, ptr)) for w in range(workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    log, finished = [], 0
    while finished < workers:
        item = rq.get()
        if item is None:
            finished += 1
            continue
        log.append({"episode": item[0], "reward": item[1], "train_return_pct": item[2], "minutes": (time.time() - t0) / 60})
        if len(log) % 8 == 0:
            recent = log[-8:]
            print(f"{agent} s{seed} 已完成 {len(log)}/{episodes} episodes：平均獎勵 "
                  f"{np.mean([x['reward'] for x in recent]):.1f}、訓練期報酬 "
                  f"{np.mean([x['train_return_pct'] for x in recent]):.2f}%、{log[-1]['minutes']:.1f} 分鐘", flush=True)
    for p in procs:
        p.join()
    res = {"agent": agent, "seed": seed, "backend": f"{device}/{diff}", "episodes": episodes, "workers": workers,
           "n_params": P.n_params(gnet), "probs": f"QLSTM_s{probs_seed}", "train_minutes": (time.time() - t0) / 60,
           "log": log, "test": E.evaluate(_Adapter(gnet), dte, pte), "train_eval": E.evaluate(_Adapter(gnet), dtr, ptr)}
    OUT.mkdir(parents=True, exist_ok=True)
    dump(res, OUT / f"rl_{agent}_s{seed}.json")
    t = res["test"]
    print(f"{agent} s{seed} 測試：總報酬 {t['total_return_pct']:.2f}%、最大回撤 {t['max_drawdown_pct']:.2f}%、"
          f"交易 {t['trades']} 次、勝率 {t['win_rate_pct']}%（論文 QA3C：11.87%、0.92%、231、56.71%）", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["qlstm", "rl"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--agent", default="QA3C", choices=["QA3C", "A3C"])
    ap.add_argument("--kind", default="QLSTM", choices=["QLSTM", "LSTM"])
    ap.add_argument("--episodes", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="lightning.qubit")
    ap.add_argument("--diff", default="adjoint")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for s in range(a.seed0, a.seed0 + a.seeds):
        if a.stage == "qlstm":
            train_qlstm(s, a.kind, device=a.device, diff=a.diff)
        else:
            train_rl(a.agent, s, a.episodes, a.workers, device=a.device, diff=a.diff)
