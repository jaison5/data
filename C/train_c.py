"""方向 C：訓練一個交易代理（非同步 A3C，8 個 worker），在驗證年挑模型，在測試年用 3 種成本評估。

和論文相同：8 workers、rollout 30、Shared Adam lr 1e-5（betas 0.92/0.999）、γ 0.995、熵係數 0.05、1,600 個 episode。
不同：動作遮罩（策略只在合法動作中抽樣，損失也用遮罩後的分布；選動作用 NumPy 計算以減少每步開銷）、交易成本、可選獎勵、walk-forward、
每 100 個 episode 在驗證年評估一次，保留驗證 Sharpe 最高的模型（避免只看最後一刻）。
代理人：QA3C（fastsim，244 參數）、A3C_small（古典 h=9，238 參數）、A3C（古典 h=128，3,332 參數，論文的基準）。

用法：python train_c.py --fold 2020 --agent QA3C --reward logret --probs qlstm --seed 0
輸出：results/rl/<agent>_<reward>_<probs>_y<fold>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "B"))
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "qfin-main"))  # 只為了 QA3C_PL（PennyLane 版 QA3C）

import numpy as np
import pandas as pd

TRAIN_COST = 0.0005
EVAL_COSTS = [0.0, 0.0005, 0.001]


def load_probs(kind, test_year, df):
    if kind == "none":
        return np.full((len(df), 2), 0.5)
    if kind == "oracle":
        import cdata
        return cdata.probs_oracle(df)
    pr = pd.read_csv(HERE / "results" / "forecaster" / f"probs_qlstm_y{test_year}.csv", parse_dates=["date"])
    return pr.set_index("date").loc[df.index, ["p_down", "p_up"]].to_numpy()


def make_agent(name):
    """QA3C／A3C／A3C_small 用 fastsim；QA3C_PL 是 PennyLane（lightning.qubit、adjoint）版的同一個 QA3C，
    同一個種子的初始參數和 fastsim 版完全相同，只拿來做交叉驗證。"""
    if name == "QA3C_PL":
        import pl_models as P
        P.configure("lightning.qubit", "adjoint", False)
        return P.QNet()
    import fastsim as F
    return F.AGENTS[name]()


def masked_logits(logits, mask):
    import torch
    return torch.where(mask, logits, torch.full_like(logits, -1e9))


def act_fn(net):
    """選動作用的前向：QA3C 用快取矩陣的快速版本（數值和 pi 相同）。"""
    return net.pi_fast if hasattr(net, "pi_fast") else net.pi


def np_policy(net):
    """選動作用的 NumPy 版前向（數值和 net.pi 相同、不含梯度）。每一步只做幾個小陣列運算，
    避免 PyTorch 每次呼叫約 0.3～0.5 ms 的固定開銷；訓練的損失與反向傳播仍用 PyTorch。"""
    g = lambda t: t.detach().cpu().numpy()
    if hasattr(net, "pi_fast"):  # QA3C：tanh(Linear) → 乘積態 → 快取的 256×256 矩陣 → |ψ|² → <Z> → Linear
        net.refresh()
        Wp, bp, Wh, bh = g(net.pa.weight), g(net.pa.bias), g(net.ha.weight), g(net.ha.bias)
        M, signs, r2 = g(net._M), g(net.signs), np.sqrt(2.0)

        def f(s):
            x = np.tanh(Wp @ s + bp)
            c, sn = np.cos(x / 2), np.sin(x / 2)
            v0, v1 = (c - sn) / r2, (sn + c) / r2
            psi = np.array([v0[0], v1[0]])
            for w in range(1, len(x)):
                psi = np.outer(psi, (v0[w], v1[w])).ravel()
            out = psi @ M
            return Wh @ ((out.real ** 2 + out.imag ** 2) @ signs) + bh
        return f
    if not hasattr(net, "pi1"):  # QA3C_PL：選動作也用 PennyLane 計算（交叉驗證時，後端是唯一的差別）
        import torch

        def f_pl(s):
            with torch.no_grad():
                return net.pi(torch.from_numpy(s[None]))[0].numpy()
        return f_pl
    W1, b1, W2, b2 = g(net.pi1.weight), g(net.pi1.bias), g(net.pi2.weight), g(net.pi2.bias)
    return lambda s: W2 @ np.tanh(W1 @ s + b1) + b2


def sample_masked(lg, m, rng):
    """只在合法動作中依 softmax 機率抽樣（遮罩掉的動作機率為 0）。"""
    lg = np.where(m, lg, -np.inf)
    p = np.exp(lg - lg.max())
    cdf = np.cumsum(p / p.sum())
    a = int(np.searchsorted(cdf, rng.random(), side="right"))
    return a if a < len(m) and m[a] else int(np.flatnonzero(m)[-1])


def worker(wid, shared, gep, rq, cfg, dtr, ptr):
    import torch
    import fastsim as F
    from env_c import EnvC
    torch.set_num_threads(1)
    params, states = shared
    gnet = make_agent(cfg["agent"])
    for p, d in zip(gnet.parameters(), params):
        p.data = d
    opt = torch.optim.Adam(gnet.parameters(), lr=cfg["lr"], betas=(0.92, 0.999), eps=1e-8)
    for p, (st, m, v) in zip(gnet.parameters(), states):
        opt.state[p] = {"step": st, "exp_avg": m, "exp_avg_sq": v}
    torch.manual_seed(cfg["seed"] * 1000 + wid)
    rng = np.random.default_rng(cfg["seed"] * 1000 + wid)
    lnet = make_agent(cfg["agent"])
    lnet.load_state_dict(gnet.state_dict())
    pol = np_policy(lnet)
    env = EnvC(dtr, ptr, cfg["cost"], cfg["reward"])
    while True:
        with gep.get_lock():
            if gep.value >= cfg["episodes"]:
                break
            gep.value += 1
            ep_id = gep.value
        s = env.reset()
        bs, ba, br, bm = [], [], [], []
        ep_r, t, done = 0.0, 0, False
        while not done:
            m = env.mask()
            a = sample_masked(pol(s[0]), m, rng)
            r, done = env.step(a)
            ep_r += r
            bs.append(s[0]); ba.append(a); br.append(r); bm.append(m)
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
                S, A = torch.tensor(np.array(bs)), torch.tensor(ba)
                Vt, Mk = torch.tensor(vt), torch.tensor(np.array(bm))
                lnet.zero_grad()
                td = Vt - lnet.v(S)[:, 0]
                dist = torch.distributions.Categorical(logits=masked_logits(lnet.pi(S), Mk))
                loss = (td.pow(2) - dist.log_prob(A) * td.detach() - cfg["beta"] * dist.entropy()).mean()
                loss.backward()
                opt.zero_grad()
                for lp, gp in zip(lnet.parameters(), gnet.parameters()):
                    gp._grad = lp.grad.clone()
                opt.step()
                lnet.load_state_dict(gnet.state_dict())
                pol = np_policy(lnet)
                bs, ba, br, bm = [], [], [], []
        rq.put((ep_id, ep_r, float(100 * (env.values[-1] / env.init - 1))))
    rq.put(None)


def greedy(net):
    import torch
    refresh = getattr(net, "refresh", None)
    if refresh:
        refresh()
    act = act_fn(net)

    def pol(s, m):
        with torch.no_grad():
            return int(masked_logits(act(torch.from_numpy(s)), torch.from_numpy(m)[None]).argmax(1)[0])
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--agent", default="QA3C", choices=["QA3C", "A3C", "A3C_small", "QA3C_PL"])
    ap.add_argument("--reward", default="logret", choices=["logret", "paper"])
    ap.add_argument("--probs", default="qlstm", choices=["qlstm", "none", "oracle"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=100)
    a = ap.parse_args()
    import torch
    import cdata
    import fastsim as F
    from env_c import evaluate
    torch.set_num_threads(1)
    out = HERE / "results" / "rl" / f"{a.agent}_{a.reward}_{a.probs}_y{a.fold}_s{a.seed}.json"
    if out.exists():
        return
    df = cdata.load_data()
    probs = load_probs(a.probs, a.fold, df)
    tr, va, te = cdata.split(df, a.fold)
    dtr, ptr, dva, pva, dte, pte = df[tr], probs[tr], df[va], probs[va], df[te], probs[te]
    torch.manual_seed(a.seed)
    gnet = make_agent(a.agent)
    gnet.share_memory()
    opt = torch.optim.Adam(gnet.parameters(), lr=1e-5, betas=(0.92, 0.999), eps=1e-8)
    for p in gnet.parameters():
        opt.state[p] = {"step": torch.zeros((), dtype=torch.float32).share_memory_(),
                        "exp_avg": torch.zeros_like(p.data).share_memory_(),
                        "exp_avg_sq": torch.zeros_like(p.data).share_memory_()}
    shared = ([p.data for p in gnet.parameters()],
              [(opt.state[p]["step"], opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]) for p in gnet.parameters()])
    cfg = {"agent": a.agent, "seed": a.seed, "episodes": a.episodes, "n_step": 30, "gamma": 0.995, "beta": 0.05,
           "lr": 1e-5, "cost": TRAIN_COST, "reward": a.reward}
    ctx = mp.get_context("spawn")
    gep, rq = ctx.Value("i", 0), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, shared, gep, rq, cfg, dtr, ptr)) for w in range(a.workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    log, val_hist, finished = [], [], 0
    best = {"sharpe": -np.inf, "state": None, "episode": 0}
    snap = lambda: {k: v.detach().clone() for k, v in gnet.state_dict().items()}

    def check_val(n_done):
        st = snap()
        net = make_agent(a.agent); net.load_state_dict(st)
        m = evaluate(greedy(net), dva, pva, TRAIN_COST)
        val_hist.append({"episode": n_done, "val_sharpe": m["sharpe_ann"], "val_return_pct": m["total_return_pct"],
                         "val_trades": m["trades"]})
        if m["sharpe_ann"] >= best["sharpe"]:
            best.update(sharpe=m["sharpe_ann"], state=st, episode=n_done)

    while finished < a.workers:
        item = rq.get()
        if item is None:
            finished += 1
            continue
        log.append({"episode": item[0], "reward": item[1], "train_return_pct": item[2], "minutes": (time.time() - t0) / 60})
        if len(log) % a.eval_every == 0:
            check_val(len(log))
            print(f"{a.agent} {a.reward} {a.probs} y{a.fold} s{a.seed}: {len(log)}/{a.episodes} episodes，"
                  f"近 100 次平均獎勵 {np.mean([x['reward'] for x in log[-100:]]):.1f}，驗證 Sharpe {val_hist[-1]['val_sharpe']:.2f}"
                  f"（交易 {val_hist[-1]['val_trades']} 次），{log[-1]['minutes']:.1f} 分鐘", flush=True)
    for p in procs:
        p.join()
    final_state = snap()
    if len(log) % a.eval_every:
        check_val(len(log))
    res = {"agent": a.agent, "reward": a.reward, "probs": a.probs, "test_year": a.fold, "seed": a.seed,
           "n_params": F.n_params(gnet), "episodes": len(log), "train_cost": TRAIN_COST,
           "best_val_episode": best["episode"], "best_val_sharpe": best["sharpe"], "minutes": (time.time() - t0) / 60,
           "train_days": int(tr.sum()), "val_days": int(va.sum()), "test_days": int(te.sum()),
           "log": log, "val_hist": val_hist, "test": {}, "test_final": {}}
    for tag, st in (("test", best["state"]), ("test_final", final_state)):
        net = make_agent(a.agent); net.load_state_dict(st)
        for c in EVAL_COSTS:
            res[tag][str(c)] = evaluate(greedy(net), dte, pte, c)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    t = res["test"][str(TRAIN_COST)]
    print(f"完成 {out.name}：測試年報酬 {t['total_return_pct']:.2f}%（成本 0.05%），Sharpe {t['sharpe_ann']:.2f}，"
          f"交易 {t['trades']} 次，持倉 {t['time_in_market_pct']:.0f}%，{res['minutes']:.1f} 分鐘", flush=True)


if __name__ == "__main__":
    main()
