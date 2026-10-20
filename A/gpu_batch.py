"""方向 A 的 GPU 批次執行器：同一個設定（匯率、資料、fold、模型）的多個種子與學習率，疊成一個張量在 GPU 上一起訓練。

為什麼：這些模型很小（3～8 qubits），一次只訓練一個時，GPU 和 CPU 的時間都花在固定開銷上。把 G 個模型的參數疊成
(G, …) 的張量，一次前向／反向就同時更新 G 個模型，每個模型仍然：
  - 用自己的種子初始化（先用 torch.manual_seed(seed) 建出和 wf.make_model 完全相同的模型，再把參數疊起來）
  - 用自己的資料順序（np.random.default_rng(seed + 1000)，同 wf.train_eval）
  - 用自己的 Adam 狀態與學習率（Adam 是逐元素的，疊起來計算和分開計算相同）
  - 各自做 early stopping（停下的模型不再更新紀錄；結果和單獨訓練的流程相同）
總損失 = 各模型損失相加，所以梯度互不影響。輸出的 JSON 格式和 run_a1.py／run_a2.py 相同，report_a.py 不用改。

用法：python gpu_batch.py --procs 6      （開 6 個行程，各自認領設定；每個行程只用 1 個 CPU 執行緒。
                                         QLSTM 疊起來放 GPU，其他模型疊起來放 CPU：實測在 CPU 忙碌時這樣最快）
      python gpu_batch.py --check         （和 CPU 版 wf.train_eval 比對）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np

A1, A2 = HERE / "results" / "a1", HERE / "results" / "a2"
LOCKS = HERE / "results" / "_locks"
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]
LRS = [0.005, 0.02, 0.05]
COST = {"QLSTM": 10, "QFWP": 3, "FWPC": 3, "LSTM3": 1, "LSTM1": 1}


# ---------------------------------------------------------------- 疊起來的模型


class Stacked:
    """把 G 個同型模型的參數疊成 (G, …) 張量；forward(X) 吃 (G, B, L, 4)，回傳 (G, B)。"""

    def __init__(self, name, seeds, device):
        import torch
        import wf
        self.name, self.dev = name, device
        ms = []
        for s in seeds:
            torch.manual_seed(s)
            ms.append(wf.make_model(name))
        self.template = ms[0]
        self.keys = [k for k, _ in ms[0].named_parameters()]
        self.P = {k: torch.stack([dict(m.named_parameters())[k].detach() for m in ms]).to(device).requires_grad_(True)
                  for k in self.keys}
        if name in ("QLSTM", "QFWP", "FWPC"):
            self.perm = ms[0].perm.to(device)
            self.signs = ms[0].signs.to(device)

    def params(self):
        return [self.P[k] for k in self.keys]

    def forward(self, X):
        import torch
        import fastsim as F
        P, n = self.P, self.name
        G, B, L, _ = X.shape
        if n == "QLSTM":
            m = self.template
            H, nq, depth = m.H, m.nq, m.depth
            W = torch.stack([P[f"w.{k}"] for k in range(4)], dim=1)  # (G, 4, depth, nq)
            h = X.new_zeros(G, B, H); c = X.new_zeros(G, B, H)
            for t in range(L):
                psi = F.product_state(torch.cat([X[:, :, t, :], h], dim=-1), plus=True).unsqueeze(1)  # (G,1,B,N)
                for d in range(depth):
                    psi = psi[..., self.perm]
                    for q in range(nq):
                        psi = F.apply_ry(psi, W[:, :, d, q][..., None], q, nq)  # (G,4,B,N)
                g = F.expval_z(psi, self.signs, range(H))  # (G,4,B,H)
                f, i, gg, o = torch.sigmoid(g[:, 0]), torch.sigmoid(g[:, 1]), torch.tanh(g[:, 2]), torch.sigmoid(g[:, 3])
                c = f * c + i * gg
                h = o * torch.tanh(c)
            return torch.einsum("gbh,goh->gbo", h, P["out.weight"])[..., 0] + P["out.bias"]
        if n in ("QFWP", "FWPC"):
            lin = lambda x, k: torch.einsum("gbi,goi->gbo", x, P[f"{k}.weight"]) + P[f"{k}.bias"][:, None, :]
            nq = 3
            theta = X.new_zeros(G, B, nq)
            y = None
            for t in range(L):
                z = lin(X[:, :, t, :], "enc")
                theta = theta + lin(z, "lay") * lin(z, "qub")
                if n == "QFWP":
                    psi = F.product_state(z, plus=False)[..., self.perm]
                    for q in range(nq):
                        psi = F.apply_ry(psi, theta[..., q], q, nq)
                    out = F.expval_z(psi, self.signs, [0])  # (G,B,1)
                else:
                    out = torch.tanh((z * theta).sum(-1, keepdim=True))
                y = lin(out, "post")
            return y[..., 0]
        # LSTM（PyTorch 的閘門順序 i, f, g, o）
        Wih, Whh = P["lstm.weight_ih_l0"], P["lstm.weight_hh_l0"]
        bias = (P["lstm.bias_ih_l0"] + P["lstm.bias_hh_l0"])[:, None, :]
        H = Whh.shape[-1]
        h = X.new_zeros(G, B, H); c = X.new_zeros(G, B, H)
        for t in range(L):
            z = torch.einsum("gbi,gji->gbj", X[:, :, t, :], Wih) + torch.einsum("gbh,gjh->gbj", h, Whh) + bias
            i, f, gg, o = z.split(H, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
            h = torch.sigmoid(o) * torch.tanh(c)
        return torch.einsum("gbh,goh->gbo", h, P["out.weight"])[..., 0] + P["out.bias"]


class StackedAdam:
    """逐元素的 Adam，每個模型有自己的學習率（數學上和每個模型各用一個 torch.optim.Adam 相同）。"""

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-8):
        import torch
        self.p, self.b1, self.b2, self.eps, self.t = params, betas[0], betas[1], eps, 0
        self.lr = torch.tensor(lrs, dtype=params[0].dtype, device=params[0].device)
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self, active=None):
        import torch
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        with torch.no_grad():
            for p, m, v in zip(self.p, self.m, self.v):
                g = p.grad
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                denom = (v.sqrt() / bc2 ** 0.5).add_(self.eps)
                lr = (self.lr / bc1).reshape(-1, *([1] * (p.dim() - 1)))
                p.addcdiv_(m * lr, denom, value=-1.0)
                p.grad = None


# ---------------------------------------------------------------- 訓練一組


def train_group(d, tr, va, te, name, lrs, seeds, device, batch=16, max_epochs=10, patience=3, fixed_steps=None):
    """回傳每個模型的結果 dict（同 wf.train_eval）。fixed_steps 給 A2 用：固定更新次數、不做 early stopping。"""
    import torch
    import wf
    G = len(seeds)
    S = Stacked(name, seeds, device)
    opt = StackedAdam(S.params(), lrs)
    Xtr = torch.tensor(d["X"][tr], device=device); ytr = torch.tensor(d["y"][tr], device=device)
    rngs = [np.random.default_rng(s + 1000) for s in seeds]
    n = len(tr)
    lo, hi, tgt = d["lo"], d["hi"], d["tgt"]

    def predict(idx):
        X = torch.tensor(d["X"][idx], device=device)
        with torch.no_grad():
            out = torch.cat([S.forward(X[s:s + 1024].unsqueeze(0).expand(G, -1, -1, -1)) for s in range(0, len(idx), 1024)], 1)
        return out.cpu().numpy() * (hi[idx] - lo[idx]) + lo[idx]  # (G, len(idx))

    def one_epoch(limit=None):
        perms = np.stack([r.permutation(n) for r in rngs])
        k = 0
        for s in range(0, n, batch):
            if limit is not None and k >= limit:
                break
            j = torch.as_tensor(perms[:, s:s + batch], device=device)  # (G, b)
            pred = S.forward(Xtr[j])
            loss = ((pred - ytr[j]) ** 2).mean(dim=1)  # (G,)
            loss.sum().backward()
            opt.step()
            k += 1
        return k

    if fixed_steps is not None:
        done = 0
        while done < fixed_steps:
            done += one_epoch(fixed_steps - done)
        return [{"pred_test": p} for p in predict(te)], done
    best = np.full(G, np.inf); best_ep = np.zeros(G, int); wait = np.zeros(G, int); stopped = np.zeros(G, bool)
    ran = np.zeros(G, int); hist = [[] for _ in range(G)]
    best_state = [None] * G
    for ep in range(1, max_epochs + 1):
        one_epoch()
        v = np.sqrt(((predict(va) - tgt[va]) ** 2).mean(axis=1))
        for g in np.flatnonzero(~stopped):
            hist[g].append(float(v[g])); ran[g] = ep
            if v[g] < best[g] - 1e-12:
                best[g], best_ep[g], wait[g] = v[g], ep, 0
                best_state[g] = {k: S.P[k][g].detach().clone() for k in S.keys}
            else:
                wait[g] += 1
                stopped[g] = wait[g] >= patience
        if stopped.all():
            break
    import torch
    with torch.no_grad():
        for g in range(G):
            if best_state[g] is not None:
                for k in S.keys:
                    S.P[k][g] = best_state[g][k]
    pt = predict(te)
    return [{"val_rmse": float(best[g]), "best_epoch": int(best_ep[g]), "epochs_run": int(ran[g]), "val_hist": hist[g],
             "pred_test": pt[g].tolist()} for g in range(G)], None


# ---------------------------------------------------------------- 認領與排程


def claim(key):
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCKS / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def a1_path(asset, variant, year, model, lr, seed):
    return A1 / f"{asset}_{variant}" / f"y{year}" / f"{model}_lr{lr:g}_s{seed}.json"


def run_a1_config(cfg, device, cache):
    import wf
    asset, variant, year, model = cfg
    if (asset, variant) not in cache:
        cache[(asset, variant)] = wf.load(asset, variant)
    d = cache[(asset, variant)]
    tr, va, te = wf.fold_idx(d, year)
    meta = {"asset": asset, "variant": variant, "test_year": year, "model": model, "n_train": int(len(tr)),
            "n_val": int(len(va)), "n_test": int(len(te)), "runner": f"gpu_batch/{device}"}

    def go(pairs):
        todo = [(lr, s) for lr, s in pairs if not a1_path(asset, variant, year, model, lr, s).exists()]
        if not todo:
            return
        t0 = time.time()
        res, _ = train_group(d, tr, va, te, model, [lr for lr, _ in todo], [s for _, s in todo], device)
        sec = (time.time() - t0) / len(todo)
        for (lr, s), r in zip(todo, res):
            p = a1_path(asset, variant, year, model, lr, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**r, **meta, "lr": lr, "seed": s, "seconds": sec, "group_size": len(todo)}))
        print(f"{asset} {variant:7s} {year} {model:5s} {len(todo):2d} 個模型一起訓練，{time.time() - t0:.0f} 秒", flush=True)

    go([(lr, s) for lr in LRS for s in range(5)])  # 階段 1
    val = {lr: np.mean([json.loads(a1_path(asset, variant, year, model, lr, s).read_text())["val_rmse"] for s in range(5)])
           for lr in LRS}
    best_lr = min(val, key=val.get)
    go([(best_lr, s) for s in range(5, 20)])  # 階段 2（只看驗證集選出的學習率）


def run_a2_group(g, device, cache):
    import wf
    variant, model, batch, budget = g
    seeds = [s for s in range(10) if not (A2 / variant / f"{model}_b{batch}_{budget}_s{s}.json").exists()]
    if not seeds:
        return
    if variant not in cache:
        cache[variant] = wf.load("EURUSD", variant)
    d = cache[variant]
    n = len(d["X"]); n_tr = int(round(0.8 * n))
    tr, te = np.arange(n_tr), np.arange(n_tr, n)
    steps = 2 * int(np.ceil(n_tr / batch)) if budget == "epochs" else 2188
    t0 = time.time()
    res, done = train_group(d, tr, None, te, model, [0.05] * len(seeds), seeds, device, batch=batch, fixed_steps=steps)
    yt, last = d["tgt"][te], d["last"][te]
    for s, r in zip(seeds, res):
        yhat = np.asarray(r["pred_test"])
        out = {"variant": variant, "model": model, "batch": batch, "budget": budget, "seed": s, "updates": done,
               "n_train": n_tr, "n_test": int(len(te)), "rmse": wf.rmse(yhat, yt), "rw_rmse": wf.rmse(last, yt),
               "da_paper": wf.da_paper(yhat, yt), "da_known": wf.da_known(yhat, yt, last),
               "seconds": (time.time() - t0) / len(seeds), "runner": f"gpu_batch/{device}"}
        (A2 / variant).mkdir(parents=True, exist_ok=True)
        (A2 / variant / f"{model}_b{batch}_{budget}_s{s}.json").write_text(json.dumps(out))
    print(f"A2 {variant:7s} {model:5s} b={batch:2d} {budget:6s} {len(seeds)} 個種子，{time.time() - t0:.0f} 秒", flush=True)


def gpu_worker(wid, device):
    import torch
    import wf
    torch.set_num_threads(1)
    cache = {}
    cfgs = sorted([(a, v, y, m) for a in wf.ASSETS for v in ("aligned", "leaky") for y in wf.TEST_YEARS for m in MODELS],
                  key=lambda c: -COST[c[3]])
    a2 = sorted([(v, m, b, bud) for m in ("QLSTM", "QFWP") for bud in ("steps", "epochs") for v in ("leaky", "aligned")
                 for b in (64, 16, 4)], key=lambda g: (-COST[g[1]], -int(g[3] == "steps")))
    jobs = [("a1", c) for c in cfgs[: len(cfgs) // 5]] + [("a2", g) for g in a2] + [("a1", c) for c in cfgs[len(cfgs) // 5:]]
    for kind, j in jobs:
        key = kind + "_" + "_".join(map(str, j))
        if not claim(key):
            continue
        model = j[3] if kind == "a1" else j[1]
        dev = device if (model == "QLSTM" and torch.cuda.is_available()) else "cpu"  # 只有 QLSTM 在 GPU 上比較划算
        try:
            (run_a1_config if kind == "a1" else run_a2_group)(j, dev, cache)
        except Exception as e:  # 失敗時放掉鎖，讓之後可以重跑
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} 失敗：{e!r}", flush=True)
            raise


def write_lr_choice():
    import wf
    choice = {}
    for a in wf.ASSETS:
        for v in ("aligned", "leaky"):
            for y in wf.TEST_YEARS:
                for m in MODELS:
                    sc = {lr: float(np.mean([json.loads(a1_path(a, v, y, m, lr, s).read_text())["val_rmse"] for s in range(5)]))
                          for lr in LRS}
                    choice[f"{a}|{v}|{y}|{m}"] = {"lr": min(sc, key=sc.get), "val_rmse": sc}
    (A1 / "lr_choice.json").write_text(json.dumps(choice, indent=1))


def check(device):
    """和 CPU 版 wf.train_eval 比對（2 個 epoch、2 個種子、兩種學習率）。"""
    import wf
    d = wf.load("EURUSD", "aligned")
    tr, va, te = wf.fold_idx(d, 2018)
    for m in MODELS:
        lrs, seeds = [0.05, 0.02], [0, 1]
        t0 = time.time()
        res, _ = train_group(d, tr, va, te, m, lrs, seeds, device, max_epochs=2)
        tg = time.time() - t0
        t0 = time.time()
        ref = [wf.train_eval(d, tr, va, te, m, lr, s, max_epochs=2) for lr, s in zip(lrs, seeds)]
        tc = time.time() - t0
        dv = max(abs(r["val_rmse"] - q["val_rmse"]) for r, q in zip(res, ref))
        dp = max(np.abs(np.array(r["pred_test"]) - np.array(q["pred_test"])).max() for r, q in zip(res, ref))
        print(f"{m:5s} 驗證 RMSE 最大差 {dv:.1e}、測試預測最大差 {dp:.1e}、best_epoch {[r['best_epoch'] for r in res]}／"
              f"{[q['best_epoch'] for q in ref]}；批次 {tg:.1f} 秒 vs CPU 逐一 {tc:.1f} 秒", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.check:
        check(a.device)
        raise SystemExit
    ps = [mp.get_context("spawn").Process(target=gpu_worker, args=(w, a.device)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    write_lr_choice()
    print("A（GPU 批次）全部完成", flush=True)
