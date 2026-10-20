"""D8：量子模型是否「必要」——規格見 PROTOCOL.md 的 D8 增補（先寫好才跑）。

② 正則化對等的古典對照（early／wd／turn）、③ 古典傅立葉特徵 LSTM、④ 資料量學習曲線（最近 K = 2、4、8 年）。
沿用 D6 的資料（QQQ）、fold、即時①時點、成本與 10 個種子；訓練函式是 D6.e2e_group 加上 L2 懲罰、換手懲罰與 epoch 上限。
用法：python d8_controls.py --procs 20 --skip QLSTM          （CPU：古典模型）
      python d8_controls.py --procs 2 --device cuda --only QLSTM（GPU：學習曲線的 QLSTM）
      python d8_controls.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

from common_d import LAST, RES, TDAYS

OUT, LOCKS = RES / "d8", RES / "_locks_d8"
CLASSICAL = ["FWPC", "LSTM3", "LSTM1"]
FF_MODELS = ["FFLSTM1", "FFLSTM3"]
LC_MODELS = ["QLSTM"] + CLASSICAL
LC_K = [2, 4, 8]
REG = {"early": None, "wd": [0.001, 0.01, 0.1], "turn": [0.05, 0.2, 0.8]}
LRS = [0.005, 0.02, 0.05]
N_SEL, N_ADD = 5, 5
CAP = 1.0
WCOST = {"QLSTM": 30, "FWPC": 3, "LSTM3": 1, "LSTM1": 1, "FFLSTM1": 1, "FFLSTM3": 1}


def patch_models():
    import wf
    import fastsim as F
    base = wf.make_model

    def make_model(name):
        if name == "FFLSTM1":
            return F.LSTM(n_in=8, H=1)
        if name == "FFLSTM3":
            return F.LSTM(n_in=8, H=3)
        return base(name)
    wf.make_model = make_model


def fold_data_window(D4, R, year, K=None):
    """同 D4.fold_data；K 不為 None 時只用驗證年之前最近 K 年訓練，縮放分位數也只用這段算。"""
    tr, va, te = D4.fold_idx(R, year)
    if K is not None:
        vstart = R["dates"][va[0]]
        tr = tr[R["dates"][tr] >= vstart - pd.DateOffset(years=K)]
    X = R["X"]
    lo = np.nanquantile(X[tr], 0.01, axis=(0, 1)); hi = np.nanquantile(X[tr], 0.99, axis=(0, 1))
    Xs = np.clip((X - lo) / (hi - lo), 0, 1)
    ylo, yhi = np.nanquantile(R["ylog"][tr], 0.01), np.nanquantile(R["ylog"][tr], 0.99)
    N = len(X)
    d = {"X": Xs, "y": (R["ylog"] - ylo) / (yhi - ylo), "lo": np.full(N, ylo), "hi": np.full(N, yhi), "tgt": R["ylog"]}
    return d, tr, va, te


def e2e_group_reg(R, d, tr, va, te, name, cap, lrs, seeds, device="cpu", max_epochs=30, patience=5, wd=None, turn=None):
    """D6.e2e_group 的複本，另外加：每個模型自己的 L2 懲罰 wd[g]、換手懲罰 turn[g]（× 年化換手），以及 epoch 上限。
    驗證年仍用不含懲罰的 Sharpe 選最好的 epoch（同 D6）。"""
    import torch
    import fastsim  # noqa: F401
    from gpu_batch import Stacked, StackedAdam
    import d6_e2e as D6
    G = len(seeds)
    S = Stacked(name, seeds, device)
    opt = StackedAdam(S.params(), lrs)
    X = torch.tensor(d["X"], device=device)
    r2, f2 = torch.tensor(R["r2"], device=device), torch.tensor(R["rf2"], device=device)
    wdv = torch.tensor(np.zeros(G) if wd is None else np.asarray(wd, float), device=device)
    tnv = torch.tensor(np.zeros(G) if turn is None else np.asarray(turn, float), device=device)
    assert np.all(np.diff(tr) == 1) and np.all(np.diff(va) == 1)
    rngs = [np.random.default_rng(s + 1000) for s in seeds]
    n = len(tr)
    SEG, NSEG = D6.SEG, D6.NSEG

    def weights(idx):
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                Xi = X[torch.as_tensor(idx[s:s + 1024], device=device)]
                out.append(cap * torch.sigmoid(S.forward(Xi.unsqueeze(0).expand(G, -1, -1, -1))))
        return torch.cat(out, 1)

    def sharpe(idx):
        w = weights(idx)
        j = torch.as_tensor(idx[1:], device=device)
        ex = D6.excess(w[:, 1:], w[:, :-1], r2[j], f2[j])
        s = (ex.mean(1) / ex.std(1) * math.sqrt(TDAYS)).cpu().numpy()
        return np.where(np.isfinite(s), s, -np.inf)

    vidx = np.r_[va[0] - 1, va]
    best = np.full(G, -np.inf); best_ep = np.zeros(G, int); wait = np.zeros(G, int); stopped = np.zeros(G, bool)
    ran = np.zeros(G, int); hist = [[] for _ in range(G)]; best_state = [None] * G
    for ep in range(1, max_epochs + 1):
        starts = []
        for g in range(G):
            off = int(rngs[g].integers(1, SEG + 1))
            starts.append(rngs[g].permutation(np.arange(off, n - SEG + 1, SEG)))
        for k in range(min(len(s) for s in starts) // NSEG):
            j = np.stack([tr[(s[k * NSEG:(k + 1) * NSEG, None] - 1 + np.arange(SEG + 1)).ravel()] for s in starts])
            jt = torch.as_tensor(j, device=device)
            w = (cap * torch.sigmoid(S.forward(X[jt]))).reshape(G, NSEG, SEG + 1)
            rr, ff = r2[jt].reshape(G, NSEG, SEG + 1)[..., 1:], f2[jt].reshape(G, NSEG, SEG + 1)[..., 1:]
            ex = D6.excess(w[..., 1:], w[..., :-1], rr, ff).reshape(G, -1)
            loss = -(ex.mean(1) / (ex.std(1) + 1e-10)) * math.sqrt(TDAYS)
            turnover = (w[..., 1:] - w[..., :-1]).abs().reshape(G, -1).mean(1) * TDAYS
            l2 = sum((S.P[k_] ** 2).reshape(G, -1).sum(1) for k_ in S.keys)
            loss = loss + tnv * turnover + wdv * l2
            loss.sum().backward()
            opt.step()
        v = sharpe(vidx)
        for g in np.flatnonzero(~stopped):
            hist[g].append(float(v[g])); ran[g] = ep
            if v[g] > best[g] + 1e-12:
                best[g], best_ep[g], wait[g] = v[g], ep, 0
                best_state[g] = {k_: S.P[k_][g].detach().clone() for k_ in S.keys}
            else:
                wait[g] += 1
                stopped[g] = wait[g] >= patience
        if stopped.all():
            break
    with torch.no_grad():
        for g in range(G):
            if best_state[g] is not None:
                for k_ in S.keys:
                    S.P[k_][g] = best_state[g][k_]
    wt = weights(te).cpu().numpy()
    ts = sharpe(tr)
    return [{"val_sharpe": float(best[g]), "best_epoch": int(best_ep[g]), "epochs_run": int(ran[g]), "val_hist": hist[g],
             "train_sharpe": float(ts[g]), "w_test": wt[g].tolist()} for g in range(G)]


# ------------------------------------------------------------ 設定與排程


def jobs():
    J = []
    for y in range(2008, 2027):
        for m in CLASSICAL:
            for tag in REG:
                J.append(("reg", tag, y, m))
        for m in FF_MODELS:
            J.append(("ff", "std", y, m))
        for K in LC_K:
            for m in LC_MODELS:
                J.append(("lc", f"K{K}", y, m))
    return sorted(J, key=lambda j: (-WCOST[j[3]], -j[2]))


def hp_grid(fam, tag):
    """回傳 (超參數名稱, 候選值)；early 只有一個固定值。"""
    if fam == "reg":
        return ("early", [0]) if tag == "early" else (tag, REG[tag])
    return ("lr", LRS)


def path(fam, tag, year, model, hpname, hp, seed):
    return OUT / f"{fam}_{tag}" / f"y{year}" / f"{model}_{hpname}{hp:g}_s{seed}.json"


def load(*a):
    return json.loads(path(*a).read_text())


def best_hp(fam, tag, year, model):
    hpname, grid = hp_grid(fam, tag)
    if len(grid) == 1:
        return grid[0], {grid[0]: float(np.mean([load(fam, tag, year, model, hpname, grid[0], s)["val_sharpe"] for s in range(N_SEL)]))}
    val = {h: float(np.mean([load(fam, tag, year, model, hpname, h, s)["val_sharpe"] for s in range(N_SEL)])) for h in grid}
    return max(val, key=val.get), val


def run_config(job, cache, device):
    import d4_vol as D4
    import d6_e2e as D6
    fam, tag, year, model = job
    if "R" not in cache:
        cache["R"] = D6.data()
    R = cache["R"]
    K = int(tag[1:]) if fam == "lc" else None
    d, tr, va, te = fold_data_window(D4, R, year, K)
    if model.startswith("FF"):
        d = {**d, "X": np.concatenate([np.cos(d["X"]), np.sin(d["X"])], axis=-1)}
    hpname, grid = hp_grid(fam, tag)

    def go(pairs):
        todo = [(h, s) for h, s in pairs if not path(fam, tag, year, model, hpname, h, s).exists()]
        if not todo:
            return
        t0 = time.time()
        if fam == "reg":
            lrs = [0.005] * len(todo)
            kw = dict(wd=[h if tag == "wd" else 0.0 for h, _ in todo], turn=[h if tag == "turn" else 0.0 for h, _ in todo])
            if tag == "early":
                kw.update(max_epochs=3, patience=1)
        else:
            lrs = [h for h, _ in todo]
            kw = {}
        res = e2e_group_reg(R, d, tr, va, te, model, CAP, lrs, [s for _, s in todo], device, **kw)
        for (h, s), r in zip(todo, res):
            p = path(fam, tag, year, model, hpname, h, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**r, "family": fam, "tag": tag, "test_year": year, "model": model, hpname: h, "seed": s,
                                     "n_train": int(len(tr)), "device": device, "seconds": (time.time() - t0) / len(todo)}))
        print(f"{fam}_{tag} {year} {model:7s} {len(todo):2d} models, {time.time() - t0:.0f} s [{device}]", flush=True)

    if len(grid) == 1:
        go([(grid[0], s) for s in range(N_SEL + N_ADD)])
    else:
        go([(h, s) for h in grid for s in range(N_SEL)])
        go([(best_hp(fam, tag, year, model)[0], s) for s in range(N_SEL, N_SEL + N_ADD)])


def claim(key):
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCKS / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(wid, device="cpu", only=None, skip=None):
    import torch
    torch.set_num_threads(1)
    patch_models()
    cache = {}
    for job in jobs():
        m = job[3]
        if only and m not in only:
            continue
        if skip and m in skip:
            continue
        key = "_".join(map(str, job))
        if not claim(key):
            continue
        dev = device if m == "QLSTM" else "cpu"
        try:
            run_config(job, cache, dev)
        except Exception as e:
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} failed: {e!r}", flush=True)
            raise


# ------------------------------------------------------------ 報表


def weights_series(R, fam, tag, model):
    import d4_vol as D4
    hpname, grid = hp_grid(fam, tag)
    a = pd.Series(np.nan, index=R["dates"]); vals = []; rows = []
    for y in range(2008, 2027):
        _, _, te = D4.fold_idx(R, y)
        h, val = best_hp(fam, tag, y, model)
        js = [load(fam, tag, y, model, hpname, h, s) for s in range(N_SEL + N_ADD)]
        a.iloc[te] = np.mean([j["w_test"] for j in js], axis=0)
        vals.append(np.mean([j["val_sharpe"] for j in js]))
        rows.append(dict(family=f"{fam}_{tag}", model=model, test_year=y, hp=h, val_sharpe=vals[-1],
                         best_epoch=np.mean([j["best_epoch"] for j in js])))
    return a.loc["2008-01-01":], float(np.mean(vals)), rows


def report():
    import d6_e2e as D6
    from realtime import ohlc, rf_series, run_close
    from d5_top_etf import boot_ratio
    from report_d1 import boot_sharpe
    from d7_ndx import stats
    patch_models()
    R = D6.data()
    df = ohlc("QQQ"); r = df["Close"].pct_change(); rf = rf_series(df.index)
    a0, a1 = "2008-01-02", LAST
    avg, _, d6lr = D6.weights_table(R)
    d6val = {m: float(d6lr[(d6lr.cap == 1.0) & (d6lr.model == m)].val_sharpe.mean()) for m in CLASSICAL}
    hist = (0.20 / (r.rolling(20).std() * math.sqrt(TDAYS))).clip(upper=1.0)
    S, valmean, lrrows = {}, {}, []

    def strat(w):
        return run_close(w.reindex(df.index).shift(2).fillna(hist.shift(2)).fillna(0.0), r, rf)[0].loc[a0:a1]
    S["D6_QLSTM"] = strat(avg[(1.0, "QLSTM")])
    for m in CLASSICAL:
        S[f"D6_{m}"] = strat(avg[(1.0, m)])
    S["plainVT"] = run_close(hist.shift(2).fillna(0.0), r, rf)[0].loc[a0:a1]
    for fam, tags, models in [("reg", list(REG), CLASSICAL), ("ff", ["std"], FF_MODELS), ("lc", [f"K{k}" for k in LC_K], LC_MODELS)]:
        for tag in tags:
            for m in models:
                w, vm, rows = weights_series(R, fam, tag, m)
                name = f"{fam}_{tag}_{m}"
                S[name] = strat(w); valmean[name] = vm; lrrows += rows
    pd.DataFrame(lrrows).to_csv(RES / "d8_hp_choice.csv", index=False)
    rows = [dict(name=k, **stats(v, rf), val_sharpe_mean=valmean.get(k, np.nan)) for k, v in S.items()]
    res = pd.DataFrame(rows); res.to_csv(RES / "d8_strategy.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(res.round(4).to_string(index=False))

    def test(a, b):
        xa, xb = S[a] - rf.loc[a0:a1], S[b] - rf.loc[a0:a1]
        ds, ps = boot_sharpe(xa, xb, reps=2000)
        dr, pr = boot_ratio(S[a], S[b])
        return dict(strategy=a, vs=b, d_sharpe=ds, p_sharpe=ps, d_ratio=dr, p_ratio=pr)
    T = []
    regs = [f"reg_{t}_{m}" for t in REG for m in CLASSICAL]
    best_reg = max(regs, key=lambda k: valmean[k])
    T.append({**test("D6_QLSTM", best_reg), "role": "② primary (best regularized classical by validation)"})
    for k in regs:
        T.append({**test("D6_QLSTM", k), "role": "② all"})
    for k in [f"ff_std_{m}" for m in FF_MODELS]:
        T.append({**test("D6_QLSTM", k), "role": "③"})
    lc = []
    for K in LC_K:
        best_c = max([f"lc_K{K}_{m}" for m in CLASSICAL], key=lambda k: valmean[k])
        t = test(f"lc_K{K}_QLSTM", best_c); t["role"] = f"④ K={K}"; T.append(t); lc.append((K, t))
    best_full = "D6_" + max(CLASSICAL, key=lambda m: d6val[m])  # D6 全資料：同樣用驗證年的平均 Sharpe 選古典模型
    t = test("D6_QLSTM", best_full); t["role"] = "④ full data (D6)"; T.append(t)
    T = pd.DataFrame(T)
    # Holm for ② all
    sub = T[T.role == "② all"].sort_values("p_sharpe")
    m_ = len(sub)
    T.loc[sub.index, "holm_p_sharpe"] = [min(1.0, max((m_ - i) * p for i, p in enumerate(sub.p_sharpe.values[:j + 1]))) for j in range(m_)]
    T.to_csv(RES / "d8_tests.csv", index=False)
    print(T.round(4).to_string(index=False))
    prim = T[T.role.str.startswith("② primary")].iloc[0]
    v2 = bool(prim.p_sharpe < 0.05 and prim.p_ratio < 0.05 and prim.d_sharpe > 0 and prim.d_ratio > 0)
    ff = T[T.role == "③"]
    v3 = bool(((ff.p_sharpe < 0.025) & (ff.d_sharpe > 0)).all())
    gaps = {K: t["d_sharpe"] for K, t in lc}
    full_gap = float(T[T.role == "④ full data (D6)"].d_sharpe.iloc[0])
    v4 = bool(all(g > 0 for g in gaps.values()) and lc[0][1]["p_sharpe"] < 0.05 and gaps[2] > full_gap)
    verdict = dict(best_regularized_classical=best_reg, criterion2_regularization=v2, criterion3_fourier=v3,
                   criterion4_less_data_more_advantage=v4, quantum_structure_necessary=v2 and v3,
                   lc_gaps={str(k): float(v) for k, v in gaps.items()}, full_gap=full_gap)
    (RES / "d8_verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        import d4_vol as D4
        import d6_e2e as D6
        import torch
        torch.set_num_threads(4)
        patch_models()
        R = D6.data()
        for K in (None, 2):
            d, tr, va, te = fold_data_window(D4, R, 2015, K)
            print("K", K, "train", len(tr), R["dates"][tr[0]].date(), "->", R["dates"][tr[-1]].date())
        d, tr, va, te = fold_data_window(D4, R, 2015, None)
        dff = {**d, "X": np.concatenate([np.cos(d["X"]), np.sin(d["X"])], axis=-1)}
        for name, dd, kw in [("LSTM1", d, dict(wd=[0.01, 0.0], turn=[0.0, 0.2])), ("FFLSTM1", dff, {})]:
            res = e2e_group_reg(R, dd, tr, va, te, name, 1.0, [0.005, 0.005], [0, 1], "cpu", max_epochs=2, patience=1, **kw)
            print(name, [round(x["val_sharpe"], 3) for x in res], "w mean", [round(float(np.mean(x["w_test"])), 3) for x in res])
        raise SystemExit
    if a.report:
        report(); raise SystemExit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    only = a.only.split(",") if a.only else None
    skip = a.skip.split(",") if a.skip else None
    ps = [mp.get_context("spawn").Process(target=worker, args=(w, a.device, only, skip)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("D8 done", flush=True)
