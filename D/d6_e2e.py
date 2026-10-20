"""D6：QLSTM／QFWP（與同規模古典模型）直接輸出 QQQ 持有比例，用扣成本後每日策略報酬的負 Sharpe 當訓練損失（端到端）。
規格見 PROTOCOL.md 的 D6 增補（先寫好才跑）。輸入、fold、purge、學習率選法、種子數都和 D4 相同，只換輸出與損失。
用法：python d6_e2e.py --procs 30     （訓練，可中斷後續跑）
      python d6_e2e.py --report       （回測、檢定、寫 results/d6_*.csv）
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

import d4_vol as D4
from common_d import LAST, RES, TDAYS

OUT, LOCKS = RES / "d6", RES / "_locks_d6"
TEST_YEARS = D4.TEST_YEARS
CAPS = [1.0, 1.5]
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1", "LINEAR"]
QUANTUM, CLASSICAL = ["QLSTM", "QFWP"], ["FWPC", "LSTM3", "LSTM1"]
LRS, N_SEL, N_ADD = D4.LRS, D4.N_SEL, D4.N_ADD
SEG, NSEG = 63, 4              # 小批次＝4 段 × 63 個連續交易日
MAX_EPOCHS, PATIENCE = 30, 5
COST, SPREAD = 0.0005, 0.005   # 單邊成本、槓桿部分的借款利差（同 realtime.run_close）
WCOST = {"QLSTM": 10, "QFWP": 3, "FWPC": 3, "LSTM3": 1, "LSTM1": 1, "LINEAR": 0.5}
A0 = "2008-01-02"


def data():
    """D4 的特徵，加上每個決策日 t 賺到的第 t+2 天 QQQ 收盤報酬與無風險利率（即時①：t 收盤後算，t+1 收盤成交）。"""
    from realtime import ohlc, rf_series
    R = D4.raw("QQQ")
    df = ohlc("QQQ")
    r, rf = df["Close"].pct_change().values, rf_series(df.index).values
    pos = df.index.get_indexer(R["dates"])
    assert (pos >= 0).all()
    p2 = pos + 2
    ok = p2 < len(df)
    R["r2"] = np.where(ok, r[np.minimum(p2, len(df) - 1)], np.nan)
    R["rf2"] = np.where(ok, rf[np.minimum(p2, len(df) - 1)], np.nan)
    return R


class StackedLinear:
    """線性參考：攤平的 20 個特徵＋常數（21 個參數），初始化同 torch.nn.Linear(20, 1)（依種子）。"""

    def __init__(self, seeds, device):
        import torch
        ms = []
        for s in seeds:
            torch.manual_seed(s)
            ms.append(torch.nn.Linear(20, 1))
        self.keys = ["weight", "bias"]
        self.P = {"weight": torch.stack([m.weight.detach()[0] for m in ms]).to(device).requires_grad_(True),
                  "bias": torch.stack([m.bias.detach()[0] for m in ms]).to(device).requires_grad_(True)}

    def params(self):
        return [self.P[k] for k in self.keys]

    def forward(self, X):
        import torch
        G, B = X.shape[:2]
        return torch.einsum("gbi,gi->gb", X.reshape(G, B, -1), self.P["weight"]) + self.P["bias"][:, None]


def excess(w, w_prev, r, rf):
    """每日策略超額報酬：w·r + (1−w)·rf − 槓桿借款利差 − 0.05%·換手，再減 rf。"""
    import torch
    net = w * r + (1 - w) * rf - torch.relu(w - 1) * SPREAD / TDAYS - COST * (w - w_prev).abs()
    return net - rf


def e2e_group(R, d, tr, va, te, name, cap, lrs, seeds, device="cpu", max_epochs=MAX_EPOCHS):
    """G 個模型（各自的種子、學習率、資料順序）一起訓練；回傳每個模型的結果 dict。"""
    import torch
    import fastsim  # noqa: F401  （設定 float64，和 D4 相同）
    from gpu_batch import Stacked, StackedAdam
    G = len(seeds)
    S = StackedLinear(seeds, device) if name == "LINEAR" else Stacked(name, seeds, device)
    opt = StackedAdam(S.params(), lrs)
    X = torch.tensor(d["X"], device=device)
    r2, f2 = torch.tensor(R["r2"], device=device), torch.tensor(R["rf2"], device=device)
    assert np.all(np.diff(tr) == 1) and np.all(np.diff(va) == 1)
    rngs = [np.random.default_rng(s + 1000) for s in seeds]
    n = len(tr)

    def weights(idx):
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                Xi = X[torch.as_tensor(idx[s:s + 1024], device=device)]
                out.append(cap * torch.sigmoid(S.forward(Xi.unsqueeze(0).expand(G, -1, -1, -1))))
        return torch.cat(out, 1)  # (G, len(idx))

    def sharpe(idx):
        """idx 為連續的決策日；第一天只用來算第二天的換手。"""
        w = weights(idx)
        j = torch.as_tensor(idx[1:], device=device)
        ex = excess(w[:, 1:], w[:, :-1], r2[j], f2[j])
        s = (ex.mean(1) / ex.std(1) * math.sqrt(TDAYS)).cpu().numpy()
        return np.where(np.isfinite(s), s, -np.inf)

    vidx = np.r_[va[0] - 1, va]
    best = np.full(G, -np.inf); best_ep = np.zeros(G, int); wait = np.zeros(G, int); stopped = np.zeros(G, bool)
    ran = np.zeros(G, int); hist = [[] for _ in range(G)]; best_state = [None] * G
    for ep in range(1, max_epochs + 1):
        starts = []
        for g in range(G):  # 每個 epoch：訓練期切成不重疊的 63 天段，起點偏移與順序依種子
            off = int(rngs[g].integers(1, SEG + 1))
            starts.append(rngs[g].permutation(np.arange(off, n - SEG + 1, SEG)))
        for k in range(min(len(s) for s in starts) // NSEG):
            j = np.stack([tr[(s[k * NSEG:(k + 1) * NSEG, None] - 1 + np.arange(SEG + 1)).ravel()] for s in starts])
            jt = torch.as_tensor(j, device=device)  # (G, NSEG*(SEG+1))
            w = (cap * torch.sigmoid(S.forward(X[jt]))).reshape(G, NSEG, SEG + 1)
            rr, ff = r2[jt].reshape(G, NSEG, SEG + 1)[..., 1:], f2[jt].reshape(G, NSEG, SEG + 1)[..., 1:]
            ex = excess(w[..., 1:], w[..., :-1], rr, ff).reshape(G, -1)
            loss = -(ex.mean(1) / (ex.std(1) + 1e-10)) * math.sqrt(TDAYS)
            loss.sum().backward()
            opt.step()
        v = sharpe(vidx)
        for g in np.flatnonzero(~stopped):
            hist[g].append(float(v[g])); ran[g] = ep
            if v[g] > best[g] + 1e-12:
                best[g], best_ep[g], wait[g] = v[g], ep, 0
                best_state[g] = {k: S.P[k][g].detach().clone() for k in S.keys}
            else:
                wait[g] += 1
                stopped[g] = wait[g] >= PATIENCE
        if stopped.all():
            break
    with torch.no_grad():
        for g in range(G):
            if best_state[g] is not None:
                for k in S.keys:
                    S.P[k][g] = best_state[g][k]
    wt = weights(te).cpu().numpy()
    ts = sharpe(tr)
    return [{"val_sharpe": float(best[g]), "best_epoch": int(best_ep[g]), "epochs_run": int(ran[g]), "val_hist": hist[g],
             "train_sharpe": float(ts[g]), "w_test": wt[g].tolist()} for g in range(G)]


# ------------------------------------------------------------ 排程（同 D4）


def path(cap, year, model, lr, seed):
    return OUT / f"cap{cap:g}" / f"y{year}" / f"{model}_lr{lr:g}_s{seed}.json"


def load(cap, year, model, lr, seed):
    return json.loads(path(cap, year, model, lr, seed).read_text())


def best_lr(cap, year, model):
    val = {lr: float(np.mean([load(cap, year, model, lr, s)["val_sharpe"] for s in range(N_SEL)])) for lr in LRS}
    return max(val, key=val.get), val


def run_config(cfg, cache, device="cpu"):
    cap, year, model = cfg
    if "R" not in cache:
        cache["R"] = data()
    R = cache["R"]
    d, tr, va, te = D4.fold_data(R, year)

    def go(pairs):
        todo = [(lr, s) for lr, s in pairs if not path(cap, year, model, lr, s).exists()]
        if not todo:
            return
        t0 = time.time()
        res = e2e_group(R, d, tr, va, te, model, cap, [lr for lr, _ in todo], [s for _, s in todo], device)
        for (lr, s), r in zip(todo, res):
            p = path(cap, year, model, lr, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**r, "asset": "QQQ", "cap": cap, "test_year": year, "model": model, "lr": lr, "seed": s,
                                     "n_train": int(len(tr)), "device": device, "seconds": (time.time() - t0) / len(todo),
                                     "group_size": len(todo)}))
        print(f"cap{cap:g} {year} {model:6s} {len(todo):2d} models, {time.time() - t0:.0f} s", flush=True)

    go([(lr, s) for lr in LRS for s in range(N_SEL)])
    go([(best_lr(cap, year, model)[0], s) for s in range(N_SEL, N_SEL + N_ADD)])


def claim(key):
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCKS / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(wid):
    import torch
    torch.set_num_threads(1)
    cache = {}
    cfgs = sorted([(c, y, m) for c in CAPS for y in TEST_YEARS for m in MODELS], key=lambda c: (-WCOST[c[2]], -c[1]))
    for c in cfgs:
        key = "_".join(map(str, c))
        if not claim(key):
            continue
        try:
            run_config(c, cache)
        except Exception as e:
            (LOCKS / f"{key}.lock").unlink(missing_ok=True)
            print(f"[{wid}] {key} failed: {e!r}", flush=True)
            raise


# ------------------------------------------------------------ 報表


def weights_table(R):
    """每個（上限, 模型）：選中學習率的 10 個種子在測試年的持有比例（決策日 t），平均版與逐種子版。"""
    dates = R["dates"]
    avg, per_seed, lrrows = {}, {}, []
    for cap in CAPS:
        for m in MODELS:
            a = pd.Series(np.nan, index=dates); ps = np.full((N_SEL + N_ADD, len(dates)), np.nan)
            for y in TEST_YEARS:
                _, _, te = D4.fold_idx(R, y)
                lr, val = best_lr(cap, y, m)
                js = [load(cap, y, m, lr, s) for s in range(N_SEL + N_ADD)]
                W = np.array([j["w_test"] for j in js])
                ps[:, te] = W
                a.iloc[te] = W.mean(0)
                lrrows.append(dict(cap=cap, model=m, test_year=y, lr=lr, **{f"val_sharpe_lr{k:g}": v for k, v in val.items()},
                                   best_epoch=np.mean([j["best_epoch"] for j in js]), epochs_run=np.mean([j["epochs_run"] for j in js]),
                                   train_sharpe=np.mean([j["train_sharpe"] for j in js]), val_sharpe=np.mean([j["val_sharpe"] for j in js]),
                                   seed_sd_of_w=float(W.std(0).mean()), mean_w=float(W.mean())))
            avg[(cap, m)] = a.loc[f"{TEST_YEARS[0]}-01-01":]
            per_seed[(cap, m)] = pd.DataFrame(ps.T, index=dates).loc[f"{TEST_YEARS[0]}-01-01":]
    return avg, per_seed, pd.DataFrame(lrrows)


def report():
    import d5_top_etf as D5
    from realtime import ohlc, rf_series, run_close, stats, vt_weight
    from report_d1 import boot_sharpe, sharpe_test
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
    R = data()
    df = ohlc("QQQ")
    r = df["Close"].pct_change()
    rf = rf_series(df.index)
    etf = {t: ohlc(t)["Close"].pct_change() for t in ("VOO", "IVV", "SPY")}
    avg, per_seed, lrtab = weights_table(R)
    lrtab.to_csv(RES / "d6_lr_choice.csv", index=False)

    def bt(w_dec, cap):
        """決策日 t 的 w → 第 t+2 天持有；2008 年頭兩天（決策日在 2007 年）用純波動度目標的比例（同 D4）。"""
        w = w_dec.reindex(df.index).shift(2)
        w = w.fillna(vt_weight(r, cap=cap).shift(2)).fillna(0.0)
        x, turn = run_close(w, r, rf, COST)
        return x, turn, w

    F = pd.read_csv(RES / "d4_forecasts_QQQ.csv", index_col=0, parse_dates=True)
    S, Wh, grp = {}, {}, {}
    for cap in CAPS:
        vt = vt_weight(r, cap=cap).shift(2).fillna(0.0)
        x, turn = run_close(vt, r, rf, COST)
        S[(cap, "plainVT")], Wh[(cap, "plainVT")], grp[(cap, "plainVT")] = (x, turn), vt, "baseline"
        for m in MODELS:
            x, turn, w = bt(avg[(cap, m)], cap)
            S[(cap, f"D6_{m}")], Wh[(cap, f"D6_{m}")] = (x, turn), w
            grp[(cap, f"D6_{m}")] = "D6 quantum" if m in QUANTUM else ("D6 classical" if m in CLASSICAL else "D6 reference")
        for m in ["hist20", "ewma", "HAR", "LINEAR"] + D4.MODELS:   # 同 d4_vol.report 的做法
            sig = np.exp(F[m]).reindex(df.index)
            w = (0.20 / sig).clip(upper=cap).shift(2)
            w = w.fillna((0.20 / np.exp(F["hist20"]).reindex(df.index)).clip(upper=cap).shift(2)).fillna(0.0)
            S[(cap, f"D4_{m}")], Wh[(cap, f"D4_{m}")], grp[(cap, f"D4_{m}")] = run_close(w, r, rf, COST), w, "D4 forecast"
    bh = (r, pd.Series(0.0, index=r.index))

    periods = {"2008-01~2026-10": (A0, LAST), "2010-09~2026-10": ("2010-09-10", LAST)}
    rows = []
    for per, (a0, a1) in periods.items():
        vtw = {cap: Wh[(cap, "plainVT")].loc[a0:a1] for cap in CAPS}
        items = [((c, n), v) for (c, n), v in S.items()] + [((np.nan, "QQQ buy&hold"), bh)] + \
                [((np.nan, t), (etf[t], pd.Series(0.0, index=etf[t].index))) for t in etf]
        for (cap, name), (x, turn) in items:
            x = x.loc[a0:a1].dropna()
            if x.index[0] > pd.Timestamp(a0) + pd.Timedelta(days=10):
                continue
            s = stats(x, rf)
            w = Wh.get((cap, name))
            rows.append(dict(period=per, cap=cap, name=name, group=grp.get((cap, name), "ETF / buy&hold"), CAGR=s["CAGR"], Vol=s["Vol"],
                             Sharpe=s["Sharpe"], teacher_sharpe=D5.ratio(x)[0], MaxDD=s["MaxDD"],
                             turnover_per_yr=float(turn.loc[a0:a1].sum() / s["years"]),
                             mean_w=float(w.loc[a0:a1].mean()) if w is not None else (1.0 if name == "QQQ buy&hold" else np.nan),
                             corr_w_plainVT=float(np.corrcoef(w.loc[a0:a1], vtw[cap])[0, 1]) if w is not None else np.nan))
    strat = pd.DataFrame(rows)
    strat.to_csv(RES / "d6_strategy.csv", index=False)

    # 檢定（2008-01～2026-10）：每個 D6 版本 vs 純波動度目標、三個古典端到端、D4 同模型、持有 QQQ
    a0, a1 = A0, LAST
    f0 = rf.loc[a0:a1]
    tests = []
    for cap in CAPS:
        for m in MODELS:
            x = S[(cap, f"D6_{m}")][0].loc[a0:a1]
            vs = [("plainVT", S[(cap, "plainVT")][0])] + [(f"D6_{c}", S[(cap, f"D6_{c}")][0]) for c in CLASSICAL if c != m] + \
                 ([(f"D4_{m}", S[(cap, f"D4_{m}")][0])] if (cap, f"D4_{m}") in S else []) + [("QQQ buy&hold", r)]
            for vn, y in vs:
                y = y.loc[a0:a1]
                ds, ps = boot_sharpe(x - f0, y - f0, reps=2000)
                dt, pt = D5.boot_ratio(x, y, reps=2000)
                tests.append(dict(cap=cap, strategy=f"D6_{m}", vs=vn, d_Sharpe=ds, p_boot=ps, p_JK=sharpe_test(x - f0, y - f0)[1],
                                  d_teacher=dt, p_boot_teacher=pt))
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "d6_tests.csv", index=False)

    # 老師夏普值 vs 前三大 ETF
    etests = []
    for per, (a0e, a1e) in periods.items():
        for cap in CAPS:
            for name in ["plainVT"] + [f"D6_{m}" for m in MODELS]:
                x = S[(cap, name)][0].loc[a0e:a1e]
                for t, y in etf.items():
                    y = y.loc[a0e:a1e].dropna()
                    if y.index[0] > pd.Timestamp(a0e) + pd.Timedelta(days=10):
                        continue
                    c = x.index.intersection(y.index)
                    dt, pt = D5.boot_ratio(x.loc[c], y.loc[c], reps=2000)
                    etests.append(dict(period=per, cap=cap, strategy=name, vs=t, d_teacher=dt, p_boot_teacher=pt))
    etests = pd.DataFrame(etests)
    etests.to_csv(RES / "d6_vs_etf.csv", index=False)

    # 逐種子（每個種子單獨用）
    srows = []
    for cap in CAPS:
        for m in MODELS:
            P = per_seed[(cap, m)]
            for s in P.columns:
                x = bt(P[s], cap)[0].loc[a0:a1]
                st = stats(x, rf)
                srows.append(dict(cap=cap, model=m, seed=s, Sharpe=st["Sharpe"], teacher_sharpe=D5.ratio(x)[0], CAGR=st["CAGR"], MaxDD=st["MaxDD"]))
    seeds = pd.DataFrame(srows)
    seeds.to_csv(RES / "d6_seeds.csv", index=False)

    # 逐年
    yrows = []
    for (cap, name), (x, _) in list(S.items()) + [((np.nan, "QQQ buy&hold"), bh)]:
        if name.startswith("D4_") and name not in ("D4_hist20", "D4_QLSTM", "D4_QFWP"):
            continue
        for y in TEST_YEARS:
            xx = x.loc[str(y)]
            ex = xx - rf.reindex(xx.index)
            yrows.append(dict(cap=cap, name=name, year=y, ret=float((1 + xx).prod() - 1), Sharpe=float(ex.mean() / ex.std() * math.sqrt(TDAYS))))
    years = pd.DataFrame(yrows)
    years.to_csv(RES / "d6_years.csv", index=False)

    # 每日持有比例（決策日 t 的 w 已移到第 t+2 天持有）
    pd.DataFrame({f"{n}_cap{c:g}": w.loc[A0:LAST] for (c, n), w in Wh.items()}).to_csv(RES / "d6_weights_QQQ.csv")

    # 事先固定的判定
    alpha = 0.05 / 4
    verdict = []
    for cap in CAPS:
        for m in QUANTUM:
            t = tests[(tests.cap == cap) & (tests.strategy == f"D6_{m}")].set_index("vs")
            c1 = bool(t.loc["plainVT", "p_boot"] < alpha and t.loc["plainVT", "p_boot_teacher"] < alpha)
            c2 = bool(all(t.loc[f"D6_{c}", "p_boot"] < alpha and t.loc[f"D6_{c}", "p_boot_teacher"] < alpha for c in CLASSICAL))
            verdict.append(dict(cap=cap, model=m, beats_plainVT=c1, beats_all_classical=c2, success=c1 and c2,
                                p_vs_plainVT=t.loc["plainVT", "p_boot"], p_teacher_vs_plainVT=t.loc["plainVT", "p_boot_teacher"],
                                max_p_vs_classical=max(t.loc[f"D6_{c}", "p_boot"] for c in CLASSICAL),
                                max_p_teacher_vs_classical=max(t.loc[f"D6_{c}", "p_boot_teacher"] for c in CLASSICAL)))
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(RES / "d6_verdict.csv", index=False)

    print(strat.drop(columns="group").round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(etests[etests.strategy.isin(["plainVT", "D6_QLSTM", "D6_QFWP"])].round(4).to_string(index=False))
    print(seeds.groupby(["cap", "model"])[["Sharpe", "teacher_sharpe"]].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(lrtab.groupby(["cap", "model"])[["lr", "best_epoch", "epochs_run", "train_sharpe", "val_sharpe", "seed_sd_of_w", "mean_w"]].mean().round(3).to_string())
    print(verdict.round(4).to_string(index=False))


def posthoc():
    """事後檢查（不在事先規格內）：D6 版本平均只持有約 4 成 QQQ，老師的夏普值（不扣國庫券）會因為多放國庫券而變高。
    這裡把純波動度目標的目標波動度調到和 D6 版本「同樣的實現波動度」，以及「固定持有同樣平均比例的 QQQ」，再比一次。
    目標波動度是看測試期結果調的（事後），只用來判斷差距從哪裡來，不是可用的策略。"""
    import d5_top_etf as D5
    from realtime import ohlc, rf_series, run_close, stats, vt_weight
    R = data()
    df = ohlc("QQQ")
    r = df["Close"].pct_change()
    rf = rf_series(df.index)
    avg, _, _ = weights_table(R)
    rows = []
    for per, (a0, a1) in {"2008-01~2026-10": (A0, LAST), "2010-09~2026-10": ("2010-09-10", LAST)}.items():
        f0 = rf.loc[a0:a1]
        for cap in CAPS:
            for m in MODELS:
                w = avg[(cap, m)].reindex(df.index).shift(2).fillna(vt_weight(r, cap=cap).shift(2)).fillna(0.0)
                x = run_close(w, r, rf, COST)[0].loc[a0:a1]
                vol, mw = x.std() * math.sqrt(TDAYS), float(w.loc[a0:a1].mean())
                lo, hi = 0.01, 0.40
                for _ in range(40):   # 二分法：純波動度目標的目標值，使實現波動度相同
                    tg = (lo + hi) / 2
                    y = run_close(vt_weight(r, target=tg, cap=cap).shift(2).fillna(0.0), r, rf, COST)[0].loc[a0:a1]
                    lo, hi = (tg, hi) if y.std() * math.sqrt(TDAYS) < vol else (lo, tg)
                z = run_close(pd.Series(mw, index=df.index), r, rf, COST)[0].loc[a0:a1]
                for name, s in [(f"D6_{m}", x), (f"plainVT target {tg:.3f} (same vol)", y), (f"constant {mw:.3f} QQQ", z)]:
                    st = stats(s, rf)
                    row = dict(period=per, cap=cap, model=m, name=name, CAGR=st["CAGR"], Vol=st["Vol"], Sharpe=st["Sharpe"],
                               teacher_sharpe=D5.ratio(s)[0], MaxDD=st["MaxDD"])
                    if not name.startswith("D6_"):
                        row["p_boot_D6_better"] = boot_sharpe_local(x - f0, s - f0)
                        row["p_boot_teacher_D6_better"] = D5.boot_ratio(x, s, reps=2000)[1]
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "d6_posthoc_exposure.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(out.round(4).to_string(index=False))


def boot_sharpe_local(a, b):
    from report_d1 import boot_sharpe
    return boot_sharpe(a, b, reps=2000)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--posthoc", action="store_true")
    ap.add_argument("--bench", default=None, help="只計時：模型名稱，訓練 1 個 epoch（2026 fold，15 個模型），不輸出任何績效")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.bench:
        import torch
        torch.set_num_threads(1)
        R = data(); d, tr, va, te = D4.fold_data(R, 2026)
        assert not np.isnan(R["r2"][tr]).any() and not np.isnan(R["r2"][va]).any() and not np.isnan(d["X"]).any()
        t0 = time.time()
        res = e2e_group(R, d, tr, va, te, a.bench, 1.0, [lr for lr in LRS for _ in range(N_SEL)], list(range(N_SEL)) * len(LRS), max_epochs=1)
        w = np.array([q["w_test"] for q in res])
        print(f"{a.bench}: 1 epoch, 15 models, n_train={len(tr)}: {time.time() - t0:.1f} s; w range [{w.min():.3f}, {w.max():.3f}]")
        raise SystemExit
    if a.report:
        report(); raise SystemExit
    if a.posthoc:
        posthoc(); raise SystemExit
    ps = [mp.get_context("spawn").Process(target=worker, args=(w,)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("D6 done", flush=True)
