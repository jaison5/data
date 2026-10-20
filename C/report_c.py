"""彙整方向 C 的結果 → results/summary_c.json、results/summary_c.md。
用法：python report_c.py"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
RES = HERE / "results"
COSTS = ["0.0", "0.0005", "0.001"]
NAMES = {"QA3C": "QA3C（244）", "A3C_small": "古典 A3C h=9（238）", "A3C": "古典 A3C h=128（3,332）",
         "QA3C_PL": "QA3C（PennyLane 後端）"}


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def load_rl(folder="rl"):
    rows = []
    for p in (RES / folder).glob("*.json"):
        r = json.loads(p.read_text())
        for c in COSTS:
            t = r["test"][c]
            rows.append({"agent": r["agent"], "reward": r["reward"], "probs": r["probs"], "year": r["test_year"],
                         "seed": r["seed"], "cost": float(c), "ret": t["total_return_pct"], "sharpe": t["sharpe_ann"],
                         "mdd": t["max_drawdown_pct"], "trades": t["trades"], "in_mkt": t["time_in_market_pct"],
                         "win": t["win_rate_pct"], "hold_share": 100 * t["action_counts"][0] / sum(t["action_counts"]),
                         "best_ep": r["best_val_episode"], "minutes": r["minutes"],
                         "train_reward_last200": float(np.mean([x["reward"] for x in r["log"][-200:]]))})
    return pd.DataFrame(rows)


def compound(rets):
    return 100 * (np.prod([1 + x / 100 for x in rets]) - 1)


def summarize(df):
    """每個設定：5 年（2020～2024）複利報酬（每個種子各算一次再平均）、每年平均報酬、Sharpe、交易次數、持倉比例。"""
    out = []
    for key, g in df.groupby(["agent", "reward", "probs", "cost"]):
        comp = [compound(gs.sort_values("year").ret) for _, gs in g.groupby("seed") if gs.year.nunique() == 5]
        out.append({"agent": key[0], "reward": key[1], "probs": key[2], "cost": key[3], "n_runs": len(g),
                    "comp_5y": float(np.mean(comp)) if comp else None, "comp_5y_sd": float(np.std(comp, ddof=1)) if len(comp) > 1 else None,
                    "ret_year": float(g.ret.mean()), "sharpe": float(g.sharpe.mean()), "mdd": float(g.mdd.mean()),
                    "trades_year": float(g.trades.mean()), "in_mkt": float(g["in_mkt"].mean()),
                    "hold_share": float(g.hold_share.mean()), "no_trade_runs": int((g.trades == 0).sum()),
                    "minutes": float(g.minutes.mean())})
    return pd.DataFrame(out)


def main():
    md, out = [], {}
    # ---- 預測器
    fc = pd.DataFrame([json.loads(p.read_text()) for p in sorted((RES / "forecaster").glob("forecaster_y*.json"))])
    n, k = int(fc.n_test.sum()), int(fc.test_correct.sum())
    acc = k / n
    ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
    out["forecaster"] = {"per_fold": fc.to_dict("records"), "pooled_acc": acc, "pooled_n": n, "ci95": ci}
    t = fc[["test_year", "n_train", "n_test", "test_acc", "test_majority"]].copy()
    t["test_acc"] = t.test_acc.map(lambda x: f"{100 * x:.1f}%"); t["test_majority"] = t.test_majority.map(lambda x: f"{100 * x:.1f}%")
    t.columns = ["測試年", "訓練筆數", "測試筆數（±1.2% 標籤）", "測試準確率", "全猜多數類別"]
    md += ["\n### QLSTM 預測器（每個 fold 重新訓練）\n", md_table(t),
           f"\n合併 5 個 fold：{k}／{n} = {100 * acc:.1f}%，95% 信賴區間 ±{100 * ci:.1f} 個百分點。\n"]
    # ---- 規則型基準
    bl = pd.DataFrame(json.loads((RES / "baselines.json").read_text()))
    bl = bl.rename(columns={"total_return_pct": "ret", "sharpe_ann": "sharpe", "max_drawdown_pct": "mdd",
                            "time_in_market_pct": "in_mkt", "test_year": "year"})
    bls = []
    for (pol, c), g in bl.groupby(["policy", "cost"]):
        if pol == "random":
            comp = [compound(gs.sort_values("year").ret) for _, gs in g.groupby("seed")]
        else:
            comp = [compound(g.sort_values("year").ret)]
        bls.append({"policy": pol, "cost": c, "comp_5y": float(np.mean(comp)), "ret_year": float(g.ret.mean()),
                    "sharpe": float(g.sharpe.mean()), "trades_year": float(g.trades.mean()), "in_mkt": float(g.in_mkt.mean())})
    bls = pd.DataFrame(bls)
    out["baselines"] = bls.to_dict("records")
    # ---- 代理人
    df = load_rl()
    s = summarize(df)
    out["rl_summary"] = s.to_dict("records")
    f1 = lambda x: "—" if x is None or x != x else f"{x:.2f}"

    def table(sel, label):
        t = sel.copy()
        return pd.DataFrame({label: t["label"], "5 年複利報酬 %": [f"{a:.2f} ± {b:.2f}" if b == b and b is not None else f1(a) for a, b in zip(t.comp_5y, t.comp_5y_sd.fillna(np.nan) if "comp_5y_sd" in t else [np.nan] * len(t))],
                             "每年報酬 %": t.ret_year.map(f1), "Sharpe": t.sharpe.map(f1),
                             "每年交易次數": t.trades_year.map(lambda x: f"{x:.1f}"), "持倉 %": t.in_mkt.map(lambda x: f"{x:.1f}")})

    for c in (0.0, 0.0005, 0.001):
        e1 = s[(s.reward == "logret") & (s.probs == "qlstm") & (s.cost == c) & (s.agent != "QA3C_PL")].copy()
        e1["label"] = e1.agent.map(NAMES)
        b = bls[bls.cost == c].copy()
        b["label"] = b.policy.map({"buy_hold": "買進持有", "ma5_ma20": "均線 MA5/MA20", "random": "隨機（20 次平均）"})
        b["comp_5y_sd"] = np.nan
        md += [f"\n### E1 主實驗：獎勵 logret、成本 {100 * c:.2f}%（2020～2024，每個代理人 5 fold × 3 種子）\n",
               md_table(pd.concat([table(e1, "策略"), table(b, "策略")], ignore_index=True))]
    e2 = s[(s.probs == "qlstm") & (s.cost == 0.0005) & s.agent.isin(["QA3C", "A3C_small"])].copy()
    e2["label"] = e2.agent.map(NAMES) + "／" + e2.reward.map({"paper": "論文獎勵", "logret": "logret"})
    t = table(e2.sort_values(["agent", "reward"]), "代理人／獎勵")
    t["持有動作比例 %"] = e2.sort_values(["agent", "reward"]).hold_share.map(lambda x: f"{x:.1f}").values
    t["完全不交易的次數"] = e2.sort_values(["agent", "reward"]).no_trade_runs.values
    md += ["\n### E2 獎勵消融（成本 0.05%）\n", md_table(t)]
    e3 = s[(s.agent == "A3C_small") & (s.reward == "logret") & (s.cost == 0.0005)].copy()
    e3["label"] = e3.probs.map({"none": "沒有預測器（0.5）", "qlstm": "QLSTM 預測器", "oracle": "理想預測器（用未來資訊，上限）"})
    md += ["\n### E3 預測器消融（古典 A3C h=9、logret、成本 0.05%）\n", md_table(table(e3.sort_values("probs"), "狀態裡的預測"))]
    # ---- 成對檢定：QA3C vs 古典（同 fold、同種子配對）
    pairs = []
    for other in ("A3C_small", "A3C"):
        for c in (0.0, 0.0005, 0.001):
            q = df[(df.agent == "QA3C") & (df.reward == "logret") & (df.probs == "qlstm") & (df.cost == c)].set_index(["year", "seed"])
            o = df[(df.agent == other) & (df.reward == "logret") & (df.probs == "qlstm") & (df.cost == c)].set_index(["year", "seed"])
            j = q.join(o, lsuffix="_q", rsuffix="_o", how="inner")
            if len(j) >= 5 and (j.ret_q - j.ret_o).abs().sum() > 0:
                pairs.append({"比較": f"QA3C vs {NAMES[other]}", "成本 %": 100 * c, "配對數": len(j),
                              "每年報酬差（QA3C − 對手）": f"{(j.ret_q - j.ret_o).mean():+.2f}",
                              "Wilcoxon p": f"{wilcoxon(j.ret_q, j.ret_o).pvalue:.3f}",
                              "QA3C 較好的次數": int((j.ret_q > j.ret_o).sum())})
    out["pairs"] = pairs
    if pairs:
        md += ["\n### QA3C 與古典 A3C 的成對比較（同 fold、同種子）\n", md_table(pd.DataFrame(pairs))]
    # ---- 交叉驗證 1：PennyLane（lightning.qubit）後端 vs 新程式（fastsim＋NumPy 選動作），2024 年、種子 0～2
    def runs(d, agent, year=None, seeds=None, cost=0.0005):
        x = d[(d.agent == agent) & (d.reward == "logret") & (d.probs == "qlstm") & (d.cost == cost)]
        if year is not None:
            x = x[x.year == year]
        if seeds is not None:
            x = x[x.seed.isin(seeds)]
        return x.sort_values(["year", "seed"])
    cols = {"ret": "測試年報酬 %", "sharpe": "Sharpe", "trades": "交易次數", "in_mkt": "持倉 %",
            "train_reward_last200": "最後 200 個 episode 平均獎勵", "best_ep": "驗證最佳 episode", "minutes": "訓練分鐘"}
    pl, fs = runs(df, "QA3C_PL", 2024, [0, 1, 2]), runs(df, "QA3C", 2024, [0, 1, 2])
    if len(pl):
        rows = []
        for tag, x in (("PennyLane", pl), ("新程式", fs)):
            for _, r in x.iterrows():
                rows.append({"後端": tag, "種子": int(r.seed), **{v: f"{r[k]:.2f}" for k, v in cols.items()}})
            agg = {v: (f"{x[k].mean():.2f} ± {x[k].std(ddof=1):.2f}" if len(x) > 1 else f"{x[k].mean():.2f}")
                   for k, v in cols.items()}
            rows.append({"後端": f"{tag} 平均 ± 標準差", "種子": "", **agg})
        out["pennylane_crosscheck"] = {"pl": pl.to_dict("records"), "fastsim": fs.to_dict("records")}
        md += ["\n### 交叉驗證：PennyLane 後端 vs 新程式（2024 年、logret、成本 0.05%、種子 0～2）\n",
               md_table(pd.DataFrame(rows))]
        if len(pl) >= 2 and len(fs) >= 2:
            from scipy.stats import mannwhitneyu
            md += [f"\n測試年報酬的 Mann–Whitney U 檢定 p = {mannwhitneyu(pl.ret, fs.ret).pvalue:.3f}"
                   "（只有 3 對 3，主要看兩邊的數值範圍是否重疊）。\n"]
    # ---- 交叉驗證 2：舊版（PyTorch 抽樣）vs 新版（NumPy 抽樣），同 fold、同種子
    old_dir = RES / "rl_old_sampler"
    if old_dir.exists() and any(old_dir.glob("*.json")):
        od = load_rl("rl_old_sampler")
        o, n_ = runs(od, "QA3C"), runs(df, "QA3C")
        j = o.merge(n_, on=["year", "seed"], suffixes=("_old", "_new"))
        if len(j):
            t = pd.DataFrame({"fold": j.year, "種子": j.seed, "舊版報酬 %": j.ret_old.map(lambda x: f"{x:.2f}"),
                              "新版報酬 %": j.ret_new.map(lambda x: f"{x:.2f}"), "舊版交易": j.trades_old,
                              "新版交易": j.trades_new})
            out["sampler_crosscheck"] = j.to_dict("records")
            md += ["\n### 交叉驗證：舊版（PyTorch 抽樣）vs 新版（NumPy 抽樣）的 QA3C（logret、成本 0.05%）\n", md_table(t),
                   f"\n平均報酬：舊版 {j.ret_old.mean():.2f}%、新版 {j.ret_new.mean():.2f}%；"
                   f"平均交易次數：舊版 {j.trades_old.mean():.1f}、新版 {j.trades_new.mean():.1f}。\n"]
    (RES / "summary_c.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (RES / "summary_c.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
