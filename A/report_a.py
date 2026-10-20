"""彙整方向 A 的結果 → results/summary_a.json、results/summary_a.md。
用法：python report_a.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import wf

HERE = Path(__file__).parent
RES = HERE / "results"
A1 = RES / "a1"
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM1", "LSTM3"]
NAMES = {"QLSTM": "QLSTM（32）", "QFWP": "QFWP（33）", "FWPC": "FWP 古典消融（33）", "LSTM1": "LSTM h=1（30）",
         "LSTM3": "LSTM h=3（112）", "Linear": "線性（21）", "Midpoint": "高低價中點（0）", "RW": "隨機漫步（0）"}
COLLAPSE = 1.2  # 測試 RMSE 超過同一 fold 隨機漫步的 1.2 倍，視為訓練崩掉


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def pooled_metrics(d, idx, pred):
    yt, last = d["tgt"][idx], d["last"][idx]
    rw = wf.rmse(last, yt)
    return {"rmse": wf.rmse(pred, yt), "rmse_ratio": wf.rmse(pred, yt) / rw,
            "da_paper": wf.da_paper(pred, yt), "da_known": wf.da_known(pred, yt, last)}


def a1():
    choice = json.loads((A1 / "lr_choice.json").read_text())
    rows, tests, folds, pairs = [], [], [], []
    for asset in wf.ASSETS:
        for variant in ("leaky", "aligned"):
            d = wf.load(asset, variant)
            idx_all, preds, base = [], {m: {s: [] for s in range(20)} for m in MODELS}, {"Linear": [], "Midpoint": []}
            fold_rw, fold_rmse = {}, {m: {} for m in MODELS}
            for y in wf.TEST_YEARS:
                tr, va, te = wf.fold_idx(d, y)
                idx_all.append(te)
                fold_rw[y] = wf.rmse(d["last"][te], d["tgt"][te])
                base["Linear"].append(np.array(wf.linear_fit(d, tr, va, te)["pred_test"]))
                base["Midpoint"].append(d["hl_mid"][te])
                for m in MODELS:
                    lr = choice[f"{asset}|{variant}|{y}|{m}"]["lr"]
                    for s in range(20):
                        r = json.loads((A1 / f"{asset}_{variant}" / f"y{y}" / f"{m}_lr{lr:g}_s{s}.json").read_text())
                        p = np.array(r["pred_test"])
                        preds[m][s].append(p)
                        fold_rmse[m][(y, s)] = wf.rmse(p, d["tgt"][te])
                    folds.append({"asset": asset, "variant": variant, "test_year": y, "model": m, "lr": lr,
                                  "rmse_ratio_mean": float(np.mean([fold_rmse[m][(y, s)] / fold_rw[y] for s in range(20)]))})
            idx = np.concatenate(idx_all)
            yt, last = d["tgt"][idx], d["last"][idx]
            per_seed = {}
            for m in MODELS:
                ms = [pooled_metrics(d, idx, np.concatenate(preds[m][s])) for s in range(20)]
                per_seed[m] = pd.DataFrame(ms)
                ens = np.mean([np.concatenate(preds[m][s]) for s in range(20)], axis=0)
                e = pooled_metrics(d, idx, ens)
                dm, dm_p = wf.dm_test(ens - yt, last - yt)
                pt, pt_p = wf.pt_test(np.sign(ens - last), np.sign(yt - last))
                coll = np.mean([fold_rmse[m][k] > COLLAPSE * fold_rw[k[0]] for k in fold_rmse[m]])
                rows.append({"asset": asset, "variant": variant, "model": m,
                             **{f"{k}_mean": float(per_seed[m][k].mean()) for k in per_seed[m]},
                             **{f"{k}_std": float(per_seed[m][k].std(ddof=1)) for k in per_seed[m]},
                             **{f"ens_{k}": v for k, v in e.items()}, "dm": dm, "dm_p": dm_p, "pt": pt, "pt_p": pt_p,
                             "collapse_rate": float(coll), "n_test": int(len(idx))})
            for b, parts in base.items():
                p = np.concatenate(parts)
                e = pooled_metrics(d, idx, p)
                dm, dm_p = wf.dm_test(p - yt, last - yt)
                pt, pt_p = wf.pt_test(np.sign(p - last), np.sign(yt - last))
                rows.append({"asset": asset, "variant": variant, "model": b, **{f"{k}_mean": v for k, v in e.items()},
                             **{f"ens_{k}": v for k, v in e.items()}, "dm": dm, "dm_p": dm_p, "pt": pt, "pt_p": pt_p,
                             "n_test": int(len(idx))})
            rows.append({"asset": asset, "variant": variant, "model": "RW", "rmse_ratio_mean": 1.0,
                         "da_paper_mean": wf.da_paper(last, yt), "da_known_mean": None, "n_test": int(len(idx))})
            for a_, b_ in (("QFWP", "QLSTM"), ("QLSTM", "LSTM1"), ("QFWP", "FWPC"), ("QFWP", "LSTM1")):
                x, z = per_seed[a_], per_seed[b_]
                pairs.append({"asset": asset, "variant": variant, "pair": f"{a_} vs {b_}",
                              "rmse_ratio_diff": float((x.rmse_ratio - z.rmse_ratio).mean()),
                              "p_rmse": float(wilcoxon(x.rmse_ratio, z.rmse_ratio).pvalue),
                              "da_paper_diff": float((x.da_paper - z.da_paper).mean()),
                              "p_da_paper": float(wilcoxon(x.da_paper, z.da_paper).pvalue),
                              "da_known_diff": float((x.da_known - z.da_known).mean()),
                              "wins_rmse": int((x.rmse_ratio < z.rmse_ratio).sum())})
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(folds), choice


def a2():
    r = pd.DataFrame([json.loads(p.read_text()) for p in (RES / "a2").glob("*/*.json")])
    r["rmse_ratio"] = r.rmse / r.rw_rmse
    g = r.groupby(["variant", "model", "budget", "batch"]).agg(
        updates=("updates", "first"), rmse_ratio=("rmse_ratio", "mean"), rmse_ratio_sd=("rmse_ratio", "std"),
        da_paper=("da_paper", "mean"), da_paper_sd=("da_paper", "std"), da_known=("da_known", "mean"),
        collapse=("rmse_ratio", lambda x: float((x > COLLAPSE).mean()))).reset_index()
    return g


def main():
    md, out = [], {}
    rows, pairs, folds, choice = a1()
    out["a1_rows"], out["a1_pairs"], out["a1_folds"] = rows.to_dict("records"), pairs.to_dict("records"), folds.to_dict("records")
    f = lambda x, n=3: "—" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{n}f}" if isinstance(x, float) else x)
    for variant, title in (("leaky", "有前視（論文的資料）"), ("aligned", "去前視")):
        for asset in wf.ASSETS:
            d = rows[(rows.asset == asset) & (rows.variant == variant)].copy()
            t = pd.DataFrame({"模型（參數）": [NAMES[m] for m in d.model],
                              "RMSE ÷ 隨機漫步（20 種子平均 ± std）": [f(r.rmse_ratio_mean, 4) + ("" if r.get("rmse_ratio_std") != r.get("rmse_ratio_std") or pd.isna(r.get("rmse_ratio_std")) else f" ± {r.rmse_ratio_std:.4f}") for _, r in d.iterrows()],
                              "DA 論文定義 %": [f(r.da_paper_mean, 2) for _, r in d.iterrows()],
                              "DA 對已知收盤 %": [f(r.da_known_mean, 2) for _, r in d.iterrows()],
                              "集成 RMSE 比": [f(r.get("ens_rmse_ratio"), 4) for _, r in d.iterrows()],
                              "DM p（vs 隨機漫步）": [f(r.get("dm_p"), 3) for _, r in d.iterrows()],
                              "PT p（方向）": [f(r.get("pt_p"), 3) for _, r in d.iterrows()],
                              "崩掉比例": [f(r.get("collapse_rate"), 3) for _, r in d.iterrows()]})
            md += [f"\n### A1：{asset}，{title}（10 個 fold 合併，測試 {int(d.n_test.iloc[0])} 天）\n", md_table(t)]
    t = pairs.copy()
    for c in ("rmse_ratio_diff", "da_paper_diff", "da_known_diff"):
        t[c] = t[c].map(lambda x: f"{x:+.4f}" if "rmse" in c else f"{x:+.2f}")
    for c in ("p_rmse", "p_da_paper"):
        t[c] = t[c].map(lambda x: f"{x:.4f}")
    t.columns = ["匯率", "資料", "比較", "RMSE 比差（負＝前者較好）", "p（RMSE）", "DA 論文定義差", "p（DA）", "DA 對已知收盤差", "前者 RMSE 較低的種子數（/20）"]
    md += ["\n### A1：成對比較（20 個種子，Wilcoxon）\n", md_table(t)]
    lrc = pd.Series([v["lr"] for v in choice.values()]).value_counts().to_dict()
    md += [f"\n驗證集選出的學習率次數：{lrc}\n"]
    g = a2()
    out["a2"] = g.to_dict("records")
    t = g.copy()
    for c in ("rmse_ratio", "rmse_ratio_sd"):
        t[c] = t[c].map(lambda x: f"{x:.4f}")
    for c in ("da_paper", "da_paper_sd", "da_known"):
        t[c] = t[c].map(lambda x: f"{x:.2f}")
    t["collapse"] = t["collapse"].map(lambda x: f"{x:.1f}")
    t.columns = ["資料", "模型", "訓練預算", "batch", "更新次數", "RMSE ÷ 隨機漫步", "std", "DA 論文定義 %", "std ", "DA 對已知收盤 %", "崩掉比例"]
    md += ["\n### A2：batch 的影響——固定 epoch vs. 固定更新次數（EUR/USD 單一切分，10 個種子）\n", md_table(t)]
    (RES / "summary_a.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    (RES / "summary_a.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
