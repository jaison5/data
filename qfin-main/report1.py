"""彙整實驗一：對應論文 Table 2（準確率）、Table 3（時間）、Table 4（加速比）。"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from common import RES

PAPER_T2 = {  # 論文 Table 2：batch → (RMSE QLSTM, RMSE QFWP, DA QLSTM, DA QFWP)
    4: (0.0034, 0.0030, 69.04, 72.72), 8: (0.0036, 0.0029, 67.89, 72.75), 16: (0.0039, 0.0029, 65.56, 72.34),
    32: (0.0043, 0.0031, 62.34, 70.18), 64: (0.0049, 0.0034, 58.67, 65.84)}
PAPER_FULL_SPEEDUP = {"QLSTM": {4: 1.41, 8: 1.67, 16: 1.90, 32: 2.08, 64: 2.23},
                      "QFWP": {4: 1.37, 8: 1.63, 16: 1.77, 32: 1.94, 64: 2.02}}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def load(variant="paper"):
    rows = [json.loads(p.read_text()) for p in (RES / f"exp1a_{variant}").glob("*_b*_s*.json")]
    return pd.DataFrame(rows), json.loads((RES / f"exp1a_{variant}" / "_data.json").read_text())


def tables(variant="paper"):
    df, base = load(variant)
    out = {"data": base, "n_runs": int(len(df))}
    acc = []
    for b in sorted(df.batch.unique()):
        r = {"batch": "non-batch" if b == 1 else int(b)}
        q = df[(df.model == "QLSTM") & (df.batch == b)].sort_values("seed")
        f = df[(df.model == "QFWP") & (df.batch == b)].sort_values("seed")
        l = df[(df.model == "LSTM") & (df.batch == b)].sort_values("seed")
        for name, d in (("QLSTM", q), ("QFWP", f), ("LSTM", l)):
            for m in ("rmse", "da_paper", "da_known"):
                r[f"{m}_{name}"] = (float(d[m].mean()), float(d[m].std(ddof=1)) if len(d) > 1 else 0.0, int(len(d)))
        common = sorted(set(q.seed) & set(f.seed))
        if len(common) >= 5:
            qa = q.set_index("seed").loc[common]; fa = f.set_index("seed").loc[common]
            r["p_rmse"] = float(wilcoxon(qa.rmse, fa.rmse).pvalue)
            r["p_da"] = float(wilcoxon(qa.da_paper, fa.da_paper).pvalue)
            r["cliff_rmse"] = float(cliffs_delta(qa.rmse, fa.rmse))
        if b in PAPER_T2:
            r["paper"] = PAPER_T2[b]
        acc.append(r)
    out["accuracy"] = acc
    tim = []
    for name in ("QLSTM", "QFWP", "LSTM"):
        nb = df[(df.model == name) & (df.batch == 1)].set_index("seed")
        for b in sorted(df.batch.unique()):
            d = df[(df.model == name) & (df.batch == b)].set_index("seed")
            row = {"model": name, "batch": int(b)}
            for c in ("train_forward_s", "backward_s", "full_train_s", "inference_s"):
                row[c] = [float(x) for x in np.percentile(d[c], [50, 25, 75])] if len(d) else None
                if b != 1 and len(nb):
                    s = (nb[c] / d[c]).dropna()
                    row[c + "_speedup"] = [float(x) for x in np.percentile(s, [50, 25, 75])] if len(s) else None
            if name in PAPER_FULL_SPEEDUP and b in PAPER_FULL_SPEEDUP[name]:
                row["paper_full_speedup"] = PAPER_FULL_SPEEDUP[name][b]
            tim.append(row)
    out["timing"] = tim
    return out


def fmt(t):
    m, s, n = t
    return f"{m:.5f} ± {s:.5f}" if m < 1 else f"{m:.2f} ± {s:.2f}"


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "paper"  # 例如 paper_adam0.05
    o = tables(v)
    print(json.dumps(o["data"], indent=1))
    print(f"runs: {o['n_runs']}")
    for r in o["accuracy"]:
        print(r["batch"], "| RMSE Q", fmt(r["rmse_QLSTM"]), "F", fmt(r["rmse_QFWP"]), "L", fmt(r["rmse_LSTM"]),
              "| DA Q", fmt(r["da_paper_QLSTM"]), "F", fmt(r["da_paper_QFWP"]), "L", fmt(r["da_paper_LSTM"]),
              "| p", round(r.get("p_rmse", np.nan), 4), "| paper", r.get("paper"))
    for r in o["timing"]:
        print(r["model"], r["batch"], "full", [round(x, 1) for x in r["full_train_s"]] if r["full_train_s"] else None,
              "speedup", [round(x, 2) for x in r.get("full_train_s_speedup") or []], "paper", r.get("paper_full_speedup"))
    (RES / f"exp1a_{v}_summary.json").write_text(json.dumps(o, indent=1, default=float))
