"""彙整實驗二：預測器準確率、QA3C vs A3C 測試成績（對應論文 Table IV）、ETF 基準（Table V）。"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import DATA, RES

import sys
OUT = RES / (sys.argv[1] if len(sys.argv) > 1 else "exp2")  # 例如 exp2_pl
PAPER_T4 = {"A3C": {"total_return_pct": 11.42, "max_drawdown_pct": 0.92, "trades": 230, "win_rate_pct": 54.78,
                    "best_trade_pct": 0.61, "worst_trade_pct": -0.88},
            "QA3C": {"total_return_pct": 11.87, "max_drawdown_pct": 0.92, "trades": 231, "win_rate_pct": 56.71,
                     "best_trade_pct": 0.61, "worst_trade_pct": -0.88}}
PAPER_T5 = {"FXB": 13.81, "FXF": 12.21, "USDU": 4.91, "FXE": 4.25, "FXC": 1.74, "FXA": -5.79, "FXY": -29.60}


def main():
    o = {"forecaster": {}, "agents": {}, "etf": []}
    for kind in ("QLSTM", "LSTM"):
        rs = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"forecaster_{kind}_s*.json"))]
        if rs:
            acc = np.array([r["test_acc"] for r in rs])
            o["forecaster"][kind] = {"seeds": [r["seed"] for r in rs], "test_acc": acc.tolist(),
                                     "mean": float(acc.mean()), "std": float(acc.std(ddof=1)) if len(acc) > 1 else 0,
                                     "majority": rs[0]["test_acc_majority"], "n_test": rs[0]["n_test"],
                                     "n_train": rs[0]["n_train"], "n_params": rs[0]["n_params"],
                                     "balanced": [r["test_balanced_acc"] for r in rs]}
    keys = ["total_return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "best_trade_pct", "worst_trade_pct",
            "sharpe_ann", "time_in_market_pct", "sum_trade_pct"]
    for agent in ("QA3C", "A3C"):
        rs = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"rl_{agent}_s*.json"))]
        if not rs:
            continue
        per = [{"seed": r["seed"], **{k: r["test"].get(k) for k in keys}, "action_counts": r["test"]["action_counts"],
                "train_return_pct": r["train_eval"]["total_return_pct"], "episodes": r["episodes"],
                "minutes": r["train_minutes"], "n_params": r["n_params"]} for r in rs]
        df = pd.DataFrame(per)
        o["agents"][agent] = {"per_seed": per,
                              "mean": {k: float(pd.to_numeric(df[k], errors="coerce").mean()) for k in keys},
                              "std": {k: float(pd.to_numeric(df[k], errors="coerce").std(ddof=1)) for k in keys},
                              "paper": PAPER_T4[agent], "buy_hold_pct": rs[0]["test"]["buy_hold_pct"],
                              "test_window": [rs[0]["test"]["start"], rs[0]["test"]["end"]]}
        # 每個種子的資產曲線另存，畫圖用
    for s, p in PAPER_T5.items():
        e = pd.read_csv(DATA / f"etf_{s}.csv", parse_dates=["date"]).set_index("date").Close.loc["2020-04-01":"2025-04-30"]
        o["etf"].append({"etf": s, "ours_pct": float(100 * (e.iloc[-1] / e.iloc[0] - 1)), "paper_pct": p})
    (RES / f"{OUT.name}_summary.json").write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in o.items() if k != "agents"}, indent=1, default=float))
    for a, v in o["agents"].items():
        print(a, {k: round(x, 2) for k, x in v["mean"].items()}, "paper", v["paper"])
        for r in v["per_seed"]:
            print("   ", r)


if __name__ == "__main__":
    main()
