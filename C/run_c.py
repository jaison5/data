"""方向 C 的總控：先訓練每個 fold 的 QLSTM 預測器、算規則型基準，再依序跑所有代理人的訓練（同時跑 K 個）。

實驗組合（訓練成本 0.05%，測試時再用 0／0.05%／0.1% 三種成本評估；5 個 fold × 3 個種子）：
  E1 主實驗   ：獎勵 logret、QLSTM 預測機率；代理人 QA3C、A3C_small、A3C
  E2 獎勵消融 ：獎勵 paper（論文 Table III，有遮罩與成本）；代理人 QA3C、A3C_small
  E3 預測器消融：獎勵 logret；A3C_small 搭配「沒有預測器（0.5）」與「理想預測器（用到未來資訊，當上限）」

用法：python run_c.py --parallel 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
SEEDS = [0, 1, 2]


def forecaster_job(year):
    import cdata
    import pandas as pd
    df = cdata.load_data()
    probs, res = cdata.train_forecaster(df, year)
    d = HERE / "results" / "forecaster"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": df.index, "p_down": probs[:, 0], "p_up": probs[:, 1]}).to_csv(d / f"probs_qlstm_y{year}.csv", index=False)
    (d / f"forecaster_y{year}.json").write_text(json.dumps(res, indent=1))
    print(f"預測器 {year}：測試準確率 {res['test_acc']:.3f}（{res['n_test']} 筆，多數類別 {res['test_majority']:.3f}）", flush=True)


def baselines():
    import numpy as np
    import cdata
    from env_c import evaluate, policy_buy_hold, make_policy_ma, make_policy_random
    from train_c import EVAL_COSTS
    df = cdata.load_data()
    out = []
    for year in cdata.TEST_YEARS:
        _, _, te = cdata.split(df, year)
        dte, pte = df[te], np.full((te.sum(), 2), 0.5)
        for c in EVAL_COSTS:
            out.append({"test_year": year, "cost": c, "policy": "buy_hold", **evaluate(policy_buy_hold, dte, pte, c)})
            out.append({"test_year": year, "cost": c, "policy": "ma5_ma20", **evaluate(make_policy_ma(dte), dte, pte, c)})
            for s in range(20):
                out.append({"test_year": year, "cost": c, "policy": "random", "seed": s,
                            **evaluate(make_policy_random(s), dte, pte, c)})
    (HERE / "results" / "baselines.json").write_text(json.dumps(out))
    print(f"規則型基準完成：{len(out)} 筆", flush=True)


def jobs():
    import cdata
    J = []
    for agent in ("QA3C", "A3C_small", "A3C"):  # QA3C 最慢，先跑
        for reward, probs, agents in (("logret", "qlstm", ("QA3C", "A3C_small", "A3C")),
                                      ("paper", "qlstm", ("QA3C", "A3C_small")),
                                      ("logret", "none", ("A3C_small",)), ("logret", "oracle", ("A3C_small",))):
            if agent not in agents:
                continue
            for y in cdata.TEST_YEARS:
                for s in SEEDS:
                    J.append((agent, reward, probs, y, s))
    return J


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import cdata
    (HERE / "logs").mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(len(cdata.TEST_YEARS)) as pool:
        pool.map(forecaster_job, cdata.TEST_YEARS)
    baselines()
    todo = [j for j in jobs() if not (HERE / "results" / "rl" / f"{j[0]}_{j[1]}_{j[2]}_y{j[3]}_s{j[4]}.json").exists()]
    print(f"代理人訓練：共 {len(jobs())} 次，待跑 {len(todo)} 次，同時 {a.parallel} 個", flush=True)
    running, done = [], 0
    t0 = time.time()
    ptxt = HERE / "parallel.txt"  # 執行中可改這個檔案調整同時跑幾個（例如方向 A 跑完後調高）
    ptxt.write_text(str(a.parallel))
    while todo or running:
        try:
            par = int(ptxt.read_text().strip())
        except (OSError, ValueError):
            par = a.parallel
        while todo and len(running) < par:
            agent, reward, probs, y, s = todo.pop(0)
            name = f"{agent}_{reward}_{probs}_y{y}_s{s}"
            f = open(HERE / "logs" / f"{name}.log", "w", encoding="utf-8")
            p = subprocess.Popen([PY, str(HERE / "train_c.py"), "--fold", str(y), "--agent", agent, "--reward", reward,
                                  "--probs", probs, "--seed", str(s)], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
            running.append((p, f, name))
        time.sleep(5)
        for item in running[:]:
            p, f, name = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                done += 1
                status = "完成" if p.returncode == 0 else f"失敗（代碼 {p.returncode}）"
                print(f"[{done}] {name} {status}，累計 {(time.time() - t0) / 60:.0f} 分鐘", flush=True)
    print("C 全部完成", flush=True)
