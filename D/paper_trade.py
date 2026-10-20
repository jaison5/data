"""一個月的即時模擬交易（real paper trading），從 2026-10-19 收盤後開始。

規則（和回測的「即時①」相同，事前固定，不再更改）：
  - 每個交易日美股收盤後跑一次。用「到今天收盤為止」的資料算出每個策略的目標持有比例，
    當作「下一個交易日收盤」的市價收盤單（MOC）；下一次跑的時候用那天的收盤價成交並記帳。
  - 第一筆單：2026-10-19 收盤後算出，2026-10-20 收盤成交；績效從 2026-10-21 起算。
  - 每個帳戶虛擬資金 1,000,000 美元；未投入的部分賺三個月國庫券利率（^IRX）；槓桿部分付國庫券＋0.5%；
    單邊成本 0.05%（成交金額）。ETF 對照組也在 2026-10-20 收盤買進，之後不動。
  - 量子版本：每天用 D4 的 2026 年模型（訓練 ≤2024、驗證 2025、學習率在驗證年選好、10 個種子）
    重新算一次波動度預測。CPU 上訓練是確定性的，同一個模型每天重算，舊日期的預測不會變，只多出新的一天。

帳戶：
  VT-20d       QQQ 波動度目標（過去 20 天 σ，不加槓桿）       ← 不用量子
  VT-20d-1.5x  同上，上限 1.5 倍                             ← 不用量子
  VT-QLSTM     QQQ 波動度目標，σ 用 QLSTM 預測（不加槓桿）   ← 用量子工具
  VT-QFWP      QQQ 波動度目標，σ 用 QFWP 預測（不加槓桿）    ← 用量子工具
  VOO / IVV / SPY / QQQ  買進持有（美國前三大 ETF＋QQQ）

用法：python paper_trade.py            （每個交易日收盤後跑一次；重跑同一天不會重複記帳）
輸出：results/paper/orders.csv、ledger.csv、summary.md
這是研究用的模擬紀錄，不是投資建議，也沒有真的下單。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

LIVE = HERE / "data" / "live"
OUT = HERE / "results" / "paper"
START_SIGNAL = pd.Timestamp("2026-10-19")   # 第一筆單用到這天收盤
CAPITAL = 1_000_000.0
COST, SPREAD, TDAYS = 0.0005, 0.005, 252
STRATS = ["VT-20d", "VT-20d-1.5x", "VT-QLSTM", "VT-QFWP"]
ETFS = ["VOO", "IVV", "SPY", "QQQ"]
QUANTUM = {"VT-QLSTM": "QLSTM", "VT-QFWP": "QFWP"}


FETCH = """
import sys, yfinance as yf
out = sys.argv[1]
for t in ["QQQ", "VOO", "IVV", "SPY", "^IRX"]:
    df = yf.download(t, start="1999-03-10", auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df[["Open", "High", "Low", "Close"]].dropna().rename_axis("date").to_csv(out + "/ohlc_" + t.replace("^", "") + ".csv")
"""


def fetch():
    """用有 yfinance 的環境（qfin-main/.venv-pl）下載；訓練用 B/.venv-gpu（和 D4 同一版 torch，結果才會一致）。"""
    import subprocess
    LIVE.mkdir(parents=True, exist_ok=True)
    py = HERE.parent / "qfin-main" / ".venv-pl" / "Scripts" / "python.exe"
    subprocess.run([str(py), "-c", FETCH, str(LIVE)], check=True)
    return pd.read_csv(LIVE / "ohlc_QQQ.csv", parse_dates=["date"])["date"].max()


def quantum_sigma(last_day):
    """D4 的 2026 模型對 2026 年每一天的 log σ̂（快取；有新日期才重算）。"""
    cache = OUT / "quantum_forecasts.csv"
    old = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else None
    if old is not None and old.index.max() >= last_day:
        return old
    import d4_vol as D
    if last_day <= pd.Timestamp("2026-10-19"):
        # 研究時（資料到 2026-10-19）已經算好的 2026 年預測，直接用
        F = pd.read_csv(HERE / "results" / "d4_forecasts_QQQ.csv", index_col=0, parse_dates=True)
        out = F.loc["2026-01-01":last_day, list(QUANTUM.values())]
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache)
        return out
    from gpu_batch import train_group
    import torch
    torch.set_num_threads(1)   # 和 D4 訓練時相同（單執行緒），結果才會逐位一致
    D.DATA, D.LAST = LIVE, last_day.strftime("%Y-%m-%d")
    R = D.raw("QQQ")
    d, tr, va, te = D.fold_data(R, 2026)
    out = pd.DataFrame(index=R["dates"][te])
    for m in QUANTUM.values():
        val = {lr: np.mean([json.loads(D.path("QQQ", 2026, m, lr, s).read_text())["val_rmse"] for s in range(D.N_SEL)]) for lr in D.LRS}
        lr = min(val, key=val.get)
        seeds = list(range(D.N_SEL + D.N_ADD))
        res, _ = train_group(d, tr, va, te, m, [lr] * len(seeds), seeds, "cpu")
        out[m] = np.mean([r["pred_test"] for r in res], axis=0)
        # 檢查：和研究時（資料到 2026-10-19）存下的預測是否一致
        ref = np.mean([json.loads(D.path("QQQ", 2026, m, lr, s).read_text())["pred_test"] for s in seeds], axis=0)
        n = min(len(ref), len(out))
        out.attrs[f"{m}_max_diff_vs_research"] = float(np.abs(out[m].values[:n] - ref[:n]).max())
        print(f"{m}: lr={lr}, 與研究時的預測最大差 {out.attrs[f'{m}_max_diff_vs_research']:.2e}")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def targets(px, qsig):
    """用到每天收盤為止的資料，算出下一個交易日收盤要持有的比例。"""
    r = px["QQQ"].pct_change()
    sig20 = r.rolling(20).std() * math.sqrt(TDAYS)
    T = pd.DataFrame(index=px.index)
    T["VT-20d"] = (0.20 / sig20).clip(upper=1.0)
    T["VT-20d-1.5x"] = (0.20 / sig20).clip(upper=1.5)
    for k, m in QUANTUM.items():
        T[k] = (0.20 / np.exp(qsig[m])).reindex(px.index).clip(upper=1.0)
    return T


def run():
    last = fetch()
    px = pd.DataFrame({t: pd.read_csv(LIVE / f"ohlc_{t}.csv", parse_dates=["date"]).set_index("date")["Close"] for t in ETFS})
    px = px.dropna()
    irx = pd.read_csv(LIVE / "ohlc_IRX.csv", parse_dates=["date"]).set_index("date")["Close"].reindex(px.index).ffill()
    rf = irx / 100 / TDAYS
    qsig = quantum_sigma(last)
    T = targets(px, qsig)
    # 已經記錄過的單不再改：用當初記下的比例；新算的若和舊紀錄不同只寫警告
    rec_file = OUT / "orders.csv"
    if rec_file.exists():
        rec = pd.read_csv(rec_file, parse_dates=["signal_date"]).pivot(index="signal_date", columns="account", values="target_weight_next_close")
        for t in rec.index:
            for k in STRATS:
                if t in T.index and abs(float(T.loc[t, k]) - float(rec.loc[t, k])) > 1e-3:
                    print(f"注意：{t:%Y-%m-%d} {k} 重算 {float(T.loc[t, k]):.4f} 和當初紀錄 {float(rec.loc[t, k]):.4f} 不同，沿用當初紀錄")
                T.loc[t, k] = float(rec.loc[t, k])

    days = px.index[px.index >= START_SIGNAL]
    rows, orders = [], []
    nav = {k: CAPITAL for k in STRATS + ETFS}
    held = {k: 0.0 for k in STRATS + ETFS}
    for i, t in enumerate(days):
        ret = px.loc[t] / px.shift(1).loc[t] - 1
        if i >= 2:   # 第一個 P&L 日：2026-10-21
            for k in STRATS + ETFS:
                w = held[k]
                cash = 1 - w
                nav[k] *= 1 + w * ret["QQQ" if k in STRATS else k] + cash * rf.loc[t] - max(-cash, 0) * SPREAD / TDAYS
        if i >= 1:   # 在 t 收盤成交前一天收盤算好的單
            for k in STRATS:
                new = float(T.loc[days[i - 1], k])
                nav[k] -= nav[k] * COST * abs(new - held[k])
                held[k] = new
            for k in ETFS:
                if held[k] == 0.0:
                    nav[k] -= nav[k] * COST
                    held[k] = 1.0
        for k in STRATS + ETFS:
            rows.append(dict(date=t.date(), account=k, weight_held=held[k], nav=round(nav[k], 2)))
        for k in STRATS:
            orders.append(dict(signal_date=t.date(), account=k, target_weight_next_close=round(float(T.loc[t, k]), 4)))
    ledger = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT / "ledger.csv", index=False)
    pd.DataFrame(orders).to_csv(OUT / "orders.csv", index=False)
    summary(ledger, rf, last, T)


def summary(ledger, rf, last, T):
    L = ledger.pivot(index="date", columns="account", values="nav")
    L.index = pd.to_datetime(L.index)
    pnl = L.loc[L.index >= pd.Timestamp("2026-10-20")]
    lines = [f"# 即時模擬交易（paper trading）", "", f"更新：{datetime.now():%Y-%m-%d %H:%M}，資料到 {last:%Y-%m-%d}（美股收盤）。",
             "第一筆單 2026-10-19 收盤後算出、2026-10-20 收盤成交；績效從 2026-10-21 起算。", ""]
    n = len(pnl) - 1
    lines.append(f"已累積 {max(n, 0)} 個交易日的績效。")
    lines.append("")
    lines.append("| 帳戶 | 用到量子工具 | 累積報酬 | 年化波動度 | 夏普值（年化報酬÷年化波動度） | 最大回撤 | 目前持有 QQQ 比例 |")
    lines.append("|---|---|---|---|---|---|---|")
    hold = ledger[ledger.date == ledger.date.max()].set_index("account")["weight_held"]
    for k in STRATS + ETFS:
        x = pnl[k].pct_change().dropna()
        cum = pnl[k].iloc[-1] / pnl[k].iloc[0] - 1 if len(pnl) > 1 else 0.0
        if len(x) >= 5:
            vol = x.std() * math.sqrt(TDAYS)
            ann = (1 + cum) ** (TDAYS / len(x)) - 1
            sh = f"{ann / vol:.2f}" if vol > 0 else "—"
            vs = f"{vol:.1%}"
        else:
            sh, vs = "（不到 5 天）", "—"
        dd = (pnl[k] / pnl[k].cummax() - 1).min() if len(pnl) > 1 else 0.0
        q = "是" if k in QUANTUM else "否"
        lines.append(f"| {k} | {q} | {cum:+.2%} | {vs} | {sh} | {dd:.2%} | {hold.get(k, 0):.0%} |")
    lines += ["", "下一個交易日收盤要調整成的持有比例：", ""]
    for k in STRATS:
        lines.append(f"- {k}：{float(T[k].dropna().iloc[-1]):.1%}")
    lines += ["", "一個月的資料很少，年化數字只能看方向，不能當統計證據。研究用的模擬紀錄，不是投資建議，沒有真的下單。"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
