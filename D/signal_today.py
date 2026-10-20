"""即時版的 QQQ 波動度目標：用到最近一個收盤為止的資料，算出下一個交易日要持有的 QQQ 比例，並記錄到
results/live_log.csv（樣本外前測：之後拿記錄和真的 QQQ 報酬比）。

和回測的「即時①」相同：只用下單前已經知道的收盤價，σ＝過去 20 個交易日報酬的年化標準差。
用法：python signal_today.py        （每個交易日收盤前或收盤後跑一次都可以，同一天重跑會覆蓋那一行）
這是研究用的訊號紀錄，不是投資建議。
"""
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

TARGET, LOOK = 0.20, 20
LOG = Path(__file__).parent / "results" / "live_log.csv"


def main():
    px = yf.download("QQQ", period="3mo", auto_adjust=True, progress=False)["Close"]
    px = px.squeeze().dropna()
    r = px.pct_change().dropna()
    sigma = r.iloc[-LOOK:].std() * math.sqrt(252)
    row = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "data_through": px.index[-1].date().isoformat(),
           "qqq_close": round(float(px.iloc[-1]), 2), "sigma_20d": round(float(sigma), 4),
           "weight_unlevered": round(min(1.0, TARGET / sigma), 4), "weight_cap_1_5": round(min(1.5, TARGET / sigma), 4)}
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    if len(log):
        log = log[log.data_through != row["data_through"]]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    LOG.parent.mkdir(exist_ok=True)
    log.to_csv(LOG, index=False)
    print(f"資料到 {row['data_through']}：QQQ 收盤 {row['qqq_close']}，過去 20 天年化波動度 {sigma:.1%}")
    print(f"下一個交易日持有 QQQ 比例：不加槓桿 {row['weight_unlevered']:.0%}；上限 1.5 倍版 {row['weight_cap_1_5']:.0%}（其餘放短期國庫券）")


if __name__ == "__main__":
    main()
