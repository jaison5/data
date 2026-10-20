"""方向 A 的跨資產前視探針：Yahoo 日線的「當日高低價」是否洩漏隔天收盤的方向？不需要訓練任何模型。

規則（零參數）：預測「明天收盤相對今天收盤」的方向 = sign(高低價中點 − 今天收盤)。
  - 當天版：中點 = (今天 High + 今天 Low) / 2
  - 前一天版：中點 = (昨天 High + 昨天 Low) / 2（沒有前視，當作對照）
如果 K 棒的 Close 真的是收盤時的價格，當天的 High/Low 都發生在 Close 之前，兩個版本的準確率應該都接近 50%；
如果 Close 其實是 K 棒開始時的價格（Yahoo 匯率日線的情況），當天的 High/Low 涵蓋了 Close 之後的走勢，當天版就會遠高於 50%。

另外量 K 棒結構：|Close − Open| 的中位數 ÷ (High − Low) 的中位數。接近 0 代表 Close 幾乎等於 Open（Close 記在 K 棒開頭）。

資產：Yahoo 匯率 22 個、股票／ETF 6 檔（交易所有明確收盤，應該沒有前視，當作對照）、加密貨幣 4 個（24 小時交易，以 UTC 切日）。
期間：2003-12-01～2024-12-31（較晚上市的資產從有資料開始）。
用法：python probe_leakage.py        （資料存到 data/probe/，結果存到 results/probe/）
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "probe"
OUT = HERE / "results" / "probe"
START, END = "2003-12-01", "2024-12-31"
ASSETS = {
    "匯率": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "CNY=X", "HKD=X", "SGD=X",
           "INR=X", "KRW=X", "TWD=X", "MXN=X", "BRL=X", "ZAR=X", "SEK=X", "NOK=X", "EURJPY=X", "EURGBP=X",
           "GBPJPY=X", "EURCHF=X"],
    "股票／ETF（對照）": ["SPY", "QQQ", "AAPL", "MSFT", "2330.TW", "0050.TW"],
    "加密貨幣": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"],
}


def download(sym):
    p = DATA / (sym.replace("=", "_").replace("^", "") + ".csv")
    if not p.exists():
        import yfinance as yf
        df = yf.Ticker(sym).history(period="max", auto_adjust=False)[["Open", "High", "Low", "Close"]]
        df.index = df.index.tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index_label="date")
    return pd.read_csv(p, parse_dates=["date"]).set_index("date")


def clean(df, fx):
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df = df[(df > 0).all(axis=1)].dropna().sort_index().loc[START:END]
    removed = 0
    if fx:  # 匯率的錯誤報價（和前後兩天平均差 25% 以上），同 qfin-main/exp2.load_data
        for _ in range(10):
            lc = np.log(df)
            bad = ((lc - (lc.shift(1) + lc.shift(-1)) / 2).abs() > 0.25).any(axis=1)
            if not bad.any():
                break
            removed += int(bad.sum())
            df = df[~bad]
    return df, removed


def probe(df):
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    nxt = np.sign(c[2:] - c[1:-1])          # 明天收盤相對今天收盤（今天 = 第 1..n-2 天）
    same = np.sign((h[1:-1] + l[1:-1]) / 2 - c[1:-1])
    prev = np.sign((h[:-2] + l[:-2]) / 2 - c[1:-1])
    m = (nxt != 0) & (same != 0) & (prev != 0)
    n = int(m.sum())
    k_same, k_prev = int((same[m] == nxt[m]).sum()), int((prev[m] == nxt[m]).sum())
    from scipy.stats import binomtest
    rng = np.median(h - l)
    return {"days": n, "da_same_day": 100 * k_same / n, "da_prev_day": 100 * k_prev / n,
            "gap_pct_points": 100 * (k_same - k_prev) / n,
            "p_same_vs_50": binomtest(k_same, n, 0.5, alternative="greater").pvalue,
            "close_open_over_range": float(np.median(np.abs(c - o)) / rng) if rng > 0 else None,
            "open_vs_prev_close_over_range": float(np.median(np.abs(o[1:] - c[:-1])) / rng) if rng > 0 else None,
            "close_outside_hl_pct": float(100 * np.mean((c > h * (1 + 1e-9)) | (c < l * (1 - 1e-9))))}


def main():
    rows = []
    for cat, syms in ASSETS.items():
        for sym in syms:
            try:
                df, removed = clean(download(sym), cat == "匯率")
            except Exception as e:
                print(f"{sym}: 下載失敗 {e!r}", flush=True)
                continue
            if len(df) < 300:
                print(f"{sym}: 資料太少（{len(df)} 天），略過", flush=True)
                continue
            r = {"類別": cat, "代號": sym, "起": str(df.index[0].date()), "迄": str(df.index[-1].date()),
                 "刪除錯誤報價": removed, **probe(df)}
            rows.append(r)
            print(f"{cat:10s} {sym:9s} {r['days']:5d} 天  當天版 {r['da_same_day']:5.1f}%  前一天版 {r['da_prev_day']:5.1f}%  "
                  f"差 {r['gap_pct_points']:+5.1f}  |C−O|/範圍 {r['close_open_over_range']:.2f}", flush=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "probe_leakage.csv", index=False, encoding="utf-8-sig")
    (OUT / "probe_leakage.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float))
    # ---- 摘要表
    lines = ["| 類別 | 代號 | 期間 | 天數 | 當天高低價規則 DA % | 前一天高低價規則 DA % | 差（百分點） | 中位數 \\|Close−Open\\| ÷ 高低價差 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['類別']} | {r['代號']} | {r['起']}～{r['迄']} | {r['days']} | {r['da_same_day']:.1f} | "
                     f"{r['da_prev_day']:.1f} | {r['gap_pct_points']:+.1f} | {r['close_open_over_range']:.2f} |")
    agg = t.groupby("類別").agg(n=("代號", "count"), same=("da_same_day", "mean"), prev=("da_prev_day", "mean"),
                              smin=("da_same_day", "min"), smax=("da_same_day", "max"), co=("close_open_over_range", "median"))
    lines += ["", "| 類別 | 資產數 | 當天版平均 DA % | 範圍 | 前一天版平均 DA % | \\|Close−Open\\| ÷ 範圍（中位數） |", "|---|---|---|---|---|---|"]
    for cat, r in agg.iterrows():
        lines.append(f"| {cat} | {r.n} | {r.same:.1f} | {r.smin:.1f}～{r.smax:.1f} | {r.prev:.1f} | {r.co:.2f} |")
    (OUT / "summary_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-(len(agg) + 2):]), flush=True)


if __name__ == "__main__":
    main()
