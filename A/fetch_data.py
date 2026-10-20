"""方向 A 的資料：Yahoo Finance 三個匯率日線（EUR/USD、GBP/USD、USD/JPY）。
用法：python fetch_data.py   輸出到 A/data/ohlc_<代號>.csv"""
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
for sym in ["EURUSD=X", "GBPUSD=X", "JPY=X"]:
    df = yf.Ticker(sym).history(period="max", auto_adjust=False)[["Open", "High", "Low", "Close"]]
    df.index = df.index.tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    name = "ohlc_" + sym.replace("=", "_") + ".csv"
    df.to_csv(OUT / name, index_label="date")
    print(f"{name:20s} {len(df):6d} 筆  {df.index.min().date()} ~ {df.index.max().date()}")
