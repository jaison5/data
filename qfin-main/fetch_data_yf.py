"""從 Yahoo Finance 下載實驗資料到 data/。

用法：python fetch_data_yf.py   （需要 pip install yfinance）
輸出：
  data/ohlc_EURUSD_X.csv   實驗一：EUR/USD 日線 OHLC
  data/ohlc_TWD_X.csv      實驗二：USD/TWD 日線 OHLC
  data/etf_<代號>.csv       實驗二：貨幣 ETF 還原權值收盤價
"""
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def save(df, name, cols):
    df = df[cols].copy()
    df.index = df.index.tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df.to_csv(OUT / name, index_label="date")
    print(f"{name:22s} {len(df):6d} 筆  {df.index.min().date()} ~ {df.index.max().date()}")


for sym, name in [("EURUSD=X", "ohlc_EURUSD_X.csv"), ("TWD=X", "ohlc_TWD_X.csv")]:
    save(yf.Ticker(sym).history(period="max", auto_adjust=False), name, ["Open", "High", "Low", "Close", "Volume"])

for etf in ["FXB", "FXF", "USDU", "FXE", "FXC", "FXA", "FXY"]:
    save(yf.Ticker(etf).history(period="max", auto_adjust=True), f"etf_{etf}.csv", ["Close"])
