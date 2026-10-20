"""Shared data loading and backtest helpers for direction D (US equity strategies vs. SPY / QQQ)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results"
TICKERS = ["SPY", "QQQ", "IEF", "^IRX"]
START, END = "1999-03-10", "2026-10-20"   # QQQ inception .. last close before today (2026-10-19)
LAST = "2026-10-19"
COST = 0.0005          # one-way cost per unit of turnover (same as direction C)
BORROW_SPREAD = 0.005  # yearly spread over T-bill on the borrowed part of a levered position
TDAYS = 252


def fetch(force=False):
    import yfinance as yf
    DATA.mkdir(exist_ok=True)
    out = DATA / "us_daily.csv"
    if out.exists() and not force:
        return
    px = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px.to_csv(out)


def load(start="2003-01-02", end=None):
    """Adjusted closes -> daily simple returns. rf = 13-week T-bill (^IRX, % p.a.) per trading day."""
    px = pd.read_csv(DATA / "us_daily.csv", index_col=0, parse_dates=True)
    irx = px.pop("^IRX").ffill()
    px = px.dropna(subset=["SPY", "QQQ"])   # IEF only starts 2002-07-30; do not cut the 2000-02 bear market
    ret = px.pct_change()
    ret["RF"] = (irx.reindex(ret.index).ffill() / 100 / TDAYS)
    ret = ret.dropna(subset=["SPY", "QQQ", "RF"])
    ret["IEF"] = ret["IEF"].fillna(ret["RF"])   # before IEF existed, its slot earns the T-bill rate
    px = px.loc[ret.index]
    sel = slice(start, end)
    return px, ret, sel


def backtest(w, ret, cost=COST):
    """w: DataFrame of weights decided at the close of day t-1 and held over day t (already lagged by caller
    or not: this function lags by one day). Columns are asset names in `ret`; the rest is cash at RF
    (negative cash = borrowing at RF + BORROW_SPREAD)."""
    w = w.reindex(ret.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * ret[w.columns]).sum(axis=1)
    cash = 1.0 - held.sum(axis=1)
    rf = ret["RF"]
    fin = cash * rf - np.maximum(-cash, 0) * BORROW_SPREAD / TDAYS
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    tc = cost * turnover.shift(1).fillna(0.0)   # pay when the new weights start being held
    return gross + fin - tc, turnover


def stats(r, rf, turnover=None):
    r = r.dropna()
    rf = rf.reindex(r.index)
    n = len(r)
    eq = (1 + r).cumprod()
    yrs = n / TDAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    ex = r - rf
    vol = r.std() * np.sqrt(TDAYS)
    sharpe = ex.mean() / ex.std() * np.sqrt(TDAYS)
    dd = (eq / eq.cummax() - 1).min()
    down = ex[ex < 0]
    sortino = ex.mean() * TDAYS / (np.sqrt((down ** 2).sum() / n) * np.sqrt(TDAYS))
    out = dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=dd,
               Calmar=cagr / abs(dd) if dd < 0 else np.nan, Final=eq.iloc[-1])
    if turnover is not None:
        out["Turnover/yr"] = turnover.reindex(r.index).sum() / yrs
    return out
