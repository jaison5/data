"""方向 C 的資料、walk-forward 切分與 QLSTM 預測器（每個 fold 重新訓練）。

資料清理、技術指標和 qfin-main/exp2.py 完全相同（讀 C/data/ohlc_TWD_X.csv）。
fold：測試年 2020～2024（涵蓋論文原本的測試期），驗證年 = 前一年，訓練 = 2004-07 起到測試年前兩年的年底。
預測器用 ../B/fastsim.py 的 QLSTM（和 PennyLane 版等價，見 B/check_fastsim.py）。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
IN_COLS = ["Open", "High", "Low", "Close", "MA5", "MA10"]
SEQ, THR = 4, 0.012


def load_data():
    df = pd.read_csv(HERE / "data" / "ohlc_TWD_X.csv", parse_dates=["date"]).set_index("date")
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df = df[(df > 0).all(axis=1)].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index().loc["2000-01-01":"2024-12-31"].copy()
    for _ in range(10):  # 刪除錯誤報價（和前後兩天平均差 25% 以上），同 exp2.load_data
        lc = np.log(df[["Open", "High", "Low", "Close"]])
        bad = ((lc - (lc.shift(1) + lc.shift(-1)) / 2).abs() > 0.25).any(axis=1)
        if not bad.any():
            break
        df = df[~bad]
    c = df["Close"]
    for n in (5, 10, 20, 60):
        df[f"MA{n}"] = c.rolling(n).mean()
    df["VOL20"] = c.pct_change().rolling(20).std()
    df["VOLREL"] = df["VOL20"] / df["VOL20"].rolling(252, min_periods=60).mean()
    return df.dropna()


def split(df, test_year):
    """回傳 (訓練, 驗證, 測試) 的布林遮罩。"""
    y = df.index.year
    return y <= test_year - 2, y == test_year - 1, y == test_year


def labels(df):
    c = df["Close"].to_numpy()
    fwd = np.full(len(df), np.nan)
    fwd[:-5] = c[5:] / c[:-5] - 1
    return np.where(fwd > THR, 1, np.where(fwd < -THR, 0, -1)), fwd


def probs_oracle(df):
    """理想預測器（上限，用到未來 5 天的資訊）：會漲 1.2% 以上給 p_up=1，會跌 1.2% 以上給 p_down=1，其他 0.5。"""
    lab, _ = labels(df)
    p = np.full((len(df), 2), 0.5)
    p[lab == 1] = [0.0, 1.0]
    p[lab == 0] = [1.0, 0.0]
    return p


def train_forecaster(df, test_year, seed=0, epochs=50, batch=32, lr=5e-3):
    """和論文相同的 QLSTM 預測器（hidden 2、dense-softmax、RMSprop 5e-3、weighted CE、50 epochs），只用該 fold 的訓練期資料。
    回傳 (每天的 [p_down, p_up], 結果 dict)。"""
    import torch
    import fastsim as F
    torch.set_num_threads(1)
    tr_m, va_m, te_m = split(df, test_year)
    raw = df[IN_COLS].to_numpy()
    lo, hi = raw[tr_m].min(axis=0), raw[tr_m].max(axis=0)
    Z = (raw - lo) / (hi - lo)
    n = len(df)
    Xall = np.zeros((n, SEQ, len(IN_COLS)))
    for t in range(SEQ - 1, n):
        Xall[t] = Z[t - SEQ + 1:t + 1]
    lab, _ = labels(df)
    idx = np.arange(n)
    last_tr = idx[tr_m].max()
    tr = (idx >= SEQ - 1) & (idx + 5 <= last_tr) & (lab >= 0)
    va = va_m & (lab >= 0) & (idx + 5 < n)
    te = te_m & (lab >= 0) & (idx + 5 < n)
    torch.manual_seed(seed)
    model = F.QLSTM(n_in=len(IN_COLS), H=2, n_out=2)
    Xtr, ytr = torch.tensor(Xall[tr]), torch.tensor(lab[tr])
    counts = np.bincount(lab[tr], minlength=2)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (2.0 * counts)))
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), batch):
            j = torch.as_tensor(perm[s:s + batch])
            opt.zero_grad()
            lossf(model(Xtr[j]), ytr[j]).backward()
            opt.step()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(Xall)), dim=1).numpy()
    probs[: SEQ - 1] = 0.5
    acc = lambda m: float(np.mean(probs[m].argmax(1) == lab[m])) if m.sum() else None
    maj = lambda m: float(max(lab[m].mean(), 1 - lab[m].mean())) if m.sum() else None
    res = {"test_year": test_year, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "val_acc": acc(va), "test_acc": acc(te), "test_majority": maj(te),
           "test_correct": int((probs[te].argmax(1) == lab[te]).sum())}
    return probs, res
