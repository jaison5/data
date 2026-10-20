from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from ad import T, Adam, RMSprop, getitem, mean, square, n_params

# 資料位置：優先用 qfin/data/；沒有的話用上一層的 research/data/（fetch_data.py 的輸出位置）
DATA = Path(__file__).parent / "data"
if not DATA.exists() and (Path(__file__).parent.parent / "data").exists():
    DATA = Path(__file__).parent.parent / "data"
# 結果位置：預設 qfin/results/；設定環境變數 QFIN_RES 可另存（例如換一台機器重跑時，不覆蓋也不略過舊結果）
RES = Path(os.environ.get("QFIN_RES") or Path(__file__).parent / "results")


def load_ohlc(sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"ohlc_{sym}.csv", parse_dates=["date"]).set_index("date")
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df = df[(df > 0).all(axis=1)].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def windows_minmax(ohlc: np.ndarray, seq: int = 5):
    """論文的資料處理：每個長度 seq 的視窗各自 min–max 正規化，目標（隔天收盤）用同一個尺度。
    回傳 X (N, seq, 4)、y (N,)（已正規化）、lo、hi、last_close、target_close、end_index。"""
    X, y, lo, hi, last, tgt, end = [], [], [], [], [], [], []
    for i in range(seq - 1, len(ohlc) - 1):
        w = ohlc[i - seq + 1:i + 1]
        mn, mx = w.min(), w.max()
        if mx - mn < 1e-12:
            continue
        X.append((w - mn) / (mx - mn))
        y.append((ohlc[i + 1, 3] - mn) / (mx - mn))
        lo.append(mn); hi.append(mx); last.append(ohlc[i, 3]); tgt.append(ohlc[i + 1, 3]); end.append(i)
    return (np.array(X), np.array(y), np.array(lo), np.array(hi), np.array(last), np.array(tgt), np.array(end))


def da_paper(yhat: np.ndarray, y: np.ndarray) -> float:
    """論文的 Directional Accuracy：sign(ŷ_t − ŷ_{t−1}) 是否等於 sign(y_t − y_{t−1})，只算 y_t ≠ y_{t−1} 的點。"""
    dy, dp = np.diff(y), np.diff(yhat)
    m = dy != 0
    return float(100 * np.mean(np.sign(dp[m]) == np.sign(dy[m])))


def da_known(yhat: np.ndarray, y: np.ndarray, last: np.ndarray) -> float:
    """標準的方向準確率：預測的隔日收盤 ŷ 和「今天已知的收盤」比，方向是否和實際一致。"""
    d = y - last
    m = d != 0
    return float(100 * np.mean(np.sign(yhat[m] - last[m]) == np.sign(d[m])))


def train_regressor(model, Xtr, ytr, batch: int, epochs: int, lr: float, seed: int, opt: str = "adam"):
    """batch=1 代表論文的 non-batch（逐筆更新）。回傳各部分花費的秒數。"""
    ps = model.params()
    o = Adam(ps, lr) if opt == "adam" else RMSprop(ps, lr)
    rng = np.random.default_rng(seed + 1000)
    tf = tb = 0.0
    t0 = time.perf_counter()
    n = len(Xtr)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s:s + batch]
            o.zero()
            a = time.perf_counter()
            loss = mean(square(_col(model(Xtr[j])) - T(ytr[j])))
            b = time.perf_counter()
            loss.backward()
            c = time.perf_counter()
            o.step()
            tf += b - a; tb += c - b
    return {"train_forward_s": tf, "backward_s": tb, "full_train_s": time.perf_counter() - t0,
            "steps_per_epoch": int(np.ceil(n / batch))}


def _col(t):
    return getitem(t, (slice(None), 0))


def predict(model, X, batch: int = 256):
    out = []
    t0 = time.perf_counter()
    for s in range(0, len(X), batch):
        out.append(model(X[s:s + batch]).data[:, 0])
    return np.concatenate(out), time.perf_counter() - t0


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
