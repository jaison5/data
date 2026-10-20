"""D5 加上 D4 的量子／古典波動度預測版本，和前三大 ETF 比老師的夏普值（2010-09～2026-10，D4 從 2008 才有預測）。"""
import numpy as np
import pandas as pd

import d5_top_etf as D5
from common_d import RES
from realtime import run_close


def extra(rets, rf):
    F = pd.read_csv(RES / "d4_forecasts_QQQ.csv", index_col=0, parse_dates=True)
    r = rets["QQQ"]
    out = {}
    for m, lab in [("QLSTM", "QLSTM"), ("QFWP", "QFWP"), ("HAR", "HAR"), ("LSTM1", "LSTM h=1")]:
        sig = np.exp(F[m]).reindex(r.index)
        w = (0.20 / sig).clip(upper=1.0).shift(2).fillna(0.0)
        out[f"QQQ 波動度目標＋{lab} 預測 σ（不加槓桿）"] = run_close(w, r, rf)[0].loc["2008-01-02":]  # D4 從 2008 才有預測
    return out


if __name__ == "__main__":
    res, tests = D5.main(extra)
    res.to_csv(RES / "d5_top_etf_with_quantum.csv", index=False)
    tests.to_csv(RES / "d5_top_etf_tests_with_quantum.csv", index=False)
