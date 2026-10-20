# 量子序列模型在金融交易上的實驗：程式碼與數據

本資料夾全部實驗實際使用的程式碼、事先登記的判定標準、原始資料快照、每一次訓練的輸出、彙整結果與執行紀錄，供檢核每個實驗的過程與數據。

## 資料夾結構

| 資料夾 | 內容 |
|---|---|
| `qfin-main/` | 重現兩篇論文（arXiv 2512.21820 QLSTM vs. QFWP；arXiv 2509.09176 QLSTM＋QA3C）的 PennyLane 程式，以及 Mac、Windows 兩台電腦各自從頭跑的結果與紀錄 |
| `A/` | Yahoo 匯率日線的前視洩漏稽核（22 個匯率與股票、加密貨幣的探針），以及無前視的 walk-forward 基準 |
| `B/` | 批次化量子電路模擬器 `fastsim.py`（純 PyTorch）、它和 PennyLane 等價的檢查 `check_fastsim.py`，以及批次訓練的效能剖析；所有量子模型都用它訓練 |
| `C/` | QA3C 強化學習交易的重新評估（有交易成本、動作遮罩、walk-forward） |
| `D/` | 用預測準確度與量子模型贏過 SPY／QQQ 與美國前三大 ETF 的交易策略實驗 D0～D9：`PROTOCOL.md`（事先寫好的判定標準）、程式、`data/`（資料快照）、`results/`（每次訓練的輸出與彙整表）、`logs/`（執行紀錄） |
| `CHECKSUMS.sha256` | 每個檔案的 SHA-256，可確認檔案沒有被改動 |

每個實驗資料夾都有自己的 `README.md`，寫明目的、結論、檔案與重跑方式。

##  D 的實驗

| 編號 | 問題 | 程式 | 主要結果檔（`D/results/`） |
|---|---|---|---|
| D0 | 每天猜漲跌要多準才贏得過買進持有 QQQ？（2003 起，另延伸到 1986 起的 Nasdaq-100） | `feasibility.py`、`threshold_long.py` | `feasibility_accuracy.csv`、`feasibility_breakeven.json`、`threshold_long*.csv` |
| D1 | QLSTM／QFWP 預測漲跌，能否改善策略？ | `run_d1.py`、`report_d1.py` | `d1_direction.csv`、`d1_strategies.csv`、`d1/`（每次訓練） |
| D2 | 波動度目標規則的穩健性（27 組參數、97 個變體、成本、多重比較） | `robustness.py` | `d2_*.csv`、`d2_meta.json` |
| D3 | 即時成交方式、真實 ETF 執行、30 年滾動視窗 | `realtime.py` | `d3_*.csv` |
| D4 | QLSTM／QFWP 預測未來 20 天波動度，放進波動度目標策略 | `d4_vol.py` | `d4_forecast_eval.csv`、`d4_strategy.csv`、`d4_forecasts_*.csv`、`d4/` |
| D5 | 和美國前三大 ETF（VOO、IVV、SPY）比年化報酬 ÷ 年化波動度 | `d5_top_etf.py`、`d5_with_quantum.py` | `d5_*.csv` |
| D6 | 量子模型直接輸出持有比例（端到端，用 Sharpe 訓練） | `d6_e2e.py` | `d6_*.csv`、`d6/` |
| D7 | 用 Nasdaq-100 指數延伸到 1995～2026（32 個年度 fold） | `d7_ndx.py` | `d7_*.csv`、`d7/` |
| D8 | 量子結構是否必要（正則化對等、傅立葉特徵、資料量學習曲線） | `d8_controls.py` | `d8_*.csv`、`d8_verdict.json`、`d8/` |
| D9 | 在硬體雜訊與有限測量次數下結果是否仍成立 | `d9_noise.py` | `d9_*.csv`、`d9_meta.json`、`d9/params/`（模型參數）、`d9/pred/`（各雜訊設定的輸出） |
| — | 即時模擬交易紀錄 | `paper_trade.py`、`signal_today.py` | `paper/`、`live_log.csv` |

各實驗的詳細結論見 `D/README.md`；判定標準見 `D/PROTOCOL.md`。

## 事先寫入（pre-registration）

D4 起的每個實驗，都先把模型、比較對象與成功標準寫進 `D/PROTOCOL.md`，再開始訓練；不論結果好壞都照實報告。下表列出 `PROTOCOL.md` 各節記載的撰寫時間，以及該實驗結果檔最早與最晚的修改時間（來自原始電腦的檔案系統，複製時保留；只供參考，不是獨立的時間證明）。

| 實驗 | PROTOCOL.md 記載的撰寫時間 | 結果檔數 | 最早修改時間 | 最晚修改時間 |
|---|---|---|---|---|
| D1 | 見 PROTOCOL.md 開頭（方向 D 的實驗規劃） | 3,838 | 2026-10-20 00:38 | 2026-10-20 01:20 |
| D4 | 2026-10-20 01:15 | 3,800 | 2026-10-20 01:24 | 2026-10-20 01:33 |
| D6 | 2026-10-20 01:40 | 4,560 | 2026-10-20 02:08 | 2026-10-20 03:32 |
| D7 | 2026-10-20 | 5,760 | 2026-10-20 10:00 | 2026-10-20 10:52 |
| D8 | 2026-10-20 10:45 | 8,170 | 2026-10-20 10:53 | 2026-10-20 11:20 |
| D9 | 2026-10-20 11:41 | 352 | 2026-10-20 11:46 | 2026-10-20 13:07 |

## 核對

1. **檔案完整性**：在本資料夾執行 `sha256sum -c CHECKSUMS.sha256`。
2. **不重新訓練，只從每次訓練的輸出重算彙整表與檢定**（數分鐘到數十分鐘；結果應與 `D/results/` 中的 CSV 相同）：
   ```bash
   cd D
   python report_d1.py
   python d4_vol.py --report
   python d6_e2e.py --report
   python d7_ndx.py --report
   python d8_controls.py --report
   python d9_noise.py --report
   python d5_top_etf.py
   python d5_with_quantum.py
   ```
3. **不需要模型的部分可直接重跑**：`python feasibility.py`、`python threshold_long.py`、`python robustness.py`、`python realtime.py`。
4. **完整重新訓練**：各程式開頭的說明列出指令（例如 `python d4_vol.py --procs 24`、`python d9_noise.py --train vol:QLSTM --device cuda`）。量子模型用 `B/fastsim.py` 精確模擬；GPU 可大幅加速，CPU 也能跑，數值只差浮點運算的最後幾位。D9 的重訓已確認和原結果差距小於 1e-14。

## 環境

- Python 3.11；主要套件版本見 `requirements.txt`（實際使用 torch 2.11.0＋CUDA 12.8、numpy 2.4.6、pandas 3.0.6、scipy 1.17.1）。
- `qfin-main/` 的 PennyLane 重現另需 `qfin-main/requirements-pl.txt`。
- 實驗電腦：AMD Ryzen 9 9950X3D、32 GB、NVIDIA RTX 5090、Windows 11。

## 資料

`*/data/` 是 Yahoo Finance 日線資料的快照（到 2026-10-19），只做核對研究結果使用；各實驗的下載程式（例如 `A/fetch_data.py`、`D/run_d1.py --fetch`、`qfin-main/fetch_data_yf.py`）可重新下載。無風險利率用 13 週國庫券殖利率（^IRX）。

回測結果不代表未來績效，資料夾的內容不是投資建議。程式以 MIT 授權。
