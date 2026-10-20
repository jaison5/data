## 實驗一：QLSTM vs. QFWP（ICASSP 2026）

**論文**：Chen, Hung, Tsai, Chen, *Batched Training for QLSTM vs. QFWP: A System-Oriented Approach to EPC-Aware RMSE-DA*，ICASSP 2026（arXiv 2512.21820）

### 設定（照論文）

| 項目 | 論文 | 本次重現 |
|---|---|---|
| 資料 | EUR/USD 日線 OHLC，2003-12～2024-12 | Yahoo `EURUSD=X`，2003-12-01～2024-12-31（5,473 列） |
| 樣本 | 序列長度 5，預測隔日收盤 | 同左，共 5,468 個視窗 |
| 切分 | 80/20 依時間 | 訓練 **4,374** 筆（論文：4,374，完全一致）、測試 1,094 筆（2020-10-21 起） |
| 正規化 | 每個視窗各自 min–max，目標用同一尺度 | 同左 |
| QLSTM | H=3、7 qubits、深度 1、\|+⟩ 起始、Ry 編碼、even-odd CNOT 階梯、量測前 H 條線；32 參數（量子 28＋古典 4） | 同左，**32 參數** |
| QFWP | s_dim 4、latent 3、3 qubits、D=1、A=1；Θ_t = Θ_{t−1} + α_t β_tᵀ；33 個古典參數 | 同左，**33 參數** |
| 訓練 | 2 epochs、無 gradient accumulation、無 LR scaling | 同左 |
| Batch | non-batch、4、8、16、32、64 | 同左（non-batch = 逐筆更新） |
| 種子 | 10 | 10 |
| 指標 | 原始尺度 RMSE、DA（sign(ŷ_t−ŷ_{t−1}) vs. sign(y_t−y_{t−1})）、Wilcoxon、Cliff's δ、各部分時間 | 同左 |
| 後端 | PennyLane `lightning.qubit` + adjoint | 相同（PennyLane 0.45.1、PyTorch 2.14.1）；另以 numpy 實作交叉驗證 |

### 論文未說明、本實作自行設定的細節

- **最佳化器與學習率**：論文沒寫。以 batch 4、16 和 3 個種子掃描 9 組設定（`sweep1.py`），選出最接近論文的 Adam、lr 0.05，損失函數為 MSE。
- **QLSTM 可訓練層的位置**：放在 CNOT 階梯之後，每條線一個 Ry（7 個參數，符合每個 VQC 7 個參數）。權重初始化用 PennyLane `TorchLayer` 的預設 U[0, 2π)。
- **QFWP 的編碼**：4 維 OHLC 放不進 3 個 qubit，所以把 encoder 輸出的 3 維 z_t 用 Ry 編碼。糾纏層為 CNOT(0,1)、CNOT(1,2)。
- **額外基準（論文沒有）**：同樣 hidden=3 的古典 LSTM（PennyLane 版 112 參數、numpy 版 100 參數）、隨機漫步（ŷ = 今天收盤）。

## 實驗二：QLSTM + QA3C 交易代理（Globecom Workshops 2025）

**論文**：Chen, Huang, Tsai, Chen, *Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading*，IEEE Globecom Workshops 2025（arXiv 2509.09176）

### 設定（照論文）

| 項目 | 論文 | 本次重現 |
|---|---|---|
| 資料 | USD/TWD 日線，2000-01-01～2025-04-30 | Yahoo `TWD=X`。Yahoo 從 2004-03 才有資料，扣除 MA60 暖機後，實際從 2004-06-21 開始 |
| 切分 | 訓練到 2020-03-31、測試 2020-04-01～2025-04-30 | 同左 |
| 特徵 | MA5、MA10、MA20、MA60、20 日波動；min–max 到 [0,1] | 同左（縮放只用訓練期間估計） |
| QLSTM 輸入 | 最近 4 天 × (O, H, L, C, MA5, MA10) | 同左 |
| QLSTM 標籤 | 5 日漲 >1.2% 為多、跌 >1.2% 為空，中間不用 | 同左：訓練 466 筆、測試 88 筆 |
| QLSTM 訓練 | hidden 2、dense-softmax、50 epochs、RMSprop 5e-3、weighted CE | 同左（batch 32，論文沒寫） |
| 狀態 | Table I 的 10 維：多空機率、現金 ÷ 初始資金、持倉市值 ÷ 初始資金、未實現損益 %、(現價 − 均價) ÷ 現價、距 MA20 %、距 MA60 %、MA5 − MA20、20 日價格 σ/μ | 同左 |
| QA3C | tanh 投影到 8 維 → 8-qubit VQC → policy／value 頭；244 參數（量子 32、古典 212） | **244 參數**（量子 32、古典 212） |
| 古典 A3C | 3,332 參數 | **3,332 參數**（actor 10→128→3、critic 10→128→1，即 MorvanZhou/pytorch-A3C 的結構） |
| A3C 超參數 | 8 workers、rollout 30、Shared Adam lr 1e-5、γ 0.995、β 0.05 | 同左（Adam betas 0.92/0.999，同 MorvanZhou）；PennyLane 版為 8 個非同步行程 |
| 動作 | 0 持有、1 買 1 單位、2 全部賣出；只做多，可加碼 | 同左 |
| 獎勵 | −0.02 + 進場（順勢 +0.5／逆勢 −2）+ 出場（獲利 +10+50×pnl%）+ 持有虧損 −5×pnl%²，截斷 [−15, 30] | 同左 |
| 交易成本 | 不計 | 不計 |
| 指標 | 總報酬、最大回撤、交易次數、勝率、最佳／最差單筆；與貨幣 ETF 比較 | 同左，另加 Sharpe、持倉時間比例、買進持有 |

### 論文未說明、本實作自行設定的細節

- **VQC 結構**：\|+⟩ → Ry(x) 編碼 → [CNOT 環 → Rz] → [CNOT 環 → Ry] → 量測 8 條線 ⟨Z⟩。actor、critic 各一個 VQC，每個 16 個量子參數；這是唯一同時符合 32 量子＋212 古典參數的配置。
- **資金與部位**：初始資金 = 10 單位（10 × 期初價格），每次買 1 單位。
- **無效動作**：沒錢還買、沒部位還賣，罰 −0.1。
- **虧損出場**：論文表格寫 −2 − 10×pnl%，但虧損時 pnl% 是負的，這樣反而會變成獎勵，應是筆誤。改用 −2 − 10×|pnl%|。
- **Episode**：跑完整個訓練期間一次為一個 episode。總共 1,600 個 episode（8 個 worker 加總），論文的獎勵曲線約在第 1,400 個 episode 收斂。
- **測試**：每一步選機率最高的動作（貪婪）。
- **pnl% 的單位**：1% 以數值 1 計算。
- **actor、critic 的結構**：論文文字寫「一個 VQC 接兩個 linear head」，但這樣古典參數只有 124 個；要符合 Table II 的 212 個，需要 actor、critic 各自一組 Linear(10→8) 與 VQC。

## 和論文不同的地方（必要的偏離）

1. **兩種後端**：除了論文使用的 PennyLane + PyTorch，另以 numpy 實作 state-vector 模擬器與反向自動微分（`ad.py`），兩者的輸出與梯度一致（`pl_check.py`）。numpy 版的訓練時間不能與論文比較；加速比以 PennyLane 版為準。
2. **numpy 版的 A3C 為同步化**：numpy 版的 8 個 worker 同步收集資料，每收集 30 步由每個 worker 各自算梯度、依序更新共享參數。PennyLane 版（README 的結果）則與論文相同，為 8 個非同步行程。
3. **USD/TWD 錯誤報價**：Yahoo 的資料有離譜的錯誤（例如 2011-10-26 單日 +1571%）。和前後兩天平均差 25% 以上的列視為錯誤報價刪除，共 10 列（2011-10、2013-09、2014-12～2015-01）。論文的資料從 2000 年開始，應該不是來自 Yahoo。

## 怎麼執行

```bash
cd research/qfin
python3 test_ad.py && python3 test_models2.py          # 驗證模擬器與梯度
python3 exp1a.py --seeds 10 --jobs 2                   # 實驗一（約 40 分鐘，2 核心）
python3 report1.py                                     # 實驗一彙整
python3 exp2.py qlstm --seeds 3 --kind QLSTM           # 實驗二：QLSTM 預測器
python3 exp2.py rl --agent A3C  --seeds 3 --episodes 1600
python3 exp2.py rl --agent QA3C --seeds 3 --episodes 1600   # 每個種子約 40 分鐘
python3 report2.py
```

需要 Python 3.10 以上、numpy、pandas、scipy、scikit-learn、matplotlib（不需要 PyTorch／PennyLane）。資料由 `research/fetch_data.py` 在 Mac 上匯出到 `research/data/`。
