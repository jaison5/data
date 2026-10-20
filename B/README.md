# 方向 B：量子序列模型批次訓練的效能剖析

**暫定題目**：Where Does the Time Go? Profiling Batched Training of Quantum Sequence Models

## 結論

- **論文一「批次加速約 2 倍就到頂」不是量子模型本身的限制，而是模擬器的實作造成的。** PennyLane 的 lightning.qubit 會把一個批次拆成「每個樣本一條電路」逐一模擬：用 PennyLane 的 Tracker 計數，16 筆的批次就執行 16 次電路。所以批次只省下 Python 呼叫的開銷，QLSTM 每筆的成本在 batch 16 以上就停在約 17 ms。
- **換成真正批次化的模擬器，每筆成本會隨批次一路下降。** 本資料夾的 `fastsim.py`（純 PyTorch）在 batch 256 時，QLSTM 每筆 0.19 ms（CPU）／0.09 ms（GPU），比 PennyLane 預設的 adjoint 快 101 倍／209 倍。
- **完整訓練也一樣快很多。** 實驗一的 2 個 epoch 訓練，QLSTM batch 64 從 189.5 秒降到 3.2 秒（CPU，快 59 倍）。批次加速比（non-batch ÷ batch）從 PennyLane 的 2.5 倍，變成 25 倍（CPU）／64 倍（GPU）。
- **換模擬器不改變結果。** fastsim 和 PennyLane 的輸出與梯度差都在 1e-15 以內，同一個種子會得到完全相同的初始參數。QLSTM 的 60 次訓練，RMSE 和 PennyLane 逐位相同（差 < 1e-16）。QFWP 在小 batch 有部分種子不同，和實驗一換電腦時一樣：QFWP 對極小的數值差異很敏感。
- **論文說的「反向是瓶頸」只在特定設定下成立。** 預設的 adjoint 在前向就把梯度算完；改用 `device_vjp` 會把時間移到反向，但整體反而更慢（0.6～0.9 倍）。parameter-shift 在 PennyLane 中不支援對批次輸入求梯度（連 batch 大小 1 的張量也不行），這兩個模型無法使用。
- **附帶發現：有一半的量子參數其實沒有作用。** 在本 repo 的電路配置下，QLSTM 的 32 個參數有 16 個、QFWP 的 33 個參數有 8 個，梯度永遠是 0。原因是它們只作用在不被量測、之後也沒有糾纏閘的線上。所以論文一「參數量相同（EPC）」的比較，實際有作用的參數是 16 對 25。

## 要解決的問題

對應論文一的痛點 #8、#9：
- 論文的速度結果取決於沒寫清楚的微分設定和硬體，別人無法重現或比較。
- lightning.qubit 的批次其實是逐筆計算，加速被卡在約 2 倍，論文把它歸因於「反向傳播是瓶頸」。

## 做了什麼

| 檔案 | 內容 |
|---|---|
| `fastsim.py` | 批次 state-vector 模擬器與模型（QLSTM、QFWP、QA3C、古典對照）。整個批次當成一個 (batch, 2ⁿ) 的張量計算，梯度由 PyTorch autograd 精確計算（等同 backprop）。CNOT 層預先合成一個排列，量測用一次矩陣乘法。參數的建立順序和 PennyLane 版相同，所以同一個種子的初始參數完全一樣。 |
| `check_fastsim.py` | 和 PennyLane 版逐一比對初始參數、輸出、梯度；並找出梯度恆為 0 的參數。 |
| `bench_step.py` | B1：每個訓練步的前向、反向時間。QLSTM、QFWP × batch 1／4／16／64／256 × 6 種後端。單執行緒，機器沒有其他工作。 |
| `train_grid.py` | B2：用 fastsim 重跑實驗一的完整訓練（2 epochs、Adam lr 0.05、6 種 batch、10 個種子），和原本的 PennyLane 結果比較時間與準確率。CPU 用 16 個平行工作（和原本 PennyLane 那次相同），GPU 單一行程、3 個種子。 |
| `report_b.py` | 彙整成 `results/summary_b.md`、`results/summary_b.json`。 |

環境：Windows 11、AMD Ryzen 9 9950X3D、NVIDIA RTX 5090；PennyLane 0.45.1、PyTorch 2.14.1（CPU）、PyTorch 2.11.0＋CUDA 12.8（GPU，另一個虛擬環境 `.venv-gpu`）。

## 結果

### B1：每筆樣本的訓練時間（前向＋反向，毫秒）

QLSTM：

| 後端 | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| PennyLane adjoint（預設，論文的設定） | 38.20 | 21.16 | 16.39 | 17.18 | 19.25 |
| PennyLane adjoint＋device_vjp | 50.83 | 32.47 | 27.13 | 27.75 | 31.38 |
| PennyLane default.qubit backprop | 90.61 | 23.36 | 7.29 | 2.35 | 1.15 |
| fastsim CPU | 6.88 | 1.87 | 0.60 | 0.28 | 0.19 |
| fastsim GPU | 24.09 | 5.97 | 1.47 | 0.37 | 0.09 |
| 古典 LSTM（參考） | 0.49 | 0.13 | 0.03 | 0.009 | 0.003 |

QFWP：

| 後端 | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| PennyLane adjoint（預設） | 6.86 | 3.43 | 2.42 | 1.95 | 2.35 |
| PennyLane adjoint＋device_vjp | 7.40 | 3.76 | 2.38 | 2.08 | 2.49 |
| PennyLane default.qubit backprop | 7.14 | 1.84 | 0.46 | 0.12 | 0.03 |
| fastsim CPU | 1.54 | 0.39 | 0.10 | 0.03 | 0.01 |
| fastsim GPU | 6.19 | 1.55 | 0.39 | 0.10 | 0.02 |

- PennyLane adjoint 的每筆時間在 batch 16 以後幾乎不變，代表批次沒有被平行計算。Tracker 的計數：lightning.qubit 對 16 筆的批次執行 32 次電路（16 筆 × 前向與梯度），default.qubit 只做 1 次模擬。
- default.qubit 原生支援批次，batch 夠大時明顯變快（QLSTM 在 batch 256 快 16.8 倍）。
- fastsim GPU 每一步固定約 24 ms（QLSTM）／6 ms（QFWP），不隨 batch 改變，代表時間都花在啟動 GPU 運算的固定開銷上。電路只有 3～8 個 qubit，所以 batch 256 以上 GPU 才贏過 CPU。
- parameter-shift：PennyLane 回報「不支援對批次參數求梯度」（PennyLane issue #4462），所以表中沒有。

### B2：實驗一的完整訓練（2 epochs）

| 模型 | Batch | PennyLane 秒 | fastsim CPU 秒 | fastsim GPU 秒 | PennyLane 批次加速 | fastsim CPU 批次加速 | fastsim GPU 批次加速 | fastsim CPU 比 PennyLane 快 |
|---|---|---|---|---|---|---|---|---|
| QLSTM | non-batch | 473.9 | 80.5 | 212.0 | — | — | — | 5.9× |
| QLSTM | 4 | 255.1 | 21.4 | 50.9 | 1.8× | 3.7× | 4.2× | 11.9× |
| QLSTM | 16 | 197.4 | 6.8 | 13.4 | 2.4× | 11.6× | 15.3× | 29.0× |
| QLSTM | 64 | 189.5 | 3.2 | 3.3 | 2.5× | 25.3× | 63.5× | 59.2× |
| QFWP | non-batch | 66.6 | 20.6 | 57.7 | — | — | — | 3.2× |
| QFWP | 4 | 32.7 | 5.4 | 14.2 | 2.0× | 3.6× | 4.1× | 6.1× |
| QFWP | 16 | 24.4 | 1.4 | 3.5 | 2.7× | 13.3× | 16.1× | 18.1× |
| QFWP | 64 | 24.1 | 0.4 | 0.9 | 2.8× | 52.4× | 64.4× | 61.1× |

（秒數與加速比都是中位數；batch 8、32 見 `results/summary_b.md`。）

準確率（10 個種子，RMSE 平均 ± 標準差）：

| 模型 | Batch | PennyLane | fastsim CPU | 同種子結果相同的比例 |
|---|---|---|---|---|
| QLSTM | 全部 6 種 | 例如 batch 16：0.00416 ± 0.00026 | 例如 batch 16：0.00416 ± 0.00026 | 60／60（差 < 1e-16） |
| QFWP | non-batch | 0.00506 ± 0.00173 | 0.00568 ± 0.00144 | 0／10 |
| QFWP | 4 | 0.00348 ± 0.00112 | 0.00390 ± 0.00135 | 4／10 |
| QFWP | 8 | 0.00317 ± 0.00027 | 0.00326 ± 0.00055 | 8／10 |
| QFWP | 16、32、64 | 例如 batch 16：0.00301 ± 0.00008 | 相同 | 30／30 |

QFWP 在更新次數多（batch 小）時會因為極小的數值差異走向不同結果（收斂或崩掉），這和 Mac 與 Windows 之間看到的現象相同，不是模擬器的錯誤：單步的輸出和梯度差都在 1e-15 量級。

### 等價性與零梯度參數

| 模型 | 參數量 | 初始參數最大差 | 輸出最大差 | 梯度最大差 | 梯度恆為 0 的參數 |
|---|---|---|---|---|---|
| QLSTM | 32 | 0 | 8.3e-17 | 4.4e-16 | 16（每個閘門最後一層 Ry 在第 3～6 條線的 4 個權重） |
| QFWP | 33 | 0 | 1.1e-16 | 8.9e-16 | 8（qubit 選擇器第 1、2 列的權重與偏差） |
| QLSTM（預測器，6 輸入、H=2） | 38 | 0 | 1.7e-16 | 1.8e-15 | — |
| QA3C | 244 | 0 | 3.3e-16 | 7.1e-15 | — |

## 預期實質成果（已產出）

- 一個支援批次維度的 state-vector 模擬器 `fastsim.py`，可直接替換 `qfin-main/pl_models.py` 的 QLSTM、QFWP、QA3C，同一個種子會得到相同的初始參數與相同的結果。方向 A、C 都用它來加速實驗。
- 各後端與微分方式的時間分解表、實測加速倍數（`results/summary_b.md`），以及可重現的計時流程（`bench_step.py`、`train_grid.py`）。
- 可以寫進論文的主張：批次加速的上限由模擬器決定，不是由反向傳播決定；選對模擬器，實驗一的訓練可以快 6～61 倍，而且結果不變。

## 限制

- lightning.gpu 不支援 Windows，沒有測；GPU 只測了 fastsim。
- 計時都在同一台電腦。B1 是單執行緒；B2 的 CPU 部分是 16 個平行工作，和原本 PennyLane 那次的條件相同。
- 只測了這兩個小電路（3～8 qubits）。qubit 數變多時 state vector 會呈指數成長，GPU 的優勢會更明顯，但這裡沒有測。

## 重跑方式

```bash
cd B
../qfin-main/.venv-pl/Scripts/python.exe check_fastsim.py
../qfin-main/.venv-pl/Scripts/python.exe bench_step.py
./.venv-gpu/Scripts/python.exe bench_step.py --backends fast_gpu
../qfin-main/.venv-pl/Scripts/python.exe train_grid.py --device cpu --jobs 16
./.venv-gpu/Scripts/python.exe train_grid.py --device cuda --seeds 3
../qfin-main/.venv-pl/Scripts/python.exe report_b.py
```

GPU 環境的建立：`py -3.11 -m venv .venv-gpu`，再 `pip install torch numpy --index-url https://download.pytorch.org/whl/cu128`。
紀錄在 `logs/`，結果在 `results/`。

## 參考文獻

相關文獻、和本方向的重疊程度、以及建議的突破實驗，見 [REFERENCES.md](REFERENCES.md)。
簡短結論：批次模擬器本身不是新貢獻（TorchQuantum 2022 已指出 PennyLane 用迴圈處理批次）；本方向的新意在於更正論文一的系統結論，以及指出「參數量相同（EPC）」比較中有一半量子參數沒有作用。
