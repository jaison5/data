# PennyLane 版：安裝與執行

和論文使用相同的工具：PennyLane `lightning.qubit` 模擬器、`diff_method="adjoint"`、PyTorch。
資料處理、交易環境、測試方式和 numpy 版共用，所以兩個版本的結果可以直接互相對照。

## 檔案

| 檔案 | 內容 |
|---|---|
| `pl_models.py` | PennyLane 電路與模型：QLSTM、QFWP、古典 LSTM、QA3C、古典 A3C |
| `pl_check.py` | **先跑這支**：核對 PennyLane 和 numpy 模擬器的輸出、梯度與參數量 |
| `pl_exp1.py` | 實驗一（多行程平行，每個訓練 1 核心） |
| `pl_exp2.py` | 實驗二（QLSTM 預測器；A3C 照論文用 8 個非同步 worker 行程） |
| `run_pl_all.sh` | 一次跑完全部步驟 |
| `requirements-pl.txt` | 需要的套件 |

## 安裝（建議 Windows 11 + WSL2 的 Ubuntu，和論文作者相同）

```bash
# Windows：在 PowerShell 執行 wsl --install，重開機後打開「Ubuntu」
sudo apt update && sudo apt install -y python3.11 python3.11-venv   # 或 Ubuntu 24.04 內建的 python3.12
cd /mnt/c/Users/<你的帳號>/…/market-pulse/research/qfin           # 或把資料夾複製到 WSL 的家目錄（速度較快）
python3.11 -m venv .venv-pl
source .venv-pl/bin/activate
pip install -r requirements-pl.txt
python pl_check.py
```

`pl_check.py` 每一行都要顯示 OK，參數量要是 QLSTM 32、QFWP 33、QA3C 244、A3C 3332，才代表 PennyLane 版和 numpy 版的電路相同。
古典 LSTM 會是 112 個參數，和 numpy 版的 100 不同，因為 PyTorch 的 LSTM 有兩組偏差；它只是參考基準，不影響論文的比較。

macOS（Apple M 系列）也可以跑：用 Homebrew 的 Python 3.11 建立虛擬環境，其餘步驟相同。

## 執行

```bash
bash run_pl_all.sh 16      # 16 = CPU 核心數（9950X3D）
```

也可以分開執行：

```bash
export OMP_NUM_THREADS=1
python pl_exp1.py --seeds 10 --jobs 16 --lr 0.05 --opt adam
python report1.py pl_adam0.05
python pl_exp2.py qlstm --seeds 3
python pl_exp2.py rl --agent A3C  --seeds 3
python pl_exp2.py rl --agent QA3C --seeds 1 --seed0 0 &   # 每個種子用 8 核心，16 核心可同時跑兩個
python pl_exp2.py rl --agent QA3C --seeds 1 --seed0 1 &
wait
python report2.py exp2_pl
```

## 注意事項

- **時間無法事先準確估計**：PennyLane 每次呼叫電路有 Python 開銷，速度可能和 numpy 版差很多。建議先用小設定試跑，例如 `python pl_exp1.py --seeds 1 --models QFWP --batches 16` 或 `python pl_exp2.py rl --agent QA3C --episodes 16`，再推估總時間。
- **GPU 用不到**：電路只有 3～8 個 qubit，CPU 的 `lightning.qubit` 最快。要試 GPU 需要另外安裝 `pennylane-lightning-gpu`（只支援 Linux／WSL2），再加 `--device lightning.gpu`；RTX 5090 需要較新版本的套件支援。
- **遇到 adjoint 相關錯誤**：某些版本對 batch 輸入加 adjoint 的支援不同。可以先改用 `--diff best` 確認程式能跑，再回報錯誤訊息給我。
- **non-batch 的寫法**：本程式的 non-batch 是用 batch 大小 1 的張量輸入；論文可能是直接輸入單一向量。數學上相同，但時間可能略有差異。
- **A3C 的 worker 行程**：Linux／macOS 用 fork；Windows 原生 Python 沒有 fork，會自動改用 spawn（在 worker 內重建網路，接上共享記憶體裡的參數和 Shared Adam 狀態），不必用 WSL2。
- **結果位置**：實驗一在 `results/exp1a_pl_adam0.05/`，實驗二在 `results/exp2_pl/`，執行紀錄在 `logs/`。
