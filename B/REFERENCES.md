# 方向 B：相關文獻、重疊程度與突破實驗

方向 B＝量子序列模型批次訓練的效能剖析（fastsim 批次模擬器、各後端計時、零梯度參數分析）。
文獻核對日期：2026-10-19。多數論文只核對了摘要、HTML 或文件頁，細節以原文為準。

## 結論：重疊判斷

- **批次模擬器本身：高度重疊，不是新貢獻。** TorchQuantum（QuantumNAS，HPCA 2022）已經指出「PennyLane 用 for 迴圈處理批次，訓練速度隨批次大小線性下降」，並報告張量化批次比 PennyLane 快 246 到 10⁴ 倍。TensorCircuit 也支援向量化（vmap）與自動微分。adjoint 梯度方法則來自 Jones 與 Gacon 2020。
- **我們的貢獻：低度重疊。** 我們用這個已知限制，更正論文一（2025）的系統結論「批次加速約 2 倍就到頂、反向是瓶頸」，並證明這個限制在 PennyLane 0.45 的 lightning.qubit 仍然存在。換成批次模擬後完整訓練快 6～61 倍，而且結果和 PennyLane 相同。另外用光錐分析指出 QLSTM 32 個參數中有 16 個、QFWP 33 個中有 8 個沒有作用，「參數量相同（EPC）」的比較並不公平。這兩點沒有找到前例。
- **建議：** B 單獨投稿的新穎性不足，適合併入主論文一節，或寫成短篇 workshop 論文。

## 相關文獻與重疊程度

| 文獻 | 做了什麼 | 和方向 B 的重疊 | 我們的差異 |
|---|---|---|---|
| [Wang 等人 2022，QuantumNAS／TorchQuantum（HPCA 2022）](https://arxiv.org/abs/2107.10845) | 原文：「PennyLane 用 'For' 迴圈處理批次，訓練速度隨批次大小線性下降」；TorchQuantum 張量化批次快 246 到 10⁴ 倍 | **高**：同樣的觀察與解法 | 他們比的是自己的 10-qubit 電路；我們指出這正是論文一系統結論的成因，並量化 QLSTM／QFWP 的實際影響 |
| [Zhang 等人，TensorCircuit](https://arxiv.org/abs/2205.10091) | 以機器學習後端做張量網路模擬，支援自動微分、JIT 與向量化 | 高：批次化模擬的工具 | 可以當作額外的對照後端（見突破實驗 1） |
| [PennyLane 文件：broadcast_expand](https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.broadcast_expand.html) | 不支援 broadcasting 的裝置會把一個批次拆成多個 tape | 高：官方文件已說明機制 | 我們用 Tracker 實測：lightning.qubit 對 16 筆的批次執行 16 次電路，default.qubit 只做 1 次模擬 |
| [PennyLane issue #4462](https://github.com/PennyLaneAI/pennylane/issues/4462) | 2023 年回報、至今仍未解決：gradient transform（例如 parameter-shift）無法對批次 QNode 求導；維護者建議用 broadcast_expand | 高：同一個限制 | 我們記錄它對 QLSTM／QFWP 的影響：parameter-shift 在批次輸入下完全無法使用 |
| [Jones 與 Gacon 2020](https://arxiv.org/abs/2009.02823) | adjoint 梯度方法：O(P) 時間、O(1) 個 state vector | 中：論文一使用的微分方法 | 我們量出 adjoint 在 lightning.qubit 的實際時間分配（預設在前向就算梯度；改 device_vjp 反而更慢） |
| [Lee 等人 2024，Light Cone Cancellation（2404.19497）](https://arxiv.org/abs/2404.19497) | 移除計算局部觀測量時用不到的閘 | 中：光錐概念 | 我們把光錐用在「參數是否有作用」，用來檢驗 EPC 是否公平 |
| [Haug、Bharti、Kim 2021（PRX Quantum）](https://spiral.imperial.ac.uk/handle/10044/1/92426) | 用有效量子維度評估電路表達力，並移除冗餘參數而不影響有效維度 | 中：冗餘參數 | 我們發現的是更極端的情況：梯度恆為 0、完全沒有作用的參數 |
| [Abbas 等人，The power of quantum neural networks（2011.00027）](https://arxiv.org/abs/2011.00027) | 以 Fisher 資訊定義有效維度 | 中：「有效參數量」的理論基礎 | 可用來定義比 EPC 更公平的比較標準（見突破實驗 2） |
| [論文一（2512.21820）](https://arxiv.org/abs/2512.21820) | 被檢驗的系統結論：批次加速約 2 倍、反向是瓶頸 | — | — |

## 我們的差異

- 用實測證明：PennyLane lightning.qubit 每筆樣本的成本在 batch 16 以上就停在約 17 ms（QLSTM），所以批次加速被卡在約 2.5 倍；這是模擬器的實作問題，不是量子模型或反向傳播的本質。
- 換成批次化模擬後，每筆成本在 batch 256 時降到 0.19 ms（CPU）／0.09 ms（GPU）；完整訓練快 6～61 倍，批次加速比變成 25 倍（CPU）／64 倍（GPU）。
- 結果不變：輸出與梯度差在 1e-15 以內；QLSTM 60 次訓練和 PennyLane 完全相同。
- `device_vjp` 讓反向變成主要耗時，但整體更慢（0.6～0.9 倍）；parameter-shift 無法用在批次輸入。
- 有效參數：在 repo 的電路配置下，QLSTM 16／32、QFWP 8／33 個參數梯度恆為 0（論文原始配置需向作者確認）。

## 建議的突破實驗

1. **和現有工具正面比較。** 在同一個 QLSTM／QFWP 工作負載上，比較 lightning.qubit、default.qubit（backprop）、TorchQuantum、TensorCircuit 與 fastsim，而不是只比自己寫的模擬器。這樣結論是「選對工具就能突破 2 倍」，貢獻更有說服力。參考：Wang 等人 2022；TensorCircuit。
2. **有效參數量（EPC-eff）。** 用光錐分析加上 Fisher 資訊矩陣的秩，定義「有效參數量」，在有效參數相同的條件下重做 QLSTM vs. QFWP vs. 古典模型的比較。再把電路改成所有參數都有作用的配置（例如可訓練層放在糾纏閘之前，或量測全部線），看準確率是否改善。參考：Abbas 等人；Haug 等人 2021；Lee 等人 2024。
3. **qubit 數擴展。** 把電路從 3～8 qubits 擴大到 12～20 qubits，找出 CPU 與 GPU 的交叉點，並比較 adjoint 與 backprop 的時間與記憶體。參考：Jones 與 Gacon 2020。
4. **達到相同準確率所需的時間。** 用「達到目標驗證誤差的時間」取代「每一步的時間」，因為大 batch 每秒處理更多樣本、但更新次數變少。
5. **給 PennyLane 的實務回饋。** 把量測結果整理成 issue 或文件建議（例如 TorchLayer 搭配 lightning.qubit 的批次行為），增加論文的實際影響。參考：PennyLane issue #4462。

## 參考文獻

- Wang, H. et al. (2022). QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits（含 TorchQuantum）. HPCA 2022. https://arxiv.org/abs/2107.10845
- Zhang, S.-X. et al. TensorCircuit: a Quantum Software Framework for the NISQ Era. https://arxiv.org/abs/2205.10091
- PennyLane documentation. pennylane.transforms.broadcast_expand. https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.broadcast_expand.html
- PennyLane issue #4462. [BUG] Differentiation of broadcasted QNode via gradient transforms. https://github.com/PennyLaneAI/pennylane/issues/4462
- Jones, T., Gacon, J. (2020). Efficient calculation of gradients in classical simulations of variational quantum algorithms. https://arxiv.org/abs/2009.02823
- Lee, X. et al. (2024). Light Cone Cancellation for Variational Quantum Eigensolver in Solving Noisy Max-Cut. https://arxiv.org/abs/2404.19497
- Haug, T., Bharti, K., Kim, M. S. (2021). Capacity and quantum geometry of parametrized quantum circuits. PRX Quantum 2, 040309. https://spiral.imperial.ac.uk/handle/10044/1/92426
- Abbas, A. et al. The power of quantum neural networks. https://arxiv.org/abs/2011.00027
- Chen, J.-H., Hung, M.-K., Tsai, Y.-C., Chen, S. Y.-C. Batched Training for QLSTM vs. QFWP（論文一）. https://arxiv.org/abs/2512.21820
