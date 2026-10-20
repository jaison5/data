# 方向 C：相關文獻、重疊程度與突破實驗

方向 C＝有交易成本、動作遮罩、walk-forward 的 QA3C vs. 古典 A3C 交易評估（含獎勵消融與預測器消融）。
文獻核對日期：2026-10-19。多數論文只核對了摘要或 HTML 版，細節以原文為準。

## 結論：重疊判斷

- **題目與架構：高度重疊。** 同一個實驗室已有三篇用 QA3C（8-qubit VQC 的 A3C）做交易的論文：論文二（USD/TWD，2509.09176）、Liu 等人 2025（QCE 2025，S&P 500 週資料，2507.12835）、Q-A3C2（ETF 選股，2512.21819）。三篇都宣稱量子版本較好。
- **評估方法：低度重疊，這是方向 C 的貢獻。** 這三篇都是**單一切分、單次執行、沒有種子變異、沒有統計檢定**。Liu 等人有交易成本但沒給數值；Q-A3C2 和論文二沒有交易成本。三篇都沒有動作遮罩，也沒有 walk-forward。方向 C 第一次用遮罩、多種成本、walk-forward、多種子、參數量相同的古典基準，以及獎勵與預測器消融，去檢驗這一系列的主張。
- **定位建議：** 因為是同一實驗室的研究線，寫成「延伸與強化評估」，不要寫成反駁；可以邀請原作者一起討論，並確認論文沒寫的設定。

## 相關文獻與重疊程度

| 文獻 | 做了什麼 | 和方向 C 的重疊 | 我們的差異 |
|---|---|---|---|
| [論文二：Chen、Huang、Tsai、Chen 2025（2509.09176）](https://arxiv.org/abs/2509.09176) | QLSTM 預測＋QA3C 交易 USD/TWD；報酬 11.87%、回撤 0.92%；單一切分、單次執行 | **高**：被檢驗的論文 | 遮罩、成本、walk-forward、多種子、同參數量古典基準、消融 |
| [Liu、Pan、Lu、Tsai、Chen 2025（QCE 2025，2507.12835）](https://arxiv.org/abs/2507.12835) | 古典 vs. 量子 A3C，有無 LSTM 預測訊號；S&P 500 週資料（2022-01～2024-04）；量子 A3C＋LSTM 累積報酬 30.13%；有成本項但沒給數值；單次執行、沒有種子變異、沒說明切分 | **高**：同樣的比較（古典 vs. 量子 A3C＋預測訊號） | 我們的成本是明確的數值並掃描三種；walk-forward；多種子＋成對檢定；參數量相同（238 vs. 244）；預測器消融含理想上限 |
| [Liu、Tsai、Chen 2025，Q-A3C2（2512.21819）](https://arxiv.org/abs/2512.21819) | VQC 加進 A3C 策略網路＋動態分群選 ETF；單一 train／validation 切分；沒有交易成本；沒有種子變異；有參數量相同的 MLP 基準但沒給數字；報酬 17.09% vs. 大盤 7.09% | 高：QA3C 在金融的應用 | 同上；我們報告成本敏感度與統計顯著性 |
| [Chen 2023，Asynchronous training of quantum reinforcement learning（2301.05096）](https://arxiv.org/abs/2301.05096) | QA3C 的原始方法（Procedia Computer Science 222）；在一般 RL 任務上和同大小的古典代理人相當或更好 | 中：方法來源 | 我們檢驗它在有成本的交易環境中是否成立 |
| [Dutta 等人 2024，QADQN（QCE 2024，2408.03088）](https://arxiv.org/abs/2408.03088) | VQC 放進 DQN，含固定交易成本；Sortino 1.28／1.19 | 中：量子 RL 交易＋成本 | 不同演算法（DQN）；我們專注 QA3C 系列並做消融 |
| [Moody 與 Saffell（NeurIPS 1998；IEEE TNN 2001）](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading) | 直接強化學習交易，以差分 Sharpe 為目標、考慮交易成本 | 中：風險調整獎勵與成本的經典做法 | 我們用 log 報酬獎勵，可再加差分 Sharpe（見突破實驗 4） |
| [Zhang、Zohren、Roberts 2020（1911.10107）](https://arxiv.org/abs/1911.10107) | DRL 交易 50 種期貨，在高交易成本下仍有正報酬 | 低：古典 DRL 的成本處理範例 | — |
| [FinRL（2011.09607）](https://arxiv.org/abs/2011.09607) | 開源 DRL 交易函式庫，環境含交易成本、流動性等摩擦 | 低：環境設計參考 | — |
| [Huang 與 Ontañón（2006.14171）](https://arxiv.org/abs/2006.14171) | 證明動作遮罩保持策略梯度有效，且比「懲罰無效動作」更能擴展 | 中：遮罩的理論依據 | 我們在 QA3C 交易上實際看到「輕罰無效動作」被鑽漏洞 |
| [Skalse 等人 2022，Defining and Characterizing Reward Hacking（2209.13085）](https://arxiv.org/abs/2209.13085) | 形式化定義 reward hacking，並指出不可被鑽漏洞的代理獎勵很少見 | 中：獎勵被鑽漏洞的理論 | 我們提供金融交易中的具體案例：從不持有、用無效賣出空手 |
| [Bailey 與 López de Prado 2014，Deflated Sharpe Ratio](https://papers.ssrn.com/abstract=2460551)、[Bailey 等人，PBO](https://papers.ssrn.com/abstract=2326253) | 修正多重測試與非常態造成的 Sharpe 高估；估計回測過度擬合機率 | 低：標準工具 | 這三篇 QA3C 論文都沒有用 |

## 我們的差異

- 動作遮罩：不允許無效動作，關掉「用無效的賣出來空手」這個漏洞。
- 明確的交易成本：訓練用 0.05%，測試時用 0／0.05%／0.1% 三種成本評估。
- walk-forward：5 個 fold（測試年 2020～2024），每個 fold 重新訓練預測器與代理人，用驗證年挑模型。
- 多種子：每個設定 3 個種子，並做成對比較。
- 公平基準：參數量相同的古典 A3C（h=9，238 個參數 vs. QA3C 244 個），加上論文的 A3C（3,332 個參數）、買進持有、均線策略、隨機策略。
- 消融：論文獎勵 vs. log 報酬獎勵；沒有預測器／QLSTM 預測器／理想預測器（上限）。

## 建議的突破實驗

1. **用同一套協定重新評估這條研究線的三篇論文。** 對論文二（USD/TWD）、Liu 等人 2025（S&P 500 週資料）、Q-A3C2（ETF）的設定，都套上遮罩、成本、walk-forward 與多種子。一次回答「QA3C 的優勢在嚴格評估下是否存在」，貢獻會比只檢驗一篇大得多。參考：2509.09176、2507.12835、2512.21819。
2. **多資產＋成本掃描＋損益兩平成本。** 加 2～3 個匯率與 S&P 500，報告每個策略「多少成本會讓報酬歸零」。參考：Zhang、Zohren、Roberts 2020；FinRL。
3. **多重測試修正。** 對所有設定計算 Deflated Sharpe 與 PBO，並用 block bootstrap 給 Sharpe 信賴區間。參考：Bailey 與 López de Prado 2014；Bailey 等人（PBO）。
4. **獎勵設計比較。** 論文獎勵 vs. log 報酬 vs. 差分 Sharpe，並用「持有比例、無效動作嘗試、交易次數」量化鑽漏洞的程度。參考：Moody 與 Saffell；Skalse 等人 2022；Huang 與 Ontañón。
5. **預測訊號的真正價值。** 沒有預測器／古典 LSTM／QLSTM／理想預測器，量化預測訊號的邊際貢獻；理想預測器的結果也能顯示這個環境最多能賺多少。參考：Liu 等人 2025 宣稱 LSTM 訊號只幫助量子 A3C，可以直接檢驗。
6. **量子是否必要。** 加入「拿掉糾纏的 VQC」版本：Bowles 等人 2024 發現拿掉糾纏常常一樣好。如果交易結果也一樣，就能說明優勢不是來自量子特性。參考：[Bowles 等人 2024](https://arxiv.org/abs/2403.07059)。

## 參考文獻

- Chen, J.-H., Huang, Y.-C., Tsai, Y.-C., Chen, S. Y.-C. (2025). Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading（論文二）. https://arxiv.org/abs/2509.09176
- Liu, Y.-K., Pan, Y.-H., Lu, P.-F., Tsai, Y.-C., Chen, S. Y.-C. (2025). Quantum-Enhanced Reinforcement Learning with LSTM Forecasting Signals for Optimizing Fintech Trading Decisions. IEEE QCE 2025. https://arxiv.org/abs/2507.12835
- Liu, Y.-K., Tsai, Y.-C., Chen, S. Y.-C. (2025). Q-A3C2: Quantum Reinforcement Learning with Time-Series Dynamic Clustering for Adaptive ETF Stock Selection. https://arxiv.org/abs/2512.21819
- Chen, S. Y.-C. (2023). Asynchronous training of quantum reinforcement learning. Procedia Computer Science 222, 321–330. https://arxiv.org/abs/2301.05096
- Dutta, S., Innan, N., Marchisio, A., Ben Yahia, S., Shafique, M. (2024). QADQN: Quantum Attention Deep Q-Network for Financial Market Prediction. IEEE QCE 2024. https://arxiv.org/abs/2408.03088
- Moody, J., Saffell, M. (1998). Reinforcement Learning for Trading. NeurIPS 1998. https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading
- Zhang, Z., Zohren, S., Roberts, S. (2020). Deep Reinforcement Learning for Trading. The Journal of Financial Data Science. https://arxiv.org/abs/1911.10107
- Liu, X.-Y. et al. (2020). FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance. https://arxiv.org/abs/2011.09607
- Huang, S., Ontañón, S. (2020/2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. https://arxiv.org/abs/2006.14171
- Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Hacking（NeurIPS 2022 標題為 Reward Gaming）. https://arxiv.org/abs/2209.13085
- Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management. https://papers.ssrn.com/abstract=2460551
- Bailey, D. H., Borwein, J., López de Prado, M., Zhu, Q. J. The Probability of Backtest Overfitting. https://papers.ssrn.com/abstract=2326253
- Bowles, J., Ahmed, S., Schuld, M. (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. https://arxiv.org/abs/2403.07059
