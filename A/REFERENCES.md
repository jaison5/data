# 方向 A：相關文獻、重疊程度與突破實驗

方向 A＝無前視、walk-forward 的 QLSTM／QFWP 評測基準（實驗 A1、A2）。
文獻核對日期：2026-10-19。多數論文只核對了摘要或 HTML 版，細節以原文為準。

## 結論：重疊判斷

- **「量子 vs. 古典」的嚴格比較：中度重疊。** 已有多篇評測發現量子模型通常不如、或只和古典模型持平（Bowles 等人 2024；2504.12416；2412.13878）。其中 2508.04488（同一實驗室）也比較了 QLSTM、QFWP 和 LSTM，2601.03802、2605.02656 則是金融資料上的量子 vs. 古典評測。
- **我們的核心貢獻：沒有找到重疊。** 沒有找到任何文獻指出 Yahoo 匯率日線的 High/Low 會洩漏隔天收盤，也沒有人用無前視的 walk-forward 去重新檢驗已發表的 QLSTM／QFWP 金融結果。同樣沒有人報告「崩掉比例」或跨硬體結果不同。這些是方向 A 最主要的新穎之處。

## 相關文獻與重疊程度

| 文獻 | 做了什麼 | 和方向 A 的重疊 | 我們的差異 |
|---|---|---|---|
| [Bowles、Ahmed、Schuld 2024](https://arxiv.org/abs/2403.07059) | 12 個量子分類器 × 6 種分類任務（160 個資料集）；古典模型整體較好，拿掉糾纏後常常一樣好或更好 | 中：同樣在問「量子到底有沒有比古典好」 | 他們做分類、不含時間序列與金融；我們做金融預測，並找出資料前視這個會誤導結論的原因 |
| [Fellner 等人 2025（2504.12416）](https://arxiv.org/abs/2504.12416) | 3 個混沌系統、27 個預測任務；廣泛調參後，量子模型常常比不上複雜度相當的簡單古典模型 | 中：時間序列的量子 vs. 古典 | 他們用混沌系統；我們用真實匯率，並處理資料前視、walk-forward 與統計檢定 |
| [Jones 等人 2024（2412.13878）](https://arxiv.org/abs/2412.13878) | 真實資料上評測 gate-based 與 annealing 量子模型；最好的古典模型勝過最好的量子模型 | 中 | 他們沒有聚焦金融、也沒有討論前視 |
| [C.-S. Chen、S. Y.-C. Chen、Tsai 2025（2508.04488）](https://arxiv.org/abs/2508.04488) | 同一實驗室；電信資料比較 LSTM、QLSTM、QFWP 等；70/15/15 單一切分、5 次平均、沒有統計檢定 | 中：同樣比較 QLSTM、QFWP、LSTM | 單一切分、沒有前視檢查；我們用 10 個 fold、20 個種子、DM／PT 檢定與崩掉比例 |
| [Ahmad 等人 2026（2601.03802）](https://arxiv.org/abs/2601.03802) | 金融預測的量子 vs. 古典評測；混合 QNN 對參數量相同的 ANN 小幅領先（AUC +3.8～4.9）；QLSTM 在 4 個 S&P 500 行情中 2 個較好 | 中：金融資料、參數量相同的基準 | 摘要沒有提到前視與 walk-forward；我們先確認資料沒有前視，再比較 |
| [Maheshwari 等人 2026（2605.02656）](https://arxiv.org/abs/2605.02656) | 金融時間序列上比較 QLSTM、量子 reservoir computing 與古典模型；單變量持平，多變量相關輸入時量子略好 | 中 | 摘要沒有提到前視；**「多變量時量子略好」可以在無前視的條件下直接檢驗**（見突破實驗 6） |
| [Kapoor 與 Narayanan 2023](https://reproducible.cs.princeton.edu/) | 調查 ML 研究中的資料洩漏：17 個領域、294 篇論文受影響，並提出洩漏分類 | 低：一般性的洩漏問題 | 我們找出金融 QML 中一個具體、可重現的洩漏來源 |
| [Hewamalage、Ackermann、Bergmeir 2023](https://arxiv.org/abs/2203.10716) | 整理資料科學家做預測評估時的常見陷阱與最佳做法 | 低：方法論背景 | 我們把這些原則實際套用到量子模型 |
| [Diebold 與 Mariano 1995](https://ideas.repec.org/a/bes/jnlbes/v13y1995i3p253-63.html)、[Pesaran 與 Timmermann 1992](https://ideas.repec.org/a/bes/jnlbes/v10y1992i4p561-65.html) | 預測準確度（DM）與方向預測（PT）的標準檢定 | 低：標準工具 | 論文一沒有和隨機漫步做檢定；我們補上 |
| [論文一（2512.21820）](https://arxiv.org/abs/2512.21820)、[QLSTM（2009.01783）](https://arxiv.org/abs/2009.01783)、[QFWP（2402.17760）](https://arxiv.org/abs/2402.17760) | 被檢驗的論文與模型原始出處 | — | — |

## 我們的差異

- 指出 Yahoo 匯率日線的 Close 約等於當天開始的價格，High/Low 卻涵蓋整天；一個零參數規則就有 DA 66.8%、RMSE 0.00323，和論文的 QFWP 相近。
- 用去前視資料＋10 個 fold 的 walk-forward＋20 個種子＋驗證集選學習率，重新檢驗「QFWP 比 QLSTM 好」與「量子比古典好」。
- 加入參數量相近的古典基準（LSTM h=1，30 個參數）與 QFWP 的古典消融（FWPC），區分「量子電路的貢獻」和「架構的貢獻」。
- 報告崩掉比例與跨硬體差異：同一份程式在 Mac 與 Windows 有 18 次 QFWP 訓練結果不同，足以翻轉顯著性。
- A2：區分「batch 大小」和「更新次數」對準確率的影響。

## 建議的突破實驗

1. **跨資產的零參數前視探針。** 對約 20 個 Yahoo 匯率、幾檔股票（交易時段明確，應沒有前視，當對照）與加密貨幣，比較「當日高低價中點」與「前一天高低價中點」的方向準確率。不需要訓練，成本很低，能證明問題不是 EUR/USD 的特例。參考：Kapoor 與 Narayanan 2023；Hewamalage 等人 2023。
2. **第二資料源驗證。** 用收盤時間明確的資料，例如 [FRED 的 H.10 匯率（紐約中午買入價）](https://fred.stlouisfed.org/series/DEXUSEU)，或以紐約 17:00 切點自建的日線，檢查 Yahoo 的 High/Low 是否也能預測這些獨立收盤價。排除「只是 Yahoo 的資料問題」的質疑。
3. **文獻稽核表。** 統計已發表的 QML／深度學習金融論文中，有多少用 Yahoo 日線的同日 OHLC 預測隔天收盤，並標記哪些可能受影響。數量越多，這篇的影響力越大。參考：Kapoor 與 Narayanan 2023（以調查呈現洩漏影響的範例）。
4. **給量子模型最好的機會。** 用 fastsim（快 6～61 倍）對量子模型做架構與超參數搜尋（深度、data re-uploading、修正零梯度參數的電路），古典模型用同樣的調參預算。參考：Bowles 等人 2024、Fellner 等人 2025 都強調公平調參。
5. **嚴格統計。** 每個模型對隨機漫步做 DM 檢定、方向做 PT 檢定，並以 fold × 種子成對比較。參考：Diebold 與 Mariano 1995；Pesaran 與 Timmermann 1992。
6. **檢驗唯一的正面主張。** 2605.02656 說量子模型在「多變量、輸入相關」時略好。在無前視資料上，用多個匯率當輸入預測其中一個，直接檢驗這個條件下量子是否真的有優勢。這是最可能找到正面結果的地方。
7. **崩掉比例與跨硬體可重現性。** 把「同程式、同種子、不同硬體結果不同」寫成一節，建議論文報告崩掉比例與種子數。

## 參考文獻

- Bowles, J., Ahmed, S., Schuld, M. (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. https://arxiv.org/abs/2403.07059
- Fellner, T., Kreplin, D., Tovey, S., Holm, C. (2025). Quantum vs. classical: A comprehensive benchmark study for predicting time series with variational quantum machine learning. https://arxiv.org/abs/2504.12416
- Jones, C. et al. (2024). Benchmarking Quantum Models for Time-series Forecasting. https://arxiv.org/abs/2412.13878
- Chen, C.-S., Chen, S. Y.-C., Tsai, Y.-C. (2025). Benchmarking Quantum and Classical Sequential Models for Urban Telecommunication Forecasting. https://arxiv.org/abs/2508.04488
- Ahmad, R., Kashif, M., Innan, N., Shafique, M. (2026). Quantum vs. Classical Machine Learning: A Benchmark Study for Financial Prediction. https://arxiv.org/abs/2601.03802
- Maheshwari, D. et al. (2026). Learning Temporal Patterns in Financial Time Series: A Comparative Study of Quantum LSTM and Quantum Reservoir Computing. https://arxiv.org/abs/2605.02656
- Kapoor, S., Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-learning-based Science. Patterns. https://reproducible.cs.princeton.edu/
- Hewamalage, H., Ackermann, K., Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. Data Mining and Knowledge Discovery. https://arxiv.org/abs/2203.10716
- Diebold, F. X., Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics. https://ideas.repec.org/a/bes/jnlbes/v13y1995i3p253-63.html
- Pesaran, M. H., Timmermann, A. (1992). A Simple Nonparametric Test of Predictive Performance. Journal of Business & Economic Statistics. https://ideas.repec.org/a/bes/jnlbes/v10y1992i4p561-65.html
- Board of Governors of the Federal Reserve System. U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU), H.10. https://fred.stlouisfed.org/series/DEXUSEU
- Chen, J.-H., Hung, M.-K., Tsai, Y.-C., Chen, S. Y.-C. Batched Training for QLSTM vs. QFWP（論文一）. https://arxiv.org/abs/2512.21820
- Chen, S. Y.-C., Yoo, S., Fang, Y.-L. L. (2022). Quantum Long Short-Term Memory. ICASSP 2022. https://arxiv.org/abs/2009.01783
- Chen, S. Y.-C. (2024). Learning to Program Variational Quantum Circuits with Fast Weights. https://arxiv.org/abs/2402.17760
