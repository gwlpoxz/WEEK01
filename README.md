  V7toV10

  本專案實作了一套基於強化學習 (Reinforcement Learning) 的高階觀察者系統原型。AI 代理人必須在 $10000 \times 10000$
  的巨大空間中，透過 $800 \times 800$ 的有限視野進行自主搜尋，並對動態目標執行像素級的精準獵殺。

  🎯 核心技術挑戰
   * 非全知視角：AI 僅具備 0.64% 的局部觀測率，需具備高效搜索策略。
   * 解耦控制：實作「視野移動」與「全視窗隨機點擊」的同步決策，不依賴傳統準星對齊。
   * 持續進化：建立模型權重繼承機制，確保訓練進度永不歸零。

  ---

  📂 專案架構 (Project Tree)
```

    0 ├── rl_human_recorder.py      # [主程式] 專家數據錄製與互動介面
    1 ├── rl_pretraining.py         # [主程式] 模仿學習預訓練系統 (BC)
    2 ├── rl_machine_training.py    # [主程式] 機器增量強化訓練系統 (PPO)
    3 ├── rl_ai_demo.py             # [主程式] AI 成果效能驗證展示介面
    4 ├── custom_ppo.py             # [核心] 自定義 PPO 演算法邏輯實現
    5 ├── model.py                  # [核心] 類神經網路模型 (CNN/MLP) 架構定義
    6 ├── hunter_latest.pth         # [權重] 訓練好的模型參數檔案 (.pth)
    7 ├── hunter_latest.zip         # [權重] 最終進化之 AI 模型大腦 (壓縮備份)
    8 ├── pretrained_hunter.zip     # [權重] 模仿人類行為的初期模型 (壓縮備份)
    9 ├── human_demo/               # [數據] 存放所有人類操作錄製檔 (.npz)
   10 ├── logs/                     # [日誌] 訓練過程數據 (TensorBoard 使用)
   11 └── performance_history.csv   # [紀錄] 訓練效能歷史數據追蹤

```
  ---

  🚀 漸進式強化流程 (Execution Flow)

  請在 PowerShell 中依序執行以下指令，以完成從數據採集到機器進化的完整閉環：

  1. 專家示範 (Data Collection)rl_human_recorder.py
  親自操作以產出人類專家的「教材」檔案。

  2. 行為模仿 (Pre-training)rl_pretraining.py
  讓AI研讀教材，獲得人類的操作直覺。

  3. 機器進化 (Incremental RL)rl_machine_training.py
  執行增量強化訓練，AI 會透過自我試錯超越人類極限。此指令可多次執行。

  4. 成果展示 (Final Demo)rl_ai_demo.py
  開啟 25 FPS 高流暢介面觀看 AI 自動化成果與 KPI 報告。

  ---

  🛠 技術規格說明 (Technical Stack)
 * 1. 核心框架與環境 (Frameworks & Env)
   * 語言版本: Python 3.x
   * 深度學習: PyTorch (核心張量運算與神經網路架構)
   * 模擬環境: Gymnasium (標準化強化學習環境介面)
   * 介面渲染: Pygame (用於 2D 視覺化錄製與 AI 展示介面)

 * 2. 神經網路架構 (Neural Network Architecture)
   * 類型: MLP (Multi-Layer Perceptron)
   * 模型類別: Actor-Critic 結構
       * Actor (策略): 負責決定獵人的移動方向與動作。
       * Critic (評價值): 負責預測當前狀態的預期獎勵。
   * 優化器: Adam Optimizer (動態學習率調整)
   * 激活函數: Tanh (用於平滑連續動作空間的輸出)

 * 3. 強化學習演算法 (RL Algorithm)
   * 演算法: PPO (Proximal Policy Optimization)
   * 實現方式: 自定義 custom_ppo.py 實現 (非直接調用封裝庫)。
   * 緩衝機制: RolloutBuffer (離線策略數據存儲，用於穩定更新)。
   * 訓練策略: 基於「預訓練模型 (BC)」的增量式強化學習。

 * 4. 數據與監控 (Data & Monitoring)
   * 數據格式: .npz (NumPy 壓縮格式，儲存錄製好的專家軌跡)。
   * 效能紀錄: Pandas / CSV (追蹤訓練過程中的平均獎勵與步數)。
   * 日誌系統: TensorBoard (視覺化訓練收斂曲線，儲存於 logs/)。

  ---

  📊 效能指標 (KPIs)
  在成果展示介面 (rl_ai_demo.py) 中，我們重點驗證以下數據：
   1. 累計獵殺次數：衡量 AI 的總產出能力。
   2. 點擊準確率 (%)：衡量 AI 對局部座標的控制精細度。
   3. 期望獵殺數 (EHK)：每分鐘預期的有效點擊產出。
   4. 平均誤差 (Error)：點擊位置與目標中心點的像素級偏差。

  ---
  開發者：Gwen (NeuroProGram Project)
  版本：v10.0 (2026-05-04)
