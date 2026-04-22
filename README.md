  V7

  本專案實作了一套基於強化學習 (Reinforcement Learning) 的高階觀察者系統原型。AI 代理人必須在 $10000 \times 10000$
  的巨大空間中，透過 $800 \times 800$ 的有限視野進行自主搜尋，並對動態目標執行像素級的精準獵殺。

  🎯 核心技術挑戰
   * 非全知視角：AI 僅具備 0.64% 的局部觀測率，需具備高效搜索策略。
   * 解耦控制：實作「視野移動」與「全視窗隨機點擊」的同步決策，不依賴傳統準星對齊。
   * 持續進化：建立模型權重繼承機制，確保訓練進度永不歸零。

  ---

  📂 專案架構 (Project Tree)
```
 1 NeuroProGram/week01/
 2 ├── rl_human_recorder.py      # [主程式] 專家數據錄製與互動介面
 3 ├── rl_pretraining.py         # [主程式] 模仿學習預訓練系統 (BC)
 4 ├── rl_machine_training.py    # [主程式] 機器增量強化訓練系統 (PPO)
 5 ├── rl_ai_demo.py             # [主程式] AI 成果效能驗證展示介面
 6 ├── hunter_env.py             # [核心] 獵人遊戲環境邏輯與獎勵機制
 7 ├── hunter_latest.zip         # [權重] 最終進化之 AI 模型大腦
 8 ├── pretrained_hunter.zip     # [權重] 模仿人類行為的初期模型
 9 ├── human_demo/               # [數據] 存放所有人類操作錄製檔 (.npz)
10 ├── logs/                     # [日誌] 訓練過程數據 (TensorBoard 使用)
11 └──  performance_history.csv   # [紀錄] 訓練效能歷史數據追蹤

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
   * 模擬器環境：基於 Gymnasium 標準封裝，整合 Pygame 渲染引擎。
   * 決策大腦：採用 PPO (Proximal Policy Optimization) 演算法搭配 MLP (多層感知器) 網路。
   * 動作空間：5 維連續空間向量（視野位移 $\times 2$、點擊座標 $\times 2$、行為觸發 $\times 1$）。
   * 獎勵機制：包含時間成本損耗、搜尋引導獎勵及精準度加權分數。

  ---

  📊 效能指標 (KPIs)
  在成果展示介面 (rl_ai_demo.py) 中，我們重點驗證以下數據：
   1. 累計獵殺次數：衡量 AI 的總產出能力。
   2. 點擊準確率 (%)：衡量 AI 對局部座標的控制精細度。
   3. 期望獵殺數 (EHK)：每分鐘預期的有效點擊產出。
   4. 平均誤差 (Error)：點擊位置與目標中心點的像素級偏差。

  ---
  開發者：Gwen (NeuroProGram Project)
  版本：v7.0 (2026-04-12)
