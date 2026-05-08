
  本專案實作了一套基於強化學習 (Reinforcement Learning) 的高階觀察者系統原型。AI 代理人必須在 $10000 \times 10000$
  的巨大空間中，透過 $800 \times 800$ 的有限視野進行自主搜尋，並對動態目標執行像素級的精準獵殺。

  🎯 核心技術挑戰
   * 非全知視角：AI 僅具備 0.64% 的局部觀測率，需具備高效搜索策略。
   * 解耦控制：實作「視野移動」與「全視窗隨機點擊」的同步決策，不依賴傳統準星對齊。
   * 持續進化：建立模型權重繼承機制，確保訓練進度永不歸零。

  ---

  📂 專案架構 (Project Tree)
```
    2 ├── rl_human_recorder.py      # [主程式] 專家數據錄製與互動介面
    3 ├── rl_pretraining.py         # [主程式] 模仿學習預訓練系統 (BC)
    4 ├── rl_machine_training.py    # [主程式] 機器增量強化訓練系統 (PPO)
    5 ├── rl_ai_demo.py             # [主程式] AI 成果效能驗證展示介面
    6 ├── custom_ppo.py             # [核心] 自定義 PPO 演算法邏輯實現
    7 ├── model.py                  # [核心] 類神經網路模型 (CNN/MLP) 架構定義
    8 ├── hunter_latest.pth         # [權重] 訓練好的模型參數檔案 (.pth)
    9 ├── hunter_latest.zip         # [權重] 最終進化之 AI 模型大腦 (壓縮備份)
   10 ├── pretrained_hunter.zip     # [權重] 模仿人類行為的初期模型 (壓縮備份)
   11 ├── human_demo/               # [數據] 存放所有人類操作錄製檔 (.npz)
   12 ├── logs/                     # [日誌] 訓練過程數據 (TensorBoard 使用)
   13 └── performance_history.csv   # [紀錄] 訓練效能歷史數據追蹤


```
  ---

  🚀 漸進式強化流程 (Execution Flow)
```
1. [數據採集] 專家演示錄製 (Expert Data Collection)
    └── 執行 `rl_human_recorder.py`
      ├── 說明：手動操控獵人捕捉目標，錄製高品質專家操作軌跡。
      └── 產出：`human_demo/*.npz` (行為數據集)
2. [模仿學習] 行為選殖預訓練 (Imitation Learning)
   └── 執行 `rl_pretraining.py`
      ├── 說明：讓 AI 讀取專家數據，快速習得「追逐」與「避障」基礎邏輯。
      └── 產出：`pretrained_hunter.zip` (具備基本智力的模型)
3. [增量強化] 機器自我進化 (Reinforcement Learning)
    └── 執行 `rl_machine_training.py`
      └── 產出：`hunter_latest.pth` / `logs/` (最終進化之 AI 權重與日誌)
4. [成果驗證] AI 效能展示 (Final Demo)
    └── 執行 `rl_ai_demo.py`
      ├── 說明：開啟 25 FPS 高流暢視覺介面，驗證 AI 在實戰中的獵殺效率。
      └── 產出：KPI 報告與自動化演示。
```
  ---

  🛠 技術規格說明 (Technical Stack)
   * 模擬器環境：基於 Gymnasium 標準封裝，整合 Pygame 渲染引擎。
   * 決策大腦：採用 PPO (Proximal Policy Optimization) 演算法搭配 MLP (多層感知器) 網路。
   * 動作空間：5 維連續空間向量（視野位移 $\times 2$、點擊座標 $\times 2$、行為觸發 $\times 1$）。
   * 獎勵機制：包含時間成本損耗、距離趨近引導、視野中心鎖定紅利及精準擊中高額加權。

  ---

  📊 效能指標 (KPIs)
  在成果展示介面 (rl_ai_demo.py) 中，我們重點驗證以下數據：
   1. 累計獵殺次數：紀錄 AI 在單場測試（1000 步）中成功擊中目標的總量，衡量 AI 的總產出能力。
   2. 點擊準確率 (%)：計算「成功擊中次數 / 總開火次數」之百分比，衡量 AI 對局部座標的控制精細度。
   3. 期望獵殺數 (EHK)：即每分鐘預期的有效點擊產出 (Estimated Hits per Minute)，反映 AI 的實戰獵殺效率。
   4. 平均誤差 (Error)：紀錄每次點擊位置與目標中心點的像素級偏差（Pixel Distance），驗證 AI 鎖定目標的精準度。

  ---
  開發者：Gwen (NeuroProGram Project)
  版本：v10.0 (2026-05-04)
