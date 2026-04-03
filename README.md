
# AI 大圖搜索與精準點擊專案 (NeuroProGram)

本專案實作了一套基於 **強化學習 (Reinforcement Learning, RL)** 的高階觀察者系統。AI 代理人必須在 $10000 \times 10000$ 的全域空間中，透過 $800 \times 800$ 的有限局部視窗搜尋動態目標。

## 🚀 快速上手
專案架構
```
   2 ├── rl_human_recorder.py      # [主程式] 專家數據錄製與互動介面
   3 ├── rl_pretraining.py         # [主程式] 模仿學習預訓練系統 (BC)
   4 ├── rl_machine_training.py    # [主程式] 機器增量強化訓練系統 (PPO)
   5 ├── rl_ai_demo.py             # [主程式] AI 成果效能驗證展示介面
   6 ├── hunter_latest.zip         # [權重] 最終進化之 AI 模型大腦
   7 ├── pretrained_hunter.zip     # [權重] 模仿人類行為的初期模型
   8 ├── human_demo/               # [數據] 存放所有人類操作錄製檔 (.npz)
   9 └── logs/                     # [日誌] 訓練過程數據 (TensorBoard 使用)
```

### 1. 環境設定
請確保您的系統已安裝 Python 3.8+。執行以下指令安裝必要套件：
```bash
pip install gymnasium stable-baselines3 pygame torch numpy
```

### 2. 開發流程
本專案遵循「從人類到 AI」的漸進式訓練流程：

#### 步驟 A：專家數據採集 (Expert Data Collection)
手動錄製人類專家的操作演示，教導 AI 基本的搜尋與點擊邏輯。
```bash
python rl_human_recorder.py
```
*   **在局部視窗內 **點擊左鍵** 進行獵殺。
*   **輸出**：數據將附帶時間戳並儲存於 `human_demo/` 資料夾。

#### 步驟 B：行為克隆預訓練 (Behavioral Cloning)
透過模仿最新錄製的人類數據，初步訓練 AI 策略。
```bash
python rl_pretraining.py
```
*   **產出**：生成 `pretrained_hunter.zip` 模型檔。

#### 步驟 C：機器增量強化訓練 (Incremental RL Fine-tuning)
讓 AI 在人類經驗的基礎上，透過自我試錯與獎勵優化超越人類表現。
```bash
# 使用預訓練權重初始化最新模型
# (Windows 指令)
copy pretrained_hunter.zip hunter_latest.zip
# (Unix/Mac 指令)
cp pretrained_hunter.zip hunter_latest.zip

# 執行增量訓練
python rl_machine_training.py
```

#### 步驟 D：效能驗證展示 (Performance Verification)
執行自動化演示，評估 AI 的搜尋效率與點擊精確度。
```bash
python rl_ai_demo.py
```

---

## 🛠 技術細節
- **強化學習演算法**：近端策略優化 (Proximal Policy Optimization, PPO)。
- **神經網路架構**：多層感知器 (Multi-Layer Perceptron, MLP)。
- **觀測空間 (Observation)**：包含歸一化全域位置、目標相對座標及基於雷達的方向引導特徵。
- **動作空間 (Action Space)**：5 維連續空間（視野平移 x2、精準點擊座標 x2、點擊觸發 x1）。

## 📊 效能指標 (KPIs)
- **累計擊中數 (Cumulative Hits)**：成功中和的目標總數。
- **點擊準確率 (%) (Click Accuracy)**：擊中次數與嘗試點擊總數的比例。
- **每分鐘期望獵殺數 (EHK)**：標準化的生產率指標。
- **平均誤差 (Pixels)**：點擊位置與目標中心點的像素級偏差。

---
**本儲存庫供工程審核與後續演算法優化使用。**
