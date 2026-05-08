import os
import sys
import numpy as np
import glob

# 啟動即時提示
print("\n[1/4] 正在載入系統組件...", flush=True)

try:
    import torch as th
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from custom_ppo import PPO
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    print(f"\n[錯誤] 缺少必要套件：{e}"); sys.exit()

class AdvancedHunterEnv(gym.Env): #環境:提供特徵與動作的維度空間 (Observation Space & Action Space)，讓神經網路可以正確初始化對應數量的神經元
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    def reset(self, seed=None): return np.zeros(7, dtype=np.float32), {}

def pretrain_model(): #主程式
    BASE_DIR = r"C:\Users\Gwen\Desktop\NeuroProGram\week01"
    DEMO_DIR = os.path.join(BASE_DIR, "human_demo")
    latest_model_path = os.path.join(BASE_DIR, "hunter_latest.pth")

    print(f"\n[2/4] 正在搜尋最新錄製數據...", flush=True)
    # 尋找資料夾下所有的 npz 檔案
    files = glob.glob(os.path.join(DEMO_DIR, "*.npz"))
    
    if not files:
        print(f"\n[錯誤] 找不到任何錄製數據於：{DEMO_DIR}")
        print("請確認您已經執行過 rl_human_recorder.py。")
        return

    # 自動挑選修改時間最晚的檔案 (最新錄製的)
    latest_file = max(files, key=os.path.getmtime)
    print(f"[系統] 已鎖定最新數據檔案：{os.path.basename(latest_file)}")

    # [新] 載入錄製數據與防呆檢查
    data = np.load(latest_file)
    obs_data = data['obs']
    act_data = data['actions']
    
    # 維度防護檢查：防止讀取到舊版維度不同的錄製檔
    env_obs_dim = AdvancedHunterEnv().observation_space.shape[0]
    if obs_data.shape[1] != env_obs_dim:
        print(f"\n[錯誤] 錄製數據的維度 (特徵數: {obs_data.shape[1]}) 與目前環境 (特徵數: {env_obs_dim}) 不符！")
        print(f"這通常是因為讀取到了舊版的錄製檔案。建議您清空 'human_demo' 資料夾中的舊檔案，")
        print(f"並重新執行 rl_human_recorder.py 錄製新數據 (請確保在畫面中操作超過 100 步才會觸發存檔)。")
        return

    obs_tensor = th.tensor(obs_data, dtype=th.float32)
    act_tensor = th.tensor(act_data, dtype=th.float32)
    
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"\n[3/4] 初始化/載入神經網路模型...", flush=True)
    if os.path.exists(latest_model_path):
        print(f"-> 偵測到現有訓練成果，將載入 {os.path.basename(latest_model_path)} 進行人類指導微調 (Fine-tuning)...", flush=True)
        model = PPO.load(latest_model_path, env=AdvancedHunterEnv())
        lr = 1e-4  # 使用較小的學習率，避免破壞原本已學好的 RL 表現
    else:
        print(f"-> 找不到現有模型，建立全新模型...", flush=True)
        model = PPO(env=AdvancedHunterEnv(), verbose=0)
        lr = 1e-3

    # 定義 MSE Loss (均方誤差)。這是監督式學習最經典的損失函數
    # 用來衡量「AI 預測的動作」跟「人類真實的動作」差距有多大
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)


    loss_fn = nn.MSELoss()

    print(f"\n[4/4] 開始人類指導學習 (行為克隆)...", flush=True) #監督式學習的核心邏輯
    policy.train()
    for epoch in range(50):# 讓 AI 把整本題庫讀 50 遍
        l_sum = 0
        for obs_b, act_b in loader:
            optimizer.zero_grad()
            # 🌟 步驟 A：讓 AI 看考題 (obs_b) 並自己作答 (action_preds)
            # 在自訂的 PPO 架構中，actor_mean 直接輸出預測的連續動作
            action_preds = policy.actor_mean(obs_b)
            # 🌟 步驟 B：計算「分數」 ( Loss ) -- 也就是差距有多大
            # 這裡我們用最直觀的「均方誤差 (MSE)」
            # 公式：(預測值 - 真實值) 的平方，全部加起來平均
            # 越小代表 AI 越會模仿人類
            loss = loss_fn(action_preds, act_b)
            # 🌟 步驟 C：回溯與修正 (反向傳播)
            # 告訴電腦：「這個差距怎麼造成的？回去改進！」
            loss.backward()
            # 更新參數 (實際修改神經網路的權重)
            optimizer.step()
            l_sum += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"   >>> 進度: [{epoch+1}/50] | 誤差: {l_sum/len(loader):.6f}", flush=True)

    model.save(latest_model_path)
    print(f"\n[成功] 訓練完成！模型已更新並儲存至：{latest_model_path}")

if __name__ == "__main__":
    pretrain_model() #呼叫pretrain_model#主程式
