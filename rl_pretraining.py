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
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    print(f"\n[錯誤] 缺少必要套件：{e}"); sys.exit()

class AdvancedHunterEnv(gym.Env):
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    def reset(self, seed=None): return np.zeros(5, dtype=np.float32), {}

def pretrain_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    DEMO_DIR = os.path.join(BASE_DIR, "human_demo")
    save_file = os.path.join(BASE_DIR, "pretrained_hunter.zip")

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

    data = np.load(latest_file)
    obs_tensor = th.tensor(data['obs'], dtype=th.float32)
    act_tensor = th.tensor(data['actions'], dtype=th.float32)
    
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"\n[3/4] 初始化神經網路模型...", flush=True)
    model = PPO("MlpPolicy", AdvancedHunterEnv(), verbose=0)
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"\n[4/4] 開始預訓練 (行為克隆)...", flush=True)
    policy.train()
    for epoch in range(50):
        l_sum = 0
        for obs_b, act_b in loader:
            optimizer.zero_grad()
            loss = loss_fn(policy.get_distribution(obs_b).mode(), act_b)
            loss.backward(); optimizer.step(); l_sum += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"   >>> 進度: [{epoch+1}/50] | 誤差: {l_sum/len(loader):.6f}", flush=True)

    model.save(save_file)
    print(f"\n[成功] 預訓練完成！模型已儲存至：{save_file}")

if __name__ == "__main__":
    pretrain_model()
