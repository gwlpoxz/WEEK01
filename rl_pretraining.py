import os
import sys
import numpy as np
import glob
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- 設定固定路徑 ---
BASE_DIR = r"C:\Users\Gwen\Desktop\NeuroProGram\week01"
DEMO_DIR = os.path.join(BASE_DIR, "human_demo")

class AdvancedHunterEnv(gym.Env):
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    def reset(self, seed=None): return np.zeros(5, dtype=np.float32), {}

def pretrain_model():
    save_file = os.path.join(BASE_DIR, "pretrained_hunter.zip")
    print("\n[1/4] 搜尋最新 5D 錄製數據...", flush=True)
    files = glob.glob(os.path.join(DEMO_DIR, "*.npz"))
    if not files: print("[錯誤] 找不到數據！"); return
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"[系統] 已鎖定檔案：{os.path.basename(latest_file)}")

    data = np.load(latest_file)
    obs_tensor = th.tensor(data['obs'], dtype=th.float32)
    act_tensor = th.tensor(data['actions'], dtype=th.float32)
    
    # 檢查維度是否匹配 5D
    if obs_tensor.shape[1] != 5:
        print(f"[錯誤] 數據維度為 {obs_tensor.shape[1]}D，非 5D。請重新錄製！")
        return

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("[2/4] 初始化 5D 策略網路...", flush=True)
    model = PPO("MlpPolicy", AdvancedHunterEnv(), verbose=0)
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("[3/4] 開始預訓練...", flush=True)
    policy.train()
    for epoch in range(50):
        l_sum = 0
        for obs_b, act_b in loader:
            optimizer.zero_grad()
            loss = loss_fn(policy.get_distribution(obs_b).mode(), act_b)
            loss.backward(); optimizer.step(); l_sum += loss.item()
        if (epoch+1)%10==0: print(f"   >>> 進度: [{epoch+1}/50] | 誤差: {l_sum/len(loader):.6f}")

    model.save(save_file)
    print(f"[4/4] 成功！模型已存至：{save_file}")

if __name__ == "__main__":
    pretrain_model()
