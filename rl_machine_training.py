import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import datetime
import pandas as pd # 確保環境有安裝 pandas

# --- 設定固定路徑 ---
BASE_DIR = r"__File__"
LOG_DIR = os.path.join(BASE_DIR, "logs")
HISTORY_FILE = os.path.join(BASE_DIR, "performance_history.csv")

# --- 檢查 TensorBoard 是否安裝 ---
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[警告] 未偵測到 TensorBoard，將停用視覺化日誌功能。若需使用請執行: pip install tensorboard")

class AdvancedHunterEnv(gym.Env):
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.num_targets, self.target_speed, self.view_speed = 60, 15.0, 600.0
        self.max_steps = 1000
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step, self.view_pos = 0, np.array([5000.0, 5000.0], dtype=np.float32)
        self.current_target_idx = 0  # 追蹤序號
        self.targets_pos = np.random.uniform(0, self.map_size, (self.num_targets, 2)).astype(np.float32)
        self.targets_vel = np.random.uniform(-self.target_speed, self.target_speed, (self.num_targets, 2)).astype(np.float32)
        return self._get_obs(), {}
    def _get_obs(self):
        # 觀察目前的順序目標
        target_pos = self.targets_pos[self.current_target_idx]
        rel_pos = target_pos - self.view_pos
        obs_rel_pos = np.clip(rel_pos / 2000.0, -1, 1) 
        seen_flag = 1.0 if np.all(np.abs(rel_pos) <= self.win_size/2) else -1.0
        return np.array([(self.view_pos[0]/self.map_size)*2-1, (self.view_pos[1]/self.map_size)*2-1,
                         obs_rel_pos[0], obs_rel_pos[1], seen_flag], dtype=np.float32)
    def step(self, action):
        self.current_step += 1
        self.view_pos = np.clip(self.view_pos + action[0:2] * self.view_speed, 0, self.map_size)
        self.targets_pos += self.targets_vel
        out_bounds = (self.targets_pos <= 0) | (self.targets_pos >= self.map_size)
        self.targets_vel[out_bounds] *= -1
        self.targets_pos = np.clip(self.targets_pos, 0, self.map_size)
        obs = self._get_obs()
        reward = -0.01 
        if obs[4] > 0: reward += 0.5 
        info = {"hit": False}
        if action[4] > -0.5:
            cp = self.view_pos + (action[2:4] * (self.win_size / 2))
            dist_to_current = np.linalg.norm(self.targets_pos[self.current_target_idx] - cp)
            if dist_to_current < 85:
                reward += 30.0; info["hit"] = True
                self.targets_pos[self.current_target_idx] = np.random.uniform(0, self.map_size, 2)
                self.current_target_idx = (self.current_target_idx + 1) % self.num_targets
            else: reward -= 0.2
        return obs, reward, False, self.current_step >= self.max_steps, info

def log_performance(step_count, mean_reward, hit_rate):
    """ 紀錄效能至 CSV，使用中原標準時間 """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "時間 (CST)": [now],
        "累積訓練步數": [step_count],
        "平均回合得分": [round(mean_reward, 2)],
        "平均每千步擊中數": [round(hit_rate, 1)]
    }
    df = pd.DataFrame(new_data)
    if not os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
    print(f"\n[日誌] 效能數據已寫入：{HISTORY_FILE}")

if __name__ == "__main__":
    model_path = os.path.join(BASE_DIR, "hunter_latest.zip")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("==========================================")
    print("   AI 自行強化訓練 (效能追蹤增強版)")
    print("==========================================")
    
    env = Monitor(AdvancedHunterEnv())
    
    tb_log = LOG_DIR if TENSORBOARD_AVAILABLE else None
    
    if os.path.exists(model_path):
        print(f"[載入] 繼承現有權重繼續強化...")
        model = PPO.load(model_path, env=env, tensorboard_log=tb_log)
        model.learning_rate = 0.0002
    else:
        print("[建立] 啟動全新模型訓練...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log=tb_log)

    # 執行訓練
    train_steps = 50000
    model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
    model.save(model_path)

    # --- 訓練後自動評測 ---
    print("\n[評測] 正在計算本次訓練成果...")
    test_env = AdvancedHunterEnv()
    total_rewards = []
    total_hits = 0
    for _ in range(3): # 測試 3 個回合
        obs, _ = test_env.reset()
        ep_reward = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, info = test_env.step(action)
            ep_reward += reward
            if info["hit"]: total_hits += 1
        total_rewards.append(ep_reward)
    
    avg_reward = np.mean(total_rewards)
    avg_hits = total_hits / 3
    
    # 紀錄到 CSV
    log_performance(model.num_timesteps, avg_reward, avg_hits)
    
    print(f"\n[成功] 訓練完成！")
    print(f"‧ 目前累積總步數: {model.num_timesteps}")
    print(f"‧ 本次平均得分: {avg_reward:.2f}")
    print(f"‧ 平均每回合擊中: {avg_hits:.1f} 次")
