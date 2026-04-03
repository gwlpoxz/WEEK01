import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

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
        self.targets_pos = np.random.uniform(0, self.map_size, (self.num_targets, 2)).astype(np.float32)
        self.targets_vel = np.random.uniform(-self.target_speed, self.target_speed, (self.num_targets, 2)).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        dists = np.linalg.norm(self.targets_pos - self.view_pos, axis=1)
        nearest_idx = np.argmin(dists)
        rel_pos = self.targets_pos[nearest_idx] - self.view_pos
        half_win = self.win_size / 2
        in_view = np.all(np.abs(rel_pos) <= half_win)
        
        if in_view:
            obs_rel_pos = rel_pos / half_win
            seen_flag = 1.0
        else:
            obs_rel_pos = np.clip(rel_pos / 2000.0, -1, 1)
            seen_flag = -1.0
        return np.array([(self.view_pos[0]/self.map_size)*2-1, (self.view_pos[1]/self.map_size)*2-1,
                         obs_rel_pos[0], obs_rel_pos[1], seen_flag], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        # 1. 視野移動
        self.view_pos = np.clip(self.view_pos + action[0:2] * self.view_speed, 0, self.map_size)
        # 2. 目標移動與反彈
        self.targets_pos += self.targets_vel
        out_bounds = (self.targets_pos <= 0) | (self.targets_pos >= self.map_size)
        self.targets_vel[out_bounds] *= -1
        self.targets_pos = np.clip(self.targets_pos, 0, self.map_size)
        
        obs = self._get_obs()
        # --- 核心獎勵修正 ---
        reward = -0.01 # 基礎時間懲罰
        
        seen_in_window = (obs[4] > 0)
        if seen_in_window:
            reward += 0.5 # [鼓勵] 只要目標在視窗內就給獎勵，引導 AI 學習搜尋
        
        info = {"hit": False}
        if action[4] > -0.5: # [修正] 降低訓練時的點擊門檻，強制 AI 嘗試點擊
            cp = self.view_pos + (action[2:4] * (self.win_size / 2))
            dists = np.linalg.norm(self.targets_pos - cp, axis=1)
            min_dist = np.min(dists)
            
            if min_dist < 85: # 擊中判定
                reward += 30.0 # [大幅加分] 擊中目標
                self.targets_pos[np.argmin(dists)] = np.random.uniform(0, self.map_size, 2)
                info["hit"] = True
            else:
                # [弱化懲罰] 減輕沒點中的懲罰，防止 AI 因為怕扣分而不敢點擊
                reward -= 0.2 
        
        return obs, reward, False, self.current_step >= self.max_steps, info

if __name__ == "__main__":
    model_path = os.path.join(BASE_DIR, "hunter_latest.zip")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("==========================================")
    print("   AI 自行強化訓練 (Turbo 探索模式)")
    print("==========================================")
    
    env = Monitor(AdvancedHunterEnv())
    
    if os.path.exists(model_path):
        print(f"[載入] 正在載入既有模型，並重置探索率...")
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
        # 強制提高學習率與探索程度
        model.learning_rate = 0.0005
        model.ent_coef = 0.05 # [關鍵] 強制 AI 亂試動作，打破 -1.0 的死局
    else:
        print("[建立] 啟動全新 PPO 高探索訓練")
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=0.0005, 
                    ent_coef=0.05, 
                    tensorboard_log=LOG_DIR)

    train_steps = 100000 # 建議一次練 10 萬步
    print(f"[執行] 正在進行強化訓練，打破 Action 4 死鎖...")
    model.learn(total_timesteps=train_steps, reset_num_timesteps=False)

    model.save(model_path)
    print(f"[成功] 訓練完成，模型已保存。請現在執行 rl_ai_demo.py 觀察 Action 4 是否開始跳動！")
