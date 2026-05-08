import gymnasium as gym
from gymnasium import spaces
import numpy as np
from custom_ppo import PPO 
from stable_baselines3.common.monitor import Monitor
import os
import datetime
import pandas as pd

# --- 設定固定路徑 ---
BASE_DIR = r"C:\Users\Gwen\Desktop\NeuroProGram\week01"
LOG_DIR = os.path.join(BASE_DIR, "logs")
HISTORY_FILE = os.path.join(BASE_DIR, "performance_history.csv")

class AdvancedHunterEnv(gym.Env): #環境
    def __init__(self): #定義地圖大小
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.num_targets, self.target_speed, self.view_speed = 60, 15.0, 600.0
        self.max_steps = 1000
        # Action (動作) 的類型：連續型動作空間
        # 宣告 AI 有 5 個旋鈕可以轉 (-1 到 1)，分別控制：
        # [油門X, 油門Y, 準星X, 準星Y, 開火(0~1)]
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        # Observation (觀察) 的類型：連續型 observation 空間
        # 宣告 AI 的眼睛只能看到 -1 到 1 之間，總共 7 個維度的數值
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step, self.view_pos = 0, np.array([5000.0, 5000.0], dtype=np.float32)
        self.current_target_idx = 0
        self.targets_pos = np.random.uniform(0, self.map_size, (self.num_targets, 2)).astype(np.float32)
        self.targets_vel = np.random.uniform(-self.target_speed, self.target_speed, (self.num_targets, 2)).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self): #觀察目標位置(取特徵值)
        target_pos = self.targets_pos[self.current_target_idx]
        rel_pos = target_pos - self.view_pos
        half_win = self.win_size / 2
        in_view = np.all(np.abs(rel_pos) <= half_win)

        if in_view:
            obs_rel_pos = rel_pos / half_win
            seen_flag = 1.0
        else:
            obs_rel_pos = np.clip(rel_pos / 2000.0, -1, 1)
            seen_flag = -1.0

        next_idx = (self.current_target_idx + 1) % self.num_targets
        next_rel_pos = self.targets_pos[next_idx] - self.view_pos
        obs_next_rel_pos = np.clip(next_rel_pos / 2000.0, -1, 1)

        return np.array([
            (self.view_pos[0]/self.map_size)*2-1,
            (self.view_pos[1]/self.map_size)*2-1,
            obs_rel_pos[0], obs_rel_pos[1],
            seen_flag,
            obs_next_rel_pos[0], obs_next_rel_pos[1]
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.view_pos = np.clip(self.view_pos + action[0:2] * self.view_speed, 0, self.map_size)
        self.targets_pos += self.targets_vel

        out_bounds = (self.targets_pos <= 0) | (self.targets_pos >= self.map_size)
        self.targets_vel[out_bounds] *= -1
        self.targets_pos = np.clip(self.targets_pos, 0, self.map_size)

        obs = self._get_obs()
        reward = -0.01

        target_pos = self.targets_pos[self.current_target_idx]
        view_center = self.view_pos + (self.win_size / 2)
        current_dist = np.linalg.norm(target_pos - view_center)

        if not hasattr(self, 'prev_dist'):
            self.prev_dist = current_dist

        dist_diff = self.prev_dist - current_dist
        reward += dist_diff * 0.05
        self.prev_dist = current_dist

        if obs[4] > 0:
            center_bonus = 1.0 - max(abs(obs[2]), abs(obs[3]))
            reward += 0.2 + (0.3 * center_bonus)

        info = {"hit": False}

        if action[4] > -0.5:
            cp = self.view_pos + (action[2:4] * (self.win_size / 2))
            # 檢查是否擊中目前的序號目標 (算直線距離: 滑鼠點擊位置 - 目前該打的目標位置)
            dist_to_current = np.linalg.norm(self.targets_pos[self.current_target_idx] - cp)

            if dist_to_current < 85: # 距離小於 85 像素就算擊中！(這就是判定半徑)
                reward += 50.0
                info["hit"] = True
                # 擊中後，原本的目標會隨機傳送到地圖上另一個角落
                self.targets_pos[self.current_target_idx] = np.random.uniform(0, self.map_size, 2)
                # 把追蹤目標換成下一個序號
                self.current_target_idx = (self.current_target_idx + 1) % self.num_targets
                self.prev_dist = np.linalg.norm(self.targets_pos[self.current_target_idx] - view_center)
            else:
                reward -= 1.5
                if dist_to_current < 400:
                    reward += 3.0 * (1 - dist_to_current / 400)

        return obs, reward, False, self.current_step >= self.max_steps, info


# ✅ 擊中率(%)
def log_performance(step_count, mean_reward, hit_rate):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hit_rate_percent = hit_rate * 100  # 轉百分比

    new_data = {
        "時間 (CST)": [now],
        "累積訓練步數": [step_count],
        "平均回合得分": [round(mean_reward, 2)],
        "擊中率(%)": [round(hit_rate_percent, 2)]
    }

    df = pd.DataFrame(new_data)

    if not os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")

    print(f"\n[日誌] 效能數據已寫入：{HISTORY_FILE}")


if __name__ == "__main__": #啟動訓練流程
    model_path = os.path.join(BASE_DIR, "hunter_latest.pth")
    os.makedirs(LOG_DIR, exist_ok=True)

    print("==========================================")
    print("   強化訓練 (PPO)")
    print("==========================================")

    env = Monitor(AdvancedHunterEnv())

    if os.path.exists(model_path):
        print(f"[載入] 繼承現有權重繼續強化...")
        model = PPO.load(model_path, env=env, verbose=1, learning_rate=0.0002)
    else:
        print("[建立] 啟動全新模型訓練...")
        model = PPO(env, verbose=1, learning_rate=0.0003, ent_coef=0.01)

    # 訓練
    train_steps = 200000
    model.learn(total_timesteps=train_steps)
    model.save(model_path)

    # --- 評測 ---
    print("\n[評測] 正在計算本次訓練成果...")
    test_env = AdvancedHunterEnv()

    total_rewards = []
    total_hits = 0
    total_steps = 0

    for _ in range(3):
        obs, _ = test_env.reset()
        ep_reward = 0

        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, info = test_env.step(action)

            ep_reward += reward
            total_steps += 1

            if info["hit"]:
                total_hits += 1

        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)

    # ✅ 擊中率
    hit_rate = total_hits / total_steps

    # 存檔
    log_performance(model.num_timesteps, avg_reward, hit_rate)

    print(f"\n[成功] 訓練完成！")
    print(f"‧ 目前累積總步數: {model.num_timesteps}")
    print(f"‧ 本次平均得分: {avg_reward:.2f}")
    print(f"‧ 擊中率: {hit_rate*100:.2f}%")