import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import os
import pygame
import sys
import datetime

# --- 設定固定路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEMO_DIR = os.path.join(BASE_DIR, "human_demo")
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class AdvancedHunterEnv(gym.Env):
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.num_targets, self.target_speed, self.view_speed = 50, 12.0, 600.0
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
        rel_pos = self.targets_pos[np.argmin(dists)] - self.view_pos
        obs_rel_pos = np.clip(rel_pos / 2000.0, -1, 1)
        seen_flag = 1.0 if np.all(np.abs(rel_pos) <= self.win_size/2) else -1.0
        return np.array([(self.view_pos[0]/self.map_size)*2-1, (self.view_pos[1]/self.map_size)*2-1,
                         obs_rel_pos[0], obs_rel_pos[1], seen_flag], dtype=np.float32)
    def step(self, action):
        self.current_step += 1
        self.view_pos = np.clip(self.view_pos + action[0:2] * self.view_speed, 0, self.map_size)
        self.targets_pos += self.targets_vel
        self.targets_vel[self.targets_pos <= 0] *= -1
        self.targets_vel[self.targets_pos >= self.map_size] *= -1
        obs = self._get_obs()
        reward = -0.01 
        info = {"hit": False, "click_pos": None}
        if action[4] > 0:
            click_pos = self.view_pos + (action[2:4] * (self.win_size / 2))
            info["click_pos"] = click_pos
            if np.min(np.linalg.norm(self.targets_pos - click_pos, axis=1)) < 65:
                self.targets_pos[np.argmin(np.linalg.norm(self.targets_pos - click_pos, axis=1))] = np.random.uniform(0, self.map_size, 2)
                info["hit"] = True
        return obs, reward, False, self.current_step >= self.max_steps, info

class InteractiveVisualizer:
    def __init__(self, model_filename):
        pygame.init()
        self.env = AdvancedHunterEnv()
        m_path = os.path.join(BASE_DIR, model_filename) if model_filename else None
        self.model = PPO.load(m_path) if m_path and os.path.exists(m_path) else None
        self.screen = pygame.display.set_mode((1300, 750))
        pygame.display.set_caption("互動錄製與展示 (數據保存版)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("microsoftjhenghei", 22, bold=True)
        self.mode = "HUMAN" 
        self.recording = []
        self.total_hits, self.total_frames, self.total_clicks = 0, 0, 0
        self.obs, _ = self.env.reset()

    def get_human_action(self):
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_w]: dy = -1
        if keys[pygame.K_s]: dy = 1
        if keys[pygame.K_a]: dx = -1
        if keys[pygame.K_d]: dx = 1
        ct, cx, cy = -1, 0, 0
        mp, mc = pygame.mouse.get_pos(), pygame.mouse.get_pressed()
        if 50 <= mp[0] <= 650 and 50 <= mp[1] <= 650:
            if mc[0]: ct, cx, cy = 1, (mp[0]-350)/300.0, (mp[1]-350)/300.0
        return np.array([dx, dy, cx, cy, ct], dtype=np.float32)

    def run(self):
        while True:
            self.total_frames += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.save_recording(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if self.model: self.mode = "AI" if self.mode == "HUMAN" else "HUMAN"
                    self.env.reset(); self.total_hits = 0
            action = self.model.predict(self.obs, deterministic=True)[0] if self.mode == "AI" else self.get_human_action()
            if action[4] > 0: self.total_clicks += 1
            if self.mode == "HUMAN": self.recording.append({"obs": self.obs, "act": action})
            self.obs, _, _, truncated, info = self.env.step(action)
            if truncated: self.obs, _ = self.env.reset()
            if info["hit"]: self.total_hits += 1
            self.screen.fill((30, 30, 30))
            pygame.draw.rect(self.screen, (50, 50, 50), (50, 50, 600, 600))
            v_p, w_s = self.env.view_pos, self.env.win_size
            for t_p in self.env.targets_pos:
                rx, ry = (t_p[0]-(v_p[0]-w_s/2))*(600/w_s), (t_p[1]-(v_p[1]-w_s/2))*(600/w_s)
                if 0<=rx<=600 and 0<=ry<=600: pygame.draw.circle(self.screen, (50,150,255), (50+int(rx), 50+int(ry)), 10)
            if action[4] > 0 and info["click_pos"] is not None:
                cx, cy = int((info["click_pos"][0]-(v_p[0]-w_s/2))*(600/w_s)), int((info["click_pos"][1]-(v_p[1]-w_s/2))*(600/w_s))
                c = (255, 50, 50) if info["hit"] else (150, 150, 150)
                pygame.draw.line(self.screen, c, (50+cx-15, 50+cy-15), (50+cx+15, 50+cy+15), 3)
                pygame.draw.line(self.screen, c, (50+cx+15, 50+cy-15), (50+cx-15, 50+cy+15), 3)
            pygame.draw.rect(self.screen, (20, 20, 20), (700, 50, 500, 500))
            for t_p in self.env.targets_pos: pygame.draw.circle(self.screen, (100, 150, 200), (700+int(t_p[0]*0.05), 50+int(t_p[1]*0.05)), 2)
            pygame.draw.rect(self.screen, (255, 255, 255), (700+int((v_p[0]-w_s/2)*0.05), 50+int((v_p[1]-w_s/2)*0.05), 40, 40), 1)
            self.screen.blit(self.font.render(f"模式: {self.mode} | 成功: {self.total_hits}", True, (255, 255, 0)), (50, 15))
            self.screen.blit(self.font.render(f"錄製數: {len(self.recording)} | 保存路徑: human_demo/", True, (0, 255, 255)), (700, 570))
            pygame.display.flip(); self.clock.tick(60)

    def save_recording(self):
        if len(self.recording) > 100:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_demo_{timestamp}.npz"
            save_path = os.path.join(DEMO_DIR, filename)
            np.savez(save_path, obs=np.array([d["obs"] for d in self.recording]), actions=np.array([d["act"] for d in self.recording]))
            print(f"\n[系統] 錄製成功！檔案已儲存至：{save_path}")

if __name__ == "__main__":
    InteractiveVisualizer("hunter_latest.zip").run()
