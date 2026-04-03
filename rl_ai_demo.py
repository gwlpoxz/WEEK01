import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os
import pygame
import sys
from collections import deque

# --- 設定固定路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class AdvancedHunterEnv(gym.Env):
    def __init__(self):
        super(AdvancedHunterEnv, self).__init__()
        self.map_size, self.win_size = 10000.0, 800.0
        self.num_targets, self.target_speed, self.view_speed = 60, 15.0, 600.0
        self.max_steps = 1000
        self.reset()

    def reset(self, seed=None):
        self.current_step = 0
        self.view_pos = np.array([5000.0, 5000.0], dtype=np.float32)
        self.targets_pos = np.random.uniform(0, self.map_size, (self.num_targets, 2)).astype(np.float32)
        self.targets_vel = np.random.uniform(-self.target_speed, self.target_speed, (self.num_targets, 2)).astype(np.float32)
        return self._get_obs()

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
        self.view_pos = np.clip(self.view_pos + action[0:2] * self.view_speed, 0, self.map_size)
        self.targets_pos += self.targets_vel
        out_bounds = (self.targets_pos <= 0) | (self.targets_pos >= self.map_size)
        self.targets_vel[out_bounds] *= -1
        self.targets_pos = np.clip(self.targets_pos, 0, self.map_size)
        
        obs = self._get_obs()
        info = {"hit": False, "click_pos": None, "dist": 999.0, "tried": (action[4]>-0.1)}
        
        if action[4] > -0.1: 
            cp = self.view_pos + (action[2:4] * (self.win_size / 2))
            info["click_pos"] = cp
            dists = np.linalg.norm(self.targets_pos - cp, axis=1)
            min_dist = np.min(dists)
            info["dist"] = min_dist
            if min_dist < 85:
                self.targets_pos[np.argmin(dists)] = np.random.uniform(0, self.map_size, 2)
                info["hit"] = True
        return obs, 0, False, self.current_step >= self.max_steps, info

class DemoApp:
    def __init__(self, model_filename):
        pygame.init()
        m_path = os.path.join(BASE_DIR, model_filename)
        self.model = PPO.load(m_path)
        self.env = AdvancedHunterEnv()
        
        # 增大視窗高度以容納下方數據區
        self.screen_w, self.screen_h = 1300, 950
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(f"AI 獵殺效能驗證展示 - {model_filename}")
        
        self.font_title = pygame.font.SysFont("microsoftjhenghei", 26, bold=True)
        self.font_stat = pygame.font.SysFont("microsoftjhenghei", 20, bold=True)
        self.font_log = pygame.font.SysFont("consolas", 16)
        self.clock = pygame.time.Clock()
        
        # 數據統計變數
        self.total_hits = 0
        self.total_tries = 0
        self.hit_history = deque(maxlen=8) # 記錄最近 8 次獵殺紀錄
        self.start_time = time.time()

    def run(self):
        obs = self.env.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, truncated, info = self.env.step(action)
            if truncated: obs = self.env.reset()
            
            if info["tried"]:
                self.total_tries += 1
                if info["hit"]:
                    self.total_hits += 1
                    res_msg = f"[SUCCESS] Hit! Error: {info['dist']:.1f}px | Pos: ({int(info['click_pos'][0])},{int(info['click_pos'][1])})"
                    self.hit_history.appendleft(res_msg)
                elif self.total_tries % 15 == 0: # 限制點空日誌的頻率
                    res_msg = f"[ MISS  ] Try! Error: {info['dist']:.1f}px"
                    self.hit_history.appendleft(res_msg)

            # --- 繪製背景 ---
            self.screen.fill((20, 20, 20))
            
            # --- 渲染視覺區域 ---
            # 1. 左側：局部視野
            pygame.draw.rect(self.screen, (40, 40, 40), (50, 80, 600, 600))
            v_p, w_s = self.env.view_pos, self.env.win_size
            for t_p in self.env.targets_pos:
                rx, ry = (t_p[0]-(v_p[0]-400))*0.75, (t_p[1]-(v_p[1]-400))*0.75
                if 0<=rx<=600 and 0<=ry<=600: pygame.draw.circle(self.screen, (50,150,255), (50+int(rx), 80+int(ry)), 10)
            
            if info["tried"] and info["click_pos"] is not None:
                cx, cy = (info["click_pos"][0]-(v_p[0]-400))*0.75, (info["click_pos"][1]-(v_p[1]-400))*0.75
                c = (255, 50, 50) if info["hit"] else (120, 120, 120)
                pygame.draw.line(self.screen, c, (50+int(cx)-20, 80+int(cy)-20), (50+int(cx)+20, 80+int(cy)+20), 4)
                pygame.draw.line(self.screen, c, (50+int(cx)+20, 80+int(cy)-20), (50+int(cx)-20, 80+int(cy)+20), 4)

            # 2. 右側：全域圖
            pygame.draw.rect(self.screen, (10, 10, 10), (700, 80, 550, 550))
            for t_p in self.env.targets_pos: pygame.draw.circle(self.screen, (80, 120, 180), (700+int(t_p[0]*0.055), 80+int(t_p[1]*0.055)), 2)
            pygame.draw.rect(self.screen, (255, 255, 255), (700+int((v_p[0]-400)*0.055), 80+int((v_p[1]-400)*0.055), 44, 44), 1)

            # --- 渲染數據面板 (底部優化) ---
            # 總體效能統計
            accuracy = (self.total_hits / self.total_tries * 100) if self.total_tries > 0 else 0
            elapsed = time.time() - self.start_time
            efficiency = (self.total_hits / (elapsed / 60)) if elapsed > 0 else 0
            
            stat_y = 700
            pygame.draw.line(self.screen, (100, 100, 100), (50, stat_y), (1250, stat_y), 2)
            self.screen.blit(self.font_stat.render(f"【總訓練效能評分】", True, (255, 215, 0)), (50, stat_y + 15))
            self.screen.blit(self.font_stat.render(f"累計獵殺: {self.total_hits} 次", True, (0, 255, 0)), (50, stat_y + 50))
            self.screen.blit(self.font_stat.render(f"點擊準確率: {accuracy:.1f}%", True, (255, 255, 255)), (300, stat_y + 50))
            self.screen.blit(self.font_stat.render(f"每分鐘期望獵殺數 (EHK): {efficiency:.1f}", True, (0, 200, 255)), (600, stat_y + 50))

            # 即時獵殺日誌區
            self.screen.blit(self.font_stat.render(f"【即時獵殺數據紀錄】", True, (200, 200, 200)), (50, stat_y + 100))
            log_y = stat_y + 130
            for i, log in enumerate(self.hit_history):
                color = (255, 80, 80) if "SUCCESS" in log else (150, 150, 150)
                self.screen.blit(self.font_log.render(log, True, color), (60, log_y + i*22))

            # 標題
            self.screen.blit(self.font_title.render("AI 高階觀察者系統：兩大核心互動機制驗證", True, (255, 255, 255)), (50, 25))

            pygame.display.flip()
            self.clock.tick(25) # 【優化】降至 25 FPS，播放速度更慢更清楚

if __name__ == "__main__":
    import time
    m_path = "hunter_latest.zip"
    if os.path.exists(os.path.join(BASE_DIR, m_path)):
        DemoApp(m_path).run()
    else:
        print("[錯誤] 找不到訓練好的模型檔案。")
