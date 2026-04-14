import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import os
import pygame
import sys
import datetime
from collections import deque
import time

# --- 設定固定路徑 ---
BASE_DIR = r"__File__"
DEMO_DIR = os.path.join(BASE_DIR, "human_demo")
os.makedirs(DEMO_DIR, exist_ok=True)

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
        self.targets_vel[self.targets_pos <= 0] *= -1
        self.targets_vel[self.targets_pos >= self.map_size] *= -1
        obs = self._get_obs()
        reward = -0.01 
        info = {"hit": False, "click_pos": None}
        if action[4] > 0:
            click_pos = self.view_pos + (action[2:4] * (self.win_size / 2))
            info["click_pos"] = click_pos
            # 檢查是否擊中目前的序號目標
            dist_to_current = np.linalg.norm(self.targets_pos[self.current_target_idx] - click_pos)
            if dist_to_current < 65:
                info["hit"] = True
                self.targets_pos[self.current_target_idx] = np.random.uniform(0, self.map_size, 2)
                self.current_target_idx = (self.current_target_idx + 1) % self.num_targets
        return obs, reward, False, self.current_step >= self.max_steps, info

class InteractiveVisualizer:
    def __init__(self, model_filename):
        pygame.init()
        self.env = AdvancedHunterEnv()
        m_path = os.path.join(BASE_DIR, model_filename) if model_filename else None
        self.model = PPO.load(m_path) if m_path and os.path.exists(m_path) else None
        
        # 視窗尺寸同步
        self.screen_w, self.screen_h = 1300, 950
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("互動錄製與展示 (數據保存版)")
        
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("microsoftjhenghei", 26, bold=True)
        self.font_stat = pygame.font.SysFont("microsoftjhenghei", 20, bold=True)
        self.font_log = pygame.font.SysFont("consolas", 16)
        
        self.mode = "HUMAN" 
        self.recording = []
        self.total_hits = 0
        self.total_tries = 0
        self.hit_history = deque(maxlen=8)
        self.start_time = time.time()
        self.obs, _ = self.env.reset()
        self.mouse_clicked = False # 用於追蹤單次點擊事件

    def get_human_action(self, click_event=False):
        keys = pygame.key.get_pressed()
        mp, mc = pygame.mouse.get_pos(), pygame.mouse.get_pressed()
        dx, dy = 0, 0
        
        # 1. 處理移動邏輯 (WASD 或 全域圖引導)
        v_p = self.env.view_pos
        # 如果有點擊事件，當前幀停止移動以確保準投，否則繼續跟隨
        if not click_event and 700 <= mp[0] <= 1250 and 80 <= mp[1] <= 630:
            target_wx = (mp[0] - 700) / 0.055
            target_wy = (mp[1] - 80) / 0.055
            dx = np.clip((target_wx - v_p[0]) / (self.env.view_speed / 5), -1.0, 1.0)
            dy = np.clip((target_wy - v_p[1]) / (self.env.view_speed / 5), -1.0, 1.0)
        else:
            if keys[pygame.K_w]: dy = -1
            if keys[pygame.K_s]: dy = 1
            if keys[pygame.K_a]: dx = -1
            if keys[pygame.K_d]: dx = 1
        
        # 2. 處理點擊邏輯
        ct, cx, cy = -1, 0, 0
        if click_event:
            ct = 1
            # 左側局部視野點擊
            if 50 <= mp[0] <= 650 and 80 <= mp[1] <= 680:
                cx, cy = (mp[0]-350)/300.0, (mp[1]-380)/300.0
            
            # 右側全域圖點擊
            elif 700 <= mp[0] <= 1250 and 80 <= mp[1] <= 630:
                wx = (mp[0] - 700) / 0.055
                wy = (mp[1] - 80) / 0.055
                half_win = self.env.win_size / 2
                cx = (wx - v_p[0]) / half_win
                cy = (wy - v_p[1]) / half_win
                
        return np.array([dx, dy, cx, cy, ct], dtype=np.float32)

    def run(self):
        while True:
            click_now = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.save_recording(); pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    click_now = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if self.model: self.mode = "AI" if self.mode == "HUMAN" else "HUMAN"
                    self.env.reset(); self.total_hits = 0; self.total_tries = 0; self.hit_history.clear()
            
            action = self.model.predict(self.obs, deterministic=True)[0] if self.mode == "AI" else self.get_human_action(click_now)
            tried = action[4] > -0.1
            if tried: self.total_tries += 1
            if self.mode == "HUMAN": self.recording.append({"obs": self.obs, "act": action})
            
            self.obs, _, _, truncated, info = self.env.step(action)
            if truncated: self.obs, _ = self.env.reset()
            
            if tried:
                if info["hit"]:
                    self.total_hits += 1
                    # 在 recorder 中，剛剛 hit 後 index 已經步進了，所以顯示目前剛被擊中的那個
                    target_num = self.env.current_target_idx if self.env.current_target_idx > 0 else self.env.num_targets
                    res_msg = f"[SUCCESS] Hit #{target_num}! Pos: ({int(info['click_pos'][0])},{int(info['click_pos'][1])})"
                    self.hit_history.appendleft(res_msg)
                else:
                    res_msg = f"[ MISS  ] Try! Target #{self.env.current_target_idx+1}"
                    self.hit_history.appendleft(res_msg)

            # --- 繪製背景 (同步配色) ---
            self.screen.fill((20, 20, 20))
            
            # --- 渲染視覺區域 (同步佈局) ---
            v_p, w_s = self.env.view_pos, self.env.win_size
            seq_font = pygame.font.SysFont("arial", 14, bold=True)
            
            # 1. 左側：局部視野
            pygame.draw.rect(self.screen, (40, 40, 40), (50, 80, 600, 600))
            for i, t_p in enumerate(self.env.targets_pos):
                rx, ry = (t_p[0]-(v_p[0]-400))*0.75, (t_p[1]-(v_p[1]-400))*0.75
                if 0<=rx<=600 and 0<=ry<=600:
                    is_current = (i == self.env.current_target_idx)
                    color = (255, 255, 0) if is_current else (50, 150, 255)
                    center = (50+int(rx), 80+int(ry))
                    radius = 16 if is_current else 14
                    pygame.draw.circle(self.screen, color, center, radius)
                    text_color = (255, 0, 0) if is_current else (255, 255, 255)
                    text_surf = seq_font.render(str(i+1), True, text_color)
                    text_rect = text_surf.get_rect(center=center)
                    self.screen.blit(text_surf, text_rect)
            
            if tried and info["click_pos"] is not None:
                # 1. 左側顯示
                cx_l, cy_l = (info["click_pos"][0]-(v_p[0]-400))*0.75, (info["click_pos"][1]-(v_p[1]-400))*0.75
                # 2. 右側顯示
                cx_r, cy_r = (info["click_pos"][0])*0.055, (info["click_pos"][1])*0.055
                
                c = (255, 50, 50) if info["hit"] else (120, 120, 120)
                # 左側繪製
                if 0<=cx_l<=600 and 0<=cy_l<=600:
                    pygame.draw.line(self.screen, c, (50+int(cx_l)-20, 80+int(cy_l)-20), (50+int(cx_l)+20, 80+int(cy_l)+20), 4)
                    pygame.draw.line(self.screen, c, (50+int(cx_l)+20, 80+int(cy_l)-20), (50+int(cx_l)-20, 80+int(cy_l)+20), 4)
                # 右側繪製
                pygame.draw.line(self.screen, c, (700+int(cx_r)-10, 80+int(cy_r)-10), (700+int(cx_r)+10, 80+int(cy_r)+10), 2)
                pygame.draw.line(self.screen, c, (700+int(cx_r)+10, 80+int(cy_r)-10), (700+int(cx_r)-10, 80+int(cy_r)+10), 2)

            # 2. 右側：全域圖
            pygame.draw.rect(self.screen, (10, 10, 10), (700, 80, 550, 550))
            for i, t_p in enumerate(self.env.targets_pos):
                is_current = (i == self.env.current_target_idx)
                color = (255, 255, 0) if is_current else (80, 120, 180)
                center = (700+int(t_p[0]*0.055), 80+int(t_p[1]*0.055))
                radius = 12 if is_current else 10
                pygame.draw.circle(self.screen, color, center, radius)
                text_color = (255, 0, 0) if is_current else (255, 255, 255)
                text_surf = seq_font.render(str(i+1), True, text_color)
                text_rect = text_surf.get_rect(center=center)
                self.screen.blit(text_surf, text_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), (700+int((v_p[0]-400)*0.055), 80+int((v_p[1]-400)*0.055), 44, 44), 1)

            # --- 渲染數據面板 (同步展示) ---
            accuracy = (self.total_hits / self.total_tries * 100) if self.total_tries > 0 else 0
            elapsed = time.time() - self.start_time
            efficiency = (self.total_hits / (elapsed / 60)) if elapsed > 0 else 0
            
            stat_y = 700
            pygame.draw.line(self.screen, (100, 100, 100), (50, stat_y), (1250, stat_y), 2)
            self.screen.blit(self.font_stat.render(f"【錄製器效能與數據】", True, (255, 215, 0)), (50, stat_y + 15))
            self.screen.blit(self.font_stat.render(f"目前模式: {self.mode}", True, (255, 255, 255)), (300, stat_y + 15))
            self.screen.blit(self.font_stat.render(f"已錄製步數: {len(self.recording)}", True, (0, 255, 255)), (600, stat_y + 15))
            
            self.screen.blit(self.font_stat.render(f"累計獵殺: {self.total_hits} 次", True, (0, 255, 0)), (50, stat_y + 50))
            self.screen.blit(self.font_stat.render(f"點擊準確率: {accuracy:.1f}%", True, (255, 255, 255)), (300, stat_y + 50))
            self.screen.blit(self.font_stat.render(f"每分鐘期望獵殺數 (EHK): {efficiency:.1f}", True, (0, 200, 255)), (600, stat_y + 50))

            # 即時日誌區
            self.screen.blit(self.font_stat.render(f"【錄製數據紀錄日誌】", True, (200, 200, 200)), (50, stat_y + 100))
            log_y = stat_y + 130
            for i, log in enumerate(self.hit_history):
                color = (255, 80, 80) if "SUCCESS" in log else (150, 150, 150)
                self.screen.blit(self.font_log.render(log, True, color), (60, log_y + i*22))

            # 標題
            self.screen.blit(self.font_title.render("人類觀察者錄製系統：數據採集與循序引導驗證", True, (255, 255, 255)), (50, 25))

            # --- 繪製靶心跟隨滑鼠 (新增邏輯) ---
            mx, my = pygame.mouse.get_pos()
            mouse_c = (200, 200, 200) # 預覽靶心顏色 (淺灰色)
            
            # 1. 檢查是否在左側視野
            if 50 <= mx <= 650 and 80 <= my <= 680:
                pygame.draw.line(self.screen, mouse_c, (mx-20, my), (mx+20, my), 2)
                pygame.draw.line(self.screen, mouse_c, (mx, my-20), (mx, my+20), 2)
                pygame.draw.circle(self.screen, mouse_c, (mx, my), 15, 1)
            
            # 2. 檢查是否在右側全域圖
            elif 700 <= mx <= 1250 and 80 <= my <= 630:
                pygame.draw.line(self.screen, mouse_c, (mx-10, my), (mx+10, my), 1)
                pygame.draw.line(self.screen, mouse_c, (mx, my-10), (mx, my+10), 1)
                pygame.draw.circle(self.screen, mouse_c, (mx, my), 8, 1)

            pygame.display.flip()
            self.clock.tick(25) 

    def save_recording(self):
        if len(self.recording) > 100:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_demo_{timestamp}.npz"
            save_path = os.path.join(DEMO_DIR, filename)
            np.savez(save_path, obs=np.array([d["obs"] for d in self.recording]), actions=np.array([d["act"] for d in self.recording]))
            print(f"\n[系統] 錄製成功！檔案已儲存至：{save_path}")

if __name__ == "__main__":
    InteractiveVisualizer("hunter_latest.zip").run()
