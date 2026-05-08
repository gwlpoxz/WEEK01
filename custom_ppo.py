import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from model import ActorCritic

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, device):
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.is_terminals = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.ptr = 0
        self.max_size = buffer_size
        
    def clear(self):
        self.ptr = 0
        
    def add(self, obs, action, logprob, reward, is_terminal, value):
        self.obs[self.ptr] = torch.tensor(obs, dtype=torch.float32)
        self.actions[self.ptr] = action.detach()
        self.logprobs[self.ptr] = logprob.detach()
        self.rewards[self.ptr] = float(reward)
        self.is_terminals[self.ptr] = float(is_terminal)
        self.values[self.ptr] = value.detach()
        self.ptr += 1

class PPO:
    def __init__(self, env, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, 
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                 target_kl=0.015):
        self.env = env
        self.verbose = verbose
        self.lr = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        
        # 決定要用 CPU 還是 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose > 0:
            print(f"[PPO] 演算法啟動，使用設備: {self.device}")
            
        # 取得環境的觀察值和動作維度
        if hasattr(env, "observation_space"):
            self.obs_dim = env.observation_space.shape[0]
            self.act_dim = env.action_space.shape[0]
        else:
            self.obs_dim = 7
            self.act_dim = 5
            
        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        self.buffer = RolloutBuffer(self.n_steps, self.obs_dim, self.act_dim, self.device)
        
        self.num_timesteps = 0

    def predict(self, obs, deterministic=False):
        """ 提供給展示器或評估時使用 (等同 SB3 的 predict) """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action_mean = self.policy.actor_mean(obs_tensor)
                action = torch.clamp(action_mean, -1, 1) # 動作裁減在 -1 到 1 之間
            else:
                action, _, _, _ = self.policy.get_action_and_value(obs_tensor)
                action = torch.clamp(action, -1, 1)
        return action.cpu().numpy()[0], None

    def learn(self, total_timesteps, reset_num_timesteps=True):
        """ PPO 主訓練迴圈 """
        global_step = 0
        obs, _ = self.env.reset()
        
        while global_step < total_timesteps:
            # 1. 收集環境互動資料 (Rollouts)
            self.policy.eval()
            self.buffer.clear()
            for step in range(self.n_steps):
                global_step += 1
                self.num_timesteps += 1
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, logprob, _, value = self.policy.get_action_and_value(obs_tensor)
                
                # 將動作轉為 NumPy 並限制在合法範圍 (-1, 1) 內輸入給環境
                action_np = action.cpu().numpy()[0]
                action_np = np.clip(action_np, -1, 1)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                
                self.buffer.add(obs, action[0], logprob[0], reward, done, value[0])
                
                obs = next_obs
                if done:
                    obs, _ = self.env.reset()
                    
            # 2. 使用 GAE 計算優勢值 (Advantages)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                next_value = self.policy.get_value(obs_tensor).squeeze(-1)
                
            advantages = torch.zeros_like(self.buffer.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.buffer.is_terminals[t+1]
                    nextvalues = self.buffer.values[t+1]
                delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + self.buffer.values
            
            # 3. 更新神經網路 (Policy and Value Networks)
            self.policy.train()
            b_obs = self.buffer.obs
            b_actions = self.buffer.actions
            b_logprobs = self.buffer.logprobs
            b_advantages = advantages
            b_returns = returns
            b_values = self.buffer.values

            # 優勢值常態化有助於訓練穩定
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            inds = np.arange(self.n_steps)
            for epoch in range(self.n_epochs):
                np.random.shuffle(inds)
                approx_kl_divs = []
                for start in range(0, self.n_steps, self.batch_size):
                    end = start + self.batch_size
                    mbinds = inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mbinds], b_actions[mbinds])
                    logratio = newlogprob - b_logprobs[mbinds]
                    ratio = logratio.exp()
                    
                    # 計算 Approximate KL Divergence
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        approx_kl_divs.append(approx_kl.item())
                    
                    mb_advantages = b_advantages[mbinds]
                    
                    # Policy Loss (裁減代理目標)
                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    
                    # Value Loss (加入價值函數裁減)
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = F.mse_loss(newvalue, b_returns[mbinds])
                    v_clipped = b_values[mbinds] + torch.clamp(
                        newvalue - b_values[mbinds],
                        -self.clip_range,
                        self.clip_range,
                    )
                    v_loss_clipped = F.mse_loss(v_clipped, b_returns[mbinds])
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                    
                    # Entropy Loss (鼓勵探索)
                    entropy_loss = entropy.mean()
                    
                    # 總損失
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
                
                # KL 散度提早煞車機制 (Early Stopping)
                if self.target_kl is not None:
                    if np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                        if self.verbose > 0:
                            print(f"      -> 提早煞車 (Early stopping) 於 Epoch {epoch}，因 KL 散度過高: {np.mean(approx_kl_divs):.4f}")
                        break
                    
            if self.verbose > 0 and (global_step // self.n_steps) % 5 == 0:
                print(f"[PPO] 進度: {global_step}/{total_timesteps} 步 | Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f} | KL: {np.mean(approx_kl_divs):.4f}")

    def save(self, path):
        # 儲存為 PyTorch .pth 格式
        torch.save(self.policy.state_dict(), path)
        if self.verbose > 0:
            print(f"[PPO] 模型已儲存至 {path}")

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        if os.path.exists(path):
            model.policy.load_state_dict(torch.load(path, map_location=model.device))
            if model.verbose > 0:
                print(f"[PPO] 成功載入手寫模型權重: {path}")
        else:
            print(f"[PPO] 找不到指定的權重檔案: {path}，將使用初始隨機權重。")
        return model
