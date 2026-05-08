import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module): #神經元 MLP架構
    def _init_weights(self, module, gain=np.sqrt(2)):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        # Actor network (策略網路: 決定動作)
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(nn.Tanh())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # 連續動作的標準差是可學習的參數
        self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic network (價值網路: 評估狀態好壞)
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(nn.Tanh())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # 套用正交初始化 (Orthogonal Initialization)
        self.actor_mean.apply(self._init_weights)
        self.critic.apply(self._init_weights)
        
        # 特別設定輸出層的 Gain 值
        self._init_weights(self.actor_mean[-1], gain=0.01)
        self._init_weights(self.critic[-1], gain=1.0)
        
    def get_value(self, x):
        return self.critic(x)

    #PPO 核心邏輯：產出action的核心機制-根據現狀，擲骰子決定下一步怎麼做  
    def get_action_and_value(self, x, action=None): #獲得策略參數
            # 1. 算出理想的動作中心點 (例如：滑鼠想往右移 0.8)
        action_mean = self.actor_mean(x)

            # 2. 算出不確定性 (例如：誤差範圍是 0.1)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        # 3. 建立一個常態分佈 (鐘形曲線)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        # 4. 關鍵動作：從這個常態分佈中「擲骰子 (Sample)」抽取出真正的 action
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
