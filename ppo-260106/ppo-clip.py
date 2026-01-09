import torch
from torch import nn
import torch.nn.functional as F
import tyro
from dataclasses import dataclass
from datetime import datetime
import os
import gym
import random
import numpy as np
import sys

# 将项目根目录加入 sys.path，避免相对导入错误
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rl_utils import train_on_policy_agent



class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.mu(x)) # tanh输出范围[-1, 1]
        std = F.softplus(self.std(x)) + 1e-5  # softplus(x) = ln(1+e^x), ensures std is positive
        return mu, std

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPO:
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.actor = ActorNet(state_dim, action_dim, action_bound, args.hidden_dim).to(args.device)
        self.critic = CriticNet(state_dim, args.hidden_dim).to(args.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.device = args.device
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.clip_eps = args.clip_eps
        self.lr_epochs = args.lr_epochs

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std) # ppo需要pi(a|s)/pi_old(a|s), 故输出是动作的概率分布
        action = dist.sample()
        return [action.item()] # 参考代码中，返回list

    def update(self, batch_dict):
        states = torch.tensor(batch_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch_dict['actions'], dtype=torch.float).view(-1,1).to(self.device) # shape
        rewards = torch.tensor(batch_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device) # shape
        next_states = torch.tensor(batch_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(batch_dict['dones'], dtype=torch.float).view(-1,1).to(self.device) # shape
        rewards = (rewards + 8.0) / 8.0 # 加此行，收敛速度显著提高且稳定！

        # ppo用优势函数A = R + gamma * V(s') - V(s)代替Q(s, a)
        # 然后计算累计优势 A_t + gamma * lmbda * A_{t+1} + (gamma * lmbda)^2 * A_{t+2} + ...
        # 累计优势作为最终的估计优势，参与到loss和BP中
        state_values_tgt = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        state_advantages = state_values_tgt - self.critic(states)
        discount_advantages = []
        discount_adv = 0.0
        for adv in state_advantages.detach().cpu().numpy()[::-1]:
            discount_adv = adv + self.gamma * self.lmbda * discount_adv
            discount_advantages.append(discount_adv)
        discount_advantages.reverse()
        discount_advantages = torch.tensor(discount_advantages, dtype=torch.float).to(self.device)

        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = dist.log_prob(actions)

        # ppo的轨迹要多次利用，多次更新
        for _ in range(self.lr_epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * discount_advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * discount_advantages
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), state_values_tgt.detach()))

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

@dataclass
class Args:
    wandb_project: str = 'Policy Based RL'
    "wandb project name"
    exp_para: str = 'base'
    "The parameter being experimented on"
    exp_name: str = f"{os.path.basename(__file__).rstrip('.py')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    "wandb experiment name"
    seed: int = 0
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_id: str = 'Pendulum-v1'
    total_episodes: int = 2000
    lr_epochs: int = 10
    "number of learning epochs per update"
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 5e-3
    gamma: float = 0.9
    lmbda: float = 0.9
    clip_eps: float = 0.2
    "ppo clip epsilon"

if __name__ == '__main__':
    args = tyro.cli(Args)
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = PPO(state_dim, action_dim, action_bound, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    return_list = train_on_policy_agent(env, agent, args.total_episodes, args)

