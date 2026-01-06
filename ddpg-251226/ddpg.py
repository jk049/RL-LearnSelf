# ddpg实现，其中actor和critic都用了target net

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import tyro
import gym
import random
import sys
from datetime import datetime

# 将项目根目录加入 sys.path，避免相对导入错误
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rl_utils import ReplayBuffer, train_off_policy_agent

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.action_bound = action_bound
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x * self.action_bound
        return x

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.state_fc = nn.Linear(state_dim, 3)
        self.fc1 = nn.Linear(3 + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        s_out = F.relu(self.state_fc(state))
        x = torch.cat([s_out, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.device = args.device
        self.sigma = args.sigma # noise std, N(0, sigma)
        self.gamma = args.gamma # discount factor
        self.tau = args.tau # target net update rate
        self.action_dim = action_dim

        self.actor = PolicyNet(state_dim, action_dim, action_bound).to(args.device)
        self.actor_target = PolicyNet(state_dim, action_dim, action_bound).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = CriticNet(state_dim, action_dim).to(args.device)
        self.critic_target = CriticNet(state_dim, action_dim).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    # a = net(s)
    # take_action要注意加噪声
    def take_action(self, state):
        # 此处的state是一个环境step的state，没有batch的维度，且不是tensor，需转换
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).item()
        # action = action + N(0, sigma), 噪声的简便设计，原论文是用Ornstein-Uhlenbeck 
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        action = action + noise
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # actor_loss = -Q(s, a); critic_loss = mse(Q(s,a), r+gamma*Q_(s',a'))
    def update(self, batch_dict):
        states = torch.tensor(batch_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(batch_dict['next_states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(batch_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        dones = torch.tensor(batch_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        qsa = self.critic(states, self.actor(states))
        actor_loss = torch.mean(-qsa)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        next_actions = self.actor_target(next_states)
        target_qsa = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), target_qsa))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

@dataclass
class Args:
    exp_para: str = 'tau'
    exp_name: str = f"{os.path.basename(__file__).rstrip('.py')}-{exp_para}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    seed: int = 0
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb_project: str = 'Policy Based RL'
    capture_video: bool = True
    save_model: bool = True
    save_code: bool = True
    env_id: str = 'Pendulum-v1'
    num_episodes: int = 300
    actor_lr: float = 3e-4
    critic_lr: float = 3e-3
    buffer_capacity: int = 10000
    minimal_size: int = 1000 # replay buffer最小容量，达到后才开始更新网络
    tau: float = 0.004
    batch_size: int = 64
    gamma: float = 0.98
    sigma: float = 0.01 # action noise std, N(0, sigma)

if __name__ == '__main__':
    args = tyro.cli(Args)
    env = gym.make(args.env_id)
    replay_buffer = ReplayBuffer(args.buffer_capacity)
    state_dim = env.observation_space.shape[0]
    aciton_dim = env.action_space.shape[0] # 环境的动作空间为连续空间，非离散，所以是shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPG(state_dim, aciton_dim, action_bound, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    return_list = train_off_policy_agent(env, agent, replay_buffer, args)

    if args.save_model:
        os.makedirs(args.exp_name, exist_ok=True)
        torch.save(agent.actor.state_dict(), os.path.join(args.exp_name, 'actor.pth'))
        torch.save(agent.critic.state_dict(), os.path.join(args.exp_name, 'critic.pth'))



