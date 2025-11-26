# reinforce算法，一种策略梯度算法

import gymnasium as gym
import tyro
import time
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import dataclass
from torch import nn
from datetime import datetime

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # F.relu会出现在模型结构中，模型结构更直观
        x = F.softmax(self.fc2(x), dim=-1) # softmax的入参dim指定在哪个维度上进行softmax计算
        return x

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.net = PolicyNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def to(self, device):
        self.net.to(device)

    def take_action(self, state):
        act_prob = self.net(state)
        action = torch.distributions.Categorical(act_prob).sample()
        return action

    def learn(self, experiences):
        episode_loss = 0
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(experiences['rewards']))):
            log_prob = torch.log(self.net(experiences['states'][i])[experiences['actions'][i]])
            G = experiences['rewards'][i] + self.gamma * G
            loss = -log_prob * G
            episode_loss += loss
            loss.backward()
        self.optimizer.step()
        return episode_loss.item()



def evaluate(model_path, eval_env):
    action_dim = eval_env.action_space.n
    state_dim = eval_env.observation_space.shape[0]
    agent = REINFORCE(state_dim, action_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    agent.net.load_state_dict(state_dict)
    agent.net.eval()
    
    state, info = eval_env.reset()
    done = False
    ep_reward = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = eval_env.step(action.item())
        done = terminated or truncated
        ep_reward += reward
        state = next_state
    return ep_reward

@dataclass
class Args:
    seed: int = 1
    track: bool = True
    project: str = "REINFORCE"
    exp: str = "base"
    env_id: str = "CartPole-v1"
    total_episodes: int = 1000
    lr: float = 1e-3
    gamma: float = 0.99


if __name__ == "__main__":
    args = tyro.cli(Args)
    env = gym.make(args.env_id)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_name = f"{args.env_id}-{args.exp}-{ts}"
    if args.track:
        wandb.init(project=args.project, name=exp_name, sync_tensorboard=True, save_code=True, monitor_gym=True, config=vars(args))
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{exp_name}", disable_logger=True)
    writer = SummaryWriter(f"runs/{exp_name}")

    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = REINFORCE(state_dim, action_dim, args.lr, args.gamma)
    agent.to(device)
    episode_rewards = []
    print('[INFO]:', agent.net)

    with tqdm(total=args.total_episodes) as pbar:
        for episode in range(args.total_episodes):
            episode_reward = 0
            exp_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward
                done = terminated or truncated
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                exp_dict['states'].append(state.clone())
                exp_dict['actions'].append(action)
                exp_dict['rewards'].append(reward)
                exp_dict['next_states'].append(next_state.clone())
                exp_dict['dones'].append(done)
                state = next_state
            
            episode_rewards.append(episode_reward)
            loss = agent.learn(exp_dict)
            writer.add_scalar('charts/episode_reward', episode_reward, episode)
            writer.add_scalar('charts/loss', loss, episode)
            pbar.set_description(f'Episode {episode+1}')
            pbar.set_postfix({'episode_reward': episode_reward, 'avg_reward': np.mean(episode_rewards[-10:])})
            pbar.update(1)
    
    save_path = f'models/{exp_name}.pth'
    torch.save(agent.net.state_dict(), save_path)
    eval_reward = evaluate(save_path, env)
    writer.add_scalar('charts/eval_reward', eval_reward, 0)
    print(f"[DONE] Model saved to {save_path}, Eval Reward: {eval_reward:.2f}")

    env.close()
    writer.close()
    if args.track:
        wandb.finish()











