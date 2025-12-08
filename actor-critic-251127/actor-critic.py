# actor-critic算法，用a2c简化版实现

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
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ActorCritic:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.value_net = ValueNet(state_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma

    def to(self, device):
        self.policy_net.to(device)
        self.value_net.to(device)

    def take_action(self, state):
        act_prob = self.policy_net(state)
        action = torch.distributions.Categorical(act_prob).sample()
        return action

    def learn(self, state, action, reward, next_state, done):
        # Update value network
        value = self.value_net(state)
        next_value = self.value_net(next_state)
        td_target = reward + self.gamma * next_value * (1 - int(done))
        td_error = td_target - value
        value_loss = td_error.pow(2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        log_prob = torch.log(self.policy_net(state)[action])
        policy_loss = -log_prob * td_error.detach()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

def evaluate(model_path, eval_env):
    action_dim = eval_env.action_space.n
    state_dim = eval_env.observation_space.shape[0]
    agent = ActorCritic(state_dim, action_dim)          
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    agent.policy_net.load_state_dict(state_dict['policy_net'])
    agent.value_net.load_state_dict(state_dict['value_net'])
    agent.policy_net.eval()
    agent.value_net.eval()

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
    env_id: str = "CartPole-v0"
    track: bool = True
    project: str = "Actor-Critic"
    exp: str = "base"       
    total_episodes: int = 1000
    lr: float = 1e-3
    gamma: float = 0.99

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.track:
        wandb.init(
            project=args.project,
            name=f"{args.exp}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(state_dim, action_dim, args.lr, args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    episode_rewards = []

    with tqdm(total=args.total_episodes) as pbar:
        for episode in range(args.total_episodes):
            state, info = env.reset()
            done = False
            ep_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action = agent.take_action(state_tensor)
                next_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

                agent.learn(state_tensor, action, reward_tensor, next_state_tensor, done)

                state = next_state
                ep_reward += reward

            episode_rewards.append(ep_reward)
            pbar.set_description(f"Episode {episode+1}, Reward: {ep_reward:.2f}")
            pbar.update(1)

            if args.track:
                wandb.log({"episode": episode + 1, "reward": ep_reward})

    # Save the model
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'value_net': agent.value_net.state_dict()
    }, f"actor_critic_{args.env_id}.pth")

    env.close()
    eval_env.close()
    if args.track:
        wandb.finish()
    print(f"Model saved to actor_critic_{args.env_id}.pth")

    eval_reward = evaluate(f"actor_critic_{args.env_id}.pth", eval_env)
    print(f"Evaluation Reward: {eval_reward:.2f}")
