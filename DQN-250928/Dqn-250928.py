#!/usr/bin/env python3
"""
DQN算法
【问题描述】：用DQN解决Pong问题
【算法原理】：用NN近似Q(s,a), 辅以经验回放和目标网络
【伪代码】：
    定义并初始化replay buffer: replay_buffer<-np.array(shape=10000, dtype=(s, a, r, s', done))  
    定义并初始化QNet
    定义并初始化dqn agent: take_action(), learn()
    for e = 1->MAX_EPISODES:
        环境复位：s <- env.reset
        while not done:
            agent选择动作：a <- agent.take_action(s)
            环境执行动作：s', r, done, info <- env.step(a)
            replay buffer 添加新样本(s, a, r, s', done)
            s <- s'
            if replay buffer样本数大于阈值：
                随机采样训练样本：batch_sample <- replay_buffer.sample()
                agent 学习：dqn agent learn(
                    训练网络选择动作：Q_sa <- agent.train_net(s, a)
                    目标网络选择动作：target_Qsa <- R + gamma * max_a' agent.target_net(s')
                    计算损失：loss = MSE(Q_sa, target_Qsa)
                    反向传播，更新训练网络：backprop loss
                    每隔C步，target_net<-train_net参数 
                )

        end while
    end for
【实验结果】：
    1. RB容量影响重大，大容量的训练速度快很多
    2. PER use less samples to achieve same performance, but each step is slower

"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from lib import replay_buffer
from lib import model
from lib import env_wrappers

class DqnAgent:
    def __init__(self, obs_shape, action_space, args):
        self.epsilon = args.epsilon_start
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.gamma = args.gamma
        self.eps_decay_slopes = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_steps
        self.PER = args.PER
        self.double = args.double
        self.noisy = args.noisy
        self.categorical = args.categorical
        self.atoms = args.atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atom_values = torch.linspace(self.vmin, self.vmax, self.atoms).to(self.device)  # shape: [atoms,]
        suffixes = [
            '_double' if args.double else '',
            '_dueling' if args.dueling else '',
            '_PER' if args.PER else '',
            '_noisy' if args.noisy else '',
            '_categorical' if args.categorical else ''
        ]
        self.model_save_name = f"{args.env}{''.join(suffixes)}.pth"
        self.train_net = model.QNet(obs_shape, action_space, args).to(self.device)
        self.target_net = model.QNet(obs_shape, action_space, args).to(self.device)
        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=args.lr)
        print(self.train_net)
        if args.resume:
            resume_path = args.resume_path if args.resume_path is not None else args.save_path + self.model_save_name
            self.train_net.load_state_dict(torch.load(resume_path, map_location=self.device))
            print(f"[INFO] Resumed model from {resume_path}")
    
    def take_action(self, state):
        if not self.noisy and rng.random() < self.epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                if self.categorical:
                    qsa_probs = self.train_net(state.unsqueeze(0))
                    atoms = self.atom_values.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, atoms]
                    qsa = (qsa_probs * atoms).sum(-1)  # shape: [1, action_space]
                    action = qsa.argmax().item()
                else:
                    action = self.train_net(state.unsqueeze(0)).argmax().item() # (4,84,84)->(1,4,84,84)
        self.epsilon = 0 if self.noisy else max(args.epsilon_end, self.epsilon - self.eps_decay_slopes)
        return action
                                           
    def dist_projection(self, dist, rewards, dones):
        delta = (self.vmax - self.vmin) / (self.atoms - 1)
        rewards = rewards.unsqueeze(1).expand(-1, self.atoms)      # [B, atoms]
        dones = dones.unsqueeze(1).expand(-1, self.atoms)          # [B, atoms]
        atom_values = self.atom_values.unsqueeze(0)                # [1, atoms]
        discount_rewards = rewards + self.gamma * (~dones).float() * atom_values  # [B, atoms]
        discount_rewards = discount_rewards.clamp(self.vmin, self.vmax)
        b = (discount_rewards - self.vmin) / delta                 # [B, atoms]
        l = b.floor().long()                                       # [B, atoms]
        u = b.ceil().long()                                        # [B, atoms]
        proj_dist = torch.zeros_like(dist)                         # [B, atoms]

        dist_l = (u -b + (l == b).float()) * dist      # l==b考虑了整点的情况
        dist_u = (b - l) * dist

        # scatter_add_: self[i, index[i,j]] += src[i,j] dim=1
        proj_dist.scatter_add_(1, l, dist_l)
        proj_dist.scatter_add_(1, u, dist_u)
        return proj_dist

    def get_train_qsa(self, state_batch, action_batch):
        train_q_values = self.train_net(state_batch) # shape: [B, action_space]
        if self.categorical:
            index = action_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.atoms) # shape: [B, 1, atoms]
            train_q_sa = train_q_values.gather(dim=1, index=index).squeeze(1) # shape: [B, atoms]
        else:
            train_q_sa = train_q_values.gather(dim=1, index=action_batch.unsqueeze(-1)).squeeze(-1) # [B,]
        return train_q_sa
    
    def get_tgt_qsa(self, next_state_batch, reward_batch, done_batch):
        with torch.no_grad():
            if self.categorical:
                if self.double:
                    next_qsa_probs = self.train_net(next_state_batch) # [B, act_num, atoms]]
                else:
                    next_qsa_probs = self.target_net(next_state_batch) # [B, act_num, atoms]]

                atoms = self.atom_values.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, atoms]
                next_action = (next_qsa_probs * atoms).sum(-1).argmax(1)  # [B,]
                index = next_action.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.atoms)  # [B, 1, atoms]
                target_q_next = self.target_net(next_state_batch).gather(dim=1, index=index).squeeze(1)  # [B, atoms]
                target_q_sa = self.dist_projection(target_q_next, reward_batch, done_batch)  # [B, atoms]
            else:
                if self.double:
                    next_action = self.train_net(next_state_batch).argmax(1)  # [B,]
                    target_q_next = self.target_net(next_state_batch).gather(dim=1, index=next_action.unsqueeze(-1)).squeeze(-1)  # [B,]
                else:
                    target_q_next = self.target_net(next_state_batch).max(1)[0]  # [B,]
                target_q_next *= ~done_batch  # if done, Q(s')=0
                target_q_sa = reward_batch + self.gamma * target_q_next  # [B,]
        return target_q_sa

    def loss(self, tgt_qsa, train_qsa, weights=None):
        if self.categorical:
            # loss = -sum(tgt_p * log(train_p))
            log_train_qsa = torch.log(train_qsa.clamp(1e-5, 1 - 1e-5))  # shape: [B, atoms]
            loss = -(tgt_qsa * log_train_qsa).sum(dim=1) # shape: [B,]
        else:
            loss = 0.5 * (tgt_qsa - train_qsa).pow(2) # shape: [B,]
        if self.PER:
            loss *= weights
        return loss

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch[:5]
        weights = batch[-1] if self.PER else None
        train_q_sa = self.get_train_qsa(states, actions)
        target_q_sa = self.get_tgt_qsa(next_states, rewards, dones)
        loss = self.loss(target_q_sa, train_q_sa, weights)
        loss_mean = loss.mean()

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return loss.detach()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed, default 47')
    parser.add_argument('--cuda', default=True, action='store_true', help='enable cuda')
    parser.add_argument('--env', default='PongNoFrameskip-v4', help='gym environment name, default PongNoFrameskip-v4')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for reward, default 0.99')
    parser.add_argument('--batch_size', type=int, default=32, help='number of transitions sampled from replay buffer, default 32')
    parser.add_argument('--rb_capacity', type=int, default=10000, help='capacity of replay buffer, default 10000')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default 1e-4')
    parser.add_argument('--sync_target_frames', type=int, default=1000, help='number of frames between target network sync, default 1000 frames')
    parser.add_argument('--replay_start_size', type=int, default=10000, help='number of transitions that must be in the replay buffer before starting training, default 10000')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='starting value of epsilon for epsilon-greedy action selection, default 1.0')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='ending value of epsilon for epsilon-greedy action selection, default 0.01')
    parser.add_argument('--epsilon_decay_steps', type=int, default=150000, help='number of steps over which to decay epsilon, default 150000')
    parser.add_argument('--rwd_bound', type=float, default=19.0, help='reward boundary for stopping criterion, default 19.0')
    parser.add_argument('--rwd_max', type=float, default=21.0, help='max reward per episode, default 21.0')
    parser.add_argument('--max_episodes', type=int, default=1000000, help='maximum number of training episodes, default 1000000')
    parser.add_argument('--save_path', type=str, default='./out/', help='path to save the trained model, default ./out/')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from saved model')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load the saved model')
    parser.add_argument('--double', default=False, action='store_true', help='enable double dqn')
    parser.add_argument('--dueling', default=False, action='store_true', help='enable dueling dqn')
    parser.add_argument('--PER', default=False, action='store_true', help='enable prioritized experience replay')
    parser.add_argument('--rb_alpha', type=float, default=0.6, help='alpha parameter for prioritized experience replay, default 0.6')
    parser.add_argument('--rb_beta', type=float, default=0.4, help='beta parameter for prioritized experience replay, default 0.4')
    parser.add_argument('--rb_eps', type=float, default=1e-6, help='epsilon parameter for prioritized experience replay, default 1e-6')
    parser.add_argument('--noisy', default=False, action='store_true', help='enable noisy dqn')
    parser.add_argument('--categorical', default=False, action='store_true', help='enable categorical dqn')
    parser.add_argument('--atoms', type=int, default=51, help='number of atoms for categorical dqn, default 51')
    parser.add_argument('--vmin', type=float, default=-10.0, help='minimum value for categorical dqn, default -10.0')
    parser.add_argument('--vmax', type=float, default=10.0, help='maximum value for categorical dqn, default 10.0')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    env = env_wrappers.make_env(args.env)
    if args.PER:
        rb = replay_buffer.PrioRB(args.rb_capacity, env.observation_space.shape, device, 
                                  alpha=args.rb_alpha, beta=args.rb_beta, eps=args.rb_eps)
    else:
        rb = replay_buffer.RB(args.rb_capacity, env.observation_space.shape, device)
    agent = DqnAgent(env.observation_space.shape, env.action_space.n, args)

    frame_cnt = 0
    best_reward = -float('inf')
    episode_rewards = []
    with tqdm(total=args.rwd_max, initial=0, desc='Mean Reward ') as pbar:
        for episode in range(args.max_episodes):
            episode_start_time = time.time()
            episode_start_frame = frame_cnt
            state = env.reset()
            state = torch.from_numpy(np.array(state)).to(device=device, dtype=torch.float32) # shape:(4,84,84)
            done = False
            episode_reward = 0
            while not done:
                if agent.noisy:
                    agent.train_net.reset_noise()
                action = agent.take_action(state) 
                next_state, reward, done, info = env.step(action)
                next_state = torch.from_numpy(np.array(next_state)).to(device=device, dtype=torch.float32)
                rb.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                frame_cnt += 1
                if frame_cnt < args.replay_start_size:
                    continue
                if frame_cnt % args.sync_target_frames == 0:
                    agent.target_net.load_state_dict(agent.train_net.state_dict())
                
                batch = rb.sample(args.batch_size)
                loss = agent.learn(batch)
                if args.PER:
                    rb.update_prio(loss, batch[5]) # batch = (s, a, r, s', done, indices, weights)

            episode_rewards.append(episode_reward)
            rwd_mean = np.mean(episode_rewards[-10:])
            episode_end_time = time.time()
            fps = (frame_cnt - episode_start_frame) / (episode_end_time - episode_start_time)
            if rwd_mean > best_reward:
                best_reward = rwd_mean 
                torch.save(agent.train_net.state_dict(), args.save_path + agent.model_save_name)
            if rwd_mean >= args.rwd_bound:
                print(f"Solved in {episode} episodes!")
                break
            progress = round(rwd_mean, 2)
            pbar.set_description(f'Episode {episode}')
            pbar.set_postfix({'Best Rwd': f'{best_reward:.2f}', 'Eps': f'{agent.epsilon:.2f}', 'FPS': f'{fps:.1f}'})
            pbar.update(progress - pbar.n)


    # plot mean reward
    plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (100 episodes)')
    plt.title('DQN on ' + args.env)
    plt.show()

    env.close()
